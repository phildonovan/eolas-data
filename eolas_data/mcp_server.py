"""eolas MCP server — stdio transport for Claude Desktop, Cursor, Grok, etc.

Thin wrapper over :class:`eolas_data.Client`. All HTTP, auth, CDC routing, and
error mapping stay in the Python client; this module only exposes typed tools
for agent hosts.

Run::

    eolas-mcp          # after pip install eolas-data[mcp]
    python -m eolas_data.mcp_server

Auth resolves the same way as the CLI: ``EOLAS_API_KEY`` env var, OS keyring
(``pip install eolas-data[secure]``), or ``~/.eolas/config.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from .client import Client, SyncResult
from .exceptions import EolasError
from .meta import _TABLE_META_KEYS

# Hard caps — never stream multi-million-row tables into model context.
_MCP_GET_ROW_CAP = 500
_MCP_SEARCH_CAP = 50

mcp = FastMCP(
    "eolas",
    instructions=(
        "Tools for the eolas.fyi statistical & geospatial data API (NZ + OECD). "
        "Use eolas_search then eolas_info before fetching unknown dataset names. "
        "For whole datasets prefer eolas_sync (keeps a local file current) or "
        "eolas_download (one-shot bulk file). Use eolas_get only for small slices."
    ),
)

_client: Optional[Client] = None


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client()
    return _client


def _set_client(client: Optional[Client]) -> None:
    """Test hook — inject a mock client."""
    global _client
    _client = client


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _df_records(df, *, max_rows: int) -> list[dict]:
    import pandas as pd

    if df is None or (hasattr(df, "empty") and df.empty):
        return []
    subset = df.head(max_rows)
    records = json.loads(
        subset.to_json(orient="records", date_format="iso", default_handler=str)
    )
    return records


def _compact_info(meta: dict) -> dict:
    out = {k: meta[k] for k in _TABLE_META_KEYS if k in meta}
    for extra in ("table", "replication_mode", "cdc_serving_tier", "pk_columns"):
        if extra in meta and extra not in out:
            out[extra] = meta[extra]
    return _json_safe(out)


def _sync_payload(result: SyncResult, *, freshness: str, fmt: str) -> dict:
    payload: dict[str, Any] = {
        "status": result.status,
        "path": str(result.path),
        "bytes_downloaded": result.bytes_downloaded,
        "sync_mode": result.sync_mode or "snapshot",
        "format": fmt,
        "freshness": freshness,
        "previous_snapshot_id": result.previous_snapshot_id,
        "current_snapshot_id": result.current_snapshot_id,
    }
    if result.sync_mode == "changelog":
        payload["previous_seq"] = result.previous_seq
        payload["current_seq"] = result.current_seq
        payload["ops_applied"] = result.ops_applied
    return _json_safe(payload)


def _map_error(exc: Exception) -> str:
    if isinstance(exc, EolasError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


@mcp.tool()
def eolas_health() -> dict:
    """Check api.eolas.fyi reachability (no API key required)."""
    import requests

    try:
        r = requests.get("https://api.eolas.fyi/health", timeout=10)
        r.raise_for_status()
        body = r.json()
        return {"ok": True, "status_code": r.status_code, "body": _json_safe(body)}
    except Exception as exc:
        return {"ok": False, "error": _map_error(exc)}


@mcp.tool()
def eolas_search(
    query: str,
    source: Optional[str] = None,
    limit: int = 25,
) -> dict:
    """Search datasets by name, title, or description (alias-aware).

    Returns compact rows: name, title, source. Use eolas_info for full metadata.
    """
    limit = max(1, min(int(limit), _MCP_SEARCH_CAP))
    try:
        df = _get_client().search(query, source=source)
    except EolasError as exc:
        return {"error": _map_error(exc), "datasets": []}

    cols = [c for c in ("name", "title", "source", "namespace") if c in df.columns]
    if cols:
        df = df[cols]
    return {
        "query": query,
        "count": int(len(df)),
        "datasets": _df_records(df, max_rows=limit),
        "truncated": len(df) > limit,
    }


@mcp.tool()
def eolas_info(name: str) -> dict:
    """Dataset metadata — includes cdc_serving_tier, bulk_export_class, row counts."""
    try:
        meta = _get_client().info(name)
    except EolasError as exc:
        return {"error": _map_error(exc), "name": name}
    return {"name": name, **_compact_info(meta)}


@mcp.tool()
def eolas_get(
    name: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 100,
) -> dict:
    """Fetch a small row slice via the live /data API.

    Hard-capped at 500 rows. For whole datasets use eolas_download or eolas_sync.
    """
    cap = max(1, min(int(limit), _MCP_GET_ROW_CAP))
    try:
        df = _get_client().get(name, start=start, end=end, limit=cap)
    except EolasError as exc:
        return {"error": _map_error(exc), "name": name, "rows": []}

    # Client.get returns a Dataset (DataFrame subclass).
    row_count = len(df)
    return {
        "name": name,
        "row_count": row_count,
        "limit_applied": cap,
        "truncated": row_count >= cap,
        "rows": _df_records(df, max_rows=cap),
    }


@mcp.tool()
def eolas_download(
    name: str,
    path: str,
    format: str = "parquet",
    freshness: str = "auto",
) -> dict:
    """One-shot bulk download to a local file (returns path + bytes, not file contents).

    format: parquet | csv_gz | geoparquet. freshness: auto | monthly | current.
    """
    fmt = format.lower()
    if fmt == "csv":
        fmt = "csv_gz"
    allowed = {"parquet", "csv_gz", "geoparquet"}
    if fmt not in allowed:
        return {
            "error": f"unsupported format {format!r}; expected parquet, csv, or geoparquet",
            "name": name,
        }

    out = Path(path).expanduser().resolve()
    try:
        result_path = _get_client().download_bulk(
            name,
            path=out,
            format=fmt,
            freshness=freshness,
            progress=False,
        )
    except EolasError as exc:
        return {"error": _map_error(exc), "name": name, "path": str(out)}

    size = out.stat().st_size if out.exists() else 0
    return _json_safe({
        "name": name,
        "path": str(result_path),
        "bytes": size,
        "format": fmt,
        "freshness": freshness,
    })


@mcp.tool()
def eolas_sync(
    name: str,
    path: str,
    format: str = "parquet",
    freshness: str = "auto",
) -> dict:
    """Keep a local file current — routes on cdc_serving_tier (snapshot or changelog CDC).

    First call downloads; later calls are cheap (HEAD check or incremental /changes).
    """
    fmt = format.lower()
    if fmt == "csv":
        fmt = "csv_gz"
    if fmt != "parquet":
        meta = {}
        try:
            meta = _get_client().info(name)
        except EolasError:
            pass
        tier = (meta.get("cdc_serving_tier") or "snapshot") if meta else "snapshot"
        if tier == "changelog":
            return {
                "error": "changelog-tier datasets only support format='parquet'",
                "name": name,
            }

    out = Path(path).expanduser().resolve()
    try:
        result = _get_client().sync(
            name,
            path=out,
            format=fmt,
            freshness=freshness,
            progress=False,
        )
    except (EolasError, ValueError) as exc:
        return {"error": _map_error(exc), "name": name, "path": str(out)}

    return _sync_payload(result, freshness=freshness, fmt=fmt)


def main() -> None:
    """Console entry point — stdio transport for MCP hosts."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()