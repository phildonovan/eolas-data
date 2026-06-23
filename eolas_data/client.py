from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Literal, Optional, Union

ProgressControl = Union[bool, str, None]
ProgressPhase = Literal["download", "read"]

import pandas as pd
import requests

from .console import nag_json_transport_once
from .dataset import Dataset
from .library import resolve_library_dir
from .meta import attach_meta, merge_provenance, split_meta
from .exceptions import (
    APIError,
    AuthenticationError,
    BulkLicenceRestricted,
    BulkNotYetAvailable,
    BulkUpgradeRequired,
    ChangesLicenceRestricted,
    ChangesUpgradeRequired,
    NotFoundError,
    RateLimitError,
    WatermarkExpired,
)

_log = logging.getLogger("eolas_data")

# Imported separately so the names module is also re-exportable for users who
# want IDE autocomplete on dataset names without instantiating a Client.
from ._dataset_names import DatasetName  # noqa: F401  (public re-export)


BASE_URL = "https://api.eolas.fyi"

_SIDECAR_SCHEMA_VERSION = 1
_SIDECAR_SCHEMA_VERSION_CDC = 2

# OS-keyring constants — service name must match the R client so a key set
# from one language is readable from the other.
_KEYRING_SERVICE  = "eolas"
_KEYRING_USERNAME = "api-key"


def _keyring_get() -> str:
    """Return the API key from the OS keyring, or an empty string.

    Silently returns ``""`` when:
    - the ``keyring`` package is not installed
    - no entry exists under service="eolas", username="api-key"
    - the keyring backend is locked / unavailable (headless CI)

    Never raises — the caller treats a falsy return as "not found" and falls
    through to the next lookup step (config file or error).
    """
    try:
        import keyring as _kr
        value = _kr.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
        return value or ""
    except Exception:
        return ""


_CONFIG_FILE = pathlib.Path.home() / ".eolas" / "config.json"


def _config_file_get() -> str:
    """Return the ``api_key`` from ``~/.eolas/config.json``, or ``""``.

    Mirrors the CLI's ``_load_api_key`` so a key saved with ``eolas auth
    set-key`` is picked up by a bare ``Client()`` in a notebook. Never raises —
    a missing/corrupt file is treated as "not found".
    """
    try:
        if _CONFIG_FILE.exists():
            return json.loads(_CONFIG_FILE.read_text()).get("api_key", "") or ""
    except (json.JSONDecodeError, OSError):
        return ""
    return ""


@dataclass
class SyncResult:
    """Result of a :meth:`Client.sync_bulk`, :meth:`Client.sync_changes`, or
    :meth:`Client.sync` call.

    Snapshot-sync fields (sync_bulk / sync when tier='snapshot'):
        status: One of ``"downloaded"`` (first time), ``"updated"``
            (new snapshot available and written), or ``"unchanged"``
            (local file is already current — no I/O performed).
        previous_snapshot_id: The snapshot id recorded in the local sidecar
            before the sync, or ``None`` when no sidecar existed.
        current_snapshot_id: The snapshot id reported by the server's
            ``X-Snapshot-Version`` response header.
        path: The local file path that was written (or preserved unchanged).
        bytes_downloaded: Bytes written in this call. ``0`` when unchanged.

    Changelog-sync fields (sync_changes / sync when tier='changelog'):
        sync_mode: ``"snapshot"`` or ``"changelog"`` — which path was taken.
            ``None`` for legacy SyncResult objects from sync_bulk calls.
        previous_seq: The watermark seq recorded in the sidecar before this
            call, or ``None`` when starting fresh (after baseline).
        current_seq: The watermark seq after this call (the highest
            ``_eolas_seq`` value seen in the pages fetched).
        ops_applied: Number of change rows merged (non-delete rows inserted
            plus rows deleted). ``0`` when no changes were available.
            ``None`` for snapshot-mode results.
    """

    status: str
    previous_snapshot_id: Optional[str]
    current_snapshot_id: str
    path: pathlib.Path
    bytes_downloaded: int
    # Changelog-specific fields — present only when sync_mode='changelog'.
    # Defaulted so existing sync_bulk callers are unaffected.
    sync_mode: Optional[str] = None
    previous_seq: Optional[int] = None
    current_seq: Optional[int] = None
    ops_applied: Optional[int] = None


def _to_geodataframe(df: "pd.DataFrame", force: bool = False):
    """Convert a DataFrame with a ``geometry_wkt`` column to a GeoDataFrame (CRS WGS84).

    Returns the GeoDataFrame on success, or ``None`` when geopandas isn't installed
    (and ``force`` is False) so the caller can fall back to the plain DataFrame.
    Raises ImportError when ``force=True`` but geopandas is missing.
    """
    try:
        import geopandas as gpd
        from shapely import wkt as _wkt
    except ImportError:
        if force:
            raise ImportError(
                "geopandas + shapely are required to return geospatial datasets "
                "as GeoDataFrames. Install with: pip install eolas-data[geo]"
            )
        return None

    geom = df["geometry_wkt"].apply(lambda s: _wkt.loads(s) if isinstance(s, str) and s else None)
    gdf = gpd.GeoDataFrame(df.drop(columns=["geometry_wkt"]), geometry=geom, crs="EPSG:4326")
    src_attrs = getattr(df, "attrs", None)
    for attr in ("eolas_name", "eolas_source", "eolas_meta", "eolas_columns"):
        if isinstance(src_attrs, dict) and attr in src_attrs:
            gdf.attrs[attr] = src_attrs[attr]
        elif hasattr(df, attr):
            gdf.attrs[attr] = getattr(df, attr)
    return gdf


class Client:
    """Client for the eolas.fyi statistical data API.

    Args:
        api_key:  Your API key. Falls back, in order, to the ``EOLAS_API_KEY``
                  env var, the OS keyring, then ``~/.eolas/config.json`` (as
                  written by ``eolas auth set-key``).
        base_url: Override the API base URL (useful for testing).
        cache:    Cache responses in memory for the lifetime of the client.
                  Useful in notebooks to avoid re-fetching on re-runs.

    Examples::

        from eolas_data import Client
        client = Client("your_api_key")

        # Source-specific helpers
        cpi = client.rbnz("rbnz_m1_prices", start="2020-01-01")
        df  = client.oecd("nz_cpi", start="2020-01-01")   # OECD YoY %, not index

        # Generic
        df = client.get("nz_gdp_growth")

        # Discovery
        all_datasets = client.list()
        nz_datasets  = client.list("Stats NZ")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        cache: bool = False,
    ):
        # Precedence: explicit arg → EOLAS_API_KEY env var → OS keyring →
        # ~/.eolas/config.json → "". The config-file step mirrors the CLI's
        # `_load_api_key` so `eolas auth set-key` carries into `Client()`.
        self._key = (
            api_key
            or os.getenv("EOLAS_API_KEY")
            or _keyring_get()
            or _config_file_get()
            or ""
        )
        self._base  = base_url.rstrip("/")
        self._cache: dict | None = {} if cache else None
        self._session = requests.Session()
        # Explicit User-Agent: good API-client hygiene, and insulation against
        # the Cloudflare edge tightening bot rules (raw default UAs can be
        # 403'd by managed rulesets — a custom UA is always allowed).
        try:
            import importlib.metadata as _md
            _ver = _md.version("eolas-data")
        except Exception:
            _ver = "1.0.0"
        self._session.headers.update({
            "X-API-Key": self._key,
            "User-Agent": f"eolas-data/{_ver} (python; +https://eolas.fyi)",
        })
        # Tri-state Arrow capability memo: None=unknown (try it), True=server
        # speaks Arrow (keep using it), False=server ignored format=arrow
        # (old server — go straight to JSON, don't waste a round-trip retrying
        # every call).
        self._arrow_supported: Optional[bool] = None
        self._meta_cache: dict[str, dict] = {}

    def __repr__(self) -> str:
        masked = self._key[:8] + "..." if len(self._key) > 8 else self._key
        cache  = " cache=on" if self._cache is not None else ""
        return f"<eolas_data.Client key={masked!r}{cache}>"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list(self, source: Optional[str] = None) -> pd.DataFrame:
        """Return metadata for all available datasets.

        Args:
            source: Optional filter, e.g. ``"Stats NZ"``, ``"OECD"``.
        """
        data = self._get("/v1/datasets")
        items = data.get("datasets", data) if isinstance(data, dict) else data
        df = pd.DataFrame(items)
        if source and not df.empty:
            df = df[df["source"] == source].reset_index(drop=True)
        return df

    def search(self, query: str, source: Optional[str] = None) -> pd.DataFrame:
        """Search datasets by name, title, or description.

        Common aliases are expanded (e.g. ``"HLFS"`` → labour-force datasets,
        ``"OCR"`` → official cash rate series).
        """
        from .search import filter_datasets

        return filter_datasets(self.list(), query, source=source)

    def info(self, name: Union[str, "DatasetName"]) -> dict:
        """Return metadata for a single dataset."""
        return self._get(f"/v1/datasets/{name}")

    def _info_cached(self, name: Union[str, "DatasetName"]) -> dict:
        key = str(name)
        if key not in self._meta_cache:
            self._meta_cache[key] = self.info(name)
        return self._meta_cache[key]

    def _apply_force(self, name: Union[str, "DatasetName"], force: bool) -> None:
        if force:
            self._meta_cache.pop(str(name), None)

    def cache_clear(
        self,
        name: Optional[Union[str, "DatasetName"]] = None,
        *,
        cache_dir: Optional[Union[str, "pathlib.Path"]] = None,
        format: Optional[str] = None,
        files: bool = True,
        meta: bool = True,
    ) -> dict:
        """Clear client-side cache (library files and/or session metadata).

        See :func:`eolas_data.library.cache_clear` for details. Pass
        ``force=True`` to :meth:`get_local` / :meth:`sync_bulk` to clear
        metadata and re-download in one step.
        """
        from .library import cache_clear as _cache_clear

        return _cache_clear(
            None if name is None else str(name),
            cache_dir=cache_dir,
            format=format,
            files=files,
            meta=meta,
            meta_cache=self._meta_cache if meta else None,
        )

    def _attach_dataset_meta(
        self,
        result: "pd.DataFrame",
        name: Union[str, "DatasetName"],
        *,
        source: str = "",
        meta: bool = True,
        provenance: Optional[dict] = None,
        data_sources: Optional[list] = None,
    ) -> "pd.DataFrame":
        table_meta: dict = {}
        column_meta = None
        if meta:
            try:
                table_meta, column_meta = split_meta(self._info_cached(name))
            except Exception:
                pass
        if provenance:
            table_meta = merge_provenance(table_meta, provenance)
        if data_sources:
            table_meta = {**table_meta, "data_sources": data_sources}
        return attach_meta(
            result,
            name=str(name),
            source=source or (table_meta.get("source") or ""),
            table_meta=table_meta,
            column_meta=column_meta,
        )

    # ------------------------------------------------------------------
    # Integrations (Enterprise plan only)
    # ------------------------------------------------------------------

    def integration(self, platform: str, datasets: list[str]) -> dict[str, str]:
        """Generate connector config files for a third-party data-pipeline tool.

        Enterprise plan only. Other plans receive an
        :class:`AuthenticationError` with the upgrade message in the detail.

        Args:
            platform: One of ``"meltano"``, ``"fivetran"``, ``"azure-data-factory"``.
            datasets: Dataset names to include in the generated config.

        Returns:
            ``{filename: file_contents}`` ready to write to disk.

        Examples::

            files = client.integration("meltano", ["nz_cpi", "nz_gdp"])
            for filename, content in files.items():
                Path("./tap-eolas") / filename).write_text(content)
        """
        if not datasets:
            raise ValueError("datasets cannot be empty")
        resp = self._get(
            f"/v1/integrations/{platform}",
            params={"datasets": ",".join(datasets)},
        )
        return resp.get("files", {})

    # ------------------------------------------------------------------
    # Bulk download
    # ------------------------------------------------------------------

    _BULK_EXTENSIONS = {
        "parquet":    ".parquet",
        "csv_gz":     ".csv.gz",
        "geoparquet": ".geo.parquet",
    }
    _LIVE_DOWNLOAD_FORMATS = {"csv", "parquet", "arrow", "json"}
    _LIVE_DOWNLOAD_EXTENSIONS = {
        "csv":     ".csv",
        "parquet": ".parquet",
        "arrow":   ".arrow",
        "json":    ".json",
    }
    # Mirrors the API guard in datasets.py — unbounded live pulls on datasets
    # above this row count (or with geometry) return HTTP 413.
    _LARGE_DATASET_ROW_THRESHOLD = 100_000

    @staticmethod
    def _bulk_export_allowed(meta: dict) -> bool:
        return (meta.get("bulk_export_class") or "").lower() not in ("", "none")

    @classmethod
    def _live_pull_blocked(cls, meta: dict) -> bool:
        """True when limit=0 with no date bounds would hit the API 413 guard."""
        row_count = int(meta.get("row_count_at_last_refresh") or 0)
        return bool(meta.get("has_geometry")) or row_count > cls._LARGE_DATASET_ROW_THRESHOLD

    @staticmethod
    def _require_bulk_export(meta: dict, name: Union[str, "DatasetName"]) -> None:
        """Raise BulkLicenceRestricted when dataset metadata blocks bulk export."""
        if (meta.get("bulk_export_class") or "").lower() == "none":
            raise BulkLicenceRestricted(
                f"{name!r} cannot be bulk-downloaded (bulk_export_class=none — "
                "typically OECD/licence-restricted). Use get() or "
                f"`eolas get {name}` for live API access instead."
            )

    def download_bulk(
        self,
        name: Union[str, "DatasetName"],
        *,
        freshness: str = "auto",
        format: str = "parquet",
        path: Optional[Union[str, "pathlib.Path"]] = None,
        progress: ProgressControl = None,
    ) -> "Union[pathlib.Path, bytes]":
        """Download a complete dataset as a single binary file.

        Wraps ``GET /v1/bulk/{namespace}/{table}`` which streams a Parquet,
        gzipped-CSV, or GeoParquet snapshot. Monthly snapshots are served from
        Cloudflare's edge cache and are typically delivered in milliseconds.
        Current snapshots are lazy-generated for Pro users on first request.

        The endpoint requires both ``namespace`` and ``table``. These are
        resolved automatically by first calling ``GET /v1/datasets/{name}`` and
        reading ``namespace`` and ``table`` off the metadata response.

        Args:
            name: Dataset identifier, e.g. ``"nz_cpi"``.
            freshness: ``"auto"`` (default) — omit the query param so the
                server picks the right level for your plan (Free → monthly,
                Pro → current). ``"monthly"`` or ``"current"`` override
                explicitly.
            format: ``"parquet"`` (default), ``"csv_gz"``, or ``"geoparquet"``.
                GeoParquet is only available on geospatial datasets.
            path: Where to write the file. ``None`` (default) returns the raw
                bytes. Pass a ``str`` or ``pathlib.Path`` to write to disk and
                return the resolved path. Parent directories are created if
                needed.
            progress: Control the download progress bar (``"download"`` phase).
                See :meth:`get_local` for the full selector vocabulary
                (``"read"`` applies only to :meth:`get_local`). When ``path``
                is ``None`` (bytes mode) progress is always disabled.

        Returns:
            ``pathlib.Path`` when ``path`` is set (the resolved, written path).
            ``bytes`` when ``path`` is ``None``.

        Raises:
            BulkUpgradeRequired: HTTP 402 — ``freshness="current"`` requires Pro.
            BulkLicenceRestricted: HTTP 403 with a licence body — dataset is
                excluded from bulk (e.g. OECD). Use ``client.get()`` instead.
            BulkNotYetAvailable: HTTP 503 — monthly snapshot not yet generated.
            NotFoundError: Dataset or namespace/table not found.
            AuthenticationError: Invalid or missing API key.

        Examples::

            # Return bytes (e.g. hand to pd.read_parquet)
            import io, pandas as pd
            raw = client.download_bulk("nz_cpi")
            df = pd.read_parquet(io.BytesIO(raw))

            # Write to a file, get the path back
            p = client.download_bulk("nz_cpi", path="nz_cpi.parquet")
            df = pd.read_parquet(p)

            # Gzipped CSV for spreadsheet users
            client.download_bulk("nz_cpi", format="csv_gz", path="nz_cpi.csv.gz")

            # Force monthly freshness (works on any plan — useful for reproducibility)
            client.download_bulk("nz_cpi", freshness="monthly", path="nz_cpi.parquet")

            # Silence the bar in a script even when run interactively
            client.download_bulk("nz_cpi", path="nz_cpi.parquet", progress=False)

        See Also:
            https://docs.eolas.fyi/bulk-downloads/
        """
        fmt = format.lower()
        if fmt not in self._BULK_EXTENSIONS:
            raise ValueError(
                f"Unknown format {format!r}. Expected one of: "
                + ", ".join(self._BULK_EXTENSIONS)
            )
        if freshness not in ("auto", "monthly", "current"):
            raise ValueError(
                f"Unknown freshness {freshness!r}. Expected 'auto', 'monthly', or 'current'."
            )

        # Resolve name → namespace + table via the datasets metadata endpoint.
        meta = self._get(f"/v1/datasets/{name}")
        self._require_bulk_export(meta, name)
        namespace = meta.get("namespace") or ""
        table     = meta.get("table") or meta.get("name") or name
        if not namespace:
            raise NotFoundError(
                f"Dataset {name!r} metadata did not include a namespace field. "
                "Cannot construct bulk URL."
            )

        params: dict = {"format": fmt}
        if freshness != "auto":
            params["freshness"] = freshness

        bulk_path = f"/v1/bulk/{namespace}/{table}"

        if path is None:
            # Bytes mode: no progress bar (no file label to show).
            resp = self._raw_bulk_get(bulk_path, params=params)
            return resp.content

        out = pathlib.Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        show = self._resolve_show_progress(progress, "download")
        resp = self._raw_bulk_get(bulk_path, params=params, stream=True)
        total = int(resp.headers.get("Content-Length", 0)) or None
        self._stream_to_file_with_progress(
            resp, out,
            total_bytes=total,
            label=f"Downloading {out.name}",
            show_progress=show,
        )
        return out

    def download(
        self,
        name: Union[str, "DatasetName"],
        *,
        path: Optional[Union[str, "pathlib.Path"]] = None,
        format: str = "csv",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
        progress: ProgressControl = None,
    ) -> "Union[pathlib.Path, bytes]":
        """Download a dataset via the live ``/v1/datasets/{name}/data`` endpoint.

        Works for **all** datasets — including OECD and other licence-restricted
        tables where bulk export is unavailable (e.g. ``nz_cpi``). For whole-dataset
        pulls on very large or geospatial tables, prefer :meth:`download_bulk` when
        bulk export is permitted.

        Args:
            name: Dataset identifier, e.g. ``"nz_cpi"``.
            path: Where to write the file. ``None`` (default) returns raw bytes.
            format: ``"csv"`` (default), ``"parquet"``, ``"arrow"``, or ``"json"``.
            start: ISO date lower bound.
            end: ISO date upper bound.
            limit: Max rows. ``None`` (default) requests the full dataset (subject
                to plan caps). Pass an integer to cap rows.
            progress: Download progress bar control (``"download"`` phase only).

        Returns:
            ``pathlib.Path`` when ``path`` is set; ``bytes`` when ``path`` is ``None``.

        Examples::

            client.download("nz_cpi", path="nz_cpi.csv")
            client.download("nz_cpi", format="parquet", path="nz_cpi.parquet")
            raw = client.download("nz_cpi", format="csv")
        """
        fmt = format.lower()
        if fmt not in self._LIVE_DOWNLOAD_FORMATS:
            raise ValueError(
                f"Unknown format {format!r}. Expected one of: "
                f"{', '.join(sorted(self._LIVE_DOWNLOAD_FORMATS))}."
            )

        params: dict = {"format": fmt}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if limit is not None:
            from .rows import resolve_fetch_limit
            fetch_limit, _ = resolve_fetch_limit(limit)
            params["limit"] = fetch_limit
        elif start is None and end is None:
            params["limit"] = 0
        else:
            params["limit"] = 0

        if path is None:
            resp = self._raw_get(f"/v1/datasets/{name}/data", params=params)
            return resp.content

        out = pathlib.Path(path).expanduser().resolve()
        if not out.suffix and fmt in self._LIVE_DOWNLOAD_EXTENSIONS:
            out = out.with_suffix(self._LIVE_DOWNLOAD_EXTENSIONS[fmt])
        out.parent.mkdir(parents=True, exist_ok=True)

        show = self._resolve_show_progress(progress, "download")
        resp = self._raw_get(
            f"/v1/datasets/{name}/data", params=params, stream=True,
        )
        total = int(resp.headers.get("Content-Length", 0)) or None
        self._stream_to_file_with_progress(
            resp, out,
            total_bytes=total,
            label=f"Downloading {out.name}",
            show_progress=show,
        )
        return out

    def sync_bulk(
        self,
        name: Union[str, "DatasetName"],
        *,
        path: Union[str, "pathlib.Path"],
        format: str = "parquet",
        freshness: str = "auto",
        progress: ProgressControl = None,
        force: bool = False,
    ) -> SyncResult:
        """Incrementally sync a bulk dataset file — only re-download when the snapshot changes.

        Wraps the same ``/v1/bulk/{namespace}/{table}`` endpoint as
        :meth:`download_bulk`, but adds a lightweight HEAD optimisation: a
        single ~200-byte HEAD request checks the ``X-Snapshot-Version``
        response header on the canonical redirect URL.  If the snapshot id
        matches what's recorded in the local sidecar, the function returns
        immediately with ``status="unchanged"`` and zero I/O on the data file.

        A sidecar file ``<path>.eolas-meta.json`` is written next to the data
        file and contains the snapshot id, download timestamp, and source URL.
        It is used on the next call to perform the no-op check cheaply.

        The replacement is **atomic**: the new file is downloaded to a
        temporary path ``<path>.eolas-tmp-<rand>`` and then renamed over the
        original with :func:`os.replace`.  Readers with the file open see no
        partial bytes.

        Args:
            name: Dataset identifier, e.g. ``"nz_cpi"``.
            path: **Required.** Where to write the data file.  The sidecar
                lives at ``str(path) + ".eolas-meta.json"``.  Parent
                directories are created if needed.
            format: ``"parquet"`` (default), ``"csv_gz"``, or
                ``"geoparquet"``.
            freshness: ``"auto"`` (default), ``"monthly"``, or ``"current"``.
                Passed verbatim to the bulk endpoint.
            progress: Control the download progress bar (``"download"`` phase).
                See :meth:`get_local` for the full selector vocabulary.
                When ``status="unchanged"`` no download bar is shown; an
                informative cached-file message is printed instead.
            force: When ``True``, skip the sidecar unchanged fast path and
                re-download even when the local snapshot id matches the server.

        Returns:
            A :class:`SyncResult` dataclass with ``status``,
            ``previous_snapshot_id``, ``current_snapshot_id``, ``path``, and
            ``bytes_downloaded``.

        Raises:
            BulkUpgradeRequired: HTTP 402.
            BulkLicenceRestricted: HTTP 403 (licence body).
            BulkNotYetAvailable: HTTP 503.
            NotFoundError: Dataset not found.
            AuthenticationError: Invalid or missing API key.

        Examples::

            from eolas_data import Client, SyncResult

            client = Client("your_api_key")

            # First call: full download
            r = client.sync_bulk("nz_cpi", path="nz_cpi.parquet")
            print(r.status)           # "downloaded"
            print(r.bytes_downloaded) # e.g. 2_100_000

            # Second call (same snapshot): no-op
            r = client.sync_bulk("nz_cpi", path="nz_cpi.parquet")
            print(r.status)           # "unchanged"
            print(r.bytes_downloaded) # 0

            # After a new ETL run: new snapshot → file replaced in-place
            r = client.sync_bulk("nz_cpi", path="nz_cpi.parquet")
            print(r.status)           # "updated"

        See Also:
            https://docs.eolas.fyi/bulk-downloads/
        """
        fmt = format.lower()
        if fmt not in self._BULK_EXTENSIONS:
            raise ValueError(
                f"Unknown format {format!r}. Expected one of: "
                + ", ".join(self._BULK_EXTENSIONS)
            )
        if freshness not in ("auto", "monthly", "current"):
            raise ValueError(
                f"Unknown freshness {freshness!r}. Expected 'auto', 'monthly', or 'current'."
            )

        self._apply_force(name, force)

        out = pathlib.Path(path).expanduser().resolve()
        sidecar = pathlib.Path(str(out) + ".eolas-meta.json")

        # Read local sidecar if present.
        prev: Optional[dict] = None
        if sidecar.exists():
            try:
                prev = json.loads(sidecar.read_text())
            except Exception:
                prev = None

        # Resolve name → namespace + table (needed to construct the bulk URL).
        meta = self._get(f"/v1/datasets/{name}")
        self._require_bulk_export(meta, name)
        namespace = meta.get("namespace") or ""
        table     = meta.get("table") or meta.get("name") or name
        if not namespace:
            raise NotFoundError(
                f"Dataset {name!r} metadata did not include a namespace field. "
                "Cannot construct bulk URL."
            )

        params: dict = {"format": fmt}
        if freshness != "auto":
            params["freshness"] = freshness

        bulk_path = f"/v1/bulk/{namespace}/{table}"
        canonical_url = f"{self._base}{bulk_path}"

        # HEAD the canonical URL to read X-Snapshot-Version cheaply.
        # We follow redirects on the HEAD so we land on the versioned CDN URL
        # that carries the header.
        current_sid = self._head_snapshot_version(canonical_url, params=params)

        # No-op fast path: snapshot hasn't changed AND file exists on disk.
        if (
            not force
            and prev is not None
            and prev.get("snapshot_id") == current_sid
            and out.exists()
        ):
            print(f"Using cached {out.name} (up to date).", file=sys.stderr)
            return SyncResult(
                status="unchanged",
                previous_snapshot_id=prev.get("snapshot_id"),
                current_snapshot_id=current_sid,
                path=out,
                bytes_downloaded=0,
            )

        # Download (atomic replace).
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + f".eolas-tmp-{os.urandom(4).hex()}")
        show = self._resolve_show_progress(progress, "download")
        try:
            resp = self._raw_bulk_get(bulk_path, params=params, stream=True)
            total = int(resp.headers.get("Content-Length", 0)) or None
            bytes_dl = self._stream_to_file_with_progress(
                resp, tmp,
                total_bytes=total,
                label=f"Downloading {out.name}",
                show_progress=show,
            )
            if bytes_dl == 0:
                raise APIError(
                    200,
                    f"Bulk download for {name!r} returned an empty body "
                    "(0 bytes). The snapshot may not exist for this dataset "
                    "or format. Use format='parquet' for non-geo datasets.",
                )
            os.replace(tmp, out)
        except Exception:
            # Best-effort cleanup of the tmp file; the original is NEVER touched
            # on failure so no 0-byte file is left at the final path.
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        # Write sidecar next to the data file.
        sidecar_data = {
            "schema_version": _SIDECAR_SCHEMA_VERSION,
            "name": str(name),
            "snapshot_id": current_sid,
            "format": fmt,
            "freshness": freshness,
            "downloaded_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_url": canonical_url + "?" + "&".join(f"{k}={v}" for k, v in params.items()),
        }
        sidecar.write_text(json.dumps(sidecar_data, indent=2) + "\n")

        return SyncResult(
            status="downloaded" if prev is None else "updated",
            previous_snapshot_id=prev.get("snapshot_id") if prev else None,
            current_snapshot_id=current_sid,
            path=out,
            bytes_downloaded=bytes_dl,
        )

    # ------------------------------------------------------------------
    # Changelog sync (CDC OUT — new in v1.2.0)
    # ------------------------------------------------------------------

    _CHANGES_PAGE_LIMIT = 50_000

    def sync_changes(
        self,
        name: Union[str, "DatasetName"],
        path: Union[str, "pathlib.Path"],
        *,
        format: str = "parquet",
        progress: Optional[bool] = None,
        force: bool = False,
    ) -> SyncResult:
        """Incrementally sync a changelog-tier dataset via the /changes feed.

        Implements the OUT half of CDC described in
        ``eolas/docs/metadata-cdc-design-2026-06-16.md`` (Client contract section).

        On the first call (cold start):
        1. Calls :meth:`sync_bulk` to download the full baseline snapshot.
        2. Fetches a tail page from ``/changes`` to record the current high-water
           seq — so the next call only asks for *new* changes, never replaying the
           whole feed.
        3. Writes a v2 sidecar (``schema_version=2``, ``sync_mode='changelog'``).

        On subsequent calls:
        1. Reads the watermark seq from the sidecar.
        2. Pages through ``GET /v1/datasets/{name}/changes?since_seq=<watermark>``
           until the server signals no more pages (``X-Eolas-Truncated: false`` or
           ``X-Eolas-Row-Count < limit``).
        3. pk-merges change rows into the local materialised file (atomic rewrite).
        4. Advances the watermark and updates the sidecar.

        On ``410 WatermarkExpired`` (since_seq predates the retained range):
        Automatically re-baselines via ``sync_bulk`` and resets the watermark,
        then returns — the next call will resume cleanly.

        Args:
            name: Dataset identifier, e.g. ``"pharmac_schedule_history"``.
            path: **Required.** Where to write the materialised data file. The
                sidecar lives at ``str(path) + ".eolas-meta.json"``.
            format: ``"parquet"`` (default). Only ``"parquet"`` is supported for
                changelog sync; other formats raise ``ValueError``.
            progress: Forwarded to :meth:`sync_bulk` for the baseline download
                progress bar (``None`` auto-detects TTY).
            force: When ``True``, re-baseline from a full bulk snapshot.

        Returns:
            A :class:`SyncResult` with ``sync_mode='changelog'``,
            ``previous_seq``, ``current_seq``, and ``ops_applied``.

        Raises:
            ChangesUpgradeRequired: HTTP 402 — changelog sync requires Pro.
            ChangesLicenceRestricted: HTTP 403 — dataset licence prohibits export.
            NotFoundError: HTTP 404 — dataset not found or tier != changelog.
            AuthenticationError: Invalid or missing API key.

        Examples::

            from eolas_data import Client

            client = Client("your_api_key")

            # First call: baseline bulk download + watermark set
            r = client.sync_changes("pharmac_schedule_history", path="pharmac.parquet")
            print(r.sync_mode)    # 'changelog'
            print(r.current_seq)  # e.g. 514000

            # Subsequent calls: apply only what changed
            r = client.sync_changes("pharmac_schedule_history", path="pharmac.parquet")
            print(r.ops_applied)  # e.g. 1200

        See Also:
            :meth:`sync` — unified dispatcher that routes on cdc_serving_tier.
        """
        fmt = format.lower()
        if fmt != "parquet":
            raise ValueError(
                f"sync_changes only supports format='parquet'; got {format!r}. "
                "Changelog feed is always delivered as Parquet."
            )

        out = pathlib.Path(path).expanduser().resolve()
        sidecar_path = pathlib.Path(str(out) + ".eolas-meta.json")
        self._apply_force(name, force)

        # Read sidecar.
        sidecar: Optional[dict] = None
        if sidecar_path.exists():
            try:
                sidecar = json.loads(sidecar_path.read_text())
            except Exception:
                sidecar = None

        # Determine if we need a cold-start baseline.
        needs_baseline = force or (
            sidecar is None
            or sidecar.get("sync_mode") != "changelog"
            or sidecar.get("watermark_seq") is None
        )

        # Fetch dataset metadata (needed for pk_columns, current_state_filter,
        # and to validate this is a changelog-tier dataset).
        meta = self._get(f"/v1/datasets/{name}")
        pk_columns: list[str] = meta.get("pk_columns") or []
        current_state_filter: Optional[str] = meta.get("current_state_filter")
        namespace = meta.get("namespace") or ""
        table = meta.get("table") or meta.get("name") or str(name)

        changes_url = f"/v1/datasets/{name}/changes"

        if needs_baseline:
            _log.info("sync_changes(%s): cold start — running baseline sync_bulk", name)
            bulk_result = self.sync_bulk(
                name,
                path=out,
                format=fmt,
                freshness="current",
                progress=progress,
                force=force,
            )
            baseline_snapshot_id = bulk_result.current_snapshot_id

            # Read the current high-water seq from the server so the next call
            # fetches only genuinely new changes. We do a single tail page
            # (since_seq=very_large_int returns empty or last page, giving us
            # the X-Eolas-Seq-High header cheaply).
            # Strategy: use a large since_seq (maxint64). The server will return
            # 0 rows with X-Eolas-Seq-High set to the current maximum seq.
            high_seq = self._fetch_changes_seq_high(changes_url, since_seq=2**62)
            watermark_seq = high_seq

            # Write v2 sidecar.
            self._write_changelog_sidecar(
                sidecar_path,
                name=str(name),
                fmt=fmt,
                pk_columns=pk_columns,
                current_state_filter=current_state_filter,
                baseline_snapshot_id=baseline_snapshot_id,
                watermark_seq=watermark_seq,
            )

            return SyncResult(
                status="downloaded",
                previous_snapshot_id=None,
                current_snapshot_id=baseline_snapshot_id,
                path=out,
                bytes_downloaded=bulk_result.bytes_downloaded,
                sync_mode="changelog",
                previous_seq=None,
                current_seq=watermark_seq,
                ops_applied=0,
            )

        # --- Incremental path: we have a valid changelog sidecar. ---
        prev_watermark = int(sidecar.get("watermark_seq", 0))
        baseline_snapshot_id = sidecar.get("baseline_snapshot_id", "")
        # Use pk_columns from sidecar if server didn't return them (registry
        # may not have been stamped yet on this dataset).
        if not pk_columns:
            pk_columns = sidecar.get("pk_columns") or []
        if not current_state_filter:
            current_state_filter = sidecar.get("current_state_filter")

        try:
            all_changes, final_seq = self._fetch_all_change_pages(
                changes_url, since_seq=prev_watermark
            )
        except WatermarkExpired:
            # 410: since_seq predates retained range. Re-baseline and reset.
            _log.warning(
                "sync_changes(%s): watermark expired (seq=%d) — re-baselining",
                name,
                prev_watermark,
            )
            bulk_result = self.sync_bulk(
                name,
                path=out,
                format=fmt,
                freshness="current",
                progress=progress,
                force=force,
            )
            baseline_snapshot_id = bulk_result.current_snapshot_id
            high_seq = self._fetch_changes_seq_high(changes_url, since_seq=2**62)
            watermark_seq = high_seq
            self._write_changelog_sidecar(
                sidecar_path,
                name=str(name),
                fmt=fmt,
                pk_columns=pk_columns,
                current_state_filter=current_state_filter,
                baseline_snapshot_id=baseline_snapshot_id,
                watermark_seq=watermark_seq,
            )
            return SyncResult(
                status="updated",
                previous_snapshot_id=sidecar.get("baseline_snapshot_id"),
                current_snapshot_id=baseline_snapshot_id,
                path=out,
                bytes_downloaded=bulk_result.bytes_downloaded,
                sync_mode="changelog",
                previous_seq=prev_watermark,
                current_seq=watermark_seq,
                ops_applied=0,
            )

        # No changes available.
        if all_changes.empty:
            return SyncResult(
                status="unchanged",
                previous_snapshot_id=sidecar.get("baseline_snapshot_id"),
                current_snapshot_id=baseline_snapshot_id,
                path=out,
                bytes_downloaded=0,
                sync_mode="changelog",
                previous_seq=prev_watermark,
                current_seq=prev_watermark,
                ops_applied=0,
            )

        # Merge changes into the local materialised file.
        from .cdc import merge_changes

        # Read the local file.
        import pyarrow.parquet as _pq
        if out.exists():
            local_df = _pq.read_table(str(out)).to_pandas()
        else:
            local_df = pd.DataFrame()

        merged_df = merge_changes(
            local_df,
            all_changes,
            pk_columns=pk_columns,
            current_state_filter=current_state_filter,
        )

        # Atomic write.
        from .cdc import df_to_parquet_bytes
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + f".eolas-tmp-{os.urandom(4).hex()}")
        try:
            tmp.write_bytes(df_to_parquet_bytes(merged_df))
            os.replace(tmp, out)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        # Count ops: non-delete insertions + deletes (rows whose PK we dropped).
        ops_applied = len(all_changes)

        # Advance watermark and update sidecar.
        self._write_changelog_sidecar(
            sidecar_path,
            name=str(name),
            fmt=fmt,
            pk_columns=pk_columns,
            current_state_filter=current_state_filter,
            baseline_snapshot_id=baseline_snapshot_id,
            watermark_seq=final_seq,
        )

        return SyncResult(
            status="updated",
            previous_snapshot_id=baseline_snapshot_id,
            current_snapshot_id=baseline_snapshot_id,
            path=out,
            bytes_downloaded=0,
            sync_mode="changelog",
            previous_seq=prev_watermark,
            current_seq=final_seq,
            ops_applied=ops_applied,
        )

    def sync(
        self,
        name: Union[str, "DatasetName"],
        path: Union[str, "pathlib.Path"],
        *,
        format: str = "parquet",
        freshness: str = "auto",
        progress: Optional[bool] = None,
        force: bool = False,
    ) -> SyncResult:
        """Unified sync dispatcher — keeps a local file current automatically.

        Reads ``cdc_serving_tier`` from ``info(name)`` and dispatches:

        - ``tier='snapshot'`` (the default for ~1480 tables) → :meth:`sync_bulk`.
          HEAD-checks the snapshot id; re-downloads only when a new ETL run has
          produced a new snapshot. Watermark = snapshot id.
        - ``tier='changelog'`` (~15-30 high-churn tables) → :meth:`sync_changes`.
          Applies only the rows that changed since the local watermark seq.
          Watermark = ``_eolas_seq`` (survives snapshot expiry).

        The tier is declared server-side in the stream registry. It is NOT
        inferred by the client — this avoids the "smart routing" footgun where a
        client guesses wrong and silently corrupts data. Call ``info(name)`` to
        inspect the tier yourself.

        Args:
            name: Dataset identifier.
            path: Local file path. Sidecar lives at ``str(path) + ".eolas-meta.json"``.
            format: ``"parquet"`` (default). Only ``"parquet"`` is supported for
                ``tier='changelog'``; other formats work for snapshot tier.
            freshness: ``"auto"`` (default), ``"monthly"``, or ``"current"``. Only
                used for the snapshot path (passed to :meth:`sync_bulk`).
            progress: Control the download progress bar.
            force: Bypass local unchanged cache and re-sync from the server.

        Returns:
            A :class:`SyncResult`. The ``sync_mode`` field is ``"snapshot"`` or
            ``"changelog"`` indicating which path was taken.

        Raises:
            The same exceptions as the dispatched method.

        Examples::

            from eolas_data import Client
            import pandas as pd

            client = Client("your_api_key")

            # Works for any dataset regardless of tier:
            r = client.sync("pharmac_schedule_history", path="pharmac.parquet")
            df = pd.read_parquet("pharmac.parquet")
        """
        meta = self._get(f"/v1/datasets/{name}")
        tier = meta.get("cdc_serving_tier", "snapshot") or "snapshot"

        if tier == "changelog":
            return self.sync_changes(name, path, format=format, progress=progress, force=force)
        else:
            result = self.sync_bulk(
                name,
                path=path,
                format=format,
                freshness=freshness,
                progress=progress,
                force=force,
            )
            # Tag the result with sync_mode so callers don't need to inspect tier.
            result.sync_mode = "snapshot"
            return result

    def _fetch_all_change_pages(
        self,
        changes_url: str,
        since_seq: int,
    ) -> tuple["pd.DataFrame", int]:
        """Fetch all pages from the /changes endpoint and concatenate them.

        Returns (changes_df, final_seq) where final_seq is the X-Eolas-Seq-High
        from the last page (the new watermark).

        Pagination stop conditions (in priority order):
        1. X-Eolas-Truncated == 'false' (server explicit — always respected).
        2. X-Eolas-Row-Count == 0 (empty page — nothing more to read).
        3. X-Eolas-Truncated header absent AND row_count < limit (fallback
           heuristic for servers that omit the header). This is secondary to
           the explicit header because a small last page can coexist with
           Truncated=true (server decided to split mid-limit).

        Raises WatermarkExpired on HTTP 410.
        """
        pages: list["pd.DataFrame"] = []
        current_seq = since_seq
        final_seq = since_seq

        while True:
            resp = self._raw_changes_get(
                changes_url,
                params={
                    "since_seq": current_seq,
                    "limit": self._CHANGES_PAGE_LIMIT,
                    "format": "parquet",
                },
            )
            seq_high = int(resp.headers.get("X-Eolas-Seq-High", current_seq))
            row_count = int(resp.headers.get("X-Eolas-Row-Count", 0))
            truncated_raw = resp.headers.get("X-Eolas-Truncated")
            # Explicit header wins. Absent header → fall back to row_count heuristic.
            if truncated_raw is not None:
                truncated = truncated_raw.lower() == "true"
            else:
                # Heuristic: if the server filled the limit exactly, assume more.
                truncated = row_count >= self._CHANGES_PAGE_LIMIT

            if row_count > 0 and resp.content:
                from .cdc import read_parquet_bytes
                page_df = read_parquet_bytes(resp.content)
                pages.append(page_df)

            final_seq = seq_high

            # Stop when server says not truncated, or when the page was empty.
            if not truncated or row_count == 0:
                break

            # Advance cursor to the high-water of this page.
            current_seq = seq_high

        if not pages:
            return pd.DataFrame(), final_seq

        return pd.concat(pages, ignore_index=True, sort=False), final_seq

    def _fetch_changes_seq_high(self, changes_url: str, since_seq: int) -> int:
        """Fetch a single /changes page and return X-Eolas-Seq-High.

        Used during cold-start to anchor the watermark at the current feed head
        without replaying any history. Returns 0 if the header is absent.
        """
        try:
            resp = self._raw_changes_get(
                changes_url,
                params={
                    "since_seq": since_seq,
                    "limit": 1,
                    "format": "parquet",
                },
            )
            return int(resp.headers.get("X-Eolas-Seq-High", 0))
        except Exception:
            return 0

    def _raw_changes_get(
        self,
        path: str,
        params: Optional[dict] = None,
    ) -> requests.Response:
        """GET the /changes endpoint, raising typed exceptions for known status codes."""
        url = f"{self._base}{path}"
        resp = self._session.get(url, params=params)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 402:
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = ""
            raise ChangesUpgradeRequired(detail) if detail else ChangesUpgradeRequired()
        if resp.status_code == 403:
            try:
                body = resp.json()
                detail = body.get("detail", "")
            except Exception:
                detail = ""
            if detail and "licence" in detail.lower():
                raise ChangesLicenceRestricted(detail)
            raise AuthenticationError(detail or "API key is inactive.")
        if resp.status_code == 410:
            try:
                body = resp.json()
                min_seq = int(body.get("min_available_seq", 0))
                detail = body.get("error", "watermark_expired")
            except Exception:
                min_seq = 0
                detail = "watermark_expired"
            raise WatermarkExpired(detail, min_available_seq=min_seq)
        self._raise_for_status(resp)
        return resp  # unreachable but satisfies type checkers

    @staticmethod
    def _write_changelog_sidecar(
        sidecar_path: "pathlib.Path",
        *,
        name: str,
        fmt: str,
        pk_columns: list,
        current_state_filter: Optional[str],
        baseline_snapshot_id: str,
        watermark_seq: int,
    ) -> None:
        """Write or update the v2 changelog sidecar file."""
        data = {
            "schema_version": _SIDECAR_SCHEMA_VERSION_CDC,
            "sync_mode": "changelog",
            "name": name,
            "format": fmt,
            "pk_columns": pk_columns,
            "current_state_filter": current_state_filter,
            "baseline_snapshot_id": baseline_snapshot_id,
            "watermark_seq": watermark_seq,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
        sidecar_path.write_text(json.dumps(data, indent=2) + "\n")

    # ------------------------------------------------------------------
    # Streaming helper
    # ------------------------------------------------------------------

    @staticmethod
    def _progress_auto_detect() -> bool:
        """Whether progress feedback should show when ``progress`` is None."""
        if os.getenv("EOLAS_NO_PROGRESS", "").strip() in ("1", "true", "yes"):
            return False
        if "ipykernel" in sys.modules:
            return True
        # tqdm writes to stderr; many terminals/IDEs pipe stdout but keep stderr
        # on the controlling TTY (Cursor, VS Code, script wrappers).
        return sys.stderr.isatty() or sys.stdout.isatty()

    @staticmethod
    def _resolve_progress_phases(progress: ProgressControl) -> dict[str, bool]:
        """Map ``progress`` to download/read phase flags.

        Accepts ``None``, ``True``/``False``, or ``"both"``, ``"download"``,
        ``"read"``, ``"none"`` (and ``"all"`` as alias for ``"both"``).
        """
        if progress is False:
            return {"download": False, "read": False}
        if progress is True:
            return {"download": True, "read": True}
        if isinstance(progress, str):
            key = progress.strip().lower()
            table = {
                "both": (True, True),
                "all": (True, True),
                "download": (True, False),
                "read": (False, True),
                "none": (False, False),
            }
            if key not in table:
                raise ValueError(
                    "progress must be True, False, None, or one of "
                    "'both', 'download', 'read', 'none'."
                )
            d, r = table[key]
            return {"download": d, "read": r}
        auto = Client._progress_auto_detect()
        return {"download": auto, "read": auto}

    @staticmethod
    def _resolve_show_progress(
        progress: ProgressControl,
        phase: ProgressPhase = "download",
    ) -> bool:
        """Resolve ``progress`` for one bulk phase (download or read)."""
        return Client._resolve_progress_phases(progress)[phase]

    @staticmethod
    @contextlib.contextmanager
    def _with_read_progress(label: str, show: bool):
        """Indeterminate spinner while materialising a cached bulk file."""
        if not show:
            yield
            return
        import tqdm.auto

        bar = tqdm.auto.tqdm(
            total=1,
            desc=f"Loading {label} from disk",
            leave=False,
            bar_format="{desc}…",
        )
        try:
            yield
        finally:
            bar.update(1)
            bar.close()

    @staticmethod
    def _stream_to_file_with_progress(
        resp: "requests.Response",
        dest: "pathlib.Path",
        *,
        total_bytes: Optional[int],
        label: str,
        show_progress: bool,
    ) -> int:
        """Stream *resp* body to *dest*, optionally displaying a tqdm progress bar.

        Parameters
        ----------
        resp:
            A streaming ``requests.Response`` object (caller must have
            passed ``stream=True`` to the ``requests.get`` / ``Session.get``
            call).
        dest:
            File path to write into.  The file is opened in ``"wb"`` mode;
            the caller is responsible for ensuring the parent directory
            exists.
        total_bytes:
            Expected file size for the bar's ``total`` parameter.
            ``None`` disables the percentage / ETA display but keeps the
            bytes-transferred counter (tqdm handles ``total=None`` cleanly).
        label:
            Short description shown left of the bar (e.g. the filename).
        show_progress:
            ``True`` → show bar.  ``False`` → silent (tqdm ``disable=True``).

        Returns
        -------
        int
            Actual bytes written.
        """
        import tqdm.auto

        bytes_written = 0
        with tqdm.auto.tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=label,
            file=sys.stderr,
            dynamic_ncols=True,
            disable=not show_progress,
            leave=True,
        ) as bar:
            with dest.open("wb") as fh:
                # 1 MiB chunks: responsive bar updates (bar refreshes ~once per
                # MiB) without excessive syscall overhead on large files.
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
                        bar.update(len(chunk))
                        bytes_written += len(chunk)
        return bytes_written

    def _head_snapshot_version(self, url: str, params: Optional[dict] = None) -> str:
        """Issue a HEAD to ``url`` (following redirects) and return ``X-Snapshot-Version``.

        The header is set by the eolas CDN/server on the canonical versioned
        bulk URL.  If the header is absent (e.g. in tests against a stub
        server), we fall back to an empty string, which will never match a
        real sidecar snapshot_id, so the full GET always fires.

        Raises the same bulk-refusal exceptions as :meth:`_raw_bulk_get` so
        that the caller doesn't need to handle HEAD errors differently.
        """
        full_url = url if url.startswith("http") else f"{self._base}{url}"
        resp = self._session.head(full_url, params=params, allow_redirects=True)
        self._raise_for_bulk_status(resp)
        return resp.headers.get("X-Snapshot-Version", "")

    def _raw_bulk_get(
        self,
        path: str,
        params: Optional[dict] = None,
        stream: bool = False,
    ) -> requests.Response:
        """Issue a GET that may 302-redirect to a canonical CDN URL.

        ``requests.Session`` follows redirects by default, which is exactly
        what we want: the bare ``/v1/bulk/{ns}/{table}`` URL redirects to the
        canonical versioned URL, and the session fetches that transparently.
        We only need special handling for the bulk-specific HTTP status codes
        (402, 503) that ``_raise_for_status`` doesn't know about.

        When ``stream=True`` the response body is not eagerly downloaded —
        callers use :meth:`_stream_to_file_with_progress` to consume it.
        """
        url  = f"{self._base}{path}"
        resp = self._session.get(url, params=params, stream=stream)
        self._raise_for_bulk_status(resp)
        return resp

    @staticmethod
    def _raise_for_bulk_status(resp: requests.Response) -> None:
        """Like ``_raise_for_status`` but handles the extra bulk refusal codes."""
        if resp.status_code == 200:
            return
        if resp.status_code == 402:
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = ""
            raise BulkUpgradeRequired(detail) if detail else BulkUpgradeRequired()
        if resp.status_code == 403:
            try:
                body   = resp.json()
                detail = body.get("detail", "")
            except Exception:
                detail = ""
            # Distinguish licence-restriction 403 from auth 403 by the presence
            # of "licence" in the server detail. A key-auth 403 goes to AuthenticationError.
            if detail and "licence" in detail.lower():
                raise BulkLicenceRestricted(detail)
            # Bulk HEAD often returns 403 with no JSON body (OECD/licence-blocked).
            if not detail and "/bulk/" in (getattr(resp, "url", "") or ""):
                raise BulkLicenceRestricted(
                    "Bulk download is not permitted for this dataset. "
                    "Use get() or `eolas get` for live API access instead."
                )
            raise AuthenticationError(detail or "API key is inactive.")
        if resp.status_code == 503:
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = ""
            raise BulkNotYetAvailable(detail) if detail else BulkNotYetAvailable()
        # Delegate everything else to the standard handler.
        Client._raise_for_status(resp)

    # ------------------------------------------------------------------
    # Source-specific helpers
    # ------------------------------------------------------------------

    def statsnz(self, name, **kwargs) -> Dataset:
        """Fetch a Stats NZ dataset."""
        return self._get_source(name, "Stats NZ", **kwargs)

    def oecd(self, name, **kwargs) -> Dataset:
        """Fetch an OECD dataset."""
        return self._get_source(name, "OECD", **kwargs)

    def rbnz(self, name, **kwargs) -> Dataset:
        """Fetch an RBNZ dataset."""
        return self._get_source(name, "RBNZ", **kwargs)

    def treasury(self, name, **kwargs) -> Dataset:
        """Fetch an NZ Treasury dataset."""
        return self._get_source(name, "NZ Treasury", **kwargs)

    def linz(self, name, **kwargs) -> Dataset:
        """Fetch a LINZ dataset."""
        return self._get_source(name, "LINZ", **kwargs)

    def statsnz_geo(self, name, **kwargs) -> Dataset:
        """Fetch a Stats NZ geospatial dataset (boundaries, census meshblocks, etc.).

        Kept as a convenience helper for discoverability — the server returns
        ``source = "Stats NZ"`` for both SDMX time series and Datafinder
        geospatial datasets, so the metadata on the returned Dataset reads
        ``"Stats NZ"`` (not ``"Stats NZ Geospatial"``).
        """
        return self._get_source(name, "Stats NZ", **kwargs)

    def mbie(self, name, **kwargs) -> Dataset:
        """Fetch an MBIE dataset."""
        return self._get_source(name, "MBIE", **kwargs)

    def nzta(self, name, **kwargs) -> Dataset:
        """Fetch a Waka Kotahi (NZTA) dataset."""
        return self._get_source(name, "Waka Kotahi", **kwargs)

    def msd(self, name, **kwargs) -> Dataset:
        """Fetch an MSD dataset."""
        return self._get_source(name, "MSD", **kwargs)

    def police(self, name, **kwargs) -> Dataset:
        """Fetch an NZ Police / MoJ dataset."""
        return self._get_source(name, "NZ Police / MoJ", **kwargs)

    def acc(self, name, **kwargs) -> Dataset:
        """Fetch an ACC dataset."""
        return self._get_source(name, "ACC", **kwargs)

    def edcounts(self, name, **kwargs) -> Dataset:
        """Fetch an Education Counts dataset."""
        return self._get_source(name, "Education Counts", **kwargs)

    def eeca(self, name, **kwargs) -> Dataset:
        """Fetch an EECA dataset (NZ energy use, EV chargers, regional heat demand).

        Examples::

            client.eeca("eeca_energy_end_use")           # NZ energy by sector x fuel x end-use x year
            client.eeca("eeca_ev_chargers_public")       # public EV charging network (point geometry)
            client.eeca("eeca_ev_metrics_district")      # EV penetration by territorial authority
            client.eeca("eeca_regional_heat_demand")     # industrial process heat by region x sector

        Notes:
            From the Energy Efficiency and Conservation Authority. EV charger streams
            refresh quarterly; Energy End Use Database is annual; Regional Heat Demand
            is an Aug 2024 snapshot. CC-BY 4.0 NZ.
        """
        return self._get_source(name, "EECA", **kwargs)

    def worksafe(self, name, **kwargs) -> Dataset:
        """Fetch a WorkSafe NZ dataset."""
        return self._get_source(name, "WorkSafe NZ", **kwargs)

    def immigration(self, name, **kwargs) -> Dataset:
        """Fetch an Immigration NZ dataset."""
        return self._get_source(name, "Immigration NZ", **kwargs)

    def geonet(self, name, **kwargs) -> Dataset:
        """Fetch a GeoNet dataset (NZ earthquakes, volcanic alert levels, strong-motion sensors).

        Examples::

            client.geonet("geonet_quakes_recent")              # rolling ~100 recent MMI>=3 quakes
            client.geonet("geonet_volcanic_alert_levels")      # 12 monitored NZ volcanoes
            client.geonet("geonet_strong_motion_sensors")      # 25 strong-motion stations

        Notes:
            Refreshed every 6 hours from api.geonet.org.nz. Earthquake catalogue is a
            rolling window of recent events, not a historical archive. CC-BY 3.0 NZ
            (Earth Sciences New Zealand, formerly GNS Science).
        """
        return self._get_source(name, "GeoNet", **kwargs)

    def lris(self, name, **kwargs) -> Dataset:
        """Fetch a Manaaki Whenua LRIS dataset (land cover, soil, protected areas).

        Examples::

            client.lris("lcdb_v6_mainland")   # current NZ land cover (~543k polygons)
            client.lris("nzlum_v03")           # NZ Land Use Management v0.3
            client.lris("pan_nz_2025_draft")   # protected areas (Draft, 2025)

        Notes:
            LCDB v3.0–v4.1 are deprecated vintages, retained for longitudinal
            analysis. LCDB v5 is superseded by v6 but still served.
            PAN-NZ 2025 was marked Draft at the time of ingestion (2026-05-12).
            Source: https://lris.scinfo.org.nz
            Licence: CC-BY 4.0 International (LCDB v5/v6, NZLUM, PBC, PAN-NZ);
            CC-BY 3.0 NZ (LCDB v3/v4 vintages). Attribution: Manaaki Whenua.
        """
        return self._get_source(name, "Manaaki Whenua / LRIS", **kwargs)

    def doc(self, name, **kwargs) -> Dataset:
        """Fetch a DOC (Department of Conservation) dataset.

        Examples::

            client.doc("doc_public_conservation_land")   # ~11k polygons of NZ public conservation land
            client.doc("doc_huts")                        # 1,429 DOC huts (Point geometry)
            client.doc("doc_tracks")                      # 3,248 DOC tracks (Polyline)

        Notes:
            Refreshed weekly from DOC's ArcGIS hub. Operational alert streams
            (track closures, hazard notices) are wired but currently blocked on
            an API key issue; they will appear automatically once resolved.
            CC-BY 4.0 International (Crown / Department of Conservation).
        """
        return self._get_source(name, "DOC", **kwargs)

    def akl_council(self, name, **kwargs) -> Dataset:
        """Fetch an Auckland Council dataset (overlays, heritage, hazards, zoning).

        Examples::

            client.akl_council("akc_notable_trees_overlay")
            client.akl_council("akc_significant_ecological_areas_overlay")
            client.akl_council("akc_historic_heritage_overlay_place")

        Notes:
            Open data from the Auckland Council ArcGIS hub. Covers district
            plan overlays, heritage areas, ecological areas, stormwater
            management zones, and more. CC-BY 4.0 (Auckland Council).
            Source: https://data-aucklandcouncil.opendata.arcgis.com
        """
        return self._get_source(name, "Auckland Council", **kwargs)

    def akl_transport(self, name, **kwargs) -> Dataset:
        """Fetch an Auckland Transport dataset (roads, public transport, cycling).

        Examples::

            client.akl_transport("akt_bus_stop")
            client.akl_transport("akt_bus_route")
            client.akl_transport("akt_cycle_facility_network")

        Notes:
            Open data from Auckland Transport (AT). Covers bus stops,
            bus routes, bridges, cycle infrastructure, and more.
            CC-BY 4.0 (Auckland Transport).
            Source: https://data-atgis.opendata.arcgis.com
        """
        return self._get_source(name, "Auckland Transport", **kwargs)

    def bay_of_plenty(self, name, **kwargs) -> Dataset:
        """Fetch a Bay of Plenty Councils dataset (hazards, resource consents, planning).

        Examples::

            client.bay_of_plenty("boprc_historic_flood_extents")
            client.bay_of_plenty("boprc_liquefaction_level_b")
            client.bay_of_plenty("boprc_rcep_ascv")

        Notes:
            Open data from Bay of Plenty Regional Council and its territorial
            authorities. Covers flood extents, liquefaction, coastal hazards,
            resource consents, and planning layers. CC-BY 4.0.
            Source: https://www.boprc.govt.nz
        """
        return self._get_source(name, "Bay of Plenty Councils", **kwargs)

    def charities(self, name, **kwargs) -> Dataset:
        """Fetch a Charities Services dataset (registered NZ charities).

        Examples::

            client.charities("charities_organisations")
            client.charities("charities_annual_returns")
            client.charities("charities_activities")

        Notes:
            Data from Charities Services (a business unit of the Department
            of Internal Affairs). Covers registered charities, officers,
            beneficiary groups, and annual financial returns.
            Open Government Licence v3.0.
            Source: https://www.charities.govt.nz
        """
        return self._get_source(name, "Charities Services", **kwargs)

    def colab_waikato(self, name, **kwargs) -> Dataset:
        """Fetch a Co-Lab Waikato dataset (planning, hazards, heritage across Waikato councils).

        Examples::

            client.colab_waikato("wmkdc_buildings")
            client.colab_waikato("tcdc_dp_coastal_environment")
            client.colab_waikato("wbopdc_coastal_erosion")

        Notes:
            Data aggregated via the Co-Lab Waikato open data hub. Covers
            district plan zones, coastal hazards, heritage, and building
            footprints across Waikato-region territorial authorities.
            CC-BY 4.0 (respective councils).
            Source: https://data-waikatolass.opendata.arcgis.com
        """
        return self._get_source(name, "Co-Lab Waikato", **kwargs)

    def ecan_canterbury(self, name, **kwargs) -> Dataset:
        """Fetch an ECan / Canterbury dataset (environment, hazards, resource consents).

        Examples::

            client.ecan_canterbury("ecan_liquefaction_susceptibility_final")
            client.ecan_canterbury("ecan_tsunami_evacuation_zones")
            client.ecan_canterbury("ecan_resource_consents_active_all")

        Notes:
            Open data from Environment Canterbury (ECan) and Canterbury-region
            councils. Covers liquefaction, earthquake faults, tsunami zones,
            water allocation, resource consents, and planning layers.
            CC-BY 4.0 (Environment Canterbury / respective councils).
            Source: https://opendata.canterburymaps.govt.nz
        """
        return self._get_source(name, "ECan / Canterbury", **kwargs)

    def hawkes_bay(self, name, **kwargs) -> Dataset:
        """Fetch a Hawke's Bay Councils dataset (hazards, planning, coastal management).

        Examples::

            client.hawkes_bay("hbrc_coastal_erosion_likely_66")
            client.hawkes_bay("hbrc_coastal_erosion_possible_33")
            client.hawkes_bay("hbrc_chb_hdc_wdc_liquefaction_severity")

        Notes:
            Open data from Hawke's Bay Regional Council and its territorial
            authorities. Covers coastal erosion, liquefaction, flood hazards,
            and district planning layers. CC-BY 4.0.
            Source: https://www.hbrc.govt.nz
        """
        return self._get_source(name, "Hawke's Bay Councils", **kwargs)

    def manawatu_whanganui(self, name, **kwargs) -> Dataset:
        """Fetch a Manawatu-Whanganui Councils dataset (airsheds, coastal, freshwater).

        Examples::

            client.manawatu_whanganui("horizons_coastal_marine_area")
            client.manawatu_whanganui("horizons_airshed_taihape")
            client.manawatu_whanganui("horizons_airshed_taumarunui")

        Notes:
            Open data from Horizons Regional Council (Manawatu-Whanganui) and
            its territorial authorities. Covers airsheds, coastal marine areas,
            freshwater, and planning layers. CC-BY 4.0.
            Source: https://www.horizons.govt.nz
        """
        return self._get_source(name, "Manawatu-Whanganui Councils", **kwargs)

    def napier_whanganui(self, name, **kwargs) -> Dataset:
        """Fetch a Napier or Whanganui city dataset (district plan, heritage, infrastructure).

        Examples::

            client.napier_whanganui("napier_heritage_buildings")
            client.napier_whanganui("napier_address_points")
            client.napier_whanganui("napier_parcels")

        Notes:
            Open data from Napier City Council and Whanganui District Council.
            Covers district plan precincts, heritage buildings and areas,
            address points, road centrelines, and parcels. CC-BY 4.0.
            Source: https://www.napier.govt.nz / https://www.whanganui.govt.nz
        """
        return self._get_source(name, "Napier + Whanganui", **kwargs)

    def northland(self, name, **kwargs) -> Dataset:
        """Fetch a Northland Councils dataset (district plans, designations, heritage).

        Examples::

            client.northland("fndc_district_plan_zones")
            client.northland("fndc_heritage_areas")
            client.northland("fndc_designations")

        Notes:
            Open data from Northland Regional Council and its territorial
            authorities (Far North, Whangarei, Kaipara). Covers district plan
            zones, designations, heritage, and environmental layers. CC-BY 4.0.
            Source: https://www.nrc.govt.nz
        """
        return self._get_source(name, "Northland Councils", **kwargs)

    def otago(self, name, **kwargs) -> Dataset:
        """Fetch an Otago Councils dataset (land use, water, planning, hazards).

        Examples::

            client.otago("orc_otago_irrigated_areas")
            client.otago("orc_otago_land_use_2024")
            client.otago("orc_floodbanks")

        Notes:
            Open data from Otago Regional Council and its territorial
            authorities (Dunedin, Queenstown-Lakes, Central Otago, Clutha,
            Waitaki). Covers land use, floodbanks, groundwater protection,
            and planning layers. CC-BY 4.0.
            Source: https://www.orc.govt.nz
        """
        return self._get_source(name, "Otago Councils", **kwargs)

    def pharmac(self, name, **kwargs) -> Dataset:
        """Fetch a PHARMAC dataset (NZ pharmaceutical subsidy schedule + hospital medicines).

        Examples::

            client.pharmac("pharmac_schedule")             # current month's funded medicines
            client.pharmac("pharmac_schedule_history")     # 2006-present subsidy archive
            client.pharmac("pharmac_hospital_medicines_list")  # current HML
            client.pharmac("pharmac_hml_history")          # 2011-present HML archive

        Notes:
            Monthly snapshots of NZ's national drug funding schedule + hospital
            formulary, from PHARMAC (Pharmaceutical Management Agency).
            Historical archives are append-mode — each month's snapshot is tagged
            with ``time_frame`` (YYYY-MM). CC-BY 3.0 NZ.
        """
        return self._get_source(name, "PHARMAC", **kwargs)

    def southland(self, name, **kwargs) -> Dataset:
        """Fetch a Southland Councils dataset (district plans, coastal, natural hazards).

        Examples::

            client.southland("sdc_southland_dp_zones")
            client.southland("sdc_southland_dp_heritage_items")
            client.southland("es_southland_land_use_2025")

        Notes:
            Open data from Environment Southland and its territorial
            authorities (Southland District, Gore, Invercargill). Covers
            district plan zones, coastal hazards, heritage, and land use.
            CC-BY 4.0.
            Source: https://www.es.govt.nz
        """
        return self._get_source(name, "Southland Councils", **kwargs)

    def taranaki(self, name, **kwargs) -> Dataset:
        """Fetch a Taranaki Councils dataset (coastal, biodiversity, district plans).

        Examples::

            client.taranaki("trc_biodiversity_coastal_mgmt_areas")
            client.taranaki("npdc_dp_operative_coastal_flooding")
            client.taranaki("npdc_dp_operative_archaeological")

        Notes:
            Open data from Taranaki Regional Council and its territorial
            authorities (New Plymouth, Stratford, South Taranaki). Covers
            biodiversity, coastal management, and district planning layers.
            CC-BY 4.0.
            Source: https://www.trc.govt.nz
        """
        return self._get_source(name, "Taranaki Councils", **kwargs)

    def top_of_south(self, name, **kwargs) -> Dataset:
        """Fetch a Gisborne / Top of South Councils dataset (coastal, planning, heritage).

        Examples::

            client.top_of_south("gdc_coastal_environment")
            client.top_of_south("gdc_coastal_erosion")
            client.top_of_south("gdc_coastal_flooding")

        Notes:
            Open data from Gisborne District Council, Marlborough District
            Council, Nelson City Council, and Tasman District Council.
            Covers coastal hazards, planning zones, and heritage layers.
            CC-BY 4.0.
            Source: https://www.gdc.govt.nz
        """
        return self._get_source(name, "Gisborne / Top of South Councils", **kwargs)

    def wellington(self, name, **kwargs) -> Dataset:
        """Fetch a Wellington Region Councils dataset (hazards, planning, infrastructure).

        Examples::

            client.wellington("wcc_district_plan_zones_2024")
            client.wellington("wcc_flood_hazard_operative")
            client.wellington("gwrc_flood_1pct_aep")

        Notes:
            Open data from Greater Wellington Regional Council and its
            territorial authorities (Wellington, Hutt, Upper Hutt, Porirua,
            Kapiti Coast). Covers flood and earthquake hazards, district plan
            zones, and coastal inundation. CC-BY 4.0.
            Source: https://www.gw.govt.nz
        """
        return self._get_source(name, "Wellington Region Councils", **kwargs)

    def list_wellington(self) -> pd.DataFrame:
        """Return metadata for all Wellington Region Councils datasets."""
        return self.list(source="Wellington Region Councils")

    def west_coast(self, name, **kwargs) -> Dataset:
        """Fetch a West Coast (Te Tai o Poutini) dataset (faults, landslides, planning).

        Examples::

            client.west_coast("wcrc_active_faults")
            client.west_coast("wcrc_alpine_fault_traces")
            client.west_coast("wcrc_landslide_catalog")

        Notes:
            Open data from West Coast Regional Council (Te Tai o Poutini) and
            its territorial authorities (Buller, Grey, Westland). Covers
            active faults, the Alpine Fault, landslide catalogs, and
            significant natural areas. CC-BY 4.0.
            Source: https://www.ttpp.nz
        """
        return self._get_source(name, "West Coast (Te Tai o Poutini)", **kwargs)

    def _get_source(self, name, source: str, **kwargs) -> Dataset:
        result = self.get(name, **kwargs)
        # When as_arrow=True the result is a pyarrow.Table which has no
        # eolas_source attribute — skip the metadata tag silently.
        try:
            result.eolas_source = source
        except (AttributeError, TypeError):
            pass
        self._warn_source_mismatch(name, source, result)
        return result

    @staticmethod
    def _warn_source_mismatch(name: str, expected_source: str, result) -> None:
        meta = getattr(result, "eolas_meta", None) or {}
        actual = (meta.get("source") or "").strip()
        if not actual or actual == expected_source:
            return
        from .console import console
        from .search import CPI_INDEX_DATASET, cpi_guidance_message

        console.print(
            f"[yellow]Warning:[/yellow] {name!r} is sourced from {actual!r}, "
            f"not {expected_source!r}. See client.info({name!r}) for canonical metadata."
        )
        if expected_source == "Stats NZ" and str(name) == "nz_cpi":
            console.print(f"[dim]{cpi_guidance_message()}[/dim]")
        elif str(name) == "nz_cpi" and expected_source == "OECD":
            console.print(
                f"[dim]For CPI index levels use {CPI_INDEX_DATASET!r} (RBNZ).[/dim]"
            )

    # ------------------------------------------------------------------
    # Bulk-cache convenience read (mirrors R eolas_get_local)
    # ------------------------------------------------------------------

    def get_local(
        self,
        name: Union[str, "DatasetName"],
        *,
        cache_dir: Optional[Union[str, "pathlib.Path"]] = None,
        format: Optional[str] = None,
        freshness: str = "auto",
        as_geo: Optional[bool] = None,
        as_arrow: bool = False,
        meta: bool = True,
        progress: ProgressControl = None,
        force: bool = False,
    ) -> "pd.DataFrame":
        """Download (or serve from cache) a whole dataset as a local DataFrame.

        The recommended path for large or geospatial datasets in a notebook
        workflow, and the Python mirror of R's ``eolas_get_local()``. On the
        first call it fetches the bulk file from CDN (milliseconds for monthly
        snapshots) and writes it to the configured library directory (see
        :func:`eolas_data.library.resolve_library_dir`). On subsequent calls a
        lightweight HEAD request checks whether the file is still current; if so
        the local copy is read directly with zero network I/O on the payload.

        If ``client.get("nz_parcels")`` on a 3-million-row geospatial dataset
        takes 15+ minutes, use ``get_local`` instead — it serves a
        pre-materialised GeoParquet from CDN, not a live Iceberg scan.

        Args:
            name: Dataset identifier, e.g. ``"nz_parcels"``.
            cache_dir: Local directory for cached files (``~``-expanded). Created
                if absent. ``None`` (default) resolves via the library
                precedence chain (``EOLAS_LIBRARY`` env → ``library_dir`` in
                ``~/.eolas/config.json`` → interactive prompt on first TTY call →
                ``~/.cache/eolas/``). An explicit value always wins.
            format: ``"parquet"``, ``"csv_gz"``, or ``"geoparquet"``. ``None``
                (default) auto-detects from dataset metadata (geo → geoparquet).
            freshness: ``"auto"`` (default), ``"monthly"``, or ``"current"`` —
                passed verbatim to :meth:`sync_bulk`.
            as_geo: When ``True`` (default) and the file is GeoParquet and
                geopandas is installed, returns a ``geopandas.GeoDataFrame``;
                otherwise a plain ``pd.DataFrame`` with the raw WKB column.
                Ignored when ``as_arrow=True``.
            as_arrow: When ``True``, return a ``pyarrow.Table`` directly (no
                geometry materialisation). Cannot be combined with
                ``as_geo=True`` (raises ``ValueError``).
            meta: When ``True`` (default), attach table/column metadata from
                :meth:`info` (session-cached). Pass ``False`` to skip the extra
                round-trip.
            progress: Control progress for two phases: **download** (streaming
                byte bar via :meth:`sync_bulk`) and **read** (spinner while
                Parquet/GeoParquet is loaded into a DataFrame). ``None``
                (default) enables both in interactive sessions. ``True``/``False``
                force both on/off. Use ``"download"``, ``"read"``, ``"both"``,
                or ``"none"`` for one phase only. Suppressed by
                ``EOLAS_NO_PROGRESS=1``. Cached snapshots skip the download bar
                and print an informative message instead.
            force: Re-download the library file even when the sidecar says the
                snapshot is current. See :meth:`cache_clear`.

        Returns:
            ``pd.DataFrame`` (tabular) or ``geopandas.GeoDataFrame`` (geo +
            ``as_geo`` + geopandas installed) or ``pyarrow.Table`` (``as_arrow``).

        Raises:
            BulkUpgradeRequired / BulkLicenceRestricted / BulkNotYetAvailable:
                propagate unchanged from :meth:`sync_bulk`.

        See Also:
            :meth:`sync_bulk` — advanced control over the sync lifecycle.
        """
        # ---- as_arrow / as_geo conflict guard (only when BOTH explicit) ------
        if as_arrow and as_geo is True:
            raise ValueError(
                "as_arrow=True and as_geo=True are mutually exclusive. "
                "as_arrow returns a pyarrow.Table (no geometry materialisation); "
                "as_geo materialises geometry as shapely objects in a GeoDataFrame. "
                "Choose one."
            )
        # Resolve as_geo: None → True (auto) unless as_arrow overrides.
        resolved_as_geo = as_geo if as_geo is not None else (not as_arrow)
        self._apply_force(name, force)

        # ---- resolve cache_dir -----------------------------------------------
        if cache_dir is not None:
            cache_path = pathlib.Path(cache_dir).expanduser().resolve()
        else:
            cache_path = resolve_library_dir()
        cache_path.mkdir(parents=True, exist_ok=True)

        # ---- auto-detect format if not specified -----------------------------
        info_meta: Optional[dict] = None
        if format is None or meta:
            try:
                info_meta = self._info_cached(name) if meta else self.info(name)
            except Exception:
                info_meta = None
        if format is None:
            try:
                gt = (info_meta or {}).get("geometry_type")
                wkt = (info_meta or {}).get("geometry_wkt")
                gt_truthy = bool(gt) and gt != "none"
                wkt_truthy = bool(wkt) and wkt != "none"
                is_geo = gt_truthy or wkt_truthy or bool((info_meta or {}).get("has_geometry"))
            except Exception:
                is_geo = False
            fmt = "geoparquet" if is_geo else "parquet"
        else:
            fmt = format.lower()
            if fmt not in self._BULK_EXTENSIONS:
                raise ValueError(
                    f"Unknown format {format!r}. Expected one of: "
                    + ", ".join(self._BULK_EXTENSIONS)
                )

        # ---- compute local file path -----------------------------------------
        ext = self._BULK_EXTENSIONS[fmt]
        file_path = cache_path / f"{name}{ext}"

        # ---- sync (download if needed, HEAD check if cached) -----------------
        self.sync_bulk(
            name, path=file_path, format=fmt, freshness=freshness,
            progress=progress, force=force,
        )

        show_read = self._resolve_show_progress(progress, "read")
        read_label = file_path.name

        # ---- read the local file into a DataFrame ----------------------------
        if as_arrow:
            import pyarrow.parquet as _pq
            with self._with_read_progress(read_label, show_read):
                if fmt == "csv_gz":
                    import pyarrow as _pa
                    return _pa.Table.from_pandas(
                        pd.read_csv(file_path), preserve_index=False,
                    )
                return _pq.read_table(file_path)

        def _read_bulk_file(path: pathlib.Path, bulk_fmt: str):
            with self._with_read_progress(read_label, show_read):
                if bulk_fmt == "csv_gz":
                    return pd.read_csv(path)
                if bulk_fmt == "geoparquet" and resolved_as_geo:
                    try:
                        import geopandas as gpd
                        return gpd.read_parquet(path)
                    except ImportError:
                        pass
                return pd.read_parquet(path)

        try:
            result = _read_bulk_file(file_path, fmt)
        except OSError as exc:
            # Bulk snapshots are written with pyarrow 24+ (Parquet 2.6). Older
            # pyarrow builds fail with "Repetition level histogram size mismatch".
            # csv_gz is always generated alongside parquet — fall back to it.
            if fmt in ("parquet", "geoparquet") and "histogram" in str(exc).lower():
                _log.warning(
                    "Parquet read failed for %s (%s) — falling back to csv_gz bulk",
                    name, exc,
                )
                csv_path = cache_path / f"{name}{self._BULK_EXTENSIONS['csv_gz']}"
                self.sync_bulk(
                    name, path=csv_path, format="csv_gz",
                    freshness=freshness, progress=progress, force=force,
                )
                result = _read_bulk_file(csv_path, "csv_gz")
            else:
                raise
        return self._attach_dataset_meta(result, name, meta=meta)

    # ------------------------------------------------------------------
    # Core data fetch
    # ------------------------------------------------------------------

    def get(
        self,
        name: Union[str, "DatasetName"],
        start: Optional[str] = None,
        end: Optional[str] = None,
        format: str = "json",
        engine: str = "pandas",
        limit: Optional[int] = None,
        as_geo: Optional[bool] = None,
        as_arrow: bool = False,
        meta: bool = True,
        envelope: bool = False,
        force: bool = False,
        progress: ProgressControl = None,
    ) -> Dataset:
        """Fetch dataset rows as a pandas (or polars / geopandas) DataFrame.

        Hits the live ``/v1/datasets/{name}/data`` endpoint directly.  Use
        :meth:`download_bulk` or :meth:`sync_bulk` for large datasets or
        whole-dataset pulls.

        Args:
            name:   Dataset identifier, e.g. ``"nz_cpi"``. Type-checked against
                    the ``DatasetName`` Literal at static-analysis time so
                    IDEs autocomplete the catalog.
            start:  ISO date lower bound, e.g. ``"2020-01-01"``.
            end:    ISO date upper bound, e.g. ``"2024-12-31"``.
            format: ``"json"`` (default) or ``"csv"``.
            engine: ``"pandas"`` (default) or ``"polars"``.
            limit:  Max rows to return. Default ``None`` requests the full dataset
                    (server enforces a 50,000-row cap on Free/Starter plans; Pro is
                    unlimited). When set with a ``date`` column, returns the
                    **most recent** N rows (not the oldest).
            as_geo: Convert geospatial datasets to a ``GeoDataFrame``.
                    ``None`` (default) auto-converts when the dataset has a
                    ``geometry_wkt`` column AND ``geopandas`` is importable.
                    ``True`` forces the conversion (raises if geopandas missing).
                    ``False`` keeps the raw WKT string column.
                    Install with ``pip install eolas-data[geo]``.
                    Cannot be combined with ``as_arrow=True``.
            as_arrow: When ``True``, return a ``pyarrow.Table`` instead of a
                    ``DataFrame`` / ``GeoDataFrame``.  Geometry stays as Arrow
                    buffers — zero-copy, no shapely allocation.  On the live
                    path the Arrow IPC response is returned directly when the
                    server supports it; otherwise a JSON response is converted
                    via ``pa.Table.from_pandas``.
                    Cannot be combined with ``as_geo=True`` (raises
                    ``ValueError``).
            meta: When ``True`` (default), attach table/column metadata from
                    :meth:`info` (session-cached on this client). Pass
                    ``False`` to skip the extra round-trip.
            envelope: When ``True``, request ``?envelope=1`` (JSON only) and
                    attach the ``data_sources`` licence block alongside rows.
                    Response ``X-Eolas-*`` headers are merged into metadata.
            force: When ``True`` and the request auto-routes to :meth:`get_local`,
                    re-download the bulk file even when the sidecar says current.
            progress: Control bulk download/read progress when auto-routed to
                    :meth:`get_local`. ``None`` auto-detects TTY; ignored on the
                    live API path.

        Returns:
            A :class:`Dataset` (pandas DataFrame subclass), a polars DataFrame
            when ``engine="polars"``, or a ``geopandas.GeoDataFrame`` when
            geometry is present and conversion is enabled.

        Examples::

            df  = client.get("nz_cpi")
            df  = client.get("nz_cpi", start="2020-01-01")
            gdf = client.get("linz_nz_parcels", limit=100)

            # Whole-dataset bulk download → use download_bulk or sync_bulk instead.
        """
        # as_arrow + as_geo conflict — check early so we fail fast.
        if as_arrow and as_geo:
            raise ValueError(
                "as_arrow=True and as_geo=True are mutually exclusive. "
                "as_arrow returns a pyarrow.Table (no geometry materialisation); "
                "as_geo materialises geometry as shapely objects in a GeoDataFrame. "
                "Choose one."
            )
        if envelope and (format != "json" or as_arrow):
            raise ValueError(
                "envelope=True requires format='json' and as_arrow=False."
            )

        from .meta import resolve_date_bounds
        from .rows import apply_row_limit, resolve_fetch_limit, sort_by_date

        # ---- start/end only apply when the dataset has a date filter column ---
        _date_bounds_info: Optional[dict] = None
        if start is not None or end is not None:
            try:
                _date_bounds_info = self._info_cached(name)
            except Exception:
                pass
            start, end, _stripped = resolve_date_bounds(_date_bounds_info, start, end)
            if _stripped:
                import warnings
                warnings.warn(
                    f"start=/end= ignored for {name!r}: this dataset has no date "
                    "filter column (not a time-series table). Use limit= for row "
                    "caps or get_local() for the full table.",
                    UserWarning,
                    stacklevel=2,
                )

        # ---- whole-dataset pull on large/geo tables → bulk cache -------------
        # Matches the API 413 guard: limit=0 with no start/end on >100k-row or
        # geometry datasets is refused. Transparently serve from get_local()
        # (CDN-backed Parquet/GeoParquet) so client.get("nz_addresses") works.
        if (
            limit is None
            and start is None
            and end is None
            and format == "json"
            and not envelope
            and not as_arrow
        ):
            try:
                info_meta = self._info_cached(name)
                if self._bulk_export_allowed(info_meta) and self._live_pull_blocked(info_meta):
                    result = self.get_local(
                        name,
                        as_geo=as_geo,
                        as_arrow=False,
                        meta=meta,
                        force=force,
                        progress=progress,
                    )
                    if engine == "polars":
                        try:
                            import polars as pl
                            return pl.from_pandas(result)
                        except ImportError:
                            raise ImportError(
                                "polars is required for engine='polars'. "
                                "Install with: pip install eolas-data[polars]"
                            )
                    return result
            except (BulkLicenceRestricted, BulkUpgradeRequired, BulkNotYetAvailable):
                raise
            except Exception:
                pass  # fall through to live path (e.g. metadata lookup failed)

        # ---- early in-memory cache check -------------------------------------
        _early_cache_key = f"{name}:{start}:{end}:{format}:{0 if limit is None else int(limit)}:{as_geo}"
        if self._cache is not None and _early_cache_key in self._cache:
            return self._cache[_early_cache_key]

        # ---- live path -------------------------------------------------------
        params: dict = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        fetch_limit, user_limit = resolve_fetch_limit(limit)
        # Positive limits on large/geo datasets must be sent to the API — the
        # client normally uses limit=0 and trims client-side for dated series,
        # but limit=0 triggers the API 413 guard on geometry / >100k tables.
        if user_limit and int(user_limit) > 0 and start is None and end is None:
            try:
                info_meta = self._info_cached(name)
                if self._live_pull_blocked(info_meta):
                    fetch_limit = int(user_limit)
            except Exception:
                pass
        params["limit"] = fetch_limit

        cache_key = f"{name}:{start}:{end}:{format}:{0 if limit is None else int(limit)}:{as_geo}"
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]

        data_sources = None
        provenance = None
        if format == "csv":
            from io import StringIO
            resp = self._raw_get(f"/v1/datasets/{name}/data", params={"format": "csv", **params})
            provenance = resp.headers
            df   = pd.read_csv(StringIO(resp.text))
        else:
            df, provenance, data_sources = self._fetch_dataframe(
                name, params, envelope=envelope,
            )
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

        df = sort_by_date(df)
        df = apply_row_limit(df, user_limit)

        # as_arrow on the live path: convert the pandas DataFrame to an Arrow
        # Table, avoiding any shapely allocation.  We convert before the
        # geo step so geometry_wkt stays as a string column in the Arrow table.
        if as_arrow:
            import pyarrow as _pa
            tbl = _pa.Table.from_pandas(df, preserve_index=False)
            if self._cache is not None:
                self._cache[cache_key] = tbl
            return tbl

        result = Dataset(df)
        result = self._attach_dataset_meta(
            result, name, meta=meta,
            provenance=provenance, data_sources=data_sources,
        )

        if engine == "polars":
            try:
                import polars as pl
                return pl.from_pandas(result)
            except ImportError:
                raise ImportError(
                    "polars is required for engine='polars'. "
                    "Install with: pip install eolas-data[polars]"
                )

        # Optional geopandas conversion. When as_geo=None we auto-convert if both
        # (a) the dataset has a geometry_wkt column AND (b) geopandas is importable.
        if as_geo is not False and "geometry_wkt" in result.columns:
            converted = _to_geodataframe(result, force=as_geo is True)
            if converted is not None:
                result = converted

        if self._cache is not None:
            self._cache[cache_key] = result

        return result

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _fetch_dataframe(
        self, name, params: dict, *, envelope: bool = False,
    ) -> tuple[pd.DataFrame, object, Optional[list]]:
        """Fetch dataset rows as a DataFrame, negotiating Arrow IPC over the
        wire (≈5x faster end-to-end, ≈82x faster parse than JSON on large
        pulls — benchmarked 2026-05-18). Transparently falls back to JSON for
        older servers, unexpected content-types, or any pyarrow issue, so the
        returned DataFrame is identical either way.

        Returns ``(dataframe, response_headers, data_sources_or_none)``.
        """
        if envelope:
            resp = self._raw_get(
                f"/v1/datasets/{name}/data",
                params={**params, "envelope": 1},
            )
            payload = resp.json()
            records = payload.get("data", payload) if isinstance(payload, dict) else payload
            sources = payload.get("data_sources") if isinstance(payload, dict) else None
            return pd.DataFrame(records), resp.headers, sources

        if self._arrow_supported is not False:
            try:
                import io
                import pyarrow as pa  # hard dependency; guarded for resilience

                resp = self._raw_get(
                    f"/v1/datasets/{name}/data",
                    params={**params, "format": "arrow"},
                )
                if "arrow" in resp.headers.get("content-type", ""):
                    self._arrow_supported = True
                    tbl = pa.ipc.open_stream(io.BytesIO(resp.content)).read_all()
                    return tbl.to_pandas(), resp.headers, None
                # Old server ignored format=arrow and returned JSON. Remember
                # so we don't pay the failed round-trip on every future call.
                self._arrow_supported = False
                nag_json_transport_once()
            except Exception:
                self._arrow_supported = False
                nag_json_transport_once()

        resp = self._raw_get(f"/v1/datasets/{name}/data", params=params)
        data = resp.json()
        records = data.get("data", data) if isinstance(data, dict) else data
        sources = data.get("data_sources") if isinstance(data, dict) else None
        return pd.DataFrame(records), resp.headers, sources

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        return self._raw_get(path, params=params).json()

    def _raw_get(
        self,
        path: str,
        params: Optional[dict] = None,
        *,
        stream: bool = False,
    ) -> requests.Response:
        url  = f"{self._base}{path}"
        resp = self._session.get(url, params=params, stream=stream)
        self._raise_for_status(resp)
        return resp

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code == 200:
            return
        if resp.status_code == 401:
            raise AuthenticationError(
                "Invalid or missing API key. Set the EOLAS_API_KEY environment "
                "variable, or run `eolas auth set-key`. "
                "Get a free key at https://eolas.fyi/signup"
            )
        if resp.status_code == 403:
            try:
                detail = resp.json().get("detail", "API key is inactive.")
            except Exception:
                detail = "API key is inactive."
            raise AuthenticationError(detail)
        if resp.status_code == 429:
            h = resp.headers
            limit  = h.get("X-RateLimit-Limit")
            retry  = h.get("Retry-After")
            reset  = h.get("X-RateLimit-Reset")
            # A 429 carrying our X-RateLimit-* headers came from the API; a 429
            # with only a cf-ray was thrown at the Cloudflare edge before origin.
            via_cf = bool(h.get("cf-ray")) and limit is None
            parts = ["Rate limit reached."]
            if limit:
                parts.append(f"Plan limit: {limit} requests.")
            if retry:
                parts.append(f"Retry after {retry}s.")
            elif reset:
                parts.append(f"Resets at {reset}.")
            if via_cf:
                parts.append(
                    f"(Blocked at the Cloudflare edge — cf-ray {h.get('cf-ray')}.)"
                )
            parts.append("Upgrade for higher limits: https://eolas.fyi/pricing")
            raise RateLimitError(" ".join(parts))
        if resp.status_code == 404:
            try:
                detail = resp.json().get("detail", "Not found.")
            except Exception:
                detail = "Not found."
            raise NotFoundError(detail)
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise APIError(resp.status_code, detail)
