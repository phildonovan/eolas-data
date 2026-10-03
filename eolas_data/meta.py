"""Dataset metadata attachment — table + column glosses from /v1/datasets/{name}."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

# Table-level fields we keep on Dataset.eolas_meta (exclude nested columns).
_PROVENANCE_HEADERS = {
    "X-Eolas-Attribution": "attribution_text",
    "X-Eolas-Licence": "licence",
    "X-Eolas-Source": "source",
    "X-Eolas-Source-URL": "source_url",
    "X-Eolas-Namespace": "namespace",
}


def provenance_from_headers(headers) -> dict:
    """Extract response provenance from X-Eolas-* headers on /data responses."""
    out: dict[str, str] = {}
    get = (
        headers.get if hasattr(headers, "get") else lambda k, d=None: headers.get(k, d)
    )
    for hdr, key in _PROVENANCE_HEADERS.items():
        val = get(hdr)
        if val:
            out[key] = str(val)
    return out


def truncation_from_headers(headers) -> dict:
    """Read the plan-cap truncation contract from /data response headers.

    The server sets ``X-Eolas-Truncated: true`` (plus ``X-Plan-Row-Cap`` and
    ``X-Plan``) when the caller's plan capped the row window below what was
    requested. Returns ``{}`` when the header is absent (old server / bulk
    path), otherwise ``{"truncated": bool, "row_cap": int | None, "plan": str}``.
    """
    if headers is None:
        return {}
    get = (
        headers.get if hasattr(headers, "get") else lambda k, d=None: headers.get(k, d)
    )
    raw = get("X-Eolas-Truncated")
    if raw is None:
        return {}
    out: dict = {"truncated": str(raw).strip().lower() == "true"}
    cap = get("X-Plan-Row-Cap")
    if cap is not None:
        try:
            out["row_cap"] = int(cap)
        except (TypeError, ValueError):
            out["row_cap"] = None
    plan = get("X-Plan")
    if plan:
        out["plan"] = str(plan)
    return out


def truncation_message(
    name: str, trunc: dict, user_limit: Optional[int] = None
) -> Optional[str]:
    """Human-readable warning for a plan-capped /data response, or ``None``."""
    if not trunc.get("truncated"):
        return None
    cap = trunc.get("row_cap")
    plan = trunc.get("plan")
    cap_txt = f"{cap:,} rows" if isinstance(cap, int) else "the plan row cap"
    plan_txt = f" ({plan} plan)" if plan else ""
    msg = (
        f"{name!r}: response truncated to {cap_txt}{plan_txt}. This is a "
        "file-order slice, NOT the full dataset"
    )
    if user_limit and int(user_limit) > 0:
        msg += (
            f"; limit={int(user_limit)} returned the latest rows WITHIN that "
            "slice, not the dataset's most recent rows"
        )
    msg += (
        ". Check df.eolas_meta['truncated']. Use start=/end= to narrow, "
        "get_local()/sync_bulk() for the whole table, or upgrade at "
        "https://eolas.nz/pricing."
    )
    return msg


def merge_provenance(table_meta: Optional[dict], headers) -> dict:
    """Merge catalogue metadata with live response headers (headers win when set).

    Also stamps ``truncated`` / ``row_cap`` from the plan-cap headers so a
    capped slice is never presented as the full dataset (C22).
    """
    merged = dict(table_meta or {})
    for key, val in provenance_from_headers(headers).items():
        if val:
            merged[key] = val
    trunc = truncation_from_headers(headers)
    if trunc:
        merged["truncated"] = trunc["truncated"]
        if trunc["truncated"]:
            merged["row_cap"] = trunc.get("row_cap")
    return merged


_TABLE_META_KEYS = (
    "name",
    "namespace",
    "source",
    "country",
    "title",
    "description",
    "bulk_export_class",
    "geometry_type",
    "has_geometry",
    "row_count_at_last_refresh",
    "attribution_text",
    "licence",
    "source_url",
    "cdc_serving_tier",
    "pk_columns",
    "current_snapshot_id",
    "refresh_cadence",
    "observation_frequency",
    "last_refreshed_at",
    "previous_snapshots",
    "date_filter_column",
    # Long/wide reshape contract (2026-07-27) — consumed by
    # eolas_data.reshape.pivot_longer()/pivot_wider(). Without these in the
    # allowlist they would be silently dropped by split_meta() before ever
    # reaching df.attrs["eolas_meta"], even though the API returns them.
    "layout",
    "time_columns",
    "id_columns",
    "value_columns",
    "measure_name_column",
)

# Mirrors app/streaming.py::detect_date_col — columns that /data?start=&end= filter on.
DATE_FILTER_CANDIDATES = (
    "date",
    "time_frame",
    "open_date",
    "awarded_date",
    "start_date",
)


def split_meta(info: dict) -> tuple[dict, Optional[pd.DataFrame]]:
    """Split a /v1/datasets/{name} response into table meta and column glossary."""
    table = {k: info.get(k) for k in _TABLE_META_KEYS if k in info}
    raw_cols = info.get("columns")
    if not raw_cols:
        return table, None
    columns = pd.DataFrame(raw_cols)
    return table, columns


def _column_meta_records(
    column_meta: Optional[pd.DataFrame | list[dict[str, Any]]],
) -> Optional[list[dict[str, Any]]]:
    if column_meta is None:
        return None
    if isinstance(column_meta, pd.DataFrame):
        if column_meta.empty:
            return None
        return column_meta.to_dict("records")
    if isinstance(column_meta, list):
        return column_meta or None
    return None


def _column_meta_dataframe(
    column_meta: Optional[pd.DataFrame | list[dict[str, Any]]],
) -> Optional[pd.DataFrame]:
    if column_meta is None:
        return None
    if isinstance(column_meta, pd.DataFrame):
        return column_meta if not column_meta.empty else None
    if isinstance(column_meta, list) and column_meta:
        return pd.DataFrame(column_meta)
    return None


def _is_geodataframe(df: pd.DataFrame) -> bool:
    try:
        import geopandas as gpd
    except ImportError:
        return False
    return isinstance(df, gpd.GeoDataFrame)


def attach_meta(
    df: pd.DataFrame,
    *,
    name: str,
    source: str = "",
    table_meta: Optional[dict] = None,
    column_meta: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Attach eolas metadata attrs to a DataFrame / Dataset / GeoDataFrame."""
    col_records = _column_meta_records(column_meta)
    col_df = _column_meta_dataframe(column_meta)
    # attrs must be JSON-serialisable — never store a DataFrame there (breaks
    # pandas repr/head on GeoDataFrames when attrs are compared with ==).
    attrs_payload = {
        "eolas_name": name,
        "eolas_source": source,
        "eolas_meta": table_meta or {},
        "eolas_columns": col_records,
    }
    attrs = getattr(df, "attrs", None)
    if isinstance(attrs, dict):
        attrs.update(attrs_payload)

    if _is_geodataframe(df):
        # GeoDataFrame: attrs only — setattr triggers geopandas/pandas warnings
        # and can break tabular repr.
        return df

    object_payload = {
        "eolas_name": name,
        "eolas_source": source,
        "eolas_meta": table_meta or {},
        "eolas_columns": col_df,
    }
    for key, val in object_payload.items():
        try:
            # object.__setattr__ bypasses pandas' NDFrame.__setattr__, which on
            # pandas 3 emits a UserWarning for every non-column attribute set
            # (fired on every get() — PY-3). The accessor (df.eolas_name) still
            # works; the value lives in df.__dict__ exactly as before.
            object.__setattr__(df, key, val)
        except (AttributeError, TypeError):
            pass
    return df


def meta_subtitle(table_meta: Optional[dict]) -> str:
    """One-line subtitle for repr: title · refreshed {cadence}."""
    if not table_meta:
        return ""
    parts: list[str] = []
    title = (table_meta.get("title") or "").strip()
    if title:
        parts.append(title)
    cadence = (table_meta.get("refresh_cadence") or "").strip()
    if cadence:
        parts.append(f"refreshed {cadence}")
    if table_meta.get("truncated"):
        cap = table_meta.get("row_cap")
        parts.append(
            f"TRUNCATED to {cap:,} rows by plan cap"
            if isinstance(cap, int)
            else "TRUNCATED by plan cap"
        )
    return " · ".join(parts)


def column_label(
    column_meta: Optional[pd.DataFrame | list[dict[str, Any]]],
    column: str,
) -> Optional[str]:
    column_meta = _column_meta_dataframe(column_meta)
    if column_meta is None or column_meta.empty or "name" not in column_meta.columns:
        return None
    rows = column_meta.loc[column_meta["name"] == column, "description"]
    if rows.empty:
        return None
    val = rows.iloc[0]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    text = str(val).strip()
    return text or None


def date_filter_column_from_info(info: dict) -> Optional[str]:
    """Return the column name used for start/end filtering, or None if unsupported."""
    if not info:
        return None
    if "date_filter_column" in info:
        col = info.get("date_filter_column")
        return col if col else None
    raw_cols = info.get("columns")
    if not raw_cols:
        return None
    if isinstance(raw_cols, pd.DataFrame):
        names = (
            set(raw_cols["name"].astype(str)) if "name" in raw_cols.columns else set()
        )
    else:
        names = {
            str(c["name"]) for c in raw_cols if isinstance(c, dict) and c.get("name")
        }
    return next((c for c in DATE_FILTER_CANDIDATES if c in names), None)


def resolve_date_bounds(
    info: Optional[dict],
    start: Optional[str],
    end: Optional[str],
) -> tuple[Optional[str], Optional[str], bool]:
    """Return (start, end, stripped) — clear bounds when the dataset has no date axis.

    When stripped is True, callers should warn that start/end were ignored.
    If metadata is unavailable, bounds are passed through unchanged (safe default).
    """
    if start is None and end is None:
        return None, None, False
    if info is None:
        return start, end, False
    if date_filter_column_from_info(info):
        return start, end, False
    return None, None, True
