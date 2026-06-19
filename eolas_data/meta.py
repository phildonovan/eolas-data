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
    get = headers.get if hasattr(headers, "get") else lambda k, d=None: headers.get(k, d)
    for hdr, key in _PROVENANCE_HEADERS.items():
        val = get(hdr)
        if val:
            out[key] = str(val)
    return out


def merge_provenance(table_meta: Optional[dict], headers) -> dict:
    """Merge catalogue metadata with live response headers (headers win when set)."""
    merged = dict(table_meta or {})
    for key, val in provenance_from_headers(headers).items():
        if val:
            merged[key] = val
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
    "last_refreshed_at",
    "previous_snapshots",
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
            setattr(df, key, val)
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