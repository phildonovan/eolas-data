"""Dataset metadata attachment — table + column glosses from /v1/datasets/{name}."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

# Table-level fields we keep on Dataset.eolas_meta (exclude nested columns).
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


def attach_meta(
    df: pd.DataFrame,
    *,
    name: str,
    source: str = "",
    table_meta: Optional[dict] = None,
    column_meta: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Attach eolas metadata attrs to a DataFrame / Dataset / GeoDataFrame."""
    payload = {
        "eolas_name": name,
        "eolas_source": source,
        "eolas_meta": table_meta or {},
        "eolas_columns": column_meta,
    }
    attrs = getattr(df, "attrs", None)
    if isinstance(attrs, dict):
        attrs.update(payload)
    for key, val in payload.items():
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


def column_label(column_meta: Optional[pd.DataFrame], column: str) -> Optional[str]:
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