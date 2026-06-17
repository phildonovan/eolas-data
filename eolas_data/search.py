"""Dataset discovery helpers — substring search with common NZ analyst aliases."""
from __future__ import annotations

from typing import Optional

import pandas as pd

# Short tokens analysts use in conversation → extra needles for list/search.
_SEARCH_ALIASES: dict[str, tuple[str, ...]] = {
    # Tight HLFS alias — avoid matching 50+ LEED "employment" tables.
    "hlfs": (
        "hlfs",
        "household labour",
        "labour force survey",
        "labour force",
        "labour market",
        "unemployment rate",
    ),
    "ocr": (
        "ocr",
        "official cash rate",
        "cash rate",
    ),
    "cpi": (
        "cpi",
        "consumer price",
        "inflation",
        "prices",
    ),
}

# Canonical CPI choices — nz_cpi is OECD YoY %, not a CPI index level.
CPI_INDEX_DATASET = "rbnz_m1_prices"
CPI_INFLATION_YOY_DATASET = "nz_cpi"

# Headline datasets surfaced first for alias searches (lower = higher).
_SEARCH_RANK: dict[str, dict[str, int]] = {
    "cpi": {
        CPI_INDEX_DATASET: 0,
        "rbnz_m1_prices_longrun": 1,
        CPI_INFLATION_YOY_DATASET: 2,
    },
    "hlfs": {
        "nz_unemployment": 0,
        "rbnz_m9_labour_market": 1,
        "poppr_lab_national": 2,
        "poppr_lab_national_chars": 3,
    },
}

_DATASET_GUIDANCE: dict[str, str] = {
    CPI_INFLATION_YOY_DATASET: (
        "OECD quarterly CPI year-on-year % change — not a CPI index level. "
        f"For index levels use {CPI_INDEX_DATASET} (RBNZ, quarterly index)."
    ),
}


def cpi_guidance_message() -> str:
    return (
        f"nz_cpi is OECD annual % change (quarterly). "
        f"For CPI index levels use {CPI_INDEX_DATASET} (RBNZ, quarterly index)."
    )


def maybe_cpi_guidance(query: str) -> Optional[str]:
    q = (query or "").strip().lower()
    if q in ("cpi", "consumer price", "consumer price index", "inflation"):
        return cpi_guidance_message()
    return None


def search_terms(query: str) -> tuple[str, ...]:
    """Expand a user query into one or more case-insensitive needles."""
    q = (query or "").strip().lower()
    if not q:
        return ()
    return _SEARCH_ALIASES.get(q, (q,))


def filter_datasets(
    df: pd.DataFrame,
    query: str,
    *,
    source: Optional[str] = None,
) -> pd.DataFrame:
    """Filter a datasets DataFrame by optional source and search query."""
    if df.empty:
        return df
    out = df
    if source:
        out = out[out["source"] == source].reset_index(drop=True)
    needles = search_terms(query)
    if not needles:
        return out
    name_col = out["name"].astype(str) if "name" in out.columns else pd.Series("", index=out.index)
    title_col = out["title"].astype(str) if "title" in out.columns else pd.Series("", index=out.index)
    desc_col = (
        out["description"].astype(str)
        if "description" in out.columns
        else pd.Series("", index=out.index)
    )
    rank_key = (query or "").strip().lower()
    search_desc = rank_key != "hlfs"
    mask = pd.Series(False, index=out.index)
    for needle in needles:
        n = needle.lower()
        mask |= (
            name_col.str.lower().str.contains(n, regex=False, na=False)
            | title_col.str.lower().str.contains(n, regex=False, na=False)
        )
        if search_desc:
            mask |= desc_col.str.lower().str.contains(n, regex=False, na=False)
    out = out.loc[mask].reset_index(drop=True)
    if not out.empty and "name" in out.columns:
        priority = _SEARCH_RANK.get(rank_key, {})
        if priority:
            out = out.assign(
                _rank=out["name"].map(lambda n: priority.get(str(n), 99))
            ).sort_values("_rank", kind="stable").drop(columns="_rank").reset_index(drop=True)
    return out