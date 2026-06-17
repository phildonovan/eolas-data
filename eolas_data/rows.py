"""Row-limit helpers — most-recent-N semantics for dated time series."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def resolve_fetch_limit(user_limit: Optional[int]) -> tuple[int, Optional[int]]:
    """Map a user ``limit`` to the server request limit.

    When the user passes a positive limit we request the full plan window
    (``0``) so we can sort by date and return the most recent rows client-side.
    """
    if user_limit is None:
        return 0, None
    ul = int(user_limit)
    if ul <= 0:
        return ul, ul
    return 0, ul


def apply_row_limit(df: pd.DataFrame, user_limit: Optional[int]) -> pd.DataFrame:
    """Return the most recent *user_limit* rows when a date column exists."""
    if user_limit is None or int(user_limit) <= 0 or df.empty:
        return df
    n = int(user_limit)
    if "date" in df.columns:
        return df.tail(n).reset_index(drop=True)
    return df.head(n).reset_index(drop=True)


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns or df.empty:
        return df
    return df.sort_values("date", kind="stable").reset_index(drop=True)