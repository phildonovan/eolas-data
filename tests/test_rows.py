import pandas as pd

from eolas_data.rows import apply_row_limit, resolve_fetch_limit, sort_by_date


def test_resolve_fetch_limit_full_pull_for_recent():
    assert resolve_fetch_limit(None) == (0, None)
    assert resolve_fetch_limit(5) == (0, 5)
    assert resolve_fetch_limit(0) == (0, 0)


def test_apply_row_limit_returns_most_recent_dated_rows():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2018-01-01", "2020-01-01", "2025-01-01"]),
        "value": [1.0, 2.0, 3.0],
    })
    out = apply_row_limit(sort_by_date(df), 2)
    assert list(out["value"]) == [2.0, 3.0]


def test_apply_row_limit_head_when_no_date():
    df = pd.DataFrame({"x": [10, 20, 30]})
    out = apply_row_limit(df, 2)
    assert list(out["x"]) == [10, 20]