import pandas as pd

from eolas_data.search import filter_datasets, search_terms


def test_search_terms_expands_hlfs():
    assert "labour force" in search_terms("HLFS")


def test_search_terms_expands_ocr():
    assert "official cash rate" in search_terms("ocr")


def test_filter_datasets_hlfs_alias():
    df = pd.DataFrame([
        {"name": "rbnz_b2_wholesale_rates_monthly", "title": "Wholesale rates", "source": "RBNZ", "description": "OCR and swaps"},
        {"name": "nz_gdp", "title": "NZ GDP", "source": "OECD", "description": "Growth"},
    ])
    out = filter_datasets(df, "OCR")
    assert set(out["name"]) == {"rbnz_b2_wholesale_rates_monthly"}


def test_filter_datasets_hlfs_is_tight_and_ranked():
    df = pd.DataFrame([
        {"name": "leed_firms_employment", "title": "LEED employment", "source": "Stats NZ"},
        {"name": "nz_unemployment", "title": "NZ Unemployment Rate", "source": "OECD"},
        {"name": "rbnz_m9_labour_market", "title": "NZ Labour Market (RBNZ M9)", "source": "RBNZ"},
        {"name": "nz_gdp", "title": "NZ GDP", "source": "OECD"},
    ])
    out = filter_datasets(df, "HLFS")
    assert "leed_firms_employment" not in set(out["name"])
    assert list(out["name"][:2]) == ["nz_unemployment", "rbnz_m9_labour_market"]


def test_filter_datasets_cpi_ranks_rbnz_first():
    df = pd.DataFrame([
        {"name": "nz_cpi", "title": "NZ CPI inflation (annual % change)", "source": "OECD"},
        {"name": "rbnz_m1_prices", "title": "NZ Prices & Inflation (RBNZ M1)", "source": "RBNZ"},
    ])
    out = filter_datasets(df, "cpi")
    assert list(out["name"]) == ["rbnz_m1_prices", "nz_cpi"]


def test_maybe_cpi_guidance():
    from eolas_data.search import maybe_cpi_guidance

    assert maybe_cpi_guidance("cpi") is not None
    assert maybe_cpi_guidance("gdp") is None


def test_filter_datasets_plain_substring():
    df = pd.DataFrame([
        {"name": "nz_cpi", "title": "NZ CPI inflation", "source": "OECD"},
        {"name": "nz_gdp", "title": "NZ GDP", "source": "OECD"},
    ])
    out = filter_datasets(df, "cpi")
    assert list(out["name"]) == ["nz_cpi"]