"""Tests for the as_arrow parameter on get() and source helpers."""
from __future__ import annotations

import pathlib
from unittest.mock import patch

import pandas as pd
import pytest

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from eolas_data import Client

pytestmark = pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")

BASE = "https://api.eolas.fyi"


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


# ---------------------------------------------------------------------------
# Conflict: as_arrow=True + as_geo=True raises ValueError
# ---------------------------------------------------------------------------

def test_get_as_arrow_and_as_geo_raises(client):
    """as_arrow=True combined with as_geo=True on get() raises a clear ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.get("nz_cpi", as_arrow=True, as_geo=True)


# ---------------------------------------------------------------------------
# get(): as_arrow on the live path (JSON response → pa.Table)
# ---------------------------------------------------------------------------

def test_get_live_as_arrow_returns_arrow_table(client):
    """as_arrow=True on the live path converts the JSON-fetched DataFrame to pa.Table."""
    fake_df = pd.DataFrame({"date": ["2023-01-01"], "value": [1100.5]})

    with patch.object(client, "_fetch_dataframe", return_value=fake_df):
        result = client.get("nz_cpi", as_arrow=True)

    assert isinstance(result, pa.Table)
    assert "value" in result.schema.names
