"""Live API smoke tests — run against api.eolas.fyi with a real key.

Skipped in the default unit CI job (``pytest -m 'not integration'``).
Enable locally or in the weekly smoke workflow::

    EOLAS_API_KEY=vs_... pytest -m integration -q
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from eolas_data import Client

pytestmark = pytest.mark.integration

_SKIP = not os.environ.get("EOLAS_API_KEY")


@pytest.fixture(scope="module")
def live_client():
    if _SKIP:
        pytest.skip("EOLAS_API_KEY not set — live smoke tests skipped")
    return Client()


def test_health_endpoint_reachable(live_client):
    import requests

    resp = requests.get(
        f"{live_client._base}/health",
        headers={"X-API-Key": live_client._key},
        timeout=30,
    )
    assert resp.status_code == 200


def test_list_returns_many_datasets(live_client):
    datasets = live_client.list()
    assert isinstance(datasets, pd.DataFrame)
    assert len(datasets) >= 100


def test_get_nz_cpi_slice(live_client):
    df = live_client.get("nz_cpi", limit=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1
    assert "value" in df.columns


def test_info_nz_cpi(live_client):
    meta = live_client.info("nz_cpi")
    assert meta["name"] == "nz_cpi"
    assert "source" in meta