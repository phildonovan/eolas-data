"""Live API smoke tests — run against api.eolas.fyi with a real key.

Mirrors eolas-r/tests/smoke-live.R and eolas/docs/client-contract.md.

Skipped in the default unit CI job (``pytest -m 'not integration'``).
Enable locally or in the weekly smoke workflow::

    EOLAS_API_KEY=vs_... pytest -m integration -q
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from eolas_data import Client

pytestmark = pytest.mark.integration

_SKIP = not os.environ.get("EOLAS_API_KEY")


@pytest.fixture()
def smoke_library(tmp_path, monkeypatch):
    """Isolated on-disk library — same idea as EOLAS_LIBRARY in R smoke."""
    lib = tmp_path / "eolas-smoke-lib"
    lib.mkdir()
    monkeypatch.setenv("EOLAS_LIBRARY", str(lib))
    return lib


@pytest.fixture()
def live_client(smoke_library):
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


def test_client_exports_cache_clear(live_client):
    assert hasattr(live_client, "cache_clear")
    assert callable(live_client.cache_clear)


def test_linz_nz_addresses_bulk_route_and_head(live_client, smoke_library):
    """User path: client.linz() — not get_local(). head() must not crash."""
    gdf = live_client.linz("nz_addresses", progress=False)
    assert len(gdf) > 100_000
    # GeoDataFrame or Dataset — repr/head must work (pandas attrs regression)
    gdf.head()
    # Bulk file landed in isolated library
    assert any(smoke_library.iterdir())
    geo_files = list(smoke_library.glob("nz_addresses*.geo.parquet"))
    assert geo_files, f"expected bulk geo file under {smoke_library}"