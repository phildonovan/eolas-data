"""Tests for Client.preview() (DRIFT-5) and get(dimensions=) (DRIFT-4)."""

from __future__ import annotations

import pytest
import responses as resp_lib

from eolas_data import Client

BASE = "https://api.eolas.fyi"


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


# ---- DRIFT-5: preview ------------------------------------------------------


@resp_lib.activate
def test_preview_hits_unauthenticated_endpoint(client):
    rows = [{"date": "2024-01-01", "value": i} for i in range(10)]
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/v1/datasets/nz_cpi/preview",
        json={"rows": rows, "hidden_columns": ["geometry_wkt"]},
    )
    df = client.preview("nz_cpi")
    assert len(df) == 10
    assert list(df.columns) == ["date", "value"]
    # It must use /preview, never the rate-limited /data path.
    urls = [c.request.url for c in resp_lib.calls]
    assert any(u.endswith("/nz_cpi/preview") for u in urls)
    assert not any(u.endswith("/data") or "/data?" in u for u in urls)


@resp_lib.activate
def test_preview_caps_at_limit(client):
    rows = [{"value": i} for i in range(10)]
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/v1/datasets/nz_cpi/preview",
        json={"rows": rows, "hidden_columns": []},
    )
    assert len(client.preview("nz_cpi", limit=3)) == 3


# ---- DRIFT-4: dimensions filter --------------------------------------------


@resp_lib.activate
def test_get_passes_dimensions_param(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/v1/datasets/building_consents/data",
        json={"data": [{"region": "Auckland", "value": 1}]},
    )
    client.get("building_consents", dimensions="auckland")
    data_reqs = [
        c
        for c in resp_lib.calls
        if c.request.url.endswith("/data") or "/data?" in c.request.url
    ]
    assert data_reqs, "expected a /data request"
    assert any("dimensions=auckland" in c.request.url for c in data_reqs)


@resp_lib.activate
def test_dimensions_forces_live_path_no_bulk_route(client):
    # With dimensions set, get() must NOT auto-route to the bulk cache (which has
    # no per-request dimension filter). A /data hit proves the live path ran.
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/v1/datasets/nz_addresses/data",
        json={"data": [{"suburb": "Ponsonby"}]},
    )
    client.get("nz_addresses", dimensions="ponsonby")
    assert any(
        c.request.url.endswith("/data") or "/data?" in c.request.url
        for c in resp_lib.calls
    )
