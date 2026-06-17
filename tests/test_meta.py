import pandas as pd
import pytest
import responses as resp_lib

from eolas_data import Client, Dataset

BASE = "https://api.eolas.fyi"

RECORDS = [
    {"date": "2023-01-01", "period": "2023Q1", "value": 100.0},
    {"date": "2023-04-01", "period": "2023Q2", "value": 101.5},
]

INFO = {
    "name": "nz_cpi",
    "title": "NZ Consumer Price Index",
    "source": "Stats NZ",
    "namespace": "statsnz",
    "description": "Official quarterly CPI from Stats NZ.",
    "refresh_cadence": "quarterly",
    "columns": [
        {"name": "date", "type": "date", "description": "Observation date"},
        {"name": "value", "type": "double", "description": "Index value"},
    ],
}


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


@resp_lib.activate
def test_get_attaches_meta_and_columns(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi", json=INFO)
    df = client.get("nz_cpi")
    assert isinstance(df, Dataset)
    assert df.eolas_meta["title"] == "NZ Consumer Price Index"
    assert df.eolas_meta["description"] == "Official quarterly CPI from Stats NZ."
    assert isinstance(df.eolas_columns, pd.DataFrame)
    assert df.eolas_columns.iloc[1]["name"] == "value"
    assert df.column_label("value") == "Index value"


@resp_lib.activate
def test_repr_shows_title_not_full_description(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi", json=INFO)
    df = client.get("nz_cpi")
    text = repr(df)
    assert "NZ Consumer Price Index" in text
    assert "refreshed quarterly" in text
    assert "Official quarterly CPI from Stats NZ." not in text


@resp_lib.activate
def test_meta_false_skips_info_fetch(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    df = client.get("nz_cpi", meta=False)
    assert df.eolas_meta == {}
    assert df.eolas_columns is None


@resp_lib.activate
def test_info_cached_on_second_get(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi", json=INFO)
    client.get("nz_cpi")
    client.get("nz_cpi", start="2020-01-01")
    info_calls = [
        c for c in resp_lib.calls
        if c.request.url == f"{BASE}/v1/datasets/nz_cpi"
    ]
    assert len(info_calls) == 1