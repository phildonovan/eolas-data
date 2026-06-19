"""Parametrised smoke tests for Client source-specific fetch helpers."""

import json
from pathlib import Path

import pytest
import responses as resp_lib

from eolas_data import Client

BASE = "https://api.eolas.fyi"
RECORDS = [
    {"date": "2023-01-01", "period": "2023Q1", "value": 100.0},
    {"date": "2023-04-01", "period": "2023Q2", "value": 101.5},
]

_SOURCE_HELPERS = json.loads(
    (Path(__file__).parent / "fixtures" / "source_helpers.json").read_text(encoding="utf-8")
)


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


@pytest.mark.parametrize(
    "method,expected_source",
    sorted(_SOURCE_HELPERS.items()),
    ids=sorted(_SOURCE_HELPERS.keys()),
)
@resp_lib.activate
def test_source_helper_sets_source(client, method, expected_source):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    df = getattr(client, method)("nz_cpi")
    assert df.eolas_source == expected_source