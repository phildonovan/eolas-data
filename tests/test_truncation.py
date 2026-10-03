"""C22 — the Free-plan row cap must never be presented as the full dataset.

The server marks a capped /data response with ``X-Eolas-Truncated: true`` and
``X-Plan-Row-Cap``. The client must warn, stamp ``eolas_meta['truncated']``,
and say that ``limit=N`` on a capped slice is not "the most recent N".
"""
from __future__ import annotations

import warnings

import pytest
import responses as resp_lib

from eolas_data import Client
from eolas_data.meta import (
    merge_provenance,
    meta_subtitle,
    truncation_from_headers,
    truncation_message,
)

BASE = "https://api.eolas.nz"

RECORDS = [
    {"date": "2018-01-01", "value": 1.0},
    {"date": "2020-01-01", "value": 2.0},
    {"date": "2024-01-01", "value": 3.0},
    {"date": "2025-01-01", "value": 4.0},
]

CAP_HEADERS = {
    "X-Eolas-Truncated": "true",
    "X-Plan-Row-Cap": "50000",
    "X-Plan": "free",
}


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_truncation_from_headers_absent_is_empty():
    assert truncation_from_headers({}) == {}
    assert truncation_from_headers(None) == {}


def test_truncation_from_headers_parses_cap_and_plan():
    out = truncation_from_headers(CAP_HEADERS)
    assert out == {"truncated": True, "row_cap": 50000, "plan": "free"}


def test_truncation_from_headers_false():
    assert truncation_from_headers({"X-Eolas-Truncated": "false"}) == {
        "truncated": False
    }


def test_truncation_message_none_when_not_truncated():
    assert truncation_message("x", {}) is None
    assert truncation_message("x", {"truncated": False}) is None


def test_truncation_message_mentions_limit_within_slice():
    msg = truncation_message("poppr_lab_national", truncation_from_headers(CAP_HEADERS), 12)
    assert "50,000 rows" in msg
    assert "NOT the full dataset" in msg
    assert "limit=12" in msg and "WITHIN" in msg


def test_merge_provenance_stamps_truncated():
    merged = merge_provenance({"title": "t"}, CAP_HEADERS)
    assert merged["truncated"] is True
    assert merged["row_cap"] == 50000
    merged = merge_provenance({"title": "t"}, {"X-Eolas-Truncated": "false"})
    assert merged["truncated"] is False
    assert "row_cap" not in merged


def test_meta_subtitle_flags_truncation():
    sub = meta_subtitle({"title": "Pop", "truncated": True, "row_cap": 50000})
    assert "TRUNCATED to 50,000 rows" in sub
    assert "TRUNCATED" not in meta_subtitle({"title": "Pop", "truncated": False})


# ---------------------------------------------------------------------------
# Client.get
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_get_warns_and_stamps_meta_when_truncated(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/poppr_lab_national",
                 json={"name": "poppr_lab_national", "source": "Stats NZ",
                       "row_count_at_last_refresh": 98000})
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/poppr_lab_national/data",
                 json={"data": RECORDS}, headers=CAP_HEADERS)

    with pytest.warns(UserWarning, match="truncated to 50,000 rows") as rec:
        df = client.get("poppr_lab_national")

    assert df.eolas_meta["truncated"] is True
    assert df.eolas_meta["row_cap"] == 50000
    assert "TRUNCATED" in repr(df)
    assert not any("WITHIN" in str(w.message) for w in rec)


@resp_lib.activate
def test_get_limit_on_capped_slice_says_not_most_recent(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/poppr_lab_national",
                 json={"name": "poppr_lab_national", "source": "Stats NZ",
                       "row_count_at_last_refresh": 98000})
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/poppr_lab_national/data",
                 json={"data": RECORDS}, headers=CAP_HEADERS)

    with pytest.warns(UserWarning, match=r"limit=2 returned the latest rows WITHIN"):
        df = client.get("poppr_lab_national", limit=2)
    assert len(df) == 2
    assert df.eolas_meta["truncated"] is True


@resp_lib.activate
def test_get_no_warning_when_not_truncated(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json={"name": "nz_cpi", "source": "Stats NZ"})
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data",
                 json={"data": RECORDS}, headers={"X-Eolas-Truncated": "false"})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = client.get("nz_cpi")
    assert df.eolas_meta["truncated"] is False


@resp_lib.activate
def test_get_no_warning_on_old_server_without_header(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json={"name": "nz_cpi", "source": "Stats NZ"})
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data",
                 json={"data": RECORDS})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = client.get("nz_cpi")
    assert "truncated" not in df.eolas_meta


@resp_lib.activate
def test_get_as_arrow_still_warns(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json={"name": "nz_cpi", "source": "Stats NZ"})
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data",
                 json={"data": RECORDS}, headers=CAP_HEADERS)
    with pytest.warns(UserWarning, match="truncated"):
        client.get("nz_cpi", as_arrow=True)


@resp_lib.activate
def test_get_csv_format_warns(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json={"name": "nz_cpi", "source": "Stats NZ"})
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data",
                 body="date,value\n2020-01-01,1.0\n", content_type="text/csv",
                 headers=CAP_HEADERS)
    with pytest.warns(UserWarning, match="truncated"):
        df = client.get("nz_cpi", format="csv")
    assert df.eolas_meta["truncated"] is True


# ---------------------------------------------------------------------------
# Client.download
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_download_bytes_warns_when_truncated(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data",
                 body="date,value\n", content_type="text/csv", headers=CAP_HEADERS)
    with pytest.warns(UserWarning, match="truncated"):
        client.download("nz_cpi", format="csv")


@resp_lib.activate
def test_download_to_path_warns_when_truncated(client, tmp_path):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data",
                 body="date,value\n", content_type="text/csv", headers=CAP_HEADERS)
    with pytest.warns(UserWarning, match="truncated"):
        client.download("nz_cpi", format="csv", path=tmp_path / "x.csv")
