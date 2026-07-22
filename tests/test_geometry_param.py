"""Tests for Client.get(geometry=False) -- omit the geometry_wkt column.

Tester feedback (Aaron, 2026-07-22): pulling TA/RC data drags a geometry_wkt
column that dwarfs the attributes; 1017/1536 datasets carry it. The API projects
the column away at the Iceberg scan so it is never read from storage. The
client's job is to send the parameter, stop mirroring the server's 413 geometry
trigger (which would otherwise keep routing these calls to a bulk download), and
keep the two variants apart in the response cache.
"""

from __future__ import annotations

import pytest
import responses as resp_lib

from eolas_data import Client

BASE = "https://api.eolas.fyi"

# Spatial, but small enough that ONLY geometry trips the large-dataset guard.
GEO_INFO = {
    "name": "nz_ta_2023",
    "namespace": "statsnz_geo",
    "source": "Stats NZ Geospatial",
    "has_geometry": True,
    "geometry_type": "polygon",
    "bulk_export_class": "materialised",
    "row_count_at_last_refresh": 67,
}

ROWS = [{"ta_name": "Auckland", "population": 1695200.0}]


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


def _register(info=None, rows=None):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_ta_2023", json=info or GEO_INFO)
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/v1/datasets/nz_ta_2023/data",
        json=rows if rows is not None else ROWS,
    )


def _data_urls():
    """Only the /data row requests.

    Match on the path segment, not a bare "/data" substring -- the metadata URL
    /v1/datasets/<name> contains "/data" inside "/datasets" and would otherwise
    be counted as a row request that is (correctly) missing the parameter.

    The live path attempts format=arrow first and falls back to JSON, so a
    single get() can legitimately produce two entries here.
    """
    out = []
    for call in resp_lib.calls:
        path = call.request.url.split("?", 1)[0]
        if path.endswith("/data"):
            out.append(call.request.url)
    return out


@resp_lib.activate
def test_geometry_false_sends_the_parameter(client):
    _register()
    client.get("nz_ta_2023", limit=10, geometry=False)
    urls = _data_urls()
    assert urls, "expected a live /data request"
    assert all("geometry=false" in u for u in urls)


@resp_lib.activate
def test_default_sends_no_geometry_parameter(client):
    # geometry=true is the server default; sending it would churn URLs and CDN
    # cache keys on nearly every call for no benefit.
    _register()
    client.get("nz_ta_2023", limit=10)
    urls = _data_urls()
    assert urls
    assert all("geometry" not in u for u in urls)


@resp_lib.activate
def test_geometry_false_keeps_whole_dataset_pull_on_the_live_path(client):
    # Without threading `geometry` into the routing mirror this dataset
    # (spatial + bulk-exportable) would divert to get_local() and never hit
    # /data -- the regression this test exists to catch.
    _register()
    client.get("nz_ta_2023", geometry=False)
    urls = _data_urls()
    assert urls, "geometry=False should stay on the live path, not route to bulk"
    assert all("geometry=false" in u for u in urls)


@resp_lib.activate
def test_default_still_routes_whole_dataset_spatial_pull_to_bulk(client, monkeypatch):
    # Existing behaviour must be untouched when geometry is not narrowed.
    _register()
    called = {}

    def fake_get_local(name, **kwargs):
        called["name"] = name
        import pandas as pd

        return pd.DataFrame(ROWS)

    monkeypatch.setattr(client, "get_local", fake_get_local)
    client.get("nz_ta_2023")
    assert called.get("name") == "nz_ta_2023"
    assert not _data_urls()


def test_geometry_false_with_as_geo_true_is_rejected(client):
    with pytest.raises(ValueError, match="contradictory"):
        client.get("nz_ta_2023", geometry=False, as_geo=True)


@resp_lib.activate
def test_cache_does_not_conflate_the_two_variants():
    """The variants differ by a whole column -- they must not share a cache key.

    Uses cache=True explicitly: the response cache is OFF by default, so a
    client from the shared fixture would make this test pass vacuously.
    """
    import json as _json

    client = Client("eolas_testkey123", base_url=BASE, cache=True)
    assert client._cache is not None, "test needs the response cache enabled"

    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_ta_2023", json=GEO_INFO)

    # Answer based on the URL, not registration order: the live path tries
    # format=arrow before falling back to JSON, so an ordered pair of responses
    # would be consumed by the first get() alone.
    def _rows(request):
        if "geometry=false" in request.url:
            body = [{"ta_name": "Auckland"}]
        else:
            body = [
                {
                    "ta_name": "Auckland",
                    "geometry_wkt": "MULTIPOLYGON(((1 1,2 2,2 1,1 1)))",
                }
            ]
        return (200, {"Content-Type": "application/json"}, _json.dumps(body))

    resp_lib.add_callback(
        resp_lib.GET,
        f"{BASE}/v1/datasets/nz_ta_2023/data",
        callback=_rows,
        content_type="application/json",
    )

    # Every other cache-key component must be IDENTICAL, otherwise the keys
    # differ for an unrelated reason and the test proves nothing -- in
    # particular as_geo, which defaults to None rather than False.
    a = client.get("nz_ta_2023", limit=10, as_geo=False)
    b = client.get("nz_ta_2023", limit=10, as_geo=False, geometry=False)

    assert "geometry_wkt" in a.columns
    assert "geometry_wkt" not in b.columns, (
        "geometry=False returned the cached geometry-bearing frame -- the cache "
        "key is missing the geometry flag"
    )


class TestLivePullBlocked:
    """The 413 mirror itself."""

    def test_geometry_alone_blocks_when_geometry_requested(self):
        assert Client._live_pull_blocked(GEO_INFO, geometry=True) is True

    def test_geometry_alone_does_not_block_when_dropped(self):
        assert Client._live_pull_blocked(GEO_INFO, geometry=False) is False

    def test_row_count_still_blocks_when_geometry_dropped(self):
        # Dropping a column doesn't reduce row count.
        big = {**GEO_INFO, "row_count_at_last_refresh": 5_000_000}
        assert Client._live_pull_blocked(big, geometry=False) is True

    def test_default_argument_preserves_old_behaviour(self):
        assert Client._live_pull_blocked(GEO_INFO) is True
