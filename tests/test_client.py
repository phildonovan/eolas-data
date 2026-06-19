import io
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest
import responses as resp_lib

from eolas_data import Client, Dataset
from eolas_data.exceptions import (
    AuthenticationError,
    BulkLicenceRestricted,
    BulkNotYetAvailable,
    BulkUpgradeRequired,
    NotFoundError,
    RateLimitError,
)

BASE = "https://api.eolas.fyi"

RECORDS = [
    {"date": "2023-01-01", "period": "2023Q1", "value": 100.0},
    {"date": "2023-04-01", "period": "2023Q2", "value": 101.5},
]

DATASET_LIST = [
    {"name": "nz_cpi",  "title": "NZ CPI",  "source": "Stats NZ", "namespace": "stats_nz"},
    {"name": "nz_gdp",  "title": "NZ GDP",  "source": "OECD",     "namespace": "oecd"},
    {"name": "nz_rbnz", "title": "NZ RBNZ", "source": "RBNZ",     "namespace": "rbnz"},
]


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


@pytest.fixture()
def cached_client():
    return Client("eolas_testkey123", base_url=BASE, cache=True)


# ---------------------------------------------------------------------------
# Client repr
# ---------------------------------------------------------------------------

def test_client_repr(client):
    assert "eolas_te" in repr(client)
    assert "..." in repr(client)


def test_cached_client_repr(cached_client):
    assert "cache=on" in repr(cached_client)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_list_returns_all_datasets(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets", json={"datasets": DATASET_LIST})
    result = client.list()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


@resp_lib.activate
def test_list_filters_by_source(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets", json={"datasets": DATASET_LIST})
    result = client.list("Stats NZ")
    assert len(result) == 1
    assert result.iloc[0]["name"] == "nz_cpi"


@resp_lib.activate
def test_list_unknown_source_returns_empty(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets", json={"datasets": DATASET_LIST})
    result = client.list("Unknown")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


@resp_lib.activate
def test_list_wellington_filters_source(client):
    datasets = [
        {"name": "kcdc_flood_extents", "source": "Wellington Region Councils"},
        {"name": "nz_cpi", "source": "OECD"},
    ]
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets", json={"datasets": datasets})
    result = client.list_wellington()
    assert len(result) == 1
    assert result.iloc[0]["name"] == "kcdc_flood_extents"


# ---------------------------------------------------------------------------
# info()
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_info_returns_meta(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json={"name": "nz_cpi", "title": "NZ Consumer Price Index"})
    meta = client.info("nz_cpi")
    assert meta["name"] == "nz_cpi"


@resp_lib.activate
def test_info_not_found_raises(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/bad", json={"detail": "Not found."}, status=404)
    with pytest.raises(NotFoundError):
        client.info("bad")


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_get_returns_dataset(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    df = client.get("nz_cpi")
    assert isinstance(df, Dataset)
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df.eolas_name == "nz_cpi"


@resp_lib.activate
def test_get_sorts_rows_by_date(client):
    # API streams Iceberg in file order, not chronological — client must sort.
    unsorted = [
        {"date": "2023-04-01", "period": "2023Q2", "value": 101.5},
        {"date": "2022-01-01", "period": "2022Q1", "value": 99.0},
        {"date": "2023-01-01", "period": "2023Q1", "value": 100.0},
    ]
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": unsorted})
    df = client.get("nz_cpi")
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2022-01-01", "2023-01-01", "2023-04-01"]


@resp_lib.activate
def test_get_limit_returns_most_recent(client):
    records = [
        {"date": "2018-01-01", "value": 1.0},
        {"date": "2020-01-01", "value": 2.0},
        {"date": "2024-01-01", "value": 3.0},
        {"date": "2025-01-01", "value": 4.0},
    ]
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": records})
    df = client.get("nz_cpi", limit=2)
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2025-01-01"]


@resp_lib.activate
def test_get_limit_requests_full_window_from_server(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json={"name": "nz_cpi", "source": "Stats NZ", "has_geometry": False,
                       "row_count_at_last_refresh": 500})
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    client.get("nz_cpi", limit=2)
    data_reqs = [c for c in resp_lib.calls if "/data?" in c.request.url or c.request.url.endswith("/data")]
    assert any("limit=0" in c.request.url for c in data_reqs)


@resp_lib.activate
def test_get_passes_date_params(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    client.get("nz_cpi", start="2023-01-01", end="2023-06-30")
    req = resp_lib.calls[0].request
    assert "start=2023-01-01" in req.url
    assert "end=2023-06-30" in req.url


@resp_lib.activate
def test_get_merges_provenance_headers(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/v1/datasets/nz_cpi/data",
        json={"data": RECORDS},
        adding_headers={
            "X-Eolas-Attribution": "Source: OECD, CC-BY 4.0.",
            "X-Eolas-Licence": "CC-BY 4.0",
            "X-Eolas-Source": "OECD",
        },
    )
    df = client.get("nz_cpi", meta=False)
    meta = df.attrs.get("eolas_meta", {})
    assert meta.get("attribution_text") == "Source: OECD, CC-BY 4.0."
    assert meta.get("licence") == "CC-BY 4.0"
    assert meta.get("source") == "OECD"


@resp_lib.activate
def test_get_envelope_attaches_data_sources(client):
    payload = {
        "data_sources": [{"name": "nz_cpi", "licence": "CC-BY 4.0"}],
        "data": RECORDS,
    }
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json=payload)
    df = client.get("nz_cpi", envelope=True, meta=False)
    meta = df.attrs.get("eolas_meta", {})
    assert meta.get("data_sources") == payload["data_sources"]
    assert "envelope=1" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_csv_returns_dataframe(client):
    csv_body = "date,period,value\n2023-01-01,2023Q1,100.0\n"
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data",
                 body=csv_body, content_type="text/csv")
    df = client.get("nz_cpi", format="csv")
    assert isinstance(df, pd.DataFrame)
    assert "value" in df.columns


# ---------------------------------------------------------------------------
# Geospatial — as_geo
# ---------------------------------------------------------------------------

GEO_RECORDS = [
    {"address_id": 1, "full_address": "1 Main Rd", "geometry_wkt": "POINT (174.78 -41.28)"},
    {"address_id": 2, "full_address": "2 Main Rd", "geometry_wkt": "POINT (174.79 -41.29)"},
]


@resp_lib.activate
def test_get_auto_converts_to_geodataframe(client):
    """When geometry_wkt is present and geopandas is importable, return a GeoDataFrame."""
    pytest.importorskip("geopandas")
    import geopandas as gpd

    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_addresses/data", json={"data": GEO_RECORDS})
    df = client.get("nz_addresses")
    assert isinstance(df, gpd.GeoDataFrame)
    assert "geometry" in df.columns
    assert "geometry_wkt" not in df.columns
    assert df.crs.to_epsg() == 4326
    assert df.geometry.iloc[0].x == pytest.approx(174.78)
    assert df.attrs.get("eolas_name") == "nz_addresses"


@resp_lib.activate
def test_get_as_geo_false_keeps_wkt(client):
    """as_geo=False returns the plain DataFrame with the WKT string column."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_addresses/data", json={"data": GEO_RECORDS})
    df = client.get("nz_addresses", as_geo=False)
    assert isinstance(df, Dataset)
    assert "geometry_wkt" in df.columns
    assert df["geometry_wkt"].iloc[0].startswith("POINT")


LARGE_GEO_META = {
    "name": "nz_addresses",
    "namespace": "linz",
    "bulk_export_class": "materialised",
    "has_geometry": True,
    "geometry_type": "point",
    "row_count_at_last_refresh": 2_418_264,
}


@resp_lib.activate
def test_get_whole_dataset_defers_to_get_local_for_large_geo(client):
    """Whole-dataset pulls on geometry tables route to get_local(), not /data."""
    sentinel = pd.DataFrame({"address_id": [1], "geometry_wkt": ["POINT (0 0)"]})
    with patch.object(client, "get_local", return_value=sentinel) as mock_local:
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_addresses",
                     json=LARGE_GEO_META, status=200)
        result = client.get("nz_addresses")
    mock_local.assert_called_once()
    assert result is sentinel


@resp_lib.activate
def test_get_limit_on_large_geo_sends_bounded_limit(client):
    """Positive limits on geometry tables must not send limit=0 (API 413)."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_addresses",
                 json=LARGE_GEO_META, status=200)
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_addresses/data",
                 json={"data": GEO_RECORDS})
    client.get("nz_addresses", limit=2)
    data_reqs = [c for c in resp_lib.calls if "/data?" in c.request.url or c.request.url.endswith("/data")]
    assert data_reqs, "expected a /data request"
    assert any("limit=2" in c.request.url for c in data_reqs)
    assert not any("limit=0" in c.request.url for c in data_reqs)


@resp_lib.activate
def test_get_no_geometry_column_returns_dataset(client):
    """Datasets without a geometry_wkt column still return a Dataset even with as_geo=None."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    df = client.get("nz_cpi")
    assert isinstance(df, Dataset)
    try:
        import geopandas as gpd
        assert not isinstance(df, gpd.GeoDataFrame)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_cache_avoids_second_request(cached_client):
    # The client negotiates Arrow first; against this JSON-only mock the first
    # get() makes an Arrow attempt + a JSON fallback (and memoises that the
    # mock doesn't speak Arrow). In mode="auto", the first get() also calls
    # info() for routing — register that endpoint too (no bulk_export_class →
    # falls through to live). The cache contract is that the SECOND get()
    # adds zero further HTTP calls — assert that, not an absolute count.
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json={"name": "nz_cpi", "source": "Stats NZ"})  # no bulk_export_class → live
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    cached_client.get("nz_cpi")
    calls_after_first = len(resp_lib.calls)
    cached_client.get("nz_cpi")
    assert len(resp_lib.calls) == calls_after_first  # served from cache, no new request


# ---------------------------------------------------------------------------
# Source-specific methods
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_statsnz_sets_source(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    df = client.statsnz("nz_cpi")
    assert isinstance(df, Dataset)
    assert df.eolas_source == "Stats NZ"


@resp_lib.activate
def test_oecd_sets_source(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_gdp/data", json={"data": RECORDS})
    df = client.oecd("nz_gdp")
    assert df.eolas_source == "OECD"


@resp_lib.activate
def test_rbnz_sets_source(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_rbnz/data", json={"data": RECORDS})
    df = client.rbnz("nz_rbnz")
    assert df.eolas_source == "RBNZ"


@resp_lib.activate
def test_treasury_sets_source(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/fiscal/data", json={"data": RECORDS})
    df = client.treasury("fiscal")
    assert df.eolas_source == "NZ Treasury"


# ---------------------------------------------------------------------------
# Dataset repr
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_dataset_repr_includes_name_and_source(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    df = client.statsnz("nz_cpi")
    r = repr(df)
    assert "nz_cpi" in r
    assert "Stats NZ" in r
    assert "2 rows" in r


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_unauthorised_raises_auth_error(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data",
                 json={"detail": "Unauthorised"}, status=401)
    with pytest.raises(AuthenticationError):
        client.get("nz_cpi")


@resp_lib.activate
def test_rate_limit_raises(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data",
                 json={"detail": "Monthly limit"}, status=429)
    with pytest.raises(RateLimitError):
        client.get("nz_cpi")


# ---------------------------------------------------------------------------
# Env var fallback
# ---------------------------------------------------------------------------

def test_key_from_env(monkeypatch):
    monkeypatch.setenv("EOLAS_API_KEY", "eolas_from_env")
    c = Client()
    assert c._key == "eolas_from_env"


# ---------------------------------------------------------------------------
# Type stubs
# ---------------------------------------------------------------------------

def test_dataset_name_literal_includes_known_dataset():
    """The Literal stub generated at release time should include nz_cpi."""
    from eolas_data._dataset_names import ALL_NAMES, CATALOG_SNAPSHOT_COUNT
    assert "nz_cpi" in ALL_NAMES
    assert CATALOG_SNAPSHOT_COUNT == len(ALL_NAMES)
    assert CATALOG_SNAPSHOT_COUNT > 100   # sanity floor; current catalog is 717


# ---------------------------------------------------------------------------
# Client.integration() — Enterprise-gated; client-side just relays
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_integration_returns_files(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/integrations/meltano",
                 json={"platform": "meltano",
                       "files": {"meltano.yml": "config", "README.md": "readme"}},
                 status=200)
    files = client.integration("meltano", ["nz_cpi", "nz_gdp"])
    assert files == {"meltano.yml": "config", "README.md": "readme"}


@resp_lib.activate
def test_integration_passes_comma_separated_datasets(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/integrations/meltano",
                 json={"platform": "meltano", "files": {}}, status=200)
    client.integration("meltano", ["nz_cpi", "nz_gdp"])
    sent = resp_lib.calls[0].request.url
    assert "datasets=nz_cpi%2Cnz_gdp" in sent or "datasets=nz_cpi,nz_gdp" in sent


def test_integration_empty_datasets_raises(client):
    with pytest.raises(ValueError, match="datasets cannot be empty"):
        client.integration("meltano", [])


@resp_lib.activate
def test_integration_403_raises_authentication_error_with_server_detail(client):
    """Non-Enterprise plan returns 403; the server detail must flow through."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/integrations/meltano",
                 json={"detail": "This endpoint is an Enterprise plan feature."},
                 status=403)
    with pytest.raises(AuthenticationError) as e:
        client.integration("meltano", ["nz_cpi"])
    assert "Enterprise" in str(e.value)


# ---------------------------------------------------------------------------
# Client.download_bulk() — /v1/bulk/{namespace}/{table}
# ---------------------------------------------------------------------------

# Minimal Parquet bytes (just needs to be non-empty binary content for the test).
FAKE_PARQUET = b"PAR1" + b"\x00" * 12 + b"PAR1"

# Dataset metadata the client fetches first (name → namespace + table).
BULK_DATASET_META = {
    "name": "nz_cpi",
    "title": "NZ Consumer Price Index",
    "source": "Stats NZ",
    "namespace": "statsnz",
    "table": "nz_cpi",
}


def _register_bulk_happy(freshness_param: str = ""):
    """Register the metadata lookup + the bulk 200 response.

    When freshness_param is empty we simulate the bare URL → 302 → 200 path
    (responses library follows redirects by default, so we register the
    final 200 directly — the redirect mechanics are tested implicitly).
    """
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=FAKE_PARQUET,
                 content_type="application/octet-stream",
                 status=200)


@resp_lib.activate
def test_download_bulk_returns_bytes_when_no_path(client):
    """With path=None download_bulk returns raw bytes."""
    _register_bulk_happy()
    result = client.download_bulk("nz_cpi")
    assert isinstance(result, bytes)
    assert result == FAKE_PARQUET


@resp_lib.activate
def test_download_bulk_writes_file_and_returns_path(client, tmp_path):
    """With path=... the file is written and the resolved Path is returned."""
    _register_bulk_happy()
    dest = tmp_path / "nz_cpi.parquet"
    result = client.download_bulk("nz_cpi", path=dest)
    import pathlib
    assert isinstance(result, pathlib.Path)
    assert result == dest
    assert dest.read_bytes() == FAKE_PARQUET


@resp_lib.activate
def test_download_bulk_creates_parent_dirs(client, tmp_path):
    """Parent directories are created automatically when path has them."""
    _register_bulk_happy()
    dest = tmp_path / "nested" / "dir" / "nz_cpi.parquet"
    result = client.download_bulk("nz_cpi", path=dest)
    assert result.exists()


@resp_lib.activate
def test_download_bulk_sends_format_param(client):
    """The format query param is sent to the bulk endpoint."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=b"csv data",
                 content_type="application/octet-stream",
                 status=200)
    client.download_bulk("nz_cpi", format="csv_gz")
    bulk_req = resp_lib.calls[1].request
    assert "format=csv_gz" in bulk_req.url


@resp_lib.activate
def test_download_bulk_auto_freshness_omits_param(client):
    """freshness='auto' must NOT send a freshness= query param (server-redirect logic)."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=FAKE_PARQUET,
                 content_type="application/octet-stream",
                 status=200)
    client.download_bulk("nz_cpi", freshness="auto")
    bulk_req = resp_lib.calls[1].request
    assert "freshness" not in bulk_req.url


@resp_lib.activate
def test_download_bulk_monthly_freshness_sends_param(client):
    """freshness='monthly' must include freshness=monthly in the request URL."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=FAKE_PARQUET,
                 content_type="application/octet-stream",
                 status=200)
    client.download_bulk("nz_cpi", freshness="monthly")
    bulk_req = resp_lib.calls[1].request
    assert "freshness=monthly" in bulk_req.url


@resp_lib.activate
def test_download_bulk_402_raises_bulk_upgrade_required(client):
    """HTTP 402 from the bulk endpoint → BulkUpgradeRequired."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 json={"detail": "Fresh bulk downloads are a Pro feature."},
                 status=402)
    with pytest.raises(BulkUpgradeRequired) as exc_info:
        client.download_bulk("nz_cpi", freshness="current")
    assert "Pro" in str(exc_info.value)


@resp_lib.activate
def test_download_bulk_403_licence_raises_bulk_licence_restricted(client):
    """HTTP 403 with 'licence' in detail → BulkLicenceRestricted (not AuthenticationError)."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/oecd_gdp",
                 json={**BULK_DATASET_META, "name": "oecd_gdp",
                       "namespace": "oecd", "table": "oecd_gdp"},
                 status=200)
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/oecd/oecd_gdp",
                 json={"detail": "This dataset is not available as a bulk download (licence: OECD)."},
                 status=403)
    with pytest.raises(BulkLicenceRestricted):
        client.download_bulk("oecd_gdp")


@resp_lib.activate
def test_download_bulk_403_auth_raises_authentication_error(client):
    """HTTP 403 without 'licence' in detail → standard AuthenticationError."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 json={"detail": "API key is inactive."},
                 status=403)
    with pytest.raises(AuthenticationError):
        client.download_bulk("nz_cpi")


@resp_lib.activate
def test_download_bulk_503_raises_bulk_not_yet_available(client):
    """HTTP 503 from the bulk endpoint → BulkNotYetAvailable."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 json={"detail": "Monthly bulk snapshots are still rolling out."},
                 status=503)
    with pytest.raises(BulkNotYetAvailable):
        client.download_bulk("nz_cpi")


@resp_lib.activate
def test_download_bulk_404_raises_not_found(client):
    """HTTP 404 from the metadata lookup → NotFoundError."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/bad_name",
                 json={"detail": "Not found."}, status=404)
    with pytest.raises(NotFoundError):
        client.download_bulk("bad_name")


def test_download_bulk_invalid_format_raises_value_error(client):
    """An unrecognised format string should raise ValueError before any HTTP call."""
    with pytest.raises(ValueError, match="Unknown format"):
        client.download_bulk("nz_cpi", format="xlsx")


def test_download_bulk_invalid_freshness_raises_value_error(client):
    """An unrecognised freshness string should raise ValueError before any HTTP call."""
    with pytest.raises(ValueError, match="Unknown freshness"):
        client.download_bulk("nz_cpi", freshness="latest")


# ---------------------------------------------------------------------------
# get() basic — live path
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_get_live_returns_dataset(client):
    """client.get('nz_cpi') hits the live API and returns a Dataset."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    df = client.get("nz_cpi")
    assert isinstance(df, Dataset)
    assert len(df) == 2
    assert df.eolas_name == "nz_cpi"
