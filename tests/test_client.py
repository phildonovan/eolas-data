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
    assert len(result) == 3


@resp_lib.activate
def test_list_filters_by_source(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets", json={"datasets": DATASET_LIST})
    result = client.list("Stats NZ")
    assert len(result) == 1
    assert result[0]["name"] == "nz_cpi"


@resp_lib.activate
def test_list_unknown_source_returns_empty(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets", json={"datasets": DATASET_LIST})
    result = client.list("Unknown")
    assert result == []


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
def test_get_passes_date_params(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    client.get("nz_cpi", start="2023-01-01", end="2023-06-30")
    req = resp_lib.calls[0].request
    assert "start=2023-01-01" in req.url
    assert "end=2023-06-30" in req.url


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


@resp_lib.activate
def test_get_as_geo_false_keeps_wkt(client):
    """as_geo=False returns the plain DataFrame with the WKT string column."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_addresses/data", json={"data": GEO_RECORDS})
    df = client.get("nz_addresses", as_geo=False)
    assert isinstance(df, Dataset)
    assert "geometry_wkt" in df.columns
    assert df["geometry_wkt"].iloc[0].startswith("POINT")


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
# get() smart-routing — mode="auto" / "live" / "cached"
# ---------------------------------------------------------------------------

# Metadata shapes used across the routing tests.
_META_SMALL_TABULAR = {
    "name": "nz_cpi", "source": "Stats NZ", "namespace": "statsnz",
    "bulk_export_class": "cc_by",
    "row_count_at_last_refresh": 145,   # below the 100_000 threshold
}
_META_LARGE_TABULAR = {
    "name": "pharmac_schedule", "source": "PHARMAC", "namespace": "pharmac",
    "bulk_export_class": "cc_by",
    "row_count_at_last_refresh": 150_000,  # above threshold
}
_META_GEO = {
    "name": "nz_parcels", "source": "LINZ", "namespace": "linz",
    "bulk_export_class": "cc_by",
    "geometry_type": "MultiPolygon",
    "row_count_at_last_refresh": 3_000_000,
}
_META_LICENCE_RESTRICTED = {
    "name": "oecd_gdp", "source": "OECD", "namespace": "oecd",
    "bulk_export_class": "none",   # licence floor — must NOT use cache path
    "row_count_at_last_refresh": 250_000,
}
# Server now returns geometry_type="none" (string) for non-geo datasets after
# the metadata-enrichment landing in commit 3e192e5. This must NOT trigger
# the geo path — "none" is semantically falsy for routing purposes.
_META_NON_GEO_WITH_NONE_STRING = {
    "name": "rbnz_b2_wholesale_rates_monthly",
    "source": "RBNZ", "namespace": "rbnz",
    "bulk_export_class": "on_demand",
    "geometry_type": "none",      # string "none" — must be treated as non-geo
    "has_geometry": None,
    "row_count_at_last_refresh": 91,   # below the 100k threshold
}

FAKE_LOCAL_DF = pd.DataFrame({"date": ["2023-01-01"], "value": [1100.5]})


def test_get_auto_small_tabular_routes_live(client):
    """mode='auto' with a small, non-geo, bulk-eligible dataset must use the live path."""
    with (
        patch.object(client, "info", return_value=_META_SMALL_TABULAR) as mock_info,
        patch.object(client, "_get_local_impl") as mock_local,
        patch.object(client, "_fetch_dataframe",
                     return_value=pd.DataFrame(RECORDS)) as mock_live,
    ):
        result = client.get("nz_cpi")

    mock_info.assert_called_once_with("nz_cpi")
    mock_local.assert_not_called()                      # cache path NOT taken
    mock_live.assert_called_once()                      # live path taken
    assert isinstance(result, Dataset)


def test_get_auto_large_tabular_routes_cached(client):
    """mode='auto' with >100k-row bulk-eligible dataset must use the cache+sync path."""
    with (
        patch.object(client, "info", return_value=_META_LARGE_TABULAR),
        patch.object(client, "_get_local_impl", return_value=FAKE_LOCAL_DF) as mock_local,
        patch.object(client, "_fetch_dataframe") as mock_live,
    ):
        result = client.get("pharmac_schedule")

    mock_local.assert_called_once()
    mock_live.assert_not_called()
    assert isinstance(result, pd.DataFrame)


def test_get_auto_geo_routes_cached(client):
    """mode='auto' with a geo dataset must use the cache+sync path regardless of row count."""
    meta_geo_small_count = {**_META_GEO, "row_count_at_last_refresh": 5_000}
    with (
        patch.object(client, "info", return_value=meta_geo_small_count),
        patch.object(client, "_get_local_impl", return_value=FAKE_LOCAL_DF) as mock_local,
        patch.object(client, "_fetch_dataframe") as mock_live,
    ):
        client.get("nz_parcels")

    mock_local.assert_called_once()
    mock_live.assert_not_called()


def test_get_auto_licence_restricted_routes_live(client):
    """Licence-restricted dataset (bulk_export_class='none') must always go live,
    even when row_count is above the threshold.  OECD must never hit the bulk path."""
    with (
        patch.object(client, "info", return_value=_META_LICENCE_RESTRICTED),
        patch.object(client, "_get_local_impl") as mock_local,
        patch.object(client, "_fetch_dataframe",
                     return_value=pd.DataFrame(RECORDS)) as mock_live,
    ):
        client.get("oecd_gdp")

    mock_local.assert_not_called()   # licence floor — must not attempt bulk
    mock_live.assert_called_once()


def test_get_slice_limit_forces_live(client):
    """Any slice kwarg (limit=) must bypass the metadata check entirely and go live."""
    with (
        patch.object(client, "info") as mock_info,
        patch.object(client, "_get_local_impl") as mock_local,
        patch.object(client, "_fetch_dataframe",
                     return_value=pd.DataFrame(RECORDS)) as mock_live,
    ):
        client.get("nz_parcels", limit=10)

    mock_info.assert_not_called()   # no metadata round-trip needed
    mock_local.assert_not_called()
    mock_live.assert_called_once()


def test_get_slice_start_forces_live(client):
    """start= kwarg must force the live path, no metadata call."""
    with (
        patch.object(client, "info") as mock_info,
        patch.object(client, "_get_local_impl") as mock_local,
        patch.object(client, "_fetch_dataframe",
                     return_value=pd.DataFrame(RECORDS)),
    ):
        client.get("nz_cpi", start="2020-01-01")

    mock_info.assert_not_called()
    mock_local.assert_not_called()


def test_get_slice_end_forces_live(client):
    """end= kwarg must force the live path, no metadata call."""
    with (
        patch.object(client, "info") as mock_info,
        patch.object(client, "_get_local_impl") as mock_local,
        patch.object(client, "_fetch_dataframe",
                     return_value=pd.DataFrame(RECORDS)),
    ):
        client.get("nz_cpi", end="2024-12-31")

    mock_info.assert_not_called()
    mock_local.assert_not_called()


def test_get_mode_live_forces_live(client):
    """mode='live' must bypass smart routing entirely and always use the live path."""
    with (
        patch.object(client, "info") as mock_info,
        patch.object(client, "_get_local_impl") as mock_local,
        patch.object(client, "_fetch_dataframe",
                     return_value=pd.DataFrame(RECORDS)),
    ):
        client.get("nz_parcels", mode="live")

    mock_info.assert_not_called()
    mock_local.assert_not_called()


def test_get_mode_cached_forces_cached(client):
    """mode='cached' must bypass smart routing and always delegate to _get_local_impl."""
    with (
        patch.object(client, "info") as mock_info,
        patch.object(client, "_get_local_impl", return_value=FAKE_LOCAL_DF) as mock_local,
        patch.object(client, "_fetch_dataframe") as mock_live,
    ):
        result = client.get("nz_cpi", mode="cached")

    mock_info.assert_not_called()
    mock_live.assert_not_called()
    mock_local.assert_called_once()
    assert isinstance(result, pd.DataFrame)


def test_get_local_alias_delegates_to_impl(client):
    """get_local(name) must call _get_local_impl — same code path as get(name, mode='cached')."""
    with patch.object(client, "_get_local_impl", return_value=FAKE_LOCAL_DF) as mock_impl:
        result_via_get_local = client.get_local("nz_cpi")
        result_via_get_cached = client.get("nz_cpi", mode="cached")

    # Both calls went through _get_local_impl
    assert mock_impl.call_count == 2


def test_get_auto_geometry_type_none_string_routes_live(client):
    """geometry_type='none' (the string) returned by the enriched metadata endpoint
    must NOT trigger the geo/cache path for a small dataset.
    Regression test for Bug A — the non-empty string was truthy before the fix."""
    with (
        patch.object(client, "info", return_value=_META_NON_GEO_WITH_NONE_STRING),
        patch.object(client, "_get_local_impl") as mock_local,
        patch.object(client, "_fetch_dataframe",
                     return_value=pd.DataFrame(RECORDS)) as mock_live,
    ):
        client.get("rbnz_b2_wholesale_rates_monthly")

    mock_local.assert_not_called()   # must NOT go to cache+sync
    mock_live.assert_called_once()   # must use the live path


def test_get_auto_info_exception_falls_through_to_live(client):
    """If info() raises for any reason, mode='auto' must silently fall back to live."""
    with (
        patch.object(client, "info", side_effect=Exception("network error")),
        patch.object(client, "_get_local_impl") as mock_local,
        patch.object(client, "_fetch_dataframe",
                     return_value=pd.DataFrame(RECORDS)) as mock_live,
    ):
        client.get("nz_cpi")

    mock_local.assert_not_called()
    mock_live.assert_called_once()


def test_get_invalid_mode_raises_value_error(client):
    """An unknown mode string must raise ValueError immediately."""
    with pytest.raises(ValueError, match="Unknown mode"):
        client.get("nz_cpi", mode="turbo")


def test_get_auto_one_time_info_log(client, caplog):
    """mode='auto' routing through cache must emit exactly one INFO log per dataset per session."""
    import logging
    # Reset the per-session notify set so the log fires in this test
    import eolas_data.client as _client_module
    _client_module._auto_route_notified.discard("nz_parcels")

    with (
        patch.object(client, "info", return_value=_META_GEO),
        patch.object(client, "_get_local_impl", return_value=FAKE_LOCAL_DF),
    ):
        with caplog.at_level(logging.INFO, logger="eolas_data"):
            client.get("nz_parcels")
            client.get("nz_parcels")   # second call must NOT re-log

    info_messages = [r for r in caplog.records
                     if r.levelno == logging.INFO and "nz_parcels" in r.message]
    assert len(info_messages) == 1
    assert "cache+sync" in info_messages[0].message
    assert "mode='live'" in info_messages[0].message


# ---------------------------------------------------------------------------
# Back-compat: existing test callsites (e.g. client.get("nz_cpi") with only
# the /data endpoint mocked) must continue to return Dataset objects.
# The auto-routing catches info() failures and falls through to live.
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_get_back_compat_small_dataset_with_only_data_endpoint_mocked(client):
    """client.get('nz_cpi') with only /data mocked:
    - auto-mode calls info() → ConnectionError (unmocked) → caught → meta={}
    - bulk_ok=False → live path taken
    - Dataset returned as before.
    """
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data", json={"data": RECORDS})
    df = client.get("nz_cpi")
    assert isinstance(df, Dataset)
    assert len(df) == 2
    assert df.eolas_name == "nz_cpi"
