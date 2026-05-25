"""Tests for the as_arrow parameter across get(), get_local(), and source helpers."""
from __future__ import annotations

import io
import pathlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from eolas_data import Client, SyncResult

pytestmark = pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")

BASE = "https://api.eolas.fyi"


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_sync_result(path: pathlib.Path, status: str = "downloaded") -> SyncResult:
    return SyncResult(
        status=status,
        previous_snapshot_id=None,
        current_snapshot_id="snap_abc123",
        path=path,
        bytes_downloaded=1024,
    )


def _write_parquet(path: pathlib.Path, df: pd.DataFrame) -> None:
    """Write a pandas DataFrame to a real Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tbl, path)


FAKE_NON_GEO_DF = pd.DataFrame({"date": ["2023-01-01"], "value": [1100.5]})
FAKE_GEO_DF = pd.DataFrame({
    "id": [1],
    "name": ["Wellington"],
    "geometry_wkt": ["POINT (174.78 -41.29)"],
})


# ---------------------------------------------------------------------------
# get_local() + _get_local_impl: as_arrow on non-geo dataset
# ---------------------------------------------------------------------------

def test_get_local_as_arrow_non_geo_returns_arrow_table(client, tmp_path):
    """as_arrow=True on a non-geo dataset returns a pyarrow.Table."""
    file_path = tmp_path / "nz_cpi.parquet"
    _write_parquet(file_path, FAKE_NON_GEO_DF)

    def fake_sync_bulk(name, *, path, format, freshness, progress=None):
        return _make_sync_result(path)

    with (
        patch.object(client, "info", return_value={"name": "nz_cpi"}),
        patch.object(client, "sync_bulk", side_effect=fake_sync_bulk),
    ):
        result = client.get_local("nz_cpi", cache_dir=str(tmp_path), as_arrow=True)

    assert isinstance(result, pa.Table)
    assert "value" in result.schema.names


# ---------------------------------------------------------------------------
# get_local(): as_arrow on geo dataset (geoparquet) returns Arrow table
# ---------------------------------------------------------------------------

def test_get_local_as_arrow_geo_returns_arrow_table(client, tmp_path):
    """as_arrow=True on a geo dataset returns a pyarrow.Table (no shapely allocation)."""
    file_path = tmp_path / "nz_addresses.geo.parquet"
    _write_parquet(file_path, FAKE_GEO_DF)

    def fake_sync_bulk(name, *, path, format, freshness, progress=None):
        return _make_sync_result(path)

    with (
        patch.object(
            client, "info",
            return_value={"name": "nz_addresses", "geometry_type": "Point"},
        ),
        patch.object(client, "sync_bulk", side_effect=fake_sync_bulk),
    ):
        result = client.get_local("nz_addresses", cache_dir=str(tmp_path), as_arrow=True)

    assert isinstance(result, pa.Table)
    # geometry_wkt stays as a string column — no shapely conversion
    assert "geometry_wkt" in result.schema.names


# ---------------------------------------------------------------------------
# get_local(): default unchanged (regression test)
# ---------------------------------------------------------------------------

def test_get_local_default_returns_dataframe(client, tmp_path):
    """Default behaviour (no as_arrow) still returns a pandas DataFrame."""
    file_path = tmp_path / "nz_cpi.parquet"
    _write_parquet(file_path, FAKE_NON_GEO_DF)

    def fake_sync_bulk(name, *, path, format, freshness, progress=None):
        return _make_sync_result(path)

    with (
        patch.object(client, "info", return_value={"name": "nz_cpi"}),
        patch.object(client, "sync_bulk", side_effect=fake_sync_bulk),
    ):
        result = client.get_local("nz_cpi", cache_dir=str(tmp_path))

    assert isinstance(result, pd.DataFrame)
    assert not isinstance(result, pa.Table)


# ---------------------------------------------------------------------------
# Conflict: as_arrow=True + as_geo=True raises ValueError
# ---------------------------------------------------------------------------

def test_get_local_as_arrow_and_as_geo_raises(client, tmp_path):
    """as_arrow=True combined with explicit as_geo=True should raise a clear ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        # Must pass as_geo=True explicitly (not the default None)
        client._get_local_impl(
            "nz_cpi",
            cache_dir=str(tmp_path),
            as_arrow=True,
            as_geo=True,
        )


def test_get_as_arrow_and_as_geo_raises(client):
    """as_arrow=True combined with as_geo=True on get() raises a clear ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.get("nz_cpi", as_arrow=True, as_geo=True)


# ---------------------------------------------------------------------------
# get(): as_arrow via smart-routing (mode="auto") cache path
# ---------------------------------------------------------------------------

def test_get_auto_as_arrow_routes_to_local_and_returns_arrow_table(client, tmp_path):
    """as_arrow=True flows through get() auto-routing to get_local, returns Arrow table."""
    file_path = tmp_path / "nz_addresses.geo.parquet"
    _write_parquet(file_path, FAKE_GEO_DF)

    def fake_sync_bulk(name, *, path, format, freshness, progress=None):
        # Write the file into tmp_path so the read succeeds
        dest = tmp_path / f"{name}.geo.parquet"
        if not dest.exists():
            _write_parquet(dest, FAKE_GEO_DF)
        return _make_sync_result(dest)

    # Patch info() to make the dataset look bulk-eligible + geo so auto-routing fires
    meta = {
        "name": "nz_addresses",
        "geometry_type": "Point",
        "bulk_export_class": "geoparquet",
        "row_count_at_last_refresh": 500_000,
    }

    with (
        patch.object(client, "info", return_value=meta),
        patch.object(client, "sync_bulk", side_effect=fake_sync_bulk),
        patch("eolas_data.client.resolve_library_dir", return_value=tmp_path),
    ):
        result = client.get("nz_addresses", as_arrow=True)

    assert isinstance(result, pa.Table)
    assert "geometry_wkt" in result.schema.names


# ---------------------------------------------------------------------------
# get(): as_arrow via mode="cached" explicit
# ---------------------------------------------------------------------------

def test_get_cached_mode_as_arrow_returns_arrow_table(client, tmp_path):
    """as_arrow=True with mode='cached' returns a pyarrow.Table."""
    file_path = tmp_path / "nz_cpi.parquet"
    _write_parquet(file_path, FAKE_NON_GEO_DF)

    def fake_sync_bulk(name, *, path, format, freshness, progress=None):
        dest = tmp_path / f"{name}.parquet"
        if not dest.exists():
            _write_parquet(dest, FAKE_NON_GEO_DF)
        return _make_sync_result(dest)

    with (
        patch.object(client, "info", return_value={"name": "nz_cpi"}),
        patch.object(client, "sync_bulk", side_effect=fake_sync_bulk),
        patch("eolas_data.client.resolve_library_dir", return_value=tmp_path),
    ):
        result = client.get(
            "nz_cpi",
            as_arrow=True,
            mode="cached",
        )

    assert isinstance(result, pa.Table)


# ---------------------------------------------------------------------------
# get(): as_arrow on the live path (JSON response → pa.Table)
# ---------------------------------------------------------------------------

def test_get_live_as_arrow_returns_arrow_table(client):
    """as_arrow=True on the live path converts the JSON-fetched DataFrame to pa.Table."""
    fake_df = pd.DataFrame({"date": ["2023-01-01"], "value": [1100.5]})

    with patch.object(client, "_fetch_dataframe", return_value=fake_df):
        result = client.get("nz_cpi", as_arrow=True, mode="live")

    assert isinstance(result, pa.Table)
    assert "value" in result.schema.names


# ---------------------------------------------------------------------------
# Source helper propagation: linz() with as_arrow
# ---------------------------------------------------------------------------

def test_source_helper_linz_as_arrow_returns_arrow_table(client, tmp_path):
    """client.linz('nz_parcels', as_arrow=True) returns a pyarrow.Table."""
    file_path = tmp_path / "nz_parcels.geo.parquet"
    _write_parquet(file_path, FAKE_GEO_DF)

    def fake_sync_bulk(name, *, path, format, freshness, progress=None):
        dest = tmp_path / f"{name}.geo.parquet"
        if not dest.exists():
            _write_parquet(dest, FAKE_GEO_DF)
        return _make_sync_result(dest)

    meta = {
        "name": "nz_parcels",
        "geometry_type": "MultiPolygon",
        "bulk_export_class": "geoparquet",
        "row_count_at_last_refresh": 3_000_000,
    }

    with (
        patch.object(client, "info", return_value=meta),
        patch.object(client, "sync_bulk", side_effect=fake_sync_bulk),
        patch("eolas_data.client.resolve_library_dir", return_value=tmp_path),
    ):
        result = client.linz("nz_parcels", as_arrow=True)

    assert isinstance(result, pa.Table)


# ---------------------------------------------------------------------------
# as_arrow=True on a csv_gz file converts via pandas intermediary
# ---------------------------------------------------------------------------

def test_get_local_as_arrow_csv_gz(client, tmp_path):
    """as_arrow=True with format='csv_gz' still returns a pyarrow.Table."""
    import gzip
    csv_path = tmp_path / "nz_cpi.csv.gz"
    with gzip.open(csv_path, "wt") as fh:
        fh.write("date,value\n2023-01-01,1100.5\n")

    def fake_sync_bulk(name, *, path, format, freshness, progress=None):
        return _make_sync_result(path)

    with (
        patch.object(client, "info", return_value={"name": "nz_cpi"}),
        patch.object(client, "sync_bulk", side_effect=fake_sync_bulk),
    ):
        result = client.get_local(
            "nz_cpi",
            cache_dir=str(tmp_path),
            format="csv_gz",
            as_arrow=True,
        )

    assert isinstance(result, pa.Table)
    assert "value" in result.schema.names
