"""Tests for Client.get_local() — the notebook-friendly whole-dataset convenience."""
from __future__ import annotations

import gzip
import io
import pathlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from eolas_data import Client, SyncResult
from eolas_data.exceptions import (
    BulkLicenceRestricted,
    BulkNotYetAvailable,
    BulkUpgradeRequired,
)

BASE = "https://api.eolas.fyi"


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sync_result(path: pathlib.Path, status: str = "downloaded") -> SyncResult:
    return SyncResult(
        status=status,
        previous_snapshot_id=None,
        current_snapshot_id="snap_abc123",
        path=path,
        bytes_downloaded=1024 if status != "unchanged" else 0,
    )


FAKE_DF = pd.DataFrame({"date": ["2023-01-01"], "value": [1100.5]})


# ---------------------------------------------------------------------------
# test_get_local_first_call_downloads_and_returns_df
# ---------------------------------------------------------------------------

def test_get_local_first_call_downloads_and_returns_df(client, tmp_path):
    """First call: sync_bulk downloads the file; get_local reads and returns DataFrame."""
    expected_path = tmp_path / "nz_cpi.parquet"

    def fake_sync_bulk(name, *, path, format, freshness):
        # create a stub file so the read doesn't fail
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return _make_sync_result(path, "downloaded")

    with (
        patch.object(client, "info", return_value={"name": "nz_cpi", "source": "Stats NZ"}),
        patch.object(client, "sync_bulk", side_effect=fake_sync_bulk),
        patch("pandas.read_parquet", return_value=FAKE_DF) as mock_read,
    ):
        result = client.get_local("nz_cpi", cache_dir=str(tmp_path))

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["date", "value"]
    # File was at expected path
    mock_read.assert_called_once_with(expected_path)


# ---------------------------------------------------------------------------
# test_get_local_subsequent_call_unchanged
# ---------------------------------------------------------------------------

def test_get_local_subsequent_call_unchanged(client, tmp_path):
    """Subsequent call: sync_bulk returns unchanged; DataFrame still returned."""
    expected_path = tmp_path / "nz_cpi.parquet"

    def fake_sync_bulk(name, *, path, format, freshness):
        return _make_sync_result(path, "unchanged")

    with (
        patch.object(client, "info", return_value={"name": "nz_cpi"}),
        patch.object(client, "sync_bulk", side_effect=fake_sync_bulk),
        patch("pandas.read_parquet", return_value=FAKE_DF),
    ):
        result = client.get_local("nz_cpi", cache_dir=str(tmp_path))

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["date", "value"]


# ---------------------------------------------------------------------------
# test_get_local_auto_format_geo
# ---------------------------------------------------------------------------

def test_get_local_auto_format_geo(client, tmp_path):
    """Dataset metadata with geometry_type -> geoparquet format, GeoDataFrame returned."""
    called_format = {}

    def fake_sync_bulk(name, *, path, format, freshness):
        called_format["format"] = format
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return _make_sync_result(path, "downloaded")

    fake_gdf = MagicMock()
    fake_gdf.__class__.__name__ = "GeoDataFrame"

    # Patch geopandas import + read_parquet inside the method
    mock_gpd = MagicMock()
    mock_gpd.read_parquet.return_value = fake_gdf

    with (
        patch.object(
            client, "info",
            return_value={"name": "nz_addresses", "geometry_type": "Point"},
        ),
        patch.object(client, "sync_bulk", side_effect=fake_sync_bulk),
        patch.dict("sys.modules", {"geopandas": mock_gpd}),
    ):
        result = client.get_local("nz_addresses", cache_dir=str(tmp_path))

    # Format auto-detected as geoparquet
    assert called_format["format"] == "geoparquet"
    # File would have .geo.parquet extension
    assert (tmp_path / "nz_addresses.geo.parquet").exists() or called_format["format"] == "geoparquet"
    # geopandas read_parquet was called
    mock_gpd.read_parquet.assert_called_once()
    assert result is fake_gdf


# ---------------------------------------------------------------------------
# test_get_local_auto_format_non_geo
# ---------------------------------------------------------------------------

def test_get_local_auto_format_non_geo(client, tmp_path):
    """No geometry in metadata -> parquet format, plain DataFrame returned."""
    called_format = {}

    def fake_sync_bulk(name, *, path, format, freshness):
        called_format["format"] = format
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return _make_sync_result(path, "downloaded")

    with (
        patch.object(
            client, "info",
            return_value={"name": "nz_cpi", "source": "Stats NZ"},  # no geometry_type
        ),
        patch.object(client, "sync_bulk", side_effect=fake_sync_bulk),
        patch("pandas.read_parquet", return_value=FAKE_DF),
    ):
        result = client.get_local("nz_cpi", cache_dir=str(tmp_path))

    assert called_format["format"] == "parquet"
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# test_get_local_explicit_format_csv_gz
# ---------------------------------------------------------------------------

def test_get_local_explicit_format_csv_gz(client, tmp_path):
    """Passing format='csv_gz' -> reads via pd.read_csv, .csv.gz extension."""
    called_format = {}
    expected_path = tmp_path / "nz_cpi.csv.gz"

    def fake_sync_bulk(name, *, path, format, freshness):
        called_format["format"] = format
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return _make_sync_result(path, "downloaded")

    with (
        patch.object(client, "info", return_value={"name": "nz_cpi"}),
        patch.object(client, "sync_bulk", side_effect=fake_sync_bulk),
        patch("pandas.read_csv", return_value=FAKE_DF) as mock_csv,
    ):
        result = client.get_local("nz_cpi", format="csv_gz", cache_dir=str(tmp_path))

    assert called_format["format"] == "csv_gz"
    mock_csv.assert_called_once_with(expected_path)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# test_get_local_cache_dir_expands_tilde
# ---------------------------------------------------------------------------

def test_get_local_cache_dir_expands_tilde(client, tmp_path, monkeypatch):
    """cache_dir with ~ prefix expands to an absolute path (no ~ in final path)."""
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_sync_bulk(name, *, path, format, freshness):
        # Verify the path passed to sync_bulk has no tilde
        assert "~" not in str(path)
        assert str(path).startswith("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return _make_sync_result(path, "downloaded")

    with (
        patch.object(client, "info", return_value={"name": "nz_cpi"}),
        patch.object(client, "sync_bulk", side_effect=fake_sync_bulk),
        patch("pandas.read_parquet", return_value=FAKE_DF),
    ):
        result = client.get_local("nz_cpi", cache_dir="~/.cache/eolas")

    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# test_get_local_passes_through_bulk_exceptions
# ---------------------------------------------------------------------------

def test_get_local_passes_through_bulk_exceptions_licence(client, tmp_path):
    """BulkLicenceRestricted from sync_bulk propagates unchanged (not wrapped)."""
    with (
        patch.object(
            client, "info",
            return_value={"name": "oecd_gdp", "source": "OECD"},
        ),
        patch.object(
            client, "sync_bulk",
            side_effect=BulkLicenceRestricted(
                "This dataset is not available as a bulk download (licence: OECD)."
            ),
        ),
    ):
        with pytest.raises(BulkLicenceRestricted):
            client.get_local("oecd_gdp", cache_dir=str(tmp_path))


def test_get_local_passes_through_bulk_exceptions_upgrade(client, tmp_path):
    """BulkUpgradeRequired from sync_bulk propagates unchanged."""
    with (
        patch.object(client, "info", return_value={"name": "nz_cpi"}),
        patch.object(
            client, "sync_bulk",
            side_effect=BulkUpgradeRequired("Fresh bulk downloads are a Pro feature."),
        ),
    ):
        with pytest.raises(BulkUpgradeRequired):
            client.get_local("nz_cpi", freshness="current", cache_dir=str(tmp_path))


def test_get_local_passes_through_bulk_exceptions_not_yet_available(client, tmp_path):
    """BulkNotYetAvailable from sync_bulk propagates unchanged."""
    with (
        patch.object(client, "info", return_value={"name": "nz_cpi"}),
        patch.object(
            client, "sync_bulk",
            side_effect=BulkNotYetAvailable("Monthly bulk snapshots are still rolling out."),
        ),
    ):
        with pytest.raises(BulkNotYetAvailable):
            client.get_local("nz_cpi", cache_dir=str(tmp_path))
