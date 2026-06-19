"""Tests for Client.get_local() — mirror of eolas-r/tests/testthat/test-get-local.R."""
from __future__ import annotations

import gzip
import pathlib
from unittest.mock import patch

import pandas as pd
import pytest

from eolas_data import Client, SyncResult
from eolas_data.exceptions import (
    BulkLicenceRestricted,
    BulkNotYetAvailable,
    BulkUpgradeRequired,
)

BASE = "https://api.eolas.fyi"
SNAPSHOT_ID = "snap_abc123"

BULK_DATASET_META = {
    "name": "nz_cpi",
    "title": "NZ CPI",
    "source": "Stats NZ",
    "namespace": "statsnz",
    "table": "nz_cpi",
}

GEO_DATASET_META = {
    "name": "nz_parcels",
    "title": "NZ Parcels",
    "source": "LINZ",
    "namespace": "linz",
    "table": "nz_parcels",
    "geometry_type": "MultiPolygon",
    "bulk_export_class": "geoparquet",
    "row_count_at_last_refresh": 3_000_000,
}


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


def _sync_downloaded(name, *, path, format, **kwargs) -> SyncResult:
    return SyncResult(
        status="downloaded",
        previous_snapshot_id=None,
        current_snapshot_id=SNAPSHOT_ID,
        path=path,
        bytes_downloaded=1024,
    )


def _sync_unchanged(name, *, path, format, **kwargs) -> SyncResult:
    return SyncResult(
        status="unchanged",
        previous_snapshot_id=SNAPSHOT_ID,
        current_snapshot_id=SNAPSHOT_ID,
        path=path,
        bytes_downloaded=0,
    )


def _write_csv_gz(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("date,value\n2023-01-01,1100.5\n")


# ---------------------------------------------------------------------------
# First call / cached read
# ---------------------------------------------------------------------------

def test_get_local_returns_dataset_csv_gz(client, tmp_path):
    """First call reads a csv_gz bulk file into a Dataset."""

    def fake_sync(name, *, path, format, **kwargs):
        _write_csv_gz(path)
        return _sync_downloaded(name, path=path, format=format)

    with (
        patch.object(client, "_info_cached", return_value=BULK_DATASET_META),
        patch.object(client, "sync_bulk", side_effect=fake_sync),
    ):
        result = client.get_local(
            "nz_cpi", format="csv_gz", cache_dir=tmp_path,
        )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["date", "value"]
    assert len(result) == 1
    assert result["value"].iloc[0] == pytest.approx(1100.5)


def test_get_local_reads_cached_file_on_subsequent_call(client, tmp_path):
    """Unchanged sync_bulk → read existing csv.gz without re-downloading."""
    csv_path = tmp_path / "nz_cpi.csv.gz"
    with gzip.open(csv_path, "wt", encoding="utf-8") as fh:
        fh.write("date,value\n2023-06-01,1105.0\n")

    with (
        patch.object(client, "_info_cached", return_value=BULK_DATASET_META),
        patch.object(client, "sync_bulk", side_effect=_sync_unchanged),
    ):
        result = client.get_local("nz_cpi", format="csv_gz", cache_dir=tmp_path)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result["value"].iloc[0] == pytest.approx(1105.0)


def test_get_local_expands_tilde_in_cache_dir(client, tmp_path, monkeypatch):
    """cache_dir='~/…' is expanded to an absolute path before sync_bulk."""
    home = tmp_path / "fakehome"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    seen_path = {"path": None}

    def fake_sync(name, *, path, format, **kwargs):
        seen_path["path"] = path
        _write_csv_gz(path)
        return _sync_downloaded(name, path=path, format=format)

    with patch.object(client, "sync_bulk", side_effect=fake_sync):
        client.get_local("nz_cpi", format="csv_gz", cache_dir="~/.cache/eolas", meta=False)

    expected = (home / ".cache" / "eolas" / "nz_cpi.csv.gz").resolve()
    assert seen_path["path"] == expected
    assert "~" not in str(seen_path["path"])


def test_get_local_explicit_cache_dir_overrides_library(client, tmp_path, monkeypatch):
    """Explicit cache_dir wins over EOLAS_LIBRARY / config resolution."""
    import eolas_data.library as lib

    env_dir = tmp_path / "from_env"
    monkeypatch.setenv("EOLAS_LIBRARY", str(env_dir))
    explicit = tmp_path / "explicit_cache"

    def fake_sync(name, *, path, format, **kwargs):
        _write_csv_gz(path)
        return _sync_downloaded(name, path=path, format=format)

    with patch.object(client, "sync_bulk", side_effect=fake_sync):
        client.get_local("nz_cpi", format="csv_gz", cache_dir=explicit, meta=False)

    assert (explicit / "nz_cpi.csv.gz").exists()
    assert not (env_dir / "nz_cpi.csv.gz").exists()
    # Sanity: library resolver would have picked env_dir
    assert lib.resolve_library_dir(interactive=False) == env_dir.resolve()


# ---------------------------------------------------------------------------
# Format auto-detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "meta,expected_fmt",
    [
        (GEO_DATASET_META, "geoparquet"),
        (BULK_DATASET_META, "parquet"),
        (
            {
                "name": "rbnz_b2_wholesale_rates_monthly",
                "namespace": "rbnz",
                "table": "rbnz_b2_wholesale_rates_monthly",
                "geometry_type": "none",
            },
            "parquet",
        ),
    ],
)
def test_get_local_auto_detects_format(client, tmp_path, meta, expected_fmt):
    """Geo metadata → geoparquet; tabular / geometry_type='none' → parquet."""
    seen = {"format": None}

    def fake_sync(name, *, path, format, **kwargs):
        seen["format"] = format
        raise OSError("skip read — format detection only")

    with (
        patch.object(client, "_info_cached", return_value=meta),
        patch.object(client, "sync_bulk", side_effect=fake_sync),
        pytest.raises(OSError, match="skip read"),
    ):
        client.get_local(meta["name"], cache_dir=tmp_path, meta=True)

    assert seen["format"] == expected_fmt


# ---------------------------------------------------------------------------
# Error propagation + validation
# ---------------------------------------------------------------------------

def test_get_local_as_arrow_and_as_geo_conflict(client, tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.get_local("nz_cpi", cache_dir=tmp_path, as_arrow=True, as_geo=True)


def test_get_local_unknown_format_raises(client, tmp_path):
    with pytest.raises(ValueError, match="Unknown format"):
        client.get_local("nz_cpi", format="xml", cache_dir=tmp_path, meta=False)


@pytest.mark.parametrize(
    "exc_cls,message",
    [
        (BulkUpgradeRequired, "Bulk upgrade required"),
        (BulkLicenceRestricted, "Bulk licence restricted"),
        (BulkNotYetAvailable, "Bulk not yet available"),
    ],
)
def test_get_local_propagates_bulk_errors(client, tmp_path, exc_cls, message):
    with (
        patch.object(client, "_info_cached", return_value=BULK_DATASET_META),
        patch.object(client, "sync_bulk", side_effect=exc_cls(message)),
        pytest.raises(exc_cls, match=message.split()[1]),  # partial match
    ):
        client.get_local("nz_cpi", cache_dir=tmp_path)


def test_get_local_parquet_histogram_fallback_to_csv_gz(client, tmp_path):
    """OSError mentioning 'histogram' triggers csv_gz re-sync + read."""
    parquet_path = tmp_path / "nz_cpi.parquet"
    csv_path = tmp_path / "nz_cpi.csv.gz"
    calls: list[tuple[str, str]] = []

    def fake_sync(name, *, path, format, **kwargs):
        calls.append((format, path.name))
        if format == "csv_gz":
            _write_csv_gz(path)
            return _sync_downloaded(name, path=path, format=format)
        path.touch()
        return _sync_downloaded(name, path=path, format=format)

    real_read_parquet = pd.read_parquet

    def flaky_read_parquet(path, *args, **kwargs):
        if path == parquet_path:
            raise OSError("Repetition level histogram size mismatch")
        return real_read_parquet(path, *args, **kwargs)

    with (
        patch.object(client, "_info_cached", return_value=BULK_DATASET_META),
        patch.object(client, "sync_bulk", side_effect=fake_sync),
        patch("eolas_data.client.pd.read_parquet", side_effect=flaky_read_parquet),
    ):
        result = client.get_local("nz_cpi", cache_dir=tmp_path, meta=True)

    assert isinstance(result, pd.DataFrame)
    assert calls[0] == ("parquet", "nz_cpi.parquet")
    assert ("csv_gz", "nz_cpi.csv.gz") in calls
    assert csv_path.exists()


def test_get_local_as_arrow_returns_table(client, tmp_path):
    """as_arrow=True returns a pyarrow.Table (no pandas materialisation)."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "nz_cpi.parquet"
    table = pa.table({"date": ["2023-01-01"], "value": [1100.5]})
    pq.write_table(table, parquet_path)

    with patch.object(client, "sync_bulk", side_effect=_sync_unchanged):
        result = client.get_local(
            "nz_cpi", cache_dir=tmp_path, as_arrow=True, meta=False,
        )

    assert isinstance(result, pa.Table)
    assert result.num_rows == 1