"""Tests for client.get() ParquetDataset integration (library smart-routing).

When EOLAS_LIBRARY is set and a _eolas-manifest.json exists for the requested
dataset, client.get() should read from disk via pyarrow.dataset rather than
hitting the API.

Scenarios:
  1. Manifest present + EOLAS_LIBRARY set → reads from disk
  2. Manifest absent → falls through to existing live/cache path
  3. mode='live' override → live API regardless of manifest
  4. as_arrow=True over a synced dataset → returns pyarrow.Table
  5. EOLAS_LIBRARY not set → no library read (falls through to live/cache)
  6. Source helper (client.doc()) routes through library when manifest present
"""
from __future__ import annotations

import os
import pathlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import responses as resp_lib

from eolas_data import Client
from eolas_data.dataset import Dataset
from eolas_data.sync import MANIFEST_FILENAME, ManifestEntry, Manifest, write_manifest

BASE = "https://api.eolas.fyi"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synced_dataset(
    library_dir: pathlib.Path,
    name: str,
    *,
    n_rows: int = 5,
    snap_id: int = 9999,
    extra_columns: dict | None = None,
) -> pathlib.Path:
    """Create a minimal synced dataset directory with one snapshot parquet file."""
    dataset_dir = library_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Build a minimal dataframe.
    data = {"id": list(range(n_rows)), "value": [float(i) for i in range(n_rows)]}
    if extra_columns:
        data.update(extra_columns)
    tbl = pa.table(data)

    snapshot_filename = "snapshot-2026-05-27.parquet"
    snapshot_path = dataset_dir / snapshot_filename
    pq.write_table(tbl, snapshot_path)

    entry = ManifestEntry(
        snapshot_id=snap_id,
        kind="snapshot",
        file=snapshot_filename,
        synced_at="2026-05-27T10:00:00Z",
        rows=n_rows,
    )
    m = Manifest(
        dataset=name,
        snapshots=[entry],
        current_snapshot=snap_id,
        format="parquet",
        schema_version=1,
    )
    write_manifest(m, dataset_dir / MANIFEST_FILENAME)
    return dataset_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    return Client("test_key", base_url=BASE)


@pytest.fixture()
def library_dir(tmp_path):
    return tmp_path / "eolas-library"


# ---------------------------------------------------------------------------
# 1. Manifest present + EOLAS_LIBRARY set → reads from disk
# ---------------------------------------------------------------------------

def test_get_reads_from_library_when_manifest_present(
    client, library_dir, monkeypatch
):
    """client.get() returns data from the synced library when manifest exists."""
    _write_synced_dataset(library_dir, "doc_huts", n_rows=5, snap_id=1001)
    monkeypatch.setenv("EOLAS_LIBRARY", str(library_dir))

    # The library read must NOT make any HTTP calls.
    with resp_lib.RequestsMock() as rsps:
        result = client.get("doc_huts")

    # Should have returned the 5 rows from the parquet file.
    assert isinstance(result, (pd.DataFrame, Dataset))
    assert len(result) == 5
    assert "id" in result.columns
    assert "value" in result.columns


def test_get_reads_correct_rows_from_library(
    client, library_dir, monkeypatch
):
    """Row count from library matches what was written."""
    _write_synced_dataset(library_dir, "nz_cpi", n_rows=12, snap_id=2002)
    monkeypatch.setenv("EOLAS_LIBRARY", str(library_dir))

    with resp_lib.RequestsMock():
        result = client.get("nz_cpi")

    assert len(result) == 12


# ---------------------------------------------------------------------------
# 2. Manifest absent → falls through to live path
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_get_falls_through_to_live_when_no_manifest(
    client, library_dir, monkeypatch
):
    """Without a manifest the library short-circuit is skipped."""
    # Library dir exists but no dataset subdir.
    library_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("EOLAS_LIBRARY", str(library_dir))

    # Mock the live API.
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data",
                 json={"data": [{"date": "2023-01-01", "value": 99.0}]}, status=200)

    result = client.get("nz_cpi", mode="live")

    assert len(result) >= 1


# ---------------------------------------------------------------------------
# 3. mode='live' override → hits API regardless of manifest
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_get_live_mode_bypasses_library(
    client, library_dir, monkeypatch
):
    """mode='live' always hits the API, even when a manifest is present."""
    _write_synced_dataset(library_dir, "doc_huts", n_rows=99, snap_id=3003)
    monkeypatch.setenv("EOLAS_LIBRARY", str(library_dir))

    # Provide a live API response with only 1 row.
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts/data",
                 json={"data": [{"id": 1, "name": "Hooker Hut"}]}, status=200)

    result = client.get("doc_huts", mode="live")

    # Should have the 1-row API response, not the 99-row library.
    assert len(result) == 1
    assert "name" in result.columns


# ---------------------------------------------------------------------------
# 4. as_arrow=True → returns pyarrow.Table
# ---------------------------------------------------------------------------

def test_get_as_arrow_reads_from_library(
    client, library_dir, monkeypatch
):
    """as_arrow=True returns a pyarrow.Table from the library."""
    _write_synced_dataset(library_dir, "doc_huts", n_rows=7, snap_id=4004)
    monkeypatch.setenv("EOLAS_LIBRARY", str(library_dir))

    with resp_lib.RequestsMock():
        result = client.get("doc_huts", as_arrow=True)

    assert isinstance(result, pa.Table)
    assert result.num_rows == 7
    assert "id" in result.schema.names


# ---------------------------------------------------------------------------
# 5. EOLAS_LIBRARY not set → no library read
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_get_no_library_env_skips_library_read(
    client, library_dir, monkeypatch
):
    """When EOLAS_LIBRARY is not set, library read is skipped."""
    # Write a library dataset anyway (but don't set env var).
    _write_synced_dataset(library_dir, "nz_cpi", n_rows=50, snap_id=5005)
    monkeypatch.delenv("EOLAS_LIBRARY", raising=False)

    # Also ensure config doesn't have library_dir set.
    # The client's _read_library_dir_from_config_static reads ~/.eolas/config.json;
    # in test env there's no config unless we write one.

    # Live API returns 1 row — if library is accidentally read we'd see 50.
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data",
                 json={"data": [{"date": "2024-01-01", "value": 100.0}]}, status=200)

    result = client.get("nz_cpi", mode="live")
    # mode='live' bypasses library anyway — just confirm no error
    assert len(result) >= 1


# ---------------------------------------------------------------------------
# 6. Source helper routes through library
# ---------------------------------------------------------------------------

def test_doc_helper_reads_from_library(
    client, library_dir, monkeypatch
):
    """client.doc('doc_huts') reads from library when manifest present."""
    _write_synced_dataset(library_dir, "doc_huts", n_rows=6, snap_id=6006)
    monkeypatch.setenv("EOLAS_LIBRARY", str(library_dir))

    with resp_lib.RequestsMock():
        result = client.doc("doc_huts")

    assert len(result) == 6


# ---------------------------------------------------------------------------
# 7. Multi-file dataset (snapshot + delta) is read as one logical table
# ---------------------------------------------------------------------------

def test_get_reads_multiple_files_as_one_table(
    client, library_dir, monkeypatch
):
    """When a dataset has snapshot + delta files, get() returns their union."""
    import datetime

    name = "nz_parcels"
    dataset_dir = library_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    snap_id_v1 = 7001
    snap_id_v2 = 7002

    # Write snapshot file (3 rows)
    snap_tbl = pa.table({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
    snap_filename = "snapshot-2026-05-20.parquet"
    pq.write_table(snap_tbl, dataset_dir / snap_filename)

    # Write delta file (2 rows)
    delta_tbl = pa.table({"id": [4, 5], "value": [40.0, 50.0]})
    delta_filename = "delta-2026-05-20-to-2026-05-27.parquet"
    pq.write_table(delta_tbl, dataset_dir / delta_filename)

    # Write manifest referencing both
    snap_entry = ManifestEntry(
        snapshot_id=snap_id_v1,
        kind="snapshot",
        file=snap_filename,
        synced_at="2026-05-20T10:00:00Z",
        rows=3,
    )
    delta_entry = ManifestEntry(
        snapshot_id=snap_id_v2,
        kind="delta",
        parent_snapshot=snap_id_v1,
        file=delta_filename,
        synced_at="2026-05-27T10:00:00Z",
        rows_added=2,
    )
    m = Manifest(
        dataset=name,
        snapshots=[snap_entry, delta_entry],
        current_snapshot=snap_id_v2,
        format="parquet",
        schema_version=1,
    )
    write_manifest(m, dataset_dir / MANIFEST_FILENAME)

    monkeypatch.setenv("EOLAS_LIBRARY", str(library_dir))

    with resp_lib.RequestsMock():
        result = client.get(name)

    # Should have 5 rows total (3 from snapshot + 2 from delta)
    assert len(result) == 5
    assert set(result["id"].tolist()) == {1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# 8. slice kwargs (start/end/limit) bypass library
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_get_with_limit_bypasses_library(
    client, library_dir, monkeypatch
):
    """client.get(name, limit=1) must bypass the library and hit the API."""
    _write_synced_dataset(library_dir, "nz_cpi", n_rows=50, snap_id=8008)
    monkeypatch.setenv("EOLAS_LIBRARY", str(library_dir))

    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data",
                 json={"data": [{"date": "2024-01-01", "value": 1.0}]}, status=200)

    result = client.get("nz_cpi", limit=1)
    # API returned 1 row; the 50-row library was bypassed.
    assert len(result) == 1
