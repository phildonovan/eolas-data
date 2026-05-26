"""Integration test for client.sync() against the live eolas.fyi API.

Run with:
    pytest tests/test_sync_integration.py -v

These tests make real HTTPS requests to api.eolas.fyi and write temporary
files to a system temp directory.  They are marked ``integration`` so they
can be excluded from fast CI with ``-m "not integration"``.

Dataset: ``doc_huts`` — 1,429 DOC huts, geo (Point), weekly refresh.
Small enough to be fast; geo so we exercise the geoparquet code path.
"""
from __future__ import annotations

import os
import pathlib
import tempfile

import pytest

from eolas_data import Client
from eolas_data.sync import MANIFEST_FILENAME, read_manifest
from eolas_data.sync.sync import SyncResult

# API key injected via environment (matches the smoke-test pattern).
_API_KEY = os.getenv("EOLAS_API_KEY", "vs_5QZ-LyljyIcJPXoJdYBdm85AwklGauBNq4VjDy8r5BA")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_client():
    return Client(_API_KEY)


@pytest.fixture(scope="module")
def tmpdir_path():
    """A single temp directory shared across tests in this module."""
    with tempfile.TemporaryDirectory(prefix="eolas-sync-integration-") as td:
        yield pathlib.Path(td)


class TestSyncIntegrationDocHuts:
    def test_first_sync_status_snapshot_full(self, live_client, tmpdir_path):
        """First sync against live API → status='snapshot_full', file appears."""
        result = live_client.sync("doc_huts", library_dir=tmpdir_path)

        assert isinstance(result, SyncResult)
        assert result.status == "snapshot_full"
        assert result.bytes_downloaded > 0
        assert result.files_added == 1
        assert result.dataset == "doc_huts"

    def test_first_sync_snapshot_file_exists(self, live_client, tmpdir_path):
        """After first sync the dataset dir must contain a snapshot parquet file."""
        dataset_dir = tmpdir_path / "doc_huts"
        assert dataset_dir.exists(), "Dataset directory must be created"

        parquet_files = (
            list(dataset_dir.glob("snapshot-*.parquet"))
            + list(dataset_dir.glob("snapshot-*.geo.parquet"))
        )
        assert parquet_files, "At least one snapshot parquet file must exist"

    def test_first_sync_manifest_shape(self, live_client, tmpdir_path):
        """Manifest must be written with correct schema."""
        manifest_path = tmpdir_path / "doc_huts" / MANIFEST_FILENAME
        assert manifest_path.exists(), "Manifest file must be created"

        m = read_manifest(manifest_path)
        assert m is not None
        assert m.dataset == "doc_huts"
        assert m.schema_version == 1
        assert len(m.snapshots) >= 1
        assert m.current_snapshot is not None
        assert m.format in ("parquet", "geoparquet")
        assert m.snapshots[0].kind == "snapshot"
        assert m.snapshots[0].rows is not None
        assert m.snapshots[0].rows > 0

    def test_second_sync_unchanged(self, live_client, tmpdir_path):
        """Second sync (snapshot unchanged since previous call in same session) → 'unchanged'."""
        # The snapshot was just downloaded in the first test; it hasn't changed.
        result = live_client.sync("doc_huts", library_dir=tmpdir_path)

        assert result.status == "unchanged"
        assert result.bytes_downloaded == 0
        assert result.rows_added == 0
        assert result.files_added == 0

    def test_second_sync_manifest_unchanged(self, live_client, tmpdir_path):
        """Manifest must not gain new entries on an unchanged sync."""
        manifest_path = tmpdir_path / "doc_huts" / MANIFEST_FILENAME
        m = read_manifest(manifest_path)
        assert m is not None
        # Should still have only 1 entry (the initial snapshot).
        assert len(m.snapshots) == 1

    def test_rows_added_reasonable(self, live_client, tmpdir_path):
        """rows_added on first sync should be close to the known ~1,429 huts."""
        manifest_path = tmpdir_path / "doc_huts" / MANIFEST_FILENAME
        m = read_manifest(manifest_path)
        assert m is not None
        total_rows = m.snapshots[0].rows
        # DOC huts count is around 1,429 — allow ±500 for new/removed huts.
        assert 900 < total_rows < 2500, (
            f"Expected ~1,429 DOC huts, got {total_rows}. "
            "Something may be wrong with the row count."
        )
