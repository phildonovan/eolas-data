"""Integration test for client.sync() / sync_all() / compact() against the live eolas.fyi API.

Run with:
    pytest tests/test_sync_integration.py -v

These tests make real HTTPS requests to api.eolas.fyi and write temporary
files to a system temp directory.  They are marked ``integration`` so they
can be excluded from fast CI with ``-m "not integration"``.

Datasets:
  - ``doc_huts``           — 1,429 DOC huts, geo (Point), weekly refresh
  - ``agr_forestry`` — DOC walking tracks, geo (LineString), weekly
  - ``agr_forestry``             — Stats NZ CPI index series, tabular, quarterly

All are small and safe to re-sync repeatedly.
"""
from __future__ import annotations

import os
import pathlib
import tempfile

import pyarrow.parquet as pq
import pytest

from eolas_data import Client
from eolas_data.sync import MANIFEST_FILENAME, read_manifest
from eolas_data.sync.compact import CompactResult
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


# ---------------------------------------------------------------------------
# sync_all() integration tests
# ---------------------------------------------------------------------------

class TestSyncAllIntegration:
    def test_sync_all_explicit_two_datasets(self, live_client, tmpdir_path):
        """sync_all with ['doc_huts', 'agr_forestry'] → both succeed."""
        results = live_client.sync_all(
            library_dir=tmpdir_path,
            datasets=["doc_huts", "agr_forestry"],
        )

        assert len(results) == 2
        for r in results:
            assert r.status in ("snapshot_full", "snapshot_delta", "unchanged"), \
                f"Unexpected status for {r.dataset}: {r.status}, error={r.error}"
            assert r.error is None, f"Unexpected error for {r.dataset}: {r.error}"

    def test_sync_all_ordered_output(self, live_client, tmpdir_path):
        """Results are in the same order as the input datasets list."""
        datasets = ["doc_huts", "agr_forestry"]
        results = live_client.sync_all(library_dir=tmpdir_path, datasets=datasets)

        assert [r.dataset for r in results] == datasets

    def test_sync_all_discovers_from_library_dir(self, live_client, tmpdir_path):
        """After syncing manually, sync_all(datasets=None) rediscovers them."""
        # Ensure at least one dataset is in the library
        live_client.sync("doc_huts", library_dir=tmpdir_path)

        results = live_client.sync_all(library_dir=tmpdir_path)

        assert len(results) >= 1
        ds_names = {r.dataset for r in results}
        assert "doc_huts" in ds_names

    def test_sync_all_second_call_unchanged(self, live_client, tmpdir_path):
        """Second sync_all on same datasets → all 'unchanged' (snapshot not yet refreshed)."""
        datasets = ["doc_huts", "agr_forestry"]
        # First call
        live_client.sync_all(library_dir=tmpdir_path, datasets=datasets)
        # Second call — snapshots haven't changed in the same test session
        results = live_client.sync_all(library_dir=tmpdir_path, datasets=datasets)

        for r in results:
            assert r.status == "unchanged", \
                f"{r.dataset} expected 'unchanged', got '{r.status}'"


# ---------------------------------------------------------------------------
# compact() integration tests
# ---------------------------------------------------------------------------

class TestCompactIntegration:
    def test_compact_after_single_sync_is_noop(self, live_client, tmpdir_path):
        """compact() after a single sync (one snapshot, no deltas) is a no-op."""
        live_client.sync("doc_huts", library_dir=tmpdir_path)

        dataset_dir = tmpdir_path / "doc_huts"
        result = live_client.compact(dataset_dir)

        assert isinstance(result, CompactResult)
        assert result.files_before == 1
        assert result.files_after == 1
        assert result.rows_before == result.rows_after
        assert result.bytes_saved == 0

    def test_compact_with_injected_delta_files(self, live_client, tmpdir_path):
        """Inject extra parquet files and verify compact merges them to 1."""
        import pyarrow as pa

        live_client.sync("doc_huts", library_dir=tmpdir_path)
        dataset_dir = tmpdir_path / "doc_huts"

        # Read manifest to get current snapshot id
        manifest = read_manifest(dataset_dir / MANIFEST_FILENAME)
        assert manifest is not None
        base_id = manifest.current_snapshot

        # Inject two synthetic delta files (minimal valid parquet)
        from eolas_data.sync.manifest import ManifestEntry, write_manifest
        for i, (delta_name, new_id) in enumerate([
            ("delta-2026-05-24-to-2026-05-25.parquet", base_id + 1001),
            ("delta-2026-05-25-to-2026-05-26.parquet", base_id + 2002),
        ]):
            tbl = pa.table({"id": [i * 10], "value": [float(i)]})
            pq.write_table(tbl, dataset_dir / delta_name)
            entry = ManifestEntry(
                snapshot_id=new_id,
                kind="delta",
                parent_snapshot=base_id if i == 0 else base_id + 1001,
                file=delta_name,
                synced_at="2026-05-27T10:00:00Z",
                rows_added=1,
            )
            manifest.snapshots.append(entry)
            manifest.current_snapshot = new_id
            base_id = new_id
        write_manifest(manifest, dataset_dir / MANIFEST_FILENAME)

        # Now compact
        result = live_client.compact(dataset_dir)

        assert result.files_before == 3   # 1 snapshot + 2 injected deltas
        assert result.files_after == 1
        # Merged file should be readable via pyarrow.
        # Use a set to deduplicate: snapshot-*.geo.parquet matches both
        # "snapshot-*.parquet" and "snapshot-*.geo.parquet" globs.
        merged_files = list({
            f for f in dataset_dir.iterdir()
            if f.is_file()
            and f.name.startswith("snapshot-")
            and (f.name.endswith(".parquet") or f.name.endswith(".geo.parquet"))
        })
        assert len(merged_files) == 1
        merged_table = pq.read_table(merged_files[0])
        assert merged_table.num_rows == result.rows_after

    def test_compact_manifest_single_entry_after_merge(self, live_client, tmpdir_path):
        """After compact, manifest has exactly 1 snapshot entry."""
        import pyarrow as pa

        live_client.sync("doc_huts", library_dir=tmpdir_path)
        dataset_dir = tmpdir_path / "doc_huts"

        manifest = read_manifest(dataset_dir / MANIFEST_FILENAME)
        assert manifest is not None
        base_id = manifest.current_snapshot

        from eolas_data.sync.manifest import ManifestEntry, write_manifest
        delta_name = "delta-2026-05-24-to-2026-05-27.parquet"
        new_id = base_id + 999
        tbl = pa.table({"id": [99], "value": [99.0]})
        pq.write_table(tbl, dataset_dir / delta_name)
        entry = ManifestEntry(
            snapshot_id=new_id,
            kind="delta",
            parent_snapshot=base_id,
            file=delta_name,
            synced_at="2026-05-27T10:00:00Z",
            rows_added=1,
        )
        manifest.snapshots.append(entry)
        manifest.current_snapshot = new_id
        write_manifest(manifest, dataset_dir / MANIFEST_FILENAME)

        live_client.compact(dataset_dir)

        post_manifest = read_manifest(dataset_dir / MANIFEST_FILENAME)
        assert post_manifest is not None
        assert len(post_manifest.snapshots) == 1
        assert post_manifest.snapshots[0].kind == "snapshot"


# ---------------------------------------------------------------------------
# CLI integration tests (shell out to the real CLI binary)
# ---------------------------------------------------------------------------

class TestCLISyncIntegration:
    """Smoke tests that exercise the real CLI binary end-to-end."""

    def test_cli_sync_creates_manifest_and_parquet(self, tmpdir_path):
        """eolas sync doc_huts --library <dir> exits 0 and creates expected files."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable, "-m", "eolas_data.cli",
                "sync", "doc_huts",
                "--library", str(tmpdir_path / "cli-lib"),
                "--api-key", _API_KEY,
                "--no-progress",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        )

        cli_lib = tmpdir_path / "cli-lib"
        dataset_dir = cli_lib / "doc_huts"
        assert dataset_dir.exists(), "Dataset directory must be created"

        parquet_files = (
            list(dataset_dir.glob("snapshot-*.parquet"))
            + list(dataset_dir.glob("snapshot-*.geo.parquet"))
        )
        assert parquet_files, "At least one snapshot parquet file must exist"

        manifest_path = dataset_dir / MANIFEST_FILENAME
        assert manifest_path.exists(), "Manifest must be created"

    def test_cli_sync_second_call_unchanged(self, tmpdir_path):
        """Second CLI sync call returns 'unchanged' in JSON output."""
        import subprocess
        import sys
        import json

        lib = tmpdir_path / "cli-lib-2"

        # First sync
        r1 = subprocess.run(
            [
                sys.executable, "-m", "eolas_data.cli",
                "sync", "doc_huts",
                "--library", str(lib),
                "--api-key", _API_KEY,
                "--no-progress",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert r1.returncode == 0, f"First sync failed: {r1.stderr}"

        # Second sync (stdout is non-TTY from subprocess → JSON mode)
        r2 = subprocess.run(
            [
                sys.executable, "-m", "eolas_data.cli",
                "sync", "doc_huts",
                "--library", str(lib),
                "--api-key", _API_KEY,
                "--no-progress",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert r2.returncode == 0, f"Second sync failed: {r2.stderr}"

        lines = [l for l in r2.stdout.splitlines() if l.strip()]
        assert lines, "Expected at least one output line"
        parsed = json.loads(lines[0])
        assert parsed["status"] == "unchanged"
        assert parsed["dataset"] == "doc_huts"

    def test_cli_compact_after_sync_exits_zero(self, tmpdir_path):
        """eolas compact <dir> after a sync exits 0."""
        import subprocess
        import sys

        lib = tmpdir_path / "cli-lib-3"

        # Sync first
        r_sync = subprocess.run(
            [
                sys.executable, "-m", "eolas_data.cli",
                "sync", "doc_huts",
                "--library", str(lib),
                "--api-key", _API_KEY,
                "--no-progress",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert r_sync.returncode == 0, f"Sync failed: {r_sync.stderr}"

        # Compact the dataset directory
        dataset_dir = lib / "doc_huts"
        r_compact = subprocess.run(
            [
                sys.executable, "-m", "eolas_data.cli",
                "compact", str(dataset_dir),
                "--api-key", _API_KEY,
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert r_compact.returncode == 0, (
            f"Compact failed: {r_compact.stderr}"
        )


class TestGetLibraryIntegration:
    """Integration tests for client.get() reading from a synced library."""

    def test_get_reads_from_library_after_sync(self, live_client, tmpdir_path):
        """After client.sync(), client.get() with EOLAS_LIBRARY set reads from disk."""
        import os

        lib = tmpdir_path / "lib-get-int"

        # First, sync doc_huts.
        sync_result = live_client.sync("doc_huts", library_dir=lib)
        assert sync_result.status in ("snapshot_full", "snapshot_delta", "unchanged")

        # Now read via get() with EOLAS_LIBRARY pointing at the library.
        old_env = os.environ.get("EOLAS_LIBRARY")
        try:
            os.environ["EOLAS_LIBRARY"] = str(lib)
            # Reset the per-session notification set so the info log fires again
            from eolas_data import client as client_module
            client_module._auto_route_notified.discard("doc_huts")

            df = live_client.get("doc_huts")
            assert len(df) > 0, "Expected rows from synced library"
            # The read should have come from disk — manifest exists.
            manifest_path = lib / "doc_huts" / MANIFEST_FILENAME
            assert manifest_path.exists()
        finally:
            if old_env is None:
                os.environ.pop("EOLAS_LIBRARY", None)
            else:
                os.environ["EOLAS_LIBRARY"] = old_env

    def test_get_as_arrow_from_synced_library(self, live_client, tmpdir_path):
        """client.get('doc_huts', as_arrow=True) returns pyarrow.Table from library."""
        import os
        import pyarrow as pa

        lib = tmpdir_path / "lib-get-arrow"
        live_client.sync("doc_huts", library_dir=lib)

        old_env = os.environ.get("EOLAS_LIBRARY")
        try:
            os.environ["EOLAS_LIBRARY"] = str(lib)
            from eolas_data import client as client_module
            client_module._auto_route_notified.discard("doc_huts")

            tbl = live_client.get("doc_huts", as_arrow=True)
            assert isinstance(tbl, pa.Table)
            assert tbl.num_rows > 0
        finally:
            if old_env is None:
                os.environ.pop("EOLAS_LIBRARY", None)
            else:
                os.environ["EOLAS_LIBRARY"] = old_env
