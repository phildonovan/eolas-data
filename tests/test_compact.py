"""Tests for client.compact() — multi-file → single snapshot compaction.

Covers:
  1. No-op: single snapshot file already, no deltas → CompactResult files_before=1 files_after=1
  2. Multiple files (snapshot + deltas) → merged, files_before=N files_after=1, old files gone
  3. Crash mid-write (mock raises during merged write) → original state intact
  4. Append-only: rows_after == rows_before
  5. Empty dataset dir (no parquet files at all) → graceful no-op
  6. No manifest → FileNotFoundError
  7. bytes_saved is computed (may be negative for tiny test files — just check it's int)
"""
from __future__ import annotations

import json
import pathlib
import shutil
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eolas_data import Client
from eolas_data.sync import (
    MANIFEST_FILENAME,
    Manifest,
    ManifestEntry,
    read_manifest,
    write_manifest,
)
from eolas_data.sync.compact import CompactResult, compact_dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SNAPSHOT_V1 = 5503437996448954328
SNAPSHOT_V2 = 7041234567890123456
SNAPSHOT_V3 = 8112345678901234567


def _make_parquet(path: pathlib.Path, num_rows: int = 10) -> None:
    """Write a minimal valid parquet file with ``num_rows`` rows."""
    table = pa.table({"id": list(range(num_rows)), "value": [float(i) for i in range(num_rows)]})
    pq.write_table(table, path)


def _write_fresh_manifest(
    dataset_dir: pathlib.Path,
    snapshot_id: int,
    filename: str,
    rows: int,
    fmt: str = "parquet",
) -> None:
    entry = ManifestEntry(
        snapshot_id=snapshot_id,
        kind="snapshot",
        file=filename,
        synced_at="2026-05-24T11:00:00Z",
        rows=rows,
    )
    m = Manifest(
        dataset="test_dataset",
        snapshots=[entry],
        current_snapshot=snapshot_id,
        format=fmt,
        schema_version=1,
    )
    write_manifest(m, dataset_dir / MANIFEST_FILENAME)


def _write_delta_manifest_entry(
    dataset_dir: pathlib.Path,
    parent_id: int,
    new_id: int,
    delta_filename: str,
    rows_added: int,
) -> None:
    """Append a delta entry to an existing manifest."""
    m = read_manifest(dataset_dir / MANIFEST_FILENAME)
    assert m is not None
    entry = ManifestEntry(
        snapshot_id=new_id,
        kind="delta",
        parent_snapshot=parent_id,
        file=delta_filename,
        synced_at="2026-05-27T12:00:00Z",
        rows_added=rows_added,
    )
    m.snapshots.append(entry)
    m.current_snapshot = new_id
    write_manifest(m, dataset_dir / MANIFEST_FILENAME)


# ---------------------------------------------------------------------------
# Test: no-op when only 1 file
# ---------------------------------------------------------------------------

class TestCompactNoOp:
    def test_single_file_returns_files_before_1_after_1(self, tmp_path):
        """One snapshot file, no deltas → compact is a no-op (files stay 1→1)."""
        ddir = tmp_path / "test_dataset"
        ddir.mkdir()
        snap_file = ddir / "snapshot-2026-05-24.parquet"
        _make_parquet(snap_file, num_rows=5)
        _write_fresh_manifest(ddir, SNAPSHOT_V1, snap_file.name, rows=5)

        result = compact_dataset(ddir)

        assert isinstance(result, CompactResult)
        assert result.files_before == 1
        assert result.files_after == 1
        assert result.rows_before == 5
        assert result.rows_after == 5
        assert result.bytes_saved == 0
        assert result.dataset == "test_dataset"

    def test_single_file_original_file_still_present(self, tmp_path):
        """No-op compact must not delete the only file."""
        ddir = tmp_path / "test_dataset"
        ddir.mkdir()
        snap_file = ddir / "snapshot-2026-05-24.parquet"
        _make_parquet(snap_file, num_rows=5)
        _write_fresh_manifest(ddir, SNAPSHOT_V1, snap_file.name, rows=5)

        compact_dataset(ddir)

        assert snap_file.exists(), "Original snapshot must still be present after no-op compact"

    def test_empty_dir_no_parquet_files(self, tmp_path):
        """Dataset dir with manifest but no parquet files → CompactResult files=0→0."""
        ddir = tmp_path / "test_dataset"
        ddir.mkdir()
        # Write manifest pointing at a file that doesn't exist (simulates edge case)
        _write_fresh_manifest(ddir, SNAPSHOT_V1, "snapshot-2026-05-24.parquet", rows=0)
        # Don't actually create the parquet file
        (ddir / "snapshot-2026-05-24.parquet").unlink(missing_ok=True)

        result = compact_dataset(ddir)

        assert result.files_before == 0
        assert result.files_after == 0
        assert result.rows_before == 0


# ---------------------------------------------------------------------------
# Test: multi-file compaction
# ---------------------------------------------------------------------------

class TestCompactMultiFile:
    def _setup_with_deltas(self, tmp_path, num_deltas: int = 3, rows_per_file: int = 10):
        """Create a dataset dir with 1 snapshot + num_deltas delta files."""
        ddir = tmp_path / "test_dataset"
        ddir.mkdir()

        # Initial snapshot
        snap_file = ddir / "snapshot-2026-05-24.parquet"
        _make_parquet(snap_file, num_rows=rows_per_file)
        _write_fresh_manifest(ddir, SNAPSHOT_V1, snap_file.name, rows=rows_per_file)

        prev_id = SNAPSHOT_V1
        for i in range(num_deltas):
            new_id = SNAPSHOT_V1 + (i + 1) * 1000
            delta_file = ddir / f"delta-2026-05-{24 + i}-to-2026-05-{25 + i}.parquet"
            _make_parquet(delta_file, num_rows=rows_per_file)
            _write_delta_manifest_entry(ddir, prev_id, new_id, delta_file.name, rows_added=rows_per_file)
            prev_id = new_id

        return ddir

    def test_compaction_files_before_N_after_1(self, tmp_path):
        """After compacting N files → exactly 1 file remains."""
        ddir = self._setup_with_deltas(tmp_path, num_deltas=3)
        result = compact_dataset(ddir)

        assert result.files_before == 4  # 1 snapshot + 3 deltas
        assert result.files_after == 1

    def test_compaction_single_file_in_dir(self, tmp_path):
        """After compaction only the merged snapshot parquet file exists (no old files)."""
        ddir = self._setup_with_deltas(tmp_path, num_deltas=3)
        compact_dataset(ddir)

        parquet_files = [
            f for f in ddir.iterdir()
            if f.is_file() and (f.name.endswith(".parquet") or f.name.endswith(".geo.parquet"))
        ]
        assert len(parquet_files) == 1, f"Expected 1 parquet file, found: {[f.name for f in parquet_files]}"

    def test_compaction_manifest_updated_to_single_snapshot(self, tmp_path):
        """Manifest must have exactly 1 entry pointing at the new snapshot file."""
        ddir = self._setup_with_deltas(tmp_path, num_deltas=3)
        compact_dataset(ddir)

        m = read_manifest(ddir / MANIFEST_FILENAME)
        assert m is not None
        assert len(m.snapshots) == 1
        assert m.snapshots[0].kind == "snapshot"
        assert m.snapshots[0].file.startswith("snapshot-")

    def test_compaction_manifest_current_snapshot_preserved(self, tmp_path):
        """current_snapshot in the compacted manifest must equal the pre-compact value."""
        ddir = self._setup_with_deltas(tmp_path, num_deltas=2)

        # Read the pre-compact current_snapshot
        pre_m = read_manifest(ddir / MANIFEST_FILENAME)
        assert pre_m is not None
        pre_current = pre_m.current_snapshot

        compact_dataset(ddir)

        post_m = read_manifest(ddir / MANIFEST_FILENAME)
        assert post_m is not None
        assert post_m.current_snapshot == pre_current

    def test_compaction_rows_after_equals_rows_before_append_only(self, tmp_path):
        """For pure append-only data rows_after must equal rows_before."""
        rows_per_file = 10
        ddir = self._setup_with_deltas(tmp_path, num_deltas=3, rows_per_file=rows_per_file)

        result = compact_dataset(ddir)

        expected_rows = rows_per_file * 4  # 1 snapshot + 3 deltas, each 10 rows
        assert result.rows_before == expected_rows
        assert result.rows_after == expected_rows

    def test_compaction_old_files_deleted(self, tmp_path):
        """All old snapshot + delta files must be removed after compact."""
        ddir = self._setup_with_deltas(tmp_path, num_deltas=2)
        old_files = {f.name for f in ddir.iterdir()
                     if f.is_file() and (f.name.startswith("snapshot-") or f.name.startswith("delta-"))}

        compact_dataset(ddir)

        remaining = {f.name for f in ddir.iterdir()
                     if f.is_file() and (f.name.endswith(".parquet") or f.name.endswith(".geo.parquet"))}
        # None of the original files should remain
        leftover_old = old_files & remaining
        assert not leftover_old, f"Old files still present after compact: {leftover_old}"

    def test_compaction_bytes_saved_is_int(self, tmp_path):
        """bytes_saved must be an integer (may be negative for tiny test data)."""
        ddir = self._setup_with_deltas(tmp_path, num_deltas=2)
        result = compact_dataset(ddir)
        assert isinstance(result.bytes_saved, int)

    def test_compaction_no_compacting_dirs_left(self, tmp_path):
        """No .compacting-* dirs should remain after a successful compact."""
        ddir = self._setup_with_deltas(tmp_path, num_deltas=2)
        compact_dataset(ddir)

        leftover = [item.name for item in ddir.iterdir()
                    if item.is_dir() and item.name.startswith(".compacting")]
        assert not leftover, f"Stale compacting dirs remain: {leftover}"


# ---------------------------------------------------------------------------
# Test: atomicity — crash during merge write leaves original state intact
# ---------------------------------------------------------------------------

class TestCompactAtomicity:
    def _setup_two_files(self, tmp_path):
        ddir = tmp_path / "test_dataset"
        ddir.mkdir()
        snap_file = ddir / "snapshot-2026-05-24.parquet"
        _make_parquet(snap_file, num_rows=10)
        _write_fresh_manifest(ddir, SNAPSHOT_V1, snap_file.name, rows=10)

        delta_file = ddir / "delta-2026-05-24-to-2026-05-27.parquet"
        _make_parquet(delta_file, num_rows=5)
        _write_delta_manifest_entry(ddir, SNAPSHOT_V1, SNAPSHOT_V2, delta_file.name, rows_added=5)
        return ddir

    def test_crash_during_merged_write_leaves_manifest_unchanged(self, tmp_path):
        """If PyArrow write_table raises, the manifest must remain unchanged."""
        ddir = self._setup_two_files(tmp_path)
        old_manifest_text = (ddir / MANIFEST_FILENAME).read_text()

        with patch("pyarrow.parquet.write_table", side_effect=RuntimeError("disk full")):
            with pytest.raises(RuntimeError, match="disk full"):
                compact_dataset(ddir)

        new_manifest_text = (ddir / MANIFEST_FILENAME).read_text()
        assert new_manifest_text == old_manifest_text, \
            "Manifest must not be mutated when the merged write fails"

    def test_crash_during_merged_write_leaves_original_files(self, tmp_path):
        """If the write fails, both original parquet files must still be present."""
        ddir = self._setup_two_files(tmp_path)
        original_files = {f.name for f in ddir.iterdir()
                          if f.is_file() and f.name.endswith(".parquet")}

        with patch("pyarrow.parquet.write_table", side_effect=RuntimeError("disk full")):
            with pytest.raises(RuntimeError, match="disk full"):
                compact_dataset(ddir)

        remaining_files = {f.name for f in ddir.iterdir()
                           if f.is_file() and f.name.endswith(".parquet")}
        assert original_files == remaining_files, \
            "Original parquet files must be intact after failed compact"

    def test_crash_during_merged_write_no_stale_staging_dir(self, tmp_path):
        """A .compacting-* dir created before the crash must be cleaned up
        on the NEXT compact run (not left forever)."""
        ddir = self._setup_two_files(tmp_path)

        # First compact: crash during write_table
        with patch("pyarrow.parquet.write_table", side_effect=RuntimeError("disk full")):
            with pytest.raises(RuntimeError):
                compact_dataset(ddir)

        # A .compacting-* dir may exist from the crash
        # Second compact: should succeed and clean up the stale dir
        result = compact_dataset(ddir)
        assert result.files_after == 1

        leftover = [item.name for item in ddir.iterdir()
                    if item.is_dir() and item.name.startswith(".compacting")]
        assert not leftover, f"Stale .compacting dirs were not cleaned up: {leftover}"

    def test_crash_after_checkpoint_before_manifest_write(self, tmp_path):
        """Simulate crash after os.replace(staging → done) but before manifest write.

        The next compact should clean up the .compacting-done-* dir and succeed.
        """
        ddir = self._setup_two_files(tmp_path)
        manifest_path = ddir / MANIFEST_FILENAME
        old_manifest_text = manifest_path.read_text()

        original_replace = __import__("os").replace
        replace_calls: list = []

        def _crash_on_manifest_replace(src, dst):
            replace_calls.append((src, dst))
            # Let staging→done rename go through; crash when replacing the manifest
            dst_str = str(dst)
            if MANIFEST_FILENAME in dst_str and not dst_str.endswith(".tmp"):
                raise OSError("simulated crash before manifest write")
            return original_replace(src, dst)

        with patch("os.replace", side_effect=_crash_on_manifest_replace):
            with pytest.raises(OSError, match="simulated crash before manifest write"):
                compact_dataset(ddir)

        # Manifest must be unchanged
        assert manifest_path.read_text() == old_manifest_text

        # Recovery: next compact run should succeed
        result = compact_dataset(ddir)
        assert result.files_after == 1


# ---------------------------------------------------------------------------
# Test: error conditions
# ---------------------------------------------------------------------------

class TestCompactErrors:
    def test_missing_dataset_dir_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compact_dataset(tmp_path / "nonexistent")

    def test_missing_manifest_raises_file_not_found(self, tmp_path):
        ddir = tmp_path / "test_dataset"
        ddir.mkdir()
        with pytest.raises(FileNotFoundError, match="manifest"):
            compact_dataset(ddir)

    def test_geoparquet_format_preserved_in_manifest(self, tmp_path):
        """Compacted manifest must preserve format='geoparquet' from the original."""
        ddir = tmp_path / "test_dataset"
        ddir.mkdir()
        snap_file = ddir / "snapshot-2026-05-24.geo.parquet"
        _make_parquet(snap_file, num_rows=3)
        entry = ManifestEntry(
            snapshot_id=SNAPSHOT_V1,
            kind="snapshot",
            file=snap_file.name,
            synced_at="2026-05-24T11:00:00Z",
            rows=3,
        )
        delta_file = ddir / "delta-2026-05-24-to-2026-05-27.geo.parquet"
        _make_parquet(delta_file, num_rows=2)
        delta_entry = ManifestEntry(
            snapshot_id=SNAPSHOT_V2,
            kind="delta",
            parent_snapshot=SNAPSHOT_V1,
            file=delta_file.name,
            synced_at="2026-05-27T09:00:00Z",
            rows_added=2,
        )
        m = Manifest(
            dataset="geo_dataset",
            snapshots=[entry, delta_entry],
            current_snapshot=SNAPSHOT_V2,
            format="geoparquet",
            schema_version=1,
        )
        write_manifest(m, ddir / MANIFEST_FILENAME)

        compact_dataset(ddir)

        post_m = read_manifest(ddir / MANIFEST_FILENAME)
        assert post_m is not None
        assert post_m.format == "geoparquet"
        # Merged file should have .geo.parquet extension
        parquet_files = list(ddir.glob("snapshot-*.geo.parquet"))
        assert parquet_files, "Expected a .geo.parquet file for geoparquet dataset"


# ---------------------------------------------------------------------------
# Test: client.compact() wrapper
# ---------------------------------------------------------------------------

class TestClientCompact:
    def test_client_compact_delegates_to_compact_dataset(self, tmp_path):
        """client.compact() must return a CompactResult."""
        client = Client("eolas_testkey123")

        ddir = tmp_path / "test_dataset"
        ddir.mkdir()
        snap_file = ddir / "snapshot-2026-05-24.parquet"
        _make_parquet(snap_file, num_rows=5)
        _write_fresh_manifest(ddir, SNAPSHOT_V1, snap_file.name, rows=5)

        # One delta to make compaction do something
        delta_file = ddir / "delta-2026-05-24-to-2026-05-27.parquet"
        _make_parquet(delta_file, num_rows=3)
        _write_delta_manifest_entry(ddir, SNAPSHOT_V1, SNAPSHOT_V2, delta_file.name, rows_added=3)

        result = client.compact(ddir)

        assert isinstance(result, CompactResult)
        assert result.files_before == 2
        assert result.files_after == 1
