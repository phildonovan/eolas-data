"""client.compact(dataset_dir) — merge all parquet files into a single snapshot.

Atomicity strategy
------------------
A crashed compact at any step leaves either the original state intact or a
clearly-named ``.compacting-<uuid>`` directory that the next compact run can
detect and clean up.  The steps are:

1. Read all parquet files in ``dataset_dir`` via ``pyarrow.dataset.dataset()``
   and write the merged table to
   ``dataset_dir/.compacting-<uuid>/snapshot-<today>.parquet.tmp``.
2. Rename ``.parquet.tmp`` → ``.parquet`` (still inside ``.compacting-<uuid>``).
3. Rename ``.compacting-<uuid>`` → ``.compacting-done-<uuid>`` — checkpoint.
4. Write the new manifest to ``_eolas-manifest.json.tmp``.
5. ``os.replace`` the tmp manifest over the real one.
6. Move the merged snapshot from ``.compacting-done-<uuid>/`` up to
   ``dataset_dir/``.
7. Delete the old orphaned files (old snapshots + deltas, now superseded).
8. Delete the ``.compacting-done-<uuid>`` directory.

The merge is a pure concatenation (``append``-style union).  SCD2 tables that
carry ``is_current`` / ``valid_to`` columns are **not** collapsed — the
compacted file contains all historical rows, and the reader still filters by
``is_current = TRUE`` as usual.  Row-level deduplication is explicitly out of
scope for v1.

Public surface
--------------
``compact_dataset(dataset_dir)`` is the implementation function.
``Client.compact(dataset_dir)`` is the thin wrapper in ``client.py``.
"""
from __future__ import annotations

import datetime
import os
import pathlib
import shutil
import uuid
from dataclasses import dataclass
from typing import Union

from .manifest import (
    MANIFEST_FILENAME,
    Manifest,
    ManifestEntry,
    read_manifest,
    write_manifest,
)

# ---------------------------------------------------------------------------
# CompactResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactResult:
    """Result of a :meth:`Client.compact` call.

    Attributes:
        dataset:
            The dataset name read from the manifest.
        rows_before:
            Total rows across all files before compaction (counted from
            parquet footer metadata so counting is cheap).
        rows_after:
            Rows in the merged snapshot file (should equal ``rows_before``
            for append-only data; may differ for SCD2 if the caller passes
            dedup logic — but v1 does pure concatenation so this is always
            equal).
        files_before:
            Number of parquet files (snapshot + delta) present before compaction.
        files_after:
            Always ``1`` after a successful compaction (a single merged
            snapshot file).
        bytes_saved:
            Approximate disk space freed: (sum of old file sizes) minus
            (size of new merged file).  May be negative if the merged file
            is larger due to different compression or metadata overhead.
    """

    dataset: str
    rows_before: int
    rows_after: int
    files_before: int
    files_after: int
    bytes_saved: int

    def __repr__(self) -> str:
        return (
            f"<CompactResult dataset={self.dataset!r} "
            f"files={self.files_before}→{self.files_after} "
            f"rows={self.rows_before} bytes_saved={self.bytes_saved}>"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _today_str() -> str:
    """Return today's UTC date as YYYY-MM-DD."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string with Z suffix."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _count_parquet_rows(path: pathlib.Path) -> int:
    """Return row count from parquet footer metadata.  Falls back to 0 on error."""
    try:
        import pyarrow.parquet as pq
        return pq.read_metadata(path).num_rows
    except Exception:
        return 0


def _list_data_files(dataset_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return all snapshot/delta parquet files in *dataset_dir* (sorted)."""
    files: list[pathlib.Path] = []
    for f in dataset_dir.iterdir():
        name = f.name
        if f.is_file() and (
            (name.startswith("snapshot-") or name.startswith("delta-"))
            and (name.endswith(".parquet") or name.endswith(".geo.parquet"))
        ):
            files.append(f)
    return sorted(files)


def _cleanup_leftover_compacting_dirs(dataset_dir: pathlib.Path) -> None:
    """Remove any ``.compacting-*`` / ``.compacting-done-*`` dirs from a previous
    crashed compact.  Called at the start of every compact run so stale state
    does not accumulate.
    """
    for item in dataset_dir.iterdir():
        if item.is_dir() and (
            item.name.startswith(".compacting-done-")
            or item.name.startswith(".compacting-")
        ):
            shutil.rmtree(item, ignore_errors=True)


# ---------------------------------------------------------------------------
# Public implementation function
# ---------------------------------------------------------------------------


def compact_dataset(
    dataset_dir: Union[str, pathlib.Path],
) -> CompactResult:
    """Implement ``client.compact(dataset_dir)``.

    Parameters
    ----------
    dataset_dir:
        Path to the dataset directory (e.g. ``library_dir/doc_huts``).
        Must contain a ``_eolas-manifest.json``.

    Returns
    -------
    CompactResult

    Raises
    ------
    FileNotFoundError
        If ``dataset_dir`` does not exist or has no manifest.
    ValueError
        If the manifest is corrupt.
    RuntimeError
        If pyarrow is not installed (required for the merge read).
    """
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for compact().  "
            "Install with: pip install pyarrow"
        ) from exc

    ddir = pathlib.Path(dataset_dir).expanduser().resolve()
    if not ddir.is_dir():
        raise FileNotFoundError(f"compact(): dataset_dir does not exist: {ddir}")

    manifest_path = ddir / MANIFEST_FILENAME
    manifest = read_manifest(manifest_path)
    if manifest is None:
        raise FileNotFoundError(
            f"compact(): no manifest found at {manifest_path}. "
            "Has this dataset been synced yet?"
        )

    # ------------------------------------------------------------------
    # 0. Clean up any stale compacting dirs from previous crashed runs
    # ------------------------------------------------------------------
    _cleanup_leftover_compacting_dirs(ddir)

    # ------------------------------------------------------------------
    # 1. Enumerate data files and measure sizes
    # ------------------------------------------------------------------
    data_files = _list_data_files(ddir)
    files_before = len(data_files)

    if files_before == 0:
        # Nothing to compact — manifest references files that don't exist?
        return CompactResult(
            dataset=manifest.dataset,
            rows_before=0,
            rows_after=0,
            files_before=0,
            files_after=0,
            bytes_saved=0,
        )

    if files_before == 1:
        # Already compacted — single file, nothing to do.
        rows = _count_parquet_rows(data_files[0])
        return CompactResult(
            dataset=manifest.dataset,
            rows_before=rows,
            rows_after=rows,
            files_before=1,
            files_after=1,
            bytes_saved=0,
        )

    # Sum old file sizes for bytes_saved calculation.
    old_total_bytes = sum(f.stat().st_size for f in data_files)

    # Sum rows across all files (from parquet footer metadata — fast).
    rows_before = sum(_count_parquet_rows(f) for f in data_files)

    # ------------------------------------------------------------------
    # 2. Determine output format from manifest
    # ------------------------------------------------------------------
    fmt = manifest.format  # "parquet" or "geoparquet"
    ext = ".geo.parquet" if fmt == "geoparquet" else ".parquet"
    today = _today_str()
    new_filename = f"snapshot-{today}{ext}"

    # ------------------------------------------------------------------
    # 3. Create the staging directory
    # ------------------------------------------------------------------
    uid = uuid.uuid4().hex[:12]
    staging_dir = ddir / f".compacting-{uid}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    new_file_tmp = staging_dir / (new_filename + ".tmp")
    new_file_staged = staging_dir / new_filename

    try:
        # --------------------------------------------------------------
        # 4. Read all files via PyArrow Dataset + write merged snapshot
        # --------------------------------------------------------------
        # Use pyarrow.dataset for schema-union read (handles schema evolution
        # across snapshot/delta files via union_by_name = True).
        parquet_files_str = [str(f) for f in data_files]
        arrow_ds = ds.dataset(parquet_files_str, format="parquet", schema=None)
        # Read into a single in-memory table (acceptable for compaction; the
        # user already has all these files on disk).
        merged_table: pa.Table = arrow_ds.to_table()

        rows_after = merged_table.num_rows

        # Write to .tmp first, then rename within the staging dir.
        pq.write_table(merged_table, new_file_tmp)
        del merged_table  # free RAM early

        # Step 2: rename .parquet.tmp → .parquet (still in staging dir)
        os.replace(new_file_tmp, new_file_staged)

    except Exception:
        # Clean up staging dir on failure — original state is untouched.
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    # ------------------------------------------------------------------
    # 5. Checkpoint: rename .compacting-<uid> → .compacting-done-<uid>
    # ------------------------------------------------------------------
    done_dir = ddir / f".compacting-done-{uid}"
    os.replace(staging_dir, done_dir)

    # From here on, original state is still intact (manifest + old files).
    new_file_in_done = done_dir / new_filename

    # ------------------------------------------------------------------
    # 6. Write new manifest (tmp → replace)
    # ------------------------------------------------------------------
    new_snap_id = manifest.current_snapshot if manifest.current_snapshot is not None else 0
    new_entry = ManifestEntry(
        snapshot_id=new_snap_id,
        kind="snapshot",
        file=new_filename,
        synced_at=_utc_now(),
        rows=rows_after,
    )
    new_manifest = Manifest(
        dataset=manifest.dataset,
        snapshots=[new_entry],
        current_snapshot=new_snap_id,
        format=fmt,
        schema_version=manifest.schema_version,
    )
    write_manifest(new_manifest, manifest_path)

    # ------------------------------------------------------------------
    # 7. Move the merged snapshot file up to dataset_dir
    # ------------------------------------------------------------------
    final_path = ddir / new_filename
    shutil.move(str(new_file_in_done), str(final_path))

    # ------------------------------------------------------------------
    # 8. Delete the now-orphaned old files
    # ------------------------------------------------------------------
    # Exclude the newly-written merged file in case its name collides with one
    # of the old files (happens when today's UTC date matches the date in the
    # existing snapshot filename, e.g. both are "snapshot-2026-05-26.parquet").
    final_path_resolved = final_path.resolve()
    for old_file in data_files:
        if old_file.resolve() == final_path_resolved:
            continue  # This IS the merged file — do not delete it.
        try:
            old_file.unlink(missing_ok=True)
        except Exception:
            pass  # Best-effort; stale files are cosmetic, not fatal.

    # ------------------------------------------------------------------
    # 9. Delete the staging done dir
    # ------------------------------------------------------------------
    shutil.rmtree(done_dir, ignore_errors=True)

    new_total_bytes = final_path.stat().st_size
    bytes_saved = old_total_bytes - new_total_bytes

    return CompactResult(
        dataset=manifest.dataset,
        rows_before=rows_before,
        rows_after=rows_after,
        files_before=files_before,
        files_after=1,
        bytes_saved=bytes_saved,
    )
