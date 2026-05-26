"""Tests for client.sync() — multi-file dataset directory model.

Covers all branches of the decision-logic flowchart:
  1. First sync (empty library_dir) → status="snapshot_full"
  2. Second sync, snapshot unchanged → status="unchanged" (no HTTP body)
  3. Second sync, new snapshot, incremental_supported=True → status="snapshot_delta"
  4. 410 from incremental endpoint → fallback to snapshot_full
  5. 400 from incremental endpoint → fallback to snapshot_full
  6. incremental_supported=False → skip incremental, snapshot_full
  7. Atomic write: simulate interrupted download → manifest NOT mutated
  8. Manifest reader/writer correctness (round-trip + validation)
"""
from __future__ import annotations

import json
import pathlib
from unittest.mock import patch

import pytest
import responses as resp_lib

from eolas_data import Client
from eolas_data.sync import (
    Manifest,
    ManifestEntry,
    MANIFEST_FILENAME,
    read_manifest,
    write_manifest,
)
from eolas_data.sync.sync import SyncResult

BASE = "https://api.eolas.fyi"

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

FAKE_PARQUET     = b"PAR1" + b"\x00" * 12 + b"PAR1"
FAKE_PARQUET_V2  = b"PAR1" + b"\x01" * 12 + b"PAR1"
FAKE_DELTA_BODY  = b"PAR1" + b"\x02" * 12 + b"PAR1"

SNAPSHOT_V1 = 5503437996448954328
SNAPSHOT_V2 = 7041234567890123456

# Non-geo metadata (incremental_supported=True by default)
META_DOC_HUTS = {
    "name": "doc_huts",
    "title": "DOC Huts",
    "source": "DOC",
    "namespace": "doc",
    "table": "doc_huts",
    "current_snapshot_id": SNAPSHOT_V1,
    "incremental_supported": True,
    "refresh_cadence": "weekly",
    "geometry_type": "Point",
    "has_geometry": True,
}

META_DOC_HUTS_V2 = {
    **META_DOC_HUTS,
    "current_snapshot_id": SNAPSHOT_V2,
}

META_NO_INCREMENTAL = {
    **META_DOC_HUTS,
    "incremental_supported": False,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


def _write_manifest_for(
    dataset_dir: pathlib.Path,
    snapshot_id: int,
    snapshot_filename: str = "snapshot-2026-05-24.geo.parquet",
) -> Manifest:
    """Write a minimal manifest into *dataset_dir* and return it."""
    entry = ManifestEntry(
        snapshot_id=snapshot_id,
        kind="snapshot",
        file=snapshot_filename,
        synced_at="2026-05-24T11:05:00Z",
        rows=1429,
    )
    m = Manifest(
        dataset="doc_huts",
        snapshots=[entry],
        current_snapshot=snapshot_id,
        format="geoparquet",
        schema_version=1,
    )
    manifest_path = dataset_dir / MANIFEST_FILENAME
    write_manifest(m, manifest_path)
    return m


# ---------------------------------------------------------------------------
# Manifest unit tests (reader / writer / validation)
# ---------------------------------------------------------------------------

class TestManifestRoundTrip:
    def test_write_and_read_snapshot_entry(self, tmp_path):
        entry = ManifestEntry(
            snapshot_id=SNAPSHOT_V1,
            kind="snapshot",
            file="snapshot-2026-05-24.parquet",
            synced_at="2026-05-24T11:05:00Z",
            rows=5000,
        )
        m = Manifest(
            dataset="linz.nz_parcels",
            snapshots=[entry],
            current_snapshot=SNAPSHOT_V1,
            format="parquet",
        )
        p = tmp_path / MANIFEST_FILENAME
        write_manifest(m, p)

        loaded = read_manifest(p)
        assert loaded is not None
        assert loaded.dataset == "linz.nz_parcels"
        assert loaded.current_snapshot == SNAPSHOT_V1
        assert loaded.format == "parquet"
        assert len(loaded.snapshots) == 1
        e = loaded.snapshots[0]
        assert e.snapshot_id == SNAPSHOT_V1
        assert e.kind == "snapshot"
        assert e.rows == 5000

    def test_write_and_read_delta_entry(self, tmp_path):
        snap_entry = ManifestEntry(
            snapshot_id=SNAPSHOT_V1,
            kind="snapshot",
            file="snapshot-2026-05-24.parquet",
            synced_at="2026-05-24T11:05:00Z",
            rows=1000,
        )
        delta_entry = ManifestEntry(
            snapshot_id=SNAPSHOT_V2,
            kind="delta",
            parent_snapshot=SNAPSHOT_V1,
            file="delta-2026-05-24-to-2026-05-31.parquet",
            synced_at="2026-05-31T11:05:00Z",
            rows_added=50,
        )
        m = Manifest(
            dataset="doc_huts",
            snapshots=[snap_entry, delta_entry],
            current_snapshot=SNAPSHOT_V2,
            format="parquet",
        )
        p = tmp_path / MANIFEST_FILENAME
        write_manifest(m, p)
        loaded = read_manifest(p)
        assert loaded.current_snapshot == SNAPSHOT_V2
        assert len(loaded.snapshots) == 2
        d = loaded.snapshots[1]
        assert d.kind == "delta"
        assert d.parent_snapshot == SNAPSHOT_V1
        assert d.rows_added == 50

    def test_read_nonexistent_returns_none(self, tmp_path):
        result = read_manifest(tmp_path / "does_not_exist.json")
        assert result is None

    def test_write_atomic_no_partial_on_failure(self, tmp_path):
        """os.replace failure must leave NO new file at the manifest path."""
        entry = ManifestEntry(
            snapshot_id=SNAPSHOT_V1,
            kind="snapshot",
            file="snapshot-2026-05-24.parquet",
            synced_at="2026-05-24T11:05:00Z",
            rows=100,
        )
        m = Manifest(
            dataset="test",
            snapshots=[entry],
            current_snapshot=SNAPSHOT_V1,
            format="parquet",
        )
        p = tmp_path / MANIFEST_FILENAME

        with patch("os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                write_manifest(m, p)

        # The canonical path must not exist.
        assert not p.exists()
        # No tmp file left over.
        assert not any(tmp_path.glob("*.tmp-*"))

    def test_validation_bad_snapshot_id(self):
        with pytest.raises(ValueError, match="snapshot_id must be int"):
            ManifestEntry(
                snapshot_id="not_an_int",  # type: ignore[arg-type]
                kind="snapshot",
                file="snapshot-2026-05-24.parquet",
                synced_at="2026-05-24T11:05:00Z",
                rows=1,
            ).validate()

    def test_validation_bad_file_name(self):
        with pytest.raises(ValueError, match="naming pattern"):
            ManifestEntry(
                snapshot_id=SNAPSHOT_V1,
                kind="snapshot",
                file="my-file.parquet",  # doesn't match expected pattern
                synced_at="2026-05-24T11:05:00Z",
                rows=1,
            ).validate()

    def test_validation_bad_synced_at(self):
        with pytest.raises(ValueError, match="ISO-8601"):
            ManifestEntry(
                snapshot_id=SNAPSHOT_V1,
                kind="snapshot",
                file="snapshot-2026-05-24.parquet",
                synced_at="2026-05-24 11:05:00",  # missing T and Z
                rows=1,
            ).validate()

    def test_geoparquet_file_accepted(self, tmp_path):
        """geo.parquet extension must pass validation."""
        entry = ManifestEntry(
            snapshot_id=SNAPSHOT_V1,
            kind="snapshot",
            file="snapshot-2026-05-24.geo.parquet",
            synced_at="2026-05-24T11:05:00Z",
            rows=500,
        )
        entry.validate()  # must not raise

    def test_delta_geoparquet_file_accepted(self, tmp_path):
        entry = ManifestEntry(
            snapshot_id=SNAPSHOT_V2,
            kind="delta",
            parent_snapshot=SNAPSHOT_V1,
            file="delta-2026-05-24-to-2026-05-31.geo.parquet",
            synced_at="2026-05-31T11:05:00Z",
            rows_added=100,
        )
        entry.validate()  # must not raise


# ---------------------------------------------------------------------------
# sync() integration-unit tests (HTTP mocked with responses library)
# ---------------------------------------------------------------------------

class TestSyncFirstDownload:
    @resp_lib.activate
    def test_first_sync_status_snapshot_full(self, client, tmp_path):
        """Empty library_dir → full download, status='snapshot_full'."""
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS, status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/doc/doc_huts",
                     body=FAKE_PARQUET,
                     content_type="application/octet-stream",
                     status=200)

        result = client.sync("doc_huts", library_dir=tmp_path)

        assert isinstance(result, SyncResult)
        assert result.status == "snapshot_full"
        assert result.dataset == "doc_huts"
        assert result.bytes_downloaded == len(FAKE_PARQUET)
        assert result.files_added == 1

    @resp_lib.activate
    def test_first_sync_manifest_written(self, client, tmp_path):
        """Manifest must be created with one snapshot entry."""
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS, status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/doc/doc_huts",
                     body=FAKE_PARQUET,
                     content_type="application/octet-stream",
                     status=200)

        client.sync("doc_huts", library_dir=tmp_path)

        manifest_path = tmp_path / "doc_huts" / MANIFEST_FILENAME
        assert manifest_path.exists()
        m = read_manifest(manifest_path)
        assert m is not None
        assert m.dataset == "doc_huts"
        assert len(m.snapshots) == 1
        assert m.snapshots[0].kind == "snapshot"
        assert m.current_snapshot == SNAPSHOT_V1

    @resp_lib.activate
    def test_first_sync_snapshot_file_exists(self, client, tmp_path):
        """The snapshot parquet file must be written."""
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS, status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/doc/doc_huts",
                     body=FAKE_PARQUET,
                     content_type="application/octet-stream",
                     status=200)

        client.sync("doc_huts", library_dir=tmp_path)

        dataset_dir = tmp_path / "doc_huts"
        parquet_files = list(dataset_dir.glob("snapshot-*.*.parquet")) + \
                        list(dataset_dir.glob("snapshot-*.parquet"))
        assert parquet_files, "At least one snapshot parquet file must be written"

    @resp_lib.activate
    def test_first_sync_geo_uses_geoparquet_format(self, client, tmp_path):
        """Geo dataset should use geoparquet format in the bulk URL."""
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS, status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/doc/doc_huts",
                     body=FAKE_PARQUET,
                     content_type="application/octet-stream",
                     status=200,
                     match_querystring=False)

        client.sync("doc_huts", library_dir=tmp_path)

        # The bulk GET request should have format=geoparquet
        bulk_calls = [
            c for c in resp_lib.calls
            if "/v1/bulk/" in c.request.url
        ]
        assert bulk_calls
        assert "geoparquet" in bulk_calls[0].request.url


class TestSyncUnchanged:
    @resp_lib.activate
    def test_unchanged_returns_status_unchanged(self, client, tmp_path):
        """When server snapshot == local manifest → no download, status='unchanged'."""
        dataset_dir = tmp_path / "doc_huts"
        dataset_dir.mkdir()
        _write_manifest_for(dataset_dir, SNAPSHOT_V1)

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS, status=200)
        # No bulk GET registered — if code tries to GET, responses raises

        result = client.sync("doc_huts", library_dir=tmp_path)

        assert result.status == "unchanged"
        assert result.bytes_downloaded == 0
        assert result.rows_added == 0
        assert result.files_added == 0

    @resp_lib.activate
    def test_unchanged_no_new_files(self, client, tmp_path):
        """No new files should appear in the dataset directory when unchanged."""
        dataset_dir = tmp_path / "doc_huts"
        dataset_dir.mkdir()
        _write_manifest_for(dataset_dir, SNAPSHOT_V1)

        before = set(dataset_dir.iterdir())

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS, status=200)

        client.sync("doc_huts", library_dir=tmp_path)

        after = set(dataset_dir.iterdir())
        assert before == after, "No new files should be created when unchanged"


class TestSyncDelta:
    @resp_lib.activate
    def test_delta_returns_snapshot_delta_status(self, client, tmp_path):
        """New snapshot + incremental_supported → fetch delta, status='snapshot_delta'."""
        dataset_dir = tmp_path / "doc_huts"
        dataset_dir.mkdir()
        _write_manifest_for(dataset_dir, SNAPSHOT_V1)

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS_V2, status=200)
        resp_lib.add(resp_lib.GET,
                     f"{BASE}/v1/datasets/doc_huts/data/incremental",
                     body=FAKE_DELTA_BODY,
                     content_type="application/octet-stream",
                     status=200,
                     headers={
                         "X-Eolas-Row-Count": "42",
                         "X-Eolas-Current-Snapshot": str(SNAPSHOT_V2),
                         "X-Eolas-Since-Snapshot": str(SNAPSHOT_V1),
                     })

        result = client.sync("doc_huts", library_dir=tmp_path)

        assert result.status == "snapshot_delta"
        assert result.bytes_downloaded == len(FAKE_DELTA_BODY)
        assert result.rows_added == 42
        assert result.files_added == 1

    @resp_lib.activate
    def test_delta_manifest_updated(self, client, tmp_path):
        """Manifest must have the new delta entry appended."""
        dataset_dir = tmp_path / "doc_huts"
        dataset_dir.mkdir()
        _write_manifest_for(dataset_dir, SNAPSHOT_V1)

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS_V2, status=200)
        resp_lib.add(resp_lib.GET,
                     f"{BASE}/v1/datasets/doc_huts/data/incremental",
                     body=FAKE_DELTA_BODY,
                     content_type="application/octet-stream",
                     status=200,
                     headers={"X-Eolas-Row-Count": "42"})

        client.sync("doc_huts", library_dir=tmp_path)

        m = read_manifest(dataset_dir / MANIFEST_FILENAME)
        assert m is not None
        assert len(m.snapshots) == 2
        assert m.snapshots[1].kind == "delta"
        assert m.snapshots[1].rows_added == 42
        assert m.current_snapshot == SNAPSHOT_V2

    @resp_lib.activate
    def test_delta_file_written(self, client, tmp_path):
        """A delta-*.parquet file must appear in the dataset dir."""
        dataset_dir = tmp_path / "doc_huts"
        dataset_dir.mkdir()
        _write_manifest_for(dataset_dir, SNAPSHOT_V1)

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS_V2, status=200)
        resp_lib.add(resp_lib.GET,
                     f"{BASE}/v1/datasets/doc_huts/data/incremental",
                     body=FAKE_DELTA_BODY,
                     content_type="application/octet-stream",
                     status=200,
                     headers={"X-Eolas-Row-Count": "42"})

        client.sync("doc_huts", library_dir=tmp_path)

        delta_files = list(dataset_dir.glob("delta-*.parquet")) + \
                      list(dataset_dir.glob("delta-*.geo.parquet"))
        assert delta_files, "Delta parquet file must be written"


class TestSyncFallbackOn410:
    @resp_lib.activate
    def test_410_triggers_full_download(self, client, tmp_path):
        """410 from incremental → fall back to full snapshot_full download."""
        dataset_dir = tmp_path / "doc_huts"
        dataset_dir.mkdir()
        _write_manifest_for(dataset_dir, SNAPSHOT_V1)

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS_V2, status=200)
        resp_lib.add(resp_lib.GET,
                     f"{BASE}/v1/datasets/doc_huts/data/incremental",
                     json={"detail": "Snapshot not found; please resync."},
                     status=410)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/doc/doc_huts",
                     body=FAKE_PARQUET_V2,
                     content_type="application/octet-stream",
                     status=200)

        result = client.sync("doc_huts", library_dir=tmp_path)

        assert result.status == "snapshot_full"
        assert result.bytes_downloaded == len(FAKE_PARQUET_V2)

    @resp_lib.activate
    def test_410_manifest_reset_to_single_entry(self, client, tmp_path):
        """After 410 fallback the manifest must have exactly one snapshot entry."""
        dataset_dir = tmp_path / "doc_huts"
        dataset_dir.mkdir()
        _write_manifest_for(dataset_dir, SNAPSHOT_V1)

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS_V2, status=200)
        resp_lib.add(resp_lib.GET,
                     f"{BASE}/v1/datasets/doc_huts/data/incremental",
                     json={"detail": "Snapshot expired."},
                     status=410)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/doc/doc_huts",
                     body=FAKE_PARQUET_V2,
                     content_type="application/octet-stream",
                     status=200)

        client.sync("doc_huts", library_dir=tmp_path)

        m = read_manifest(dataset_dir / MANIFEST_FILENAME)
        assert m is not None
        assert len(m.snapshots) == 1
        assert m.snapshots[0].kind == "snapshot"
        assert m.current_snapshot == SNAPSHOT_V2


class TestSyncFallbackOn400:
    @resp_lib.activate
    def test_400_triggers_full_download(self, client, tmp_path):
        """400 from incremental (incremental_supported=false at request time) → snapshot_full."""
        dataset_dir = tmp_path / "doc_huts"
        dataset_dir.mkdir()
        _write_manifest_for(dataset_dir, SNAPSHOT_V1)

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS_V2, status=200)
        resp_lib.add(resp_lib.GET,
                     f"{BASE}/v1/datasets/doc_huts/data/incremental",
                     json={"detail": "incremental_supported=false"},
                     status=400)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/doc/doc_huts",
                     body=FAKE_PARQUET_V2,
                     content_type="application/octet-stream",
                     status=200)

        result = client.sync("doc_huts", library_dir=tmp_path)

        assert result.status == "snapshot_full"


class TestSyncIncrementalUnsupported:
    @resp_lib.activate
    def test_incremental_supported_false_skips_incremental(self, client, tmp_path):
        """incremental_supported=False in metadata → no incremental request, direct bulk."""
        dataset_dir = tmp_path / "doc_huts"
        dataset_dir.mkdir()
        _write_manifest_for(dataset_dir, SNAPSHOT_V1)

        meta_v2_no_incr = {**META_NO_INCREMENTAL, "current_snapshot_id": SNAPSHOT_V2}
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=meta_v2_no_incr, status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/doc/doc_huts",
                     body=FAKE_PARQUET_V2,
                     content_type="application/octet-stream",
                     status=200)
        # Do NOT register an incremental route — if code tries it, responses raises

        result = client.sync("doc_huts", library_dir=tmp_path)

        assert result.status == "snapshot_full"
        # No call was made to the incremental endpoint
        incremental_calls = [
            c for c in resp_lib.calls
            if "/data/incremental" in c.request.url
        ]
        assert not incremental_calls, "incremental endpoint must NOT be called"


class TestSyncAtomicWrite:
    @resp_lib.activate
    def test_interrupted_snapshot_download_leaves_manifest_intact(self, client, tmp_path):
        """If os.replace fails mid-download, the manifest must NOT be mutated."""
        # Write existing manifest pointing to SNAPSHOT_V1
        dataset_dir = tmp_path / "doc_huts"
        dataset_dir.mkdir()
        _write_manifest_for(dataset_dir, SNAPSHOT_V1)

        # Read the old manifest content for comparison
        old_manifest_text = (dataset_dir / MANIFEST_FILENAME).read_text()

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS_V2, status=200)
        # incremental returns 410 so we fall to full download
        resp_lib.add(resp_lib.GET,
                     f"{BASE}/v1/datasets/doc_huts/data/incremental",
                     json={"detail": "Expired."},
                     status=410)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/doc/doc_huts",
                     body=FAKE_PARQUET_V2,
                     content_type="application/octet-stream",
                     status=200)

        replace_calls = []

        original_replace = __import__("os").replace

        def _exploding_replace(src, dst):
            replace_calls.append((src, dst))
            # Only explode on the parquet file write, not the manifest write.
            if str(dst).endswith(".parquet") or str(dst).endswith(".geo.parquet"):
                raise OSError("simulated disk full")
            return original_replace(src, dst)

        with patch("os.replace", side_effect=_exploding_replace):
            with pytest.raises(OSError, match="simulated disk full"):
                client.sync("doc_huts", library_dir=tmp_path)

        # The manifest must be unchanged (the new snapshot write failed,
        # so write_manifest was never called for the new state).
        new_manifest_text = (dataset_dir / MANIFEST_FILENAME).read_text()
        assert new_manifest_text == old_manifest_text, (
            "Manifest must not be mutated when the data file write fails"
        )

    @resp_lib.activate
    def test_interrupted_delta_download_leaves_manifest_intact(self, client, tmp_path):
        """If delta file write fails, manifest must still point to old snapshot."""
        dataset_dir = tmp_path / "doc_huts"
        dataset_dir.mkdir()
        _write_manifest_for(dataset_dir, SNAPSHOT_V1)

        old_manifest_text = (dataset_dir / MANIFEST_FILENAME).read_text()

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS_V2, status=200)
        resp_lib.add(resp_lib.GET,
                     f"{BASE}/v1/datasets/doc_huts/data/incremental",
                     body=FAKE_DELTA_BODY,
                     content_type="application/octet-stream",
                     status=200,
                     headers={"X-Eolas-Row-Count": "42"})

        replace_calls = []
        original_replace = __import__("os").replace

        def _exploding_replace(src, dst):
            replace_calls.append((src, dst))
            if "delta-" in str(dst):
                raise OSError("simulated disk full on delta")
            return original_replace(src, dst)

        with patch("os.replace", side_effect=_exploding_replace):
            with pytest.raises(OSError, match="simulated disk full"):
                client.sync("doc_huts", library_dir=tmp_path)

        # Manifest must still show old snapshot
        new_manifest_text = (dataset_dir / MANIFEST_FILENAME).read_text()
        assert new_manifest_text == old_manifest_text, (
            "Manifest must not be updated when the delta write fails"
        )


# ---------------------------------------------------------------------------
# sync_all() tests
# ---------------------------------------------------------------------------

# Additional metadata fixture for a second dataset (non-geo, no incremental)
META_NZ_CPI = {
    "name": "nz_cpi",
    "title": "NZ CPI",
    "source": "Stats NZ",
    "namespace": "statsnz",
    "table": "nz_cpi",
    "current_snapshot_id": SNAPSHOT_V1,
    "incremental_supported": False,
    "refresh_cadence": "quarterly",
    "geometry_type": None,
    "has_geometry": False,
}

META_NZ_CPI_V2 = {**META_NZ_CPI, "current_snapshot_id": SNAPSHOT_V2}


class TestSyncAll:
    @resp_lib.activate
    def test_sync_all_explicit_datasets_returns_ordered_results(self, client, tmp_path):
        """sync_all with explicit list → results in same order as input."""
        # Register mock responses for both datasets
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS, status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/doc/doc_huts",
                     body=FAKE_PARQUET,
                     content_type="application/octet-stream",
                     status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                     json=META_NZ_CPI, status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                     body=FAKE_PARQUET_V2,
                     content_type="application/octet-stream",
                     status=200)

        results = client.sync_all(
            library_dir=tmp_path,
            datasets=["doc_huts", "nz_cpi"],
        )

        assert len(results) == 2
        assert results[0].dataset == "doc_huts"
        assert results[1].dataset == "nz_cpi"

    @resp_lib.activate
    def test_sync_all_explicit_datasets_all_succeed(self, client, tmp_path):
        """sync_all with explicit list → all results have non-error status."""
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS, status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/doc/doc_huts",
                     body=FAKE_PARQUET,
                     content_type="application/octet-stream",
                     status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                     json=META_NZ_CPI, status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                     body=FAKE_PARQUET_V2,
                     content_type="application/octet-stream",
                     status=200)

        results = client.sync_all(
            library_dir=tmp_path,
            datasets=["doc_huts", "nz_cpi"],
        )

        for r in results:
            assert r.status != "error", f"Expected no errors, got: {r}"

    @resp_lib.activate
    def test_sync_all_discovers_manifests_when_datasets_none(self, client, tmp_path):
        """sync_all(datasets=None) discovers all sub-dirs with manifests."""
        # Pre-populate two dataset dirs with manifests
        for ds_name, meta in [("doc_huts", META_DOC_HUTS), ("nz_cpi", META_NZ_CPI)]:
            ddir = tmp_path / ds_name
            ddir.mkdir()
            _write_manifest_for(ddir, SNAPSHOT_V1)

        # Both datasets are already at SNAPSHOT_V1 on the server → unchanged
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS, status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                     json=META_NZ_CPI, status=200)

        results = client.sync_all(library_dir=tmp_path)

        assert len(results) == 2
        dataset_names = {r.dataset for r in results}
        assert "doc_huts" in dataset_names
        assert "nz_cpi" in dataset_names

    def test_sync_all_empty_library_dir_returns_empty_list(self, client, tmp_path):
        """sync_all on a library with no manifests (datasets=None) → empty list."""
        results = client.sync_all(library_dir=tmp_path)
        assert results == []

    @resp_lib.activate
    def test_sync_all_one_failure_does_not_kill_batch(self, client, tmp_path):
        """One failing dataset must not prevent the others from completing."""
        # doc_huts will succeed
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json=META_DOC_HUTS, status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/doc/doc_huts",
                     body=FAKE_PARQUET,
                     content_type="application/octet-stream",
                     status=200)
        # nz_cpi will return 500 → exception
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                     json={"detail": "internal server error"},
                     status=500)

        results = client.sync_all(
            library_dir=tmp_path,
            datasets=["doc_huts", "nz_cpi"],
        )

        assert len(results) == 2
        # doc_huts should succeed
        doc_result = next(r for r in results if r.dataset == "doc_huts")
        assert doc_result.status == "snapshot_full"

        # nz_cpi should be error, not raise
        cpi_result = next(r for r in results if r.dataset == "nz_cpi")
        assert cpi_result.status == "error"
        assert cpi_result.error is not None
        assert len(cpi_result.error) > 0

    @resp_lib.activate
    def test_sync_all_error_result_has_error_field(self, client, tmp_path):
        """A failed dataset must produce SyncResult with status='error' and error string."""
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                     json={"detail": "not found"},
                     status=404)

        results = client.sync_all(library_dir=tmp_path, datasets=["doc_huts"])

        assert len(results) == 1
        r = results[0]
        assert r.status == "error"
        assert r.error is not None
        assert r.bytes_downloaded == 0
        assert r.rows_added == 0
        assert r.files_added == 0

    @resp_lib.activate
    def test_sync_all_concurrency_limit_respected(self, client, tmp_path):
        """max_concurrent=2 with 4 datasets should complete without error."""
        datasets = ["doc_huts", "doc_huts", "doc_huts", "doc_huts"]
        # Register enough responses for 4 calls
        for _ in range(4):
            resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/doc_huts",
                         json=META_DOC_HUTS, status=200)
            resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/doc/doc_huts",
                         body=FAKE_PARQUET,
                         content_type="application/octet-stream",
                         status=200)

        results = client.sync_all(
            library_dir=tmp_path,
            datasets=datasets,
            max_concurrent=2,
        )

        assert len(results) == 4
        # All should succeed (no errors despite concurrency limit)
        assert all(r.status != "error" for r in results)

    def test_sync_result_error_field_default_none(self):
        """SyncResult must have error=None by default (backward compat)."""
        from eolas_data.sync.sync import SyncResult
        r = SyncResult(
            status="unchanged",
            dataset="test",
            library_dir=pathlib.Path("/tmp"),
            bytes_downloaded=0,
            rows_added=0,
            files_added=0,
        )
        assert r.error is None
