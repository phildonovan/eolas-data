"""Tests for Client.sync_changes(), Client.sync() dispatcher, and the CDC merge logic.

Ship gate: the merge must satisfy emit -> merge == bulk.

Coverage:
  Unit (pure merge logic, no HTTP):
  - Pure-append (I-only) feed merges correctly
  - Update (D+I for one pk) replaces the old row
  - Delete (D only) removes the row
  - Multi-column PK handling
  - current_state_filter machine-enforcement (is_current = true)
  - CDC meta-columns are stripped from the materialised output
  - Seq ordering is enforced even if feed arrives out of order

  Integration (mocked HTTP via `responses`):
  - sync_changes cold start: baseline sync_bulk + watermark anchor + v2 sidecar
  - sync_changes incremental: fetch pages, merge, advance watermark
  - sync_changes pagination: loop until Truncated=false
  - sync_changes unchanged: no pages, no file write, status='unchanged'
  - sync_changes 410 WatermarkExpired: re-baseline, sidecar reset
  - sync() dispatcher: routes snapshot tier -> sync_bulk (sync_mode='snapshot')
  - sync() dispatcher: routes changelog tier -> sync_changes (sync_mode='changelog')
"""
from __future__ import annotations

import io
import json
import pathlib
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import responses as resp_lib

from eolas_data import Client, SyncResult
from eolas_data.cdc import (
    apply_current_state_filter,
    merge_changes,
    read_parquet_bytes,
    df_to_parquet_bytes,
)
from eolas_data.exceptions import WatermarkExpired

BASE = "https://api.eolas.fyi"

# ---------------------------------------------------------------------------
# Helpers — build test DataFrames and Parquet blobs
# ---------------------------------------------------------------------------

def _df(**cols) -> pd.DataFrame:
    """Shorthand: _df(id=[1,2], val=['a','b']) -> DataFrame."""
    return pd.DataFrame(cols)


def _to_parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="snappy")
    return buf.getvalue()


def _changes_df(
    rows: list[dict],
    seq_start: int = 1,
) -> pd.DataFrame:
    """Build a changes DataFrame with CDC meta-columns.

    Each dict in rows should have at minimum 'op' and the base columns. The
    _eolas_seq is assigned sequentially from seq_start.
    """
    records = []
    for i, row in enumerate(rows):
        op = row.pop("op")
        rec = {
            "_eolas_seq": seq_start + i,
            "_eolas_op": op,
            "_eolas_committed_at": "2026-06-17T00:00:00Z",
            "_eolas_snapshot_id": "snap_001",
            **row,
        }
        records.append(rec)
    return pd.DataFrame(records)


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


# ---------------------------------------------------------------------------
# Unit tests — pure merge_changes() function
# ---------------------------------------------------------------------------

class TestMergeChanges:
    """Correctness tests for the pure merge function. No HTTP involved."""

    def test_pure_append_i_only(self):
        """I-only feed: new rows are added; existing rows are untouched."""
        local = _df(id=[1, 2], val=["a", "b"])
        changes = _changes_df([
            {"op": "I", "id": 3, "val": "c"},
            {"op": "I", "id": 4, "val": "d"},
        ], seq_start=100)

        result = merge_changes(local, changes, pk_columns=["id"])

        assert set(result["id"].tolist()) == {1, 2, 3, 4}
        assert len(result) == 4
        # CDC cols stripped
        for col in ("_eolas_seq", "_eolas_op", "_eolas_committed_at", "_eolas_snapshot_id"):
            assert col not in result.columns

    def test_update_d_then_i_replaces_row(self):
        """D+I for same PK: old row dropped, new row inserted. Equivalent to bulk snap."""
        local = _df(id=[1, 2, 3], val=["a", "b", "c"])
        # Update pk=2: D at seq=10, I at seq=11 with new val
        changes = _changes_df([
            {"op": "D", "id": 2, "val": "b"},   # seq=10 — drop old
            {"op": "I", "id": 2, "val": "b_new"},  # seq=11 — insert new
        ], seq_start=10)

        result = merge_changes(local, changes, pk_columns=["id"])

        assert len(result) == 3
        row2 = result[result["id"] == 2]
        assert len(row2) == 1
        assert row2.iloc[0]["val"] == "b_new"
        # Other rows untouched
        assert result[result["id"] == 1].iloc[0]["val"] == "a"
        assert result[result["id"] == 3].iloc[0]["val"] == "c"

    def test_delete_removes_row(self):
        """D op removes a row; result has one fewer row."""
        local = _df(id=[1, 2, 3], val=["a", "b", "c"])
        changes = _changes_df([{"op": "D", "id": 2, "val": "b"}], seq_start=20)

        result = merge_changes(local, changes, pk_columns=["id"])

        assert set(result["id"].tolist()) == {1, 3}
        assert len(result) == 2

    def test_multi_column_pk(self):
        """Composite PK: (pharmacode, time_frame) — the pharmac pattern."""
        local = _df(
            pharmacode=["P001", "P001", "P002"],
            time_frame=["2026-01", "2026-02", "2026-01"],
            price=[10.0, 11.0, 5.0],
        )
        # Update (P001, 2026-02): D then I with new price
        changes = _changes_df([
            {"op": "D", "pharmacode": "P001", "time_frame": "2026-02", "price": 11.0},
            {"op": "I", "pharmacode": "P001", "time_frame": "2026-02", "price": 12.5},
            # Also insert a new row
            {"op": "I", "pharmacode": "P002", "time_frame": "2026-02", "price": 6.0},
        ], seq_start=50)

        result = merge_changes(local, changes, pk_columns=["pharmacode", "time_frame"])

        assert len(result) == 4
        updated = result[
            (result["pharmacode"] == "P001") & (result["time_frame"] == "2026-02")
        ]
        assert len(updated) == 1
        assert updated.iloc[0]["price"] == 12.5

    def test_seq_ordering_enforced(self):
        """Out-of-order feed: D before I even when seq is scrambled in the input."""
        local = _df(id=[1], val=["old"])
        # Provide I before D (reversed) — the merge must re-sort by seq.
        changes = pd.DataFrame([
            {
                "_eolas_seq": 20,
                "_eolas_op": "I",
                "_eolas_committed_at": "2026-06-17T00:00:00Z",
                "_eolas_snapshot_id": "s",
                "id": 1,
                "val": "new",
            },
            {
                "_eolas_seq": 19,
                "_eolas_op": "D",
                "_eolas_committed_at": "2026-06-17T00:00:00Z",
                "_eolas_snapshot_id": "s",
                "id": 1,
                "val": "old",
            },
        ])
        result = merge_changes(local, changes, pk_columns=["id"])

        # After sort: D at seq=19 drops the old row, I at seq=20 inserts new
        assert len(result) == 1
        assert result.iloc[0]["val"] == "new"

    def test_current_state_filter_applied_after_merge(self):
        """SCD2 tables: is_current=false rows are stripped from the materialised file."""
        local = _df(
            id=[1, 2],
            val=["a", "b"],
            is_current=[True, True],
        )
        # SCD2 update of id=1: D (mark old), I with is_current=True + old row re-inserted
        # with is_current=False (as the server would send it).
        changes = _changes_df([
            {"op": "D", "id": 1, "val": "a", "is_current": True},
            {"op": "I", "id": 1, "val": "a_old", "is_current": False},  # expired row
            {"op": "I", "id": 1, "val": "a_new", "is_current": True},   # current row
        ], seq_start=30)

        result = merge_changes(
            local,
            changes,
            pk_columns=["id"],
            current_state_filter="is_current = true",
        )

        # Only current rows kept
        assert all(result["is_current"] == True)
        r1 = result[result["id"] == 1]
        assert len(r1) == 1
        assert r1.iloc[0]["val"] == "a_new"

    def test_current_state_filter_missing_column_is_noop(self):
        """If the filter column is absent (non-SCD2 table), filter is silently skipped."""
        local = _df(id=[1, 2], val=["a", "b"])
        changes = _changes_df([{"op": "I", "id": 3, "val": "c"}], seq_start=5)

        result = merge_changes(
            local,
            changes,
            pk_columns=["id"],
            current_state_filter="is_current = true",
        )
        # All 3 rows kept — filter skipped because 'is_current' not in DataFrame
        assert len(result) == 3

    def test_cdc_meta_columns_stripped(self):
        """All _eolas_* columns are absent from the merged output."""
        local = _df(id=[1], val=["x"])
        changes = _changes_df([{"op": "I", "id": 2, "val": "y"}], seq_start=1)

        result = merge_changes(local, changes, pk_columns=["id"])

        for col in result.columns:
            assert not col.startswith("_eolas_"), f"CDC column {col!r} leaked into output"

    def test_empty_local_append_only(self):
        """Cold start with empty local DataFrame — all change rows inserted."""
        local = pd.DataFrame(columns=["id", "val"])
        changes = _changes_df([
            {"op": "I", "id": 1, "val": "a"},
            {"op": "I", "id": 2, "val": "b"},
        ], seq_start=1)

        result = merge_changes(local, changes, pk_columns=["id"])
        assert len(result) == 2

    def test_merge_equals_bulk_snapshot(self):
        """The ship gate: merge(baseline + changes_feed) == bulk_snap1.

        Snapshot 0 (baseline): rows id=1..5
        Changes feed (snap0 -> snap1):
          - Delete id=3
          - Update id=4 (D+I)
          - Insert id=6
        Expected snap1: id=1,2,4(updated),5,6
        """
        snap0 = _df(
            id=[1, 2, 3, 4, 5],
            val=["a", "b", "c", "d_old", "e"],
        )
        changes = _changes_df([
            {"op": "D", "id": 3, "val": "c"},           # delete
            {"op": "D", "id": 4, "val": "d_old"},        # update old
            {"op": "I", "id": 4, "val": "d_new"},        # update new
            {"op": "I", "id": 6, "val": "f"},            # insert
        ], seq_start=1)

        # Expected snap1 — this is what a full re-download should return
        snap1_expected = _df(
            id=[1, 2, 4, 5, 6],
            val=["a", "b", "d_new", "e", "f"],
        )

        result = merge_changes(snap0, changes, pk_columns=["id"])

        # Same PK set
        assert set(result["id"].tolist()) == set(snap1_expected["id"].tolist())
        # No duplicates
        assert len(result) == len(result["id"].drop_duplicates())
        # Value for updated row
        assert result[result["id"] == 4].iloc[0]["val"] == "d_new"
        # Deleted row absent
        assert 3 not in result["id"].tolist()
        # Inserted row present
        assert 6 in result["id"].tolist()
        # Untouched rows intact
        for pk, val in [(1, "a"), (2, "b"), (5, "e")]:
            assert result[result["id"] == pk].iloc[0]["val"] == val


# ---------------------------------------------------------------------------
# Integration tests — mocked HTTP with `responses`
# ---------------------------------------------------------------------------

# Shared mock metadata for the test dataset (changelog-tier)
CHANGELOG_META = {
    "name": "pharmac_schedule_history",
    "title": "PHARMAC Schedule History",
    "source": "PHARMAC",
    "namespace": "pharmac",
    "table": "pharmac_schedule_history",
    "cdc_serving_tier": "changelog",
    "pk_columns": ["pharmacode", "time_frame"],
    "current_state_filter": None,
}

SNAPSHOT_META = {
    "name": "nz_cpi",
    "title": "NZ CPI",
    "source": "Stats NZ",
    "namespace": "statsnz",
    "table": "nz_cpi",
    "cdc_serving_tier": "snapshot",
    "pk_columns": ["period"],
    "current_state_filter": None,
}

_BASELINE_SNAP_ID = "snap_baseline_001"
_BASELINE_DATA = _df(
    pharmacode=["P001", "P002"],
    time_frame=["2026-01", "2026-01"],
    price=[10.0, 5.0],
)
_BASELINE_PARQUET = _to_parquet(_BASELINE_DATA)

_FAKE_CPI_PARQUET = b"PAR1" + b"\x00" * 12 + b"PAR1"
_FAKE_CPI_SNAP = "cpi_snap_001"


def _make_changes_parquet(rows: list[dict], seq_start: int = 100) -> bytes:
    df = _changes_df([dict(r) for r in rows], seq_start=seq_start)
    return _to_parquet(df)


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


class TestSyncChanges:
    """HTTP-mocked integration tests for sync_changes()."""

    @resp_lib.activate
    def test_cold_start_downloads_baseline_and_sets_watermark(self, client, tmp_path):
        """Cold start (no sidecar): sync_bulk called, v2 sidecar written, status='downloaded'."""
        dest = tmp_path / "pharmac.parquet"

        # Dataset metadata
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history",
                     json=CHANGELOG_META, status=200)
        # sync_bulk path: HEAD + GET
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history",
                     json=CHANGELOG_META, status=200)
        resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/pharmac/pharmac_schedule_history",
                     body=b"", status=200,
                     headers={"X-Snapshot-Version": _BASELINE_SNAP_ID})
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/pharmac/pharmac_schedule_history",
                     body=_BASELINE_PARQUET,
                     content_type="application/octet-stream",
                     status=200)
        # _fetch_changes_seq_high: GET /changes with large since_seq
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history/changes",
                     body=b"",
                     status=200,
                     headers={
                         "X-Eolas-Seq-High": "514000",
                         "X-Eolas-Row-Count": "0",
                         "X-Eolas-Truncated": "false",
                     })

        result = client.sync_changes(
            "pharmac_schedule_history",
            path=dest,
        )

        assert result.status == "downloaded"
        assert result.sync_mode == "changelog"
        assert result.current_seq == 514000
        assert result.previous_seq is None
        assert result.ops_applied == 0

        # v2 sidecar written
        sidecar_path = pathlib.Path(str(dest) + ".eolas-meta.json")
        assert sidecar_path.exists()
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["schema_version"] == 2
        assert sidecar["sync_mode"] == "changelog"
        assert sidecar["watermark_seq"] == 514000
        assert sidecar["pk_columns"] == ["pharmacode", "time_frame"]
        assert sidecar["baseline_snapshot_id"] == _BASELINE_SNAP_ID

    @resp_lib.activate
    def test_incremental_applies_changes_and_advances_watermark(self, client, tmp_path):
        """Incremental call: changes fetched, merged, watermark advanced, status='updated'."""
        dest = tmp_path / "pharmac.parquet"
        dest.write_bytes(_BASELINE_PARQUET)

        # Write a v2 sidecar with watermark at 514000
        sidecar_path = pathlib.Path(str(dest) + ".eolas-meta.json")
        sidecar_path.write_text(json.dumps({
            "schema_version": 2,
            "sync_mode": "changelog",
            "name": "pharmac_schedule_history",
            "format": "parquet",
            "pk_columns": ["pharmacode", "time_frame"],
            "current_state_filter": None,
            "baseline_snapshot_id": _BASELINE_SNAP_ID,
            "watermark_seq": 514000,
            "updated_at": "2026-06-17T00:00:00Z",
        }) + "\n")

        # Dataset metadata
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history",
                     json=CHANGELOG_META, status=200)

        # Changes page: one new insert (P003, 2026-02)
        change_rows = [
            {"op": "I", "pharmacode": "P003", "time_frame": "2026-02", "price": 7.5},
        ]
        changes_parquet = _make_changes_parquet(change_rows, seq_start=514001)

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history/changes",
                     body=changes_parquet,
                     content_type="application/octet-stream",
                     status=200,
                     headers={
                         "X-Eolas-Seq-High": "514001",
                         "X-Eolas-Row-Count": "1",
                         "X-Eolas-Truncated": "false",
                     })

        result = client.sync_changes("pharmac_schedule_history", path=dest)

        assert result.status == "updated"
        assert result.sync_mode == "changelog"
        assert result.previous_seq == 514000
        assert result.current_seq == 514001
        assert result.ops_applied == 1

        # File updated: now has 3 rows
        merged = pq.read_table(str(dest)).to_pandas()
        assert len(merged) == 3
        assert set(merged["pharmacode"].tolist()) == {"P001", "P002", "P003"}

        # Sidecar watermark advanced
        sidecar_updated = json.loads(sidecar_path.read_text())
        assert sidecar_updated["watermark_seq"] == 514001

    @resp_lib.activate
    def test_pagination_loops_until_not_truncated(self, client, tmp_path):
        """Multi-page feed: loop continues while Truncated=true, stops at Truncated=false."""
        dest = tmp_path / "pharmac.parquet"
        dest.write_bytes(_BASELINE_PARQUET)

        sidecar_path = pathlib.Path(str(dest) + ".eolas-meta.json")
        sidecar_path.write_text(json.dumps({
            "schema_version": 2,
            "sync_mode": "changelog",
            "name": "pharmac_schedule_history",
            "format": "parquet",
            "pk_columns": ["pharmacode", "time_frame"],
            "current_state_filter": None,
            "baseline_snapshot_id": _BASELINE_SNAP_ID,
            "watermark_seq": 100,
            "updated_at": "2026-06-17T00:00:00Z",
        }) + "\n")

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history",
                     json=CHANGELOG_META, status=200)

        # Page 1 — truncated, high_seq=150
        page1_changes = _make_changes_parquet([
            {"op": "I", "pharmacode": "P010", "time_frame": "2026-01", "price": 1.0},
        ], seq_start=101)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history/changes",
                     body=page1_changes,
                     content_type="application/octet-stream",
                     status=200,
                     headers={
                         "X-Eolas-Seq-High": "150",
                         "X-Eolas-Row-Count": "1",
                         # Simulate large enough count that truncated=true triggers another page
                         "X-Eolas-Truncated": "true",
                     })

        # Page 2 — not truncated, high_seq=200
        page2_changes = _make_changes_parquet([
            {"op": "I", "pharmacode": "P011", "time_frame": "2026-01", "price": 2.0},
        ], seq_start=151)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history/changes",
                     body=page2_changes,
                     content_type="application/octet-stream",
                     status=200,
                     headers={
                         "X-Eolas-Seq-High": "200",
                         "X-Eolas-Row-Count": "1",
                         "X-Eolas-Truncated": "false",
                     })

        result = client.sync_changes("pharmac_schedule_history", path=dest)

        assert result.current_seq == 200
        assert result.ops_applied == 2  # two change rows total

        merged = pq.read_table(str(dest)).to_pandas()
        # Baseline (P001, P002) + P010 + P011
        assert len(merged) == 4

        sidecar_updated = json.loads(sidecar_path.read_text())
        assert sidecar_updated["watermark_seq"] == 200

    @resp_lib.activate
    def test_unchanged_when_no_changes(self, client, tmp_path):
        """Server returns 0 rows: file untouched, status='unchanged'."""
        dest = tmp_path / "pharmac.parquet"
        dest.write_bytes(_BASELINE_PARQUET)

        sidecar_path = pathlib.Path(str(dest) + ".eolas-meta.json")
        sidecar_path.write_text(json.dumps({
            "schema_version": 2,
            "sync_mode": "changelog",
            "name": "pharmac_schedule_history",
            "format": "parquet",
            "pk_columns": ["pharmacode", "time_frame"],
            "current_state_filter": None,
            "baseline_snapshot_id": _BASELINE_SNAP_ID,
            "watermark_seq": 514000,
            "updated_at": "2026-06-17T00:00:00Z",
        }) + "\n")

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history",
                     json=CHANGELOG_META, status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history/changes",
                     body=b"",
                     status=200,
                     headers={
                         "X-Eolas-Seq-High": "514000",
                         "X-Eolas-Row-Count": "0",
                         "X-Eolas-Truncated": "false",
                     })

        result = client.sync_changes("pharmac_schedule_history", path=dest)

        assert result.status == "unchanged"
        assert result.sync_mode == "changelog"
        assert result.ops_applied == 0
        assert result.current_seq == 514000  # unchanged

        # File content must be untouched
        assert dest.read_bytes() == _BASELINE_PARQUET

    @resp_lib.activate
    def test_410_watermark_expired_rebaselines(self, client, tmp_path):
        """410 WatermarkExpired triggers a re-baseline via sync_bulk, sidecar reset."""
        dest = tmp_path / "pharmac.parquet"
        dest.write_bytes(_BASELINE_PARQUET)

        sidecar_path = pathlib.Path(str(dest) + ".eolas-meta.json")
        sidecar_path.write_text(json.dumps({
            "schema_version": 2,
            "sync_mode": "changelog",
            "name": "pharmac_schedule_history",
            "format": "parquet",
            "pk_columns": ["pharmacode", "time_frame"],
            "current_state_filter": None,
            "baseline_snapshot_id": "old_baseline",
            "watermark_seq": 1,  # ancient watermark, will expire
            "updated_at": "2026-01-01T00:00:00Z",
        }) + "\n")

        # Dataset metadata fetch
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history",
                     json=CHANGELOG_META, status=200)

        # /changes returns 410
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history/changes",
                     json={
                         "error": "watermark_expired",
                         "min_available_seq": 50000,
                         "resync": "bulk",
                     },
                     status=410)

        # Re-baseline sync_bulk
        new_snap_id = "snap_new_baseline"
        new_baseline_data = _df(
            pharmacode=["P001", "P002", "P999"],
            time_frame=["2026-01", "2026-01", "2026-06"],
            price=[10.0, 5.0, 99.0],
        )
        new_baseline_parquet = _to_parquet(new_baseline_data)

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history",
                     json=CHANGELOG_META, status=200)
        resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/pharmac/pharmac_schedule_history",
                     body=b"", status=200,
                     headers={"X-Snapshot-Version": new_snap_id})
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/pharmac/pharmac_schedule_history",
                     body=new_baseline_parquet,
                     content_type="application/octet-stream",
                     status=200)

        # New high-water seq after re-baseline
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history/changes",
                     body=b"",
                     status=200,
                     headers={
                         "X-Eolas-Seq-High": "600000",
                         "X-Eolas-Row-Count": "0",
                         "X-Eolas-Truncated": "false",
                     })

        result = client.sync_changes("pharmac_schedule_history", path=dest)

        assert result.status == "updated"
        assert result.sync_mode == "changelog"
        assert result.current_seq == 600000
        assert result.previous_seq == 1  # the expired watermark
        assert result.ops_applied == 0   # no change rows applied (re-baselined)

        # File now has new baseline content
        merged = pq.read_table(str(dest)).to_pandas()
        assert len(merged) == 3
        assert "P999" in merged["pharmacode"].tolist()

        # Sidecar reset with new baseline + watermark
        sidecar_updated = json.loads(sidecar_path.read_text())
        assert sidecar_updated["baseline_snapshot_id"] == new_snap_id
        assert sidecar_updated["watermark_seq"] == 600000

    @resp_lib.activate
    def test_v1_sidecar_triggers_cold_start(self, client, tmp_path):
        """A v1 (snapshot) sidecar means we need a changelog cold start."""
        dest = tmp_path / "pharmac.parquet"
        dest.write_bytes(_BASELINE_PARQUET)

        # Write a v1 snapshot sidecar (no sync_mode)
        sidecar_path = pathlib.Path(str(dest) + ".eolas-meta.json")
        sidecar_path.write_text(json.dumps({
            "schema_version": 1,
            "name": "pharmac_schedule_history",
            "snapshot_id": "old_snap",
            "format": "parquet",
        }) + "\n")

        # Same sequence as cold start test
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history",
                     json=CHANGELOG_META, status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history",
                     json=CHANGELOG_META, status=200)
        resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/pharmac/pharmac_schedule_history",
                     body=b"", status=200,
                     headers={"X-Snapshot-Version": _BASELINE_SNAP_ID})
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/pharmac/pharmac_schedule_history",
                     body=_BASELINE_PARQUET,
                     content_type="application/octet-stream",
                     status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history/changes",
                     body=b"",
                     status=200,
                     headers={
                         "X-Eolas-Seq-High": "514000",
                         "X-Eolas-Row-Count": "0",
                         "X-Eolas-Truncated": "false",
                     })

        result = client.sync_changes("pharmac_schedule_history", path=dest)

        assert result.status == "downloaded"
        assert result.sync_mode == "changelog"
        sidecar_updated = json.loads(sidecar_path.read_text())
        assert sidecar_updated["schema_version"] == 2


class TestSyncDispatcher:
    """Tests for the unified sync() dispatcher."""

    @resp_lib.activate
    def test_dispatch_snapshot_tier_calls_sync_bulk(self, client, tmp_path):
        """tier='snapshot' -> sync_bulk path, result.sync_mode='snapshot'."""
        dest = tmp_path / "nz_cpi.parquet"

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                     json=SNAPSHOT_META, status=200)
        # sync_bulk needs: meta GET + HEAD + GET
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                     json=SNAPSHOT_META, status=200)
        resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                     body=b"", status=200,
                     headers={"X-Snapshot-Version": _FAKE_CPI_SNAP})
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                     body=_FAKE_CPI_PARQUET,
                     content_type="application/octet-stream",
                     status=200)

        result = client.sync("nz_cpi", path=dest)

        assert isinstance(result, SyncResult)
        assert result.sync_mode == "snapshot"
        assert result.status == "downloaded"
        assert result.current_snapshot_id == _FAKE_CPI_SNAP

    @resp_lib.activate
    def test_dispatch_changelog_tier_calls_sync_changes(self, client, tmp_path):
        """tier='changelog' -> sync_changes path, result.sync_mode='changelog'."""
        dest = tmp_path / "pharmac.parquet"

        # sync() reads meta first
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history",
                     json=CHANGELOG_META, status=200)
        # sync_changes then re-reads meta for pk_columns etc.
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history",
                     json=CHANGELOG_META, status=200)
        # Cold-start bulk
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history",
                     json=CHANGELOG_META, status=200)
        resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/pharmac/pharmac_schedule_history",
                     body=b"", status=200,
                     headers={"X-Snapshot-Version": _BASELINE_SNAP_ID})
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/pharmac/pharmac_schedule_history",
                     body=_BASELINE_PARQUET,
                     content_type="application/octet-stream",
                     status=200)
        # High-water seq
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/pharmac_schedule_history/changes",
                     body=b"",
                     status=200,
                     headers={
                         "X-Eolas-Seq-High": "514000",
                         "X-Eolas-Row-Count": "0",
                         "X-Eolas-Truncated": "false",
                     })

        result = client.sync("pharmac_schedule_history", path=dest)

        assert result.sync_mode == "changelog"
        assert result.current_seq == 514000

    @resp_lib.activate
    def test_dispatch_missing_tier_defaults_to_snapshot(self, client, tmp_path):
        """Datasets without cdc_serving_tier get snapshot behaviour."""
        dest = tmp_path / "old_dataset.parquet"

        meta_no_tier = {
            "name": "old_dataset",
            "namespace": "statsnz",
            "table": "old_dataset",
            # no cdc_serving_tier key
        }

        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/old_dataset",
                     json=meta_no_tier, status=200)
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/old_dataset",
                     json=meta_no_tier, status=200)
        resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/statsnz/old_dataset",
                     body=b"", status=200,
                     headers={"X-Snapshot-Version": "snap_old"})
        resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/statsnz/old_dataset",
                     body=_FAKE_CPI_PARQUET,
                     content_type="application/octet-stream",
                     status=200)

        result = client.sync("old_dataset", path=dest)

        assert result.sync_mode == "snapshot"


# ---------------------------------------------------------------------------
# Parquet round-trip helper tests
# ---------------------------------------------------------------------------

class TestParquetHelpers:
    def test_round_trip(self):
        df = _df(id=[1, 2, 3], val=["a", "b", "c"])
        raw = df_to_parquet_bytes(df)
        back = read_parquet_bytes(raw)
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True),
            back.reset_index(drop=True),
        )
