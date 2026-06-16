"""Round-trip invariant for the CDC OUT path: merge(baseline + feed) == full re-download.

This is the load-bearing client correctness claim of the changelog sync — a client that
starts from a bulk baseline at snapshot S0 and applies the /changes feed up to S1 must end
up byte-for-byte (frame-for-frame) identical to a fresh bulk download at S1.

The existing test_sync_changes.py::test_merge_equals_bulk_snapshot checks PK sets + spot
values; this asserts the FULL frame equality grok flagged as missing, and adds the cases it
didn't cover: the append pilot's real multi-column PK, the SCD2 current_state_filter path
(which Phase 2b's server-side SCD2 emit will rely on — validated here on the client side
independently of when that emit lands), and idempotent replay of overlapping pages.

Placement note: this lives in eolas-data (not eolas/api) on purpose — eolas and eolas-data
are SEPARATE GitHub repos, so an eolas-CI job cannot import eolas_data. The client merge is
the invariant under test, so it belongs where merge_changes lives and its CI runs.
"""
from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from eolas_data.cdc import merge_changes


def _changes_df(rows: list[dict], seq_start: int = 1) -> pd.DataFrame:
    """Build a /changes feed frame: each row dict carries 'op' + payload columns."""
    records = []
    for i, row in enumerate(rows):
        r = dict(row)
        op = r.pop("op")
        records.append({
            "_eolas_seq": seq_start + i,
            "_eolas_op": op,
            "_eolas_committed_at": "2026-06-17T00:00:00Z",
            "_eolas_snapshot_id": "snap_001",
            **r,
        })
    return pd.DataFrame(records)


def _canon(df: pd.DataFrame, pk: list[str]) -> pd.DataFrame:
    """Canonicalise for frame comparison: sort rows by pk, sort columns, reset index.

    merge result row/column order is an implementation detail; the invariant is set+content
    equality with a full re-download, so we normalise both sides the same way.
    """
    return (
        df.sort_values(pk)
          .reset_index(drop=True)
          .reindex(sorted(df.columns), axis=1)
    )


# ---------------------------------------------------------------------------
# Append-only (the fuel_prices_weekly pilot) — real multi-column PK, I-only feed
# ---------------------------------------------------------------------------
_FUEL_PK = ["week", "fuel", "variable", "unit"]


def test_roundtrip_append_multicol_pk_frame_equal():
    # bulk @ S0: two weeks already materialised locally.
    baseline = pd.DataFrame({
        "week":     ["2026-W01", "2026-W01", "2026-W02", "2026-W02"],
        "fuel":     ["91",       "diesel",   "91",       "diesel"],
        "variable": ["price",    "price",    "price",    "price"],
        "unit":     ["nzd_per_l"] * 4,
        "value":    [2.50, 2.10, 2.55, 2.12],
    })
    # feed S0 -> S1: a new week appended (append-only never updates/deletes existing rows).
    changes = _changes_df([
        {"op": "I", "week": "2026-W03", "fuel": "91",     "variable": "price", "unit": "nzd_per_l", "value": 2.60},
        {"op": "I", "week": "2026-W03", "fuel": "diesel", "variable": "price", "unit": "nzd_per_l", "value": 2.15},
    ])
    # bulk @ S1: what a full re-download returns.
    bulk_s1 = pd.concat([baseline, pd.DataFrame({
        "week":     ["2026-W03", "2026-W03"],
        "fuel":     ["91", "diesel"],
        "variable": ["price", "price"],
        "unit":     ["nzd_per_l", "nzd_per_l"],
        "value":    [2.60, 2.15],
    })], ignore_index=True)

    merged = merge_changes(baseline, changes, pk_columns=_FUEL_PK)

    assert_frame_equal(_canon(merged, _FUEL_PK), _canon(bulk_s1, _FUEL_PK))
    # No CDC meta-columns leak into the materialised output.
    assert not any(c.startswith("_eolas_") for c in merged.columns)


# ---------------------------------------------------------------------------
# Update (D+I) + Delete + Insert — full frame equality (single PK)
# ---------------------------------------------------------------------------
def test_roundtrip_update_delete_insert_frame_equal():
    baseline = pd.DataFrame({"id": [1, 2, 3, 4, 5], "val": ["a", "b", "c", "d_old", "e"]})
    changes = _changes_df([
        {"op": "D", "id": 3, "val": "c"},        # delete id 3
        {"op": "D", "id": 4, "val": "d_old"},    # update id 4 (expire old) ...
        {"op": "I", "id": 4, "val": "d_new"},    # ... insert new version
        {"op": "I", "id": 6, "val": "f"},        # insert id 6
    ])
    bulk_s1 = pd.DataFrame({"id": [1, 2, 4, 5, 6], "val": ["a", "b", "d_new", "e", "f"]})

    merged = merge_changes(baseline, changes, pk_columns=["id"])

    assert_frame_equal(_canon(merged, ["id"]), _canon(bulk_s1, ["id"]))


# ---------------------------------------------------------------------------
# SCD2 current_state_filter — pre-validates the Phase 2b SCD2 emit on the client side
# ---------------------------------------------------------------------------
def test_roundtrip_scd2_current_state_filter_frame_equal():
    # The client materialised snapshot is current-state only (is_current = true). An SCD2
    # update arrives as D (the old current row, now expired) + I (the new current row). The
    # feed payload carries is_current; the merge drops the old PK and the filter keeps only
    # the live row. This is exactly the shape write_stream_scd2 must emit in Step 2.
    baseline = pd.DataFrame({
        "id":         [1, 2, 3],
        "name":       ["p1", "p2_old", "p3"],
        "is_current": [True, True, True],
    })
    changes = _changes_df([
        {"op": "D", "id": 2, "name": "p2_old", "is_current": True},   # expire old current
        {"op": "I", "id": 2, "name": "p2_new", "is_current": True},   # new current version
        {"op": "D", "id": 3, "name": "p3",     "is_current": True},   # delete id 3
        {"op": "I", "id": 4, "name": "p4",     "is_current": True},   # insert id 4
    ])
    bulk_s1 = pd.DataFrame({
        "id":         [1, 2, 4],
        "name":       ["p1", "p2_new", "p4"],
        "is_current": [True, True, True],
    })

    merged = merge_changes(baseline, changes, pk_columns=["id"],
                           current_state_filter="is_current = true")

    assert_frame_equal(_canon(merged, ["id"]), _canon(bulk_s1, ["id"]))


# ---------------------------------------------------------------------------
# Idempotent replay — clients re-fetch overlapping pages; applying twice is a no-op
# ---------------------------------------------------------------------------
def test_roundtrip_idempotent_replay():
    baseline = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
    changes = _changes_df([
        {"op": "D", "id": 2, "val": "b"},
        {"op": "I", "id": 2, "val": "b_new"},
        {"op": "I", "id": 3, "val": "c"},
    ])
    once = merge_changes(baseline, changes, pk_columns=["id"])
    twice = merge_changes(once, changes, pk_columns=["id"])  # replay the same feed

    assert_frame_equal(_canon(once, ["id"]), _canon(twice, ["id"]))
