"""CDC merge utilities for eolas-data changelog sync (OUT half of CDC).

This module contains the pure merge logic that is separated from the HTTP
layer so it can be unit-tested independently of the network.

Design contract (from metadata-cdc-design-2026-06-16.md, Client contract):
- Change rows arrive sorted by _eolas_seq ascending.
- Ops: 'I' (insert), 'U' (update, represented as D+I pair at the server),
  'D' (delete).
- Merge algorithm (pk-keyed, seq-ordered):
  1. Sort the change batch by _eolas_seq (already guaranteed by server, but
     enforce for correctness).
  2. Collect the set of PKs touched by any change row.
  3. Drop ALL local rows whose PK value matches any touched PK.
  4. Append all change rows where _eolas_op != 'D' (i.e. 'I' or 'U').
     The result is the current state for those PKs.
  5. Apply current_state_filter if defined (e.g. "is_current = true" for SCD2
     tables) — filter on the column/value pair to keep only live rows.
  6. Strip the CDC meta-columns (_eolas_seq, _eolas_op, _eolas_committed_at,
     _eolas_snapshot_id) before writing the materialised file.

PK key rule: key STRICTLY on pk_columns. Geometry columns are never part of
the merge key — geometry equality is expensive and semantically wrong here.

Thread safety: this module is stateless; callers own the file lock.
"""
from __future__ import annotations

import io
import re
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# CDC meta-columns that the server attaches to every change row.
# These are stripped before writing the materialised local file.
_CDC_COLS = {"_eolas_seq", "_eolas_op", "_eolas_committed_at", "_eolas_snapshot_id"}


def _parse_current_state_filter(filter_expr: Optional[str]) -> Optional[tuple[str, object]]:
    """Parse a simple 'column = value' filter expression.

    Returns (column_name, value) or None if filter_expr is None/empty.

    Only the exact form '<col> = <val>' is supported (as documented in the
    stream registry spec). More complex expressions are ignored with a warning
    so existing datasets with complex filters don't silently break.

    Examples:
        'is_current = true'  -> ('is_current', True)
        'is_current = false' -> ('is_current', False)
        'status = active'    -> ('status', 'active')
    """
    if not filter_expr:
        return None
    m = re.match(r"^\s*(\w+)\s*=\s*(.+?)\s*$", filter_expr.strip())
    if not m:
        return None
    col, raw_val = m.group(1), m.group(2).strip()
    # Coerce booleans and integers; keep everything else as a string.
    if raw_val.lower() == "true":
        return (col, True)
    if raw_val.lower() == "false":
        return (col, False)
    try:
        return (col, int(raw_val))
    except ValueError:
        pass
    return (col, raw_val)


def apply_current_state_filter(
    df: pd.DataFrame,
    filter_expr: Optional[str],
) -> pd.DataFrame:
    """Apply a current_state_filter (e.g. 'is_current = true') to a DataFrame.

    When the filter column is absent from the DataFrame (e.g. the dataset has
    no SCD2 history column) the filter is silently skipped — this keeps
    append-only tables (which have no is_current) working correctly.

    Comparison is type-tolerant (and must stay in lockstep with the R client's
    eolas_apply_current_state_filter): a boolean filter value matches the column
    whether it is stored as boolean, "true"/"false" string, etc. — case-insensitive.
    A plain ``df[col] == True`` would silently drop EVERY row if is_current were
    ever served as a string (schema drift / a bad page), and would diverge from R.
    """
    parsed = _parse_current_state_filter(filter_expr)
    if parsed is None:
        return df
    col, val = parsed
    if col not in df.columns:
        return df
    if isinstance(val, bool):
        keep = df[col].astype(str).str.lower() == str(val).lower()
    else:
        keep = df[col].astype(str) == str(val)
    return df[keep].reset_index(drop=True)


def merge_changes(
    local_df: pd.DataFrame,
    changes_df: pd.DataFrame,
    pk_columns: list[str],
    current_state_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Merge a batch of change rows into the current local materialised snapshot.

    Parameters
    ----------
    local_df:
        Current local materialised state (may be empty DataFrame with correct
        columns for the first call after a fresh baseline).
    changes_df:
        Change rows from the /changes feed. Must contain _eolas_seq and
        _eolas_op columns. May also contain _eolas_committed_at and
        _eolas_snapshot_id.
    pk_columns:
        Primary-key column names. The merge is keyed on these columns only.
        Must be a non-empty list.
    current_state_filter:
        Optional filter expression from stream registry, e.g. 'is_current = true'.
        Applied after merge to keep only the current-state rows.

    Returns
    -------
    pd.DataFrame
        The merged current-state DataFrame, with CDC meta-columns stripped.

    Raises
    ------
    ValueError
        If pk_columns is empty or if required CDC columns are absent from
        changes_df.
    """
    if not pk_columns:
        raise ValueError("pk_columns must be a non-empty list")

    required_cdc = {"_eolas_seq", "_eolas_op"}
    missing = required_cdc - set(changes_df.columns)
    if missing:
        raise ValueError(
            f"changes_df is missing required CDC columns: {sorted(missing)}"
        )

    # Sort changes by seq — the server guarantees this order but enforce it for
    # correctness: a U is represented as D then I in ascending seq order, so
    # applying them out of order would produce stale state.
    changes_sorted = changes_df.sort_values("_eolas_seq", kind="stable").reset_index(drop=True)

    # Identify the set of PKs touched by any change (including deletes).
    # Build a tuple key for multi-column PKs.
    def _pk_tuple(df: pd.DataFrame) -> pd.Index:
        """Return a MultiIndex or single Index of PK values."""
        if len(pk_columns) == 1:
            return df[pk_columns[0]]
        return pd.MultiIndex.from_frame(df[pk_columns])

    touched_pk_set: set
    if len(pk_columns) == 1:
        touched_pk_set = set(changes_sorted[pk_columns[0]].tolist())
    else:
        touched_pk_set = set(
            tuple(row) for row in changes_sorted[pk_columns].itertuples(index=False, name=None)
        )

    # Drop all local rows whose PK matches any touched PK.
    if local_df.empty or not touched_pk_set:
        surviving_local = local_df.copy()
    else:
        if len(pk_columns) == 1:
            pk_col = pk_columns[0]
            if pk_col in local_df.columns:
                mask = local_df[pk_col].isin(touched_pk_set)
                surviving_local = local_df[~mask].reset_index(drop=True)
            else:
                surviving_local = local_df.copy()
        else:
            # Multi-column PK: build a tuple key column for the mask.
            if all(c in local_df.columns for c in pk_columns):
                local_pk_tuples = [
                    tuple(row)
                    for row in local_df[pk_columns].itertuples(index=False, name=None)
                ]
                mask = pd.Series(
                    [t in touched_pk_set for t in local_pk_tuples],
                    index=local_df.index,
                )
                surviving_local = local_df[~mask].reset_index(drop=True)
            else:
                surviving_local = local_df.copy()

    # Keep only non-delete rows from the change batch (I and U ops).
    # The server represents UPDATE as D then I; the D drops the old state above
    # and the I inserts the new state here.
    insertions = changes_sorted[changes_sorted["_eolas_op"] != "D"].copy()

    # Strip CDC meta-columns from insertions before appending.
    insertion_cols_to_drop = [c for c in _CDC_COLS if c in insertions.columns]
    insertions = insertions.drop(columns=insertion_cols_to_drop)

    # Also strip any CDC meta-cols that might have leaked into the local snapshot
    # (shouldn't happen after the first proper merge, but defensive).
    local_cols_to_drop = [c for c in _CDC_COLS if c in surviving_local.columns]
    surviving_local = surviving_local.drop(columns=local_cols_to_drop)

    # Concatenate: surviving local rows first, then new insertions.
    if insertions.empty:
        merged = surviving_local
    elif surviving_local.empty:
        merged = insertions.reset_index(drop=True)
    else:
        merged = pd.concat([surviving_local, insertions], ignore_index=True, sort=False)

    # Apply current_state_filter (machine-enforced, per spec).
    merged = apply_current_state_filter(merged, current_state_filter)

    return merged


def read_parquet_bytes(raw: bytes) -> pd.DataFrame:
    """Read a parquet blob (bytes) into a DataFrame."""
    return pq.read_table(io.BytesIO(raw)).to_pandas()


def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to parquet bytes (snappy compressed)."""
    buf = io.BytesIO()
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tbl, buf, compression="snappy")
    return buf.getvalue()
