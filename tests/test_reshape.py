"""Tests for eolas_data.reshape.pivot_longer()/pivot_wider() — the long/wide
reshape helpers (2026-07-27 contract, see
eolas/docs/long-format-toggle-plan-2026-07-27.md). Mirrors
eolas-r/tests/testthat/test-reshape.R.
"""

from __future__ import annotations

import gzip
import pathlib
from unittest.mock import patch

import pandas as pd
import pytest
import responses as resp_lib

from eolas_data import Client, Dataset, SyncResult, pivot_longer, pivot_wider
from eolas_data.exceptions import ReshapeError
from eolas_data.meta import attach_meta

BASE = "https://api.eolas.fyi"

# ---------------------------------------------------------------------------
# Fixtures — a wide RBNZ-style FX table and its long equivalent
# ---------------------------------------------------------------------------

WIDE_TABLE_META = {
    "name": "rbnz_fx_test",
    "title": "Test FX rates (wide)",
    "source": "RBNZ",
    "namespace": "rbnz",
    "table": "rbnz_fx_test",
    "layout": "wide",
    "time_columns": ["date"],
    "id_columns": ["date"],
    "value_columns": ["usd", "aud", "eur"],
}

WIDE_COLUMN_META = [
    {"name": "date", "type": "date", "description": "Observation date"},
    {
        "name": "usd",
        "type": "double",
        "description": "USD rate",
        "series_id": "RBNZD.SUSD",
    },
    {
        "name": "aud",
        "type": "double",
        "description": "AUD rate",
        "series_id": "RBNZD.SAUD",
    },
    {
        "name": "eur",
        "type": "double",
        "description": "EUR rate",
        "series_id": "RBNZD.SEUR",
    },
]

LONG_TABLE_META = {
    "name": "rbnz_fx_test_long",
    "title": "Test FX rates (long)",
    "source": "RBNZ",
    "namespace": "rbnz",
    "table": "rbnz_fx_test_long",
    "layout": "long",
    "time_columns": ["date"],
    "id_columns": ["date"],
    "value_columns": ["value"],
    "measure_name_column": "currency",
}


def _wide_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-02-01"],
            "usd": [0.61, 0.62],
            "aud": [0.93, 0.94],
            "eur": [0.56, None],
        }
    )
    return attach_meta(
        Dataset(df),
        name="rbnz_fx_test",
        source="RBNZ",
        table_meta=WIDE_TABLE_META,
        column_meta=pd.DataFrame(WIDE_COLUMN_META),
    )


def _long_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
            "currency": ["usd", "aud", "usd", "aud"],
            "value": [0.61, 0.93, 0.62, 0.94],
        }
    )
    return attach_meta(
        Dataset(df),
        name="rbnz_fx_test_long",
        source="RBNZ",
        table_meta=LONG_TABLE_META,
        column_meta=None,
    )


# ---------------------------------------------------------------------------
# pivot_longer — happy path
# ---------------------------------------------------------------------------


def test_pivot_longer_melts_wide_table_and_preserves_series_id():
    wide = _wide_frame()
    long = pivot_longer(wide)

    assert set(long.columns) == {"date", "measure", "value", "series_id"}
    assert set(long["measure"]) == {"usd", "aud", "eur"}
    # eur/2023-02-01 is NaN in the source — dropped (values_drop_na semantics).
    assert len(long) == 5
    row = long[(long["date"] == "2023-01-01") & (long["measure"] == "usd")].iloc[0]
    assert row["series_id"] == "RBNZD.SUSD"


def test_pivot_longer_respects_explicit_names_to():
    wide = _wide_frame()
    long = pivot_longer(wide, names_to="currency")
    assert "currency" in long.columns
    assert "measure" not in long.columns


# ---------------------------------------------------------------------------
# pivot_wider — happy path
# ---------------------------------------------------------------------------


def test_pivot_wider_derives_names_and_values_from_metadata():
    long = _long_frame()
    wide = pivot_wider(long)
    assert set(wide.columns) == {"date", "usd", "aud"}
    row = wide[wide["date"] == "2023-01-01"].iloc[0]
    assert row["usd"] == pytest.approx(0.61)
    assert row["aud"] == pytest.approx(0.93)


def test_pivot_wider_explicit_names_from_values_from():
    long = _long_frame()
    wide = pivot_wider(long, names_from="currency", values_from="value")
    assert set(wide.columns) == {"date", "usd", "aud"}


def test_roundtrip_pivot_longer_then_wider():
    wide = _wide_frame()
    long = pivot_longer(wide, names_to="currency")
    back = pivot_wider(
        attach_meta(
            Dataset(long),
            name="rbnz_fx_test_roundtrip",
            table_meta={**LONG_TABLE_META, "id_columns": ["date"]},
        )
    )
    assert set(back.columns) == {"date", "usd", "aud", "eur"}


# ---------------------------------------------------------------------------
# Refuse: no metadata
# ---------------------------------------------------------------------------


def test_pivot_longer_refuses_without_metadata():
    df = pd.DataFrame({"date": ["2023-01-01"], "value": [1.0]})
    with pytest.raises(ReshapeError, match="no eolas layout metadata"):
        pivot_longer(df)


def test_pivot_wider_refuses_without_metadata():
    df = pd.DataFrame({"date": ["2023-01-01"], "measure": ["x"], "value": [1.0]})
    with pytest.raises(ReshapeError, match="no eolas layout metadata"):
        pivot_wider(df)


def test_pivot_longer_refuses_when_layout_field_missing_but_meta_present():
    df = attach_meta(
        Dataset(pd.DataFrame({"date": ["2023-01-01"], "value": [1.0]})),
        name="no_layout_dataset",
        table_meta={"name": "no_layout_dataset", "title": "No layout"},
    )
    with pytest.raises(ReshapeError, match="no `layout` metadata"):
        pivot_longer(df)


# ---------------------------------------------------------------------------
# Refuse: feature / entity layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["feature", "entity"])
def test_pivot_longer_refuses_feature_and_entity_layout(layout):
    df = attach_meta(
        Dataset(pd.DataFrame({"id": [1], "value": [1.0]})),
        name="feature_or_entity",
        table_meta={"name": "feature_or_entity", "layout": layout},
    )
    with pytest.raises(ReshapeError, match=f"layout is {layout!r}"):
        pivot_longer(df)


@pytest.mark.parametrize("layout", ["feature", "entity"])
def test_pivot_wider_refuses_feature_and_entity_layout(layout):
    df = attach_meta(
        Dataset(pd.DataFrame({"id": [1], "value": [1.0]})),
        name="feature_or_entity",
        table_meta={"name": "feature_or_entity", "layout": layout},
    )
    with pytest.raises(ReshapeError, match=f"layout is {layout!r}"):
        pivot_wider(df)


# ---------------------------------------------------------------------------
# Refuse: geometry present
# ---------------------------------------------------------------------------


def test_pivot_longer_refuses_geometry_wkt_column():
    df = attach_meta(
        Dataset(
            pd.DataFrame(
                {
                    "id": [1],
                    "geometry_wkt": ["POINT (1 1)"],
                    "value": [1.0],
                }
            )
        ),
        name="geo_table",
        table_meta={
            "name": "geo_table",
            "layout": "wide",
            "value_columns": ["value"],
        },
    )
    with pytest.raises(ReshapeError, match="geometry column is present"):
        pivot_longer(df)


def test_pivot_wider_refuses_geometry_wkt_column():
    df = attach_meta(
        Dataset(
            pd.DataFrame(
                {
                    "id": [1],
                    "geometry_wkt": ["POINT (1 1)"],
                    "measure": ["a"],
                    "value": [1.0],
                }
            )
        ),
        name="geo_table_long",
        table_meta={
            "name": "geo_table_long",
            "layout": "long",
            "measure_name_column": "measure",
            "value_columns": ["value"],
        },
    )
    with pytest.raises(ReshapeError, match="geometry column is present"):
        pivot_wider(df)


# ---------------------------------------------------------------------------
# Refuse: wrong-direction layout / ambiguous columns
# ---------------------------------------------------------------------------


def test_pivot_longer_refuses_already_long_layout():
    long = _long_frame()
    with pytest.raises(ReshapeError, match="already long"):
        pivot_longer(long)


def test_pivot_wider_refuses_already_wide_layout():
    wide = _wide_frame()
    with pytest.raises(ReshapeError, match="already wide"):
        pivot_wider(wide)


def test_pivot_wider_refuses_ambiguous_names_values():
    df = attach_meta(
        Dataset(
            pd.DataFrame(
                {
                    "date": ["2023-01-01"],
                    "recipients": [10],
                    "cancels": [2],
                }
            )
        ),
        name="ambiguous_long",
        table_meta={
            "name": "ambiguous_long",
            "layout": "long",
            "id_columns": ["date"],
            "value_columns": ["recipients", "cancels"],
        },
    )
    with pytest.raises(ReshapeError, match="cannot be derived unambiguously"):
        pivot_wider(df)


# ---------------------------------------------------------------------------
# Contract parity: live path (Client.get) vs bulk path (Client.get_local)
# attach identical layout metadata and pivot identically.
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


LIVE_WIDE_RECORDS = [
    {"date": "2023-01-01", "usd": 0.61, "aud": 0.93},
    {"date": "2023-02-01", "usd": 0.62, "aud": 0.94},
]

LIVE_WIDE_META = {
    "name": "rbnz_fx_live_test",
    "title": "Test FX rates (wide, live)",
    "source": "RBNZ",
    "namespace": "rbnz",
    "table": "rbnz_fx_live_test",
    "layout": "wide",
    "id_columns": ["date"],
    "value_columns": ["usd", "aud"],
    "columns": [
        {"name": "date", "type": "date"},
        {"name": "usd", "type": "double", "series_id": "RBNZD.SUSD"},
        {"name": "aud", "type": "double", "series_id": "RBNZD.SAUD"},
    ],
}


@resp_lib.activate
def test_live_path_pivot_longer(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/v1/datasets/rbnz_fx_live_test/data",
        json={"data": LIVE_WIDE_RECORDS},
    )
    resp_lib.add(
        resp_lib.GET, f"{BASE}/v1/datasets/rbnz_fx_live_test", json=LIVE_WIDE_META
    )
    df = client.get("rbnz_fx_live_test")
    long = pivot_longer(df)

    assert set(long.columns) == {"date", "measure", "value", "series_id"}
    assert len(long) == 4
    usd_rows = long[long["measure"] == "usd"]
    assert len(usd_rows) == 2
    assert set(usd_rows["series_id"]) == {"RBNZD.SUSD"}


def test_bulk_path_pivot_longer_matches_live_path_shape(client, tmp_path):
    """The same DatasetMeta served over the bulk (get_local) path reshapes to
    the same shape as the live path — the contract-parity requirement. (The
    'date' column's dtype differs between the two paths for an unrelated,
    pre-existing reason — get_local()'s CSV reader doesn't parse dates — so
    this compares structure/series_id, not a byte-for-byte frame diff.)
    """

    def _write_wide_csv(path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write("date,usd,aud\n2023-01-01,0.61,0.93\n2023-02-01,0.62,0.94\n")

    def fake_sync(name, *, path, format, **kwargs):
        _write_wide_csv(path)
        return SyncResult(
            status="downloaded",
            previous_snapshot_id=None,
            current_snapshot_id="snap_1",
            path=path,
            bytes_downloaded=1024,
        )

    with (
        patch.object(client, "_info_cached", return_value=LIVE_WIDE_META),
        patch.object(client, "sync_bulk", side_effect=fake_sync),
    ):
        bulk_df = client.get_local(
            "rbnz_fx_live_test", format="csv_gz", cache_dir=tmp_path
        )

    long_bulk = pivot_longer(bulk_df)

    assert set(long_bulk.columns) == {"date", "measure", "value", "series_id"}
    assert len(long_bulk) == 4
    usd_rows = long_bulk[long_bulk["measure"] == "usd"]
    assert len(usd_rows) == 2
    assert set(usd_rows["series_id"]) == {"RBNZD.SUSD"}
    assert set(long_bulk["measure"]) == {"usd", "aud"}
