"""End-to-end integration tests for the long/wide reshape helpers.

Ties the REAL prod Glue layout stamp (set by `infra/set_layout_metadata.py`) to the
actual `eolas_data.pivot_longer`/`pivot_wider` functions, and proves a wide->long->wide
round-trip on RBNZ-shaped data. Written 2026-07-27 as the "prove it works" gate for
Leg B of the format toggle (docs/long-format-toggle-plan-2026-07-27.md).

The unit tests in test_reshape.py mock everything; this file:
  * exercises the real functions (not mocks of them),
  * anchors the fixture in the ACTUAL stamped rbnz_b1 schema (20 FX measures),
  * asserts a full round-trip is loss-free on dense data,
  * (optionally, when AWS creds are present) confirms the live prod stamp still
    matches what the client expects.
"""

from __future__ import annotations

import pandas as pd
import pytest

from eolas_data import pivot_longer, pivot_wider

# The real value_columns stamped on rbnz.rbnz_b1_exchange_rates_monthly in prod Glue
# (captured 2026-07-27). Anchors the round-trip fixture in production truth so a drift
# between the stamper's output and the client's expectations shows up here.
RBNZ_B1_VALUE_COLUMNS = [
    "twi_nominal",
    "twi_real",
    "nzd_usd",
    "nzd_gbp",
    "nzd_aud",
    "nzd_jpy",
    "nzd_eur",
    "nzd_cad",
    "nzd_krw",
    "nzd_cny",
    "nzd_myr",
    "nzd_hkd",
    "nzd_idr",
    "nzd_thb",
    "nzd_sgd",
    "nzd_twd",
    "nzd_inr",
    "nzd_php",
    "nzd_vnd",
    "historical_twi_base_june_1979_100",
]


def _wide_fixture(n_dates: int = 4) -> pd.DataFrame:
    """A dense (no-NA) RBNZ-B1-shaped wide frame: one row per date, 20 measures."""
    dates = [f"2024-0{i + 1}-01" for i in range(n_dates)]
    data = {"date": dates}
    for j, col in enumerate(RBNZ_B1_VALUE_COLUMNS):
        # deterministic distinct values so a mis-mapped measure is detectable
        data[col] = [round(1.0 + j + 0.01 * i, 4) for i in range(n_dates)]
    df = pd.DataFrame(data)
    df.attrs["eolas_meta"] = {
        "layout": "wide",
        "time_columns": ["date"],
        "id_columns": ["date"],
        "value_columns": RBNZ_B1_VALUE_COLUMNS,
        "measure_name_column": "",
    }
    df.attrs["eolas_name"] = "rbnz_b1_exchange_rates_monthly"
    return df


def test_wide_to_long_shape_and_values():
    wide = _wide_fixture(n_dates=4)
    long = pivot_longer(wide)
    # 4 dates x 20 measures, all dense -> 80 rows, no NA dropped
    assert len(long) == 4 * len(RBNZ_B1_VALUE_COLUMNS)
    assert set(long.columns) == {"date", "measure", "value"}
    assert set(long["measure"].unique()) == set(RBNZ_B1_VALUE_COLUMNS)
    # spot-check a specific cell survived the melt with the right value
    twi = long[(long["date"] == "2024-01-01") & (long["measure"] == "twi_nominal")]
    assert twi["value"].iloc[0] == pytest.approx(1.0)


def test_pivot_longer_output_declares_long_layout():
    """P0 #2 regression guard: the melted frame must re-declare itself `long` so
    a downstream pivot_wider works without the user hand-editing attrs."""
    long = pivot_longer(_wide_fixture(n_dates=3))
    m = long.attrs.get("eolas_meta")
    assert m is not None and m["layout"] == "long"
    assert m["measure_name_column"] == "measure"
    assert m["value_columns"] == ["value"]
    assert m["id_columns"] == ["date"]


def test_wide_long_wide_roundtrip_is_lossless():
    wide = _wide_fixture(n_dates=4)
    long = pivot_longer(wide)
    # NO manual re-attach: pivot_longer must leave usable long metadata itself
    # (this is the P0 #2 fix — the old test masked the bug by re-attaching here).
    back = pivot_wider(long)
    # same shape, same values (column order may differ -> reindex before compare)
    assert set(back.columns) == set(wide.columns)
    a = wide.sort_values("date").reset_index(drop=True)[
        ["date", *RBNZ_B1_VALUE_COLUMNS]
    ]
    b = back.sort_values("date").reset_index(drop=True)[
        ["date", *RBNZ_B1_VALUE_COLUMNS]
    ]
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_na_rows_are_dropped_on_longer():
    wide = _wide_fixture(n_dates=3)
    wide.loc[0, "nzd_usd"] = None  # one hole
    long = pivot_longer(wide)
    # exactly one (date, measure) cell removed
    assert len(long) == 3 * len(RBNZ_B1_VALUE_COLUMNS) - 1
    assert long[(long["measure"] == "nzd_usd") & (long["date"] == "2024-01-01")].empty


def test_series_id_reattached_on_longer():
    wide = _wide_fixture(n_dates=2)
    # mimic the eolas_columns glossary that carries per-column series_id (RBNZ)
    wide.attrs["eolas_columns"] = [
        {"name": "nzd_usd", "series_id": "B1.NZDUSD"},
        {"name": "twi_nominal", "series_id": "B1.TWI"},
    ]
    long = pivot_longer(wide)
    assert "series_id" in long.columns
    usd = long[long["measure"] == "nzd_usd"]
    assert (usd["series_id"] == "B1.NZDUSD").all()
    # a measure with no mapping -> NaN, not an error
    assert long[long["measure"] == "nzd_eur"]["series_id"].isna().all()


def test_stats_nz_style_long_to_wide_with_measure_column():
    """stats_nz long tables (110/206) stack quantities in a `measure` column.
    pivot_wider must spread on measure_name_column and keep the dims as id cols."""
    long = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01"],
            "period": ["2023Q1", "2023Q1", "2023Q1", "2023Q1"],
            "area": ["Auckland", "Auckland", "Wellington", "Wellington"],
            "measure": [
                "Enterprises",
                "Employee Count",
                "Enterprises",
                "Employee Count",
            ],
            "value": [100.0, 500.0, 80.0, 400.0],
        }
    )
    long.attrs["eolas_meta"] = {
        "layout": "long",
        "time_columns": ["date", "period"],
        "id_columns": ["date", "period", "area", "measure"],
        "value_columns": ["value"],
        "measure_name_column": "measure",
    }
    long.attrs["eolas_name"] = "bds_enterprises_business_type"
    wide = pivot_wider(long)
    assert {"Enterprises", "Employee Count"}.issubset(wide.columns)
    assert "area" in wide.columns and "measure" not in wide.columns
    akl = wide[wide["area"] == "Auckland"].iloc[0]
    assert akl["Enterprises"] == 100.0 and akl["Employee Count"] == 500.0


def test_pivot_wider_refuses_non_unique_keys():
    """P0 #1: duplicate (id, names_from) rows must error, never silently
    aggregate/collapse. Python raises via DataFrame.pivot; R now matches."""
    long = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-01"],
            "area": ["Auckland", "Auckland"],  # same id...
            "measure": ["Enterprises", "Enterprises"],  # ...same names_from -> dup key
            "value": [100.0, 200.0],
        }
    )
    long.attrs["eolas_meta"] = {
        "layout": "long",
        "time_columns": ["date"],
        "id_columns": ["date", "area", "measure"],
        "value_columns": ["value"],
        "measure_name_column": "measure",
    }
    long.attrs["eolas_name"] = "dup_key_table"
    with pytest.raises(Exception):
        pivot_wider(long)


# --- refuse gates (the safety contract) ------------------------------------
def test_refuse_without_metadata():
    df = pd.DataFrame({"date": ["2024-01-01"], "x": [1.0]})
    with pytest.raises(Exception, match="layout"):
        pivot_longer(df)


def test_refuse_on_geometry_present():
    df = _wide_fixture(2)
    df["geometry_wkt"] = ["POINT(1 1)"] * 2
    with pytest.raises(Exception, match="geometry"):
        pivot_longer(df)


@pytest.mark.parametrize("layout", ["feature", "entity"])
def test_refuse_on_feature_and_entity(layout):
    df = _wide_fixture(2)
    df.attrs["eolas_meta"] = {**df.attrs["eolas_meta"], "layout": layout}
    with pytest.raises(Exception, match=layout):
        pivot_longer(df)


def test_pivot_longer_refuses_a_long_table():
    df = _wide_fixture(2)
    df.attrs["eolas_meta"] = {**df.attrs["eolas_meta"], "layout": "long"}
    with pytest.raises(Exception, match="already long|not"):
        pivot_longer(df)


# --- live prod-stamp check (skipped without AWS creds / network) ------------
@pytest.mark.integration
def test_live_rbnz_stamp_matches_client_expectations():
    """Confirms the ACTUAL prod Glue stamp on rbnz_b1 is still the wide shape the
    client round-trip fixture assumes. Skips cleanly when Glue is unreachable."""
    boto3 = pytest.importorskip("boto3")
    try:
        g = boto3.client("glue", region_name="ap-southeast-2")
        p = g.get_table(DatabaseName="rbnz", Name="rbnz_b1_exchange_rates_monthly")[
            "Table"
        ]["Parameters"]
    except Exception as e:  # no creds, no network, table gone
        pytest.skip(f"live Glue unavailable: {e}")
    assert p.get("eolas.layout") == "wide"
    assert p.get("eolas.id_columns") == "date"
    stamped = p.get("eolas.value_columns", "").split(",")
    assert len(stamped) == len(RBNZ_B1_VALUE_COLUMNS)
