"""Tests for start/end date-bound resolution on non-temporal datasets."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest
import responses as resp_lib
from unittest.mock import patch

from eolas_data import Client
from eolas_data.meta import (
    date_filter_column_from_info,
    resolve_date_bounds,
)

BASE = "https://api.eolas.fyi"

TEMPORAL_INFO = {
    "name": "nz_cpi",
    "date_filter_column": "date",
    "columns": [{"name": "date"}, {"name": "value"}],
}

NON_TEMPORAL_INFO = {
    "name": "nz_addresses",
    "date_filter_column": None,
    "has_geometry": True,
    "bulk_export_class": "materialised",
    "row_count_at_last_refresh": 2_418_264,
    "columns": [{"name": "address_id"}, {"name": "geometry_wkt"}],
}


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


def test_date_filter_column_from_explicit_field():
    assert date_filter_column_from_info(TEMPORAL_INFO) == "date"
    assert date_filter_column_from_info(NON_TEMPORAL_INFO) is None


def test_date_filter_column_inferred_from_columns():
    info = {"columns": [{"name": "awarded_date"}, {"name": "amount"}]}
    assert date_filter_column_from_info(info) == "awarded_date"


def test_resolve_date_bounds_keeps_temporal():
    start, end, stripped = resolve_date_bounds(TEMPORAL_INFO, "2020-01-01", "2024-12-31")
    assert start == "2020-01-01"
    assert end == "2024-12-31"
    assert stripped is False


def test_resolve_date_bounds_strips_non_temporal():
    start, end, stripped = resolve_date_bounds(NON_TEMPORAL_INFO, "2020-01-01", None)
    assert start is None
    assert end is None
    assert stripped is True


def test_resolve_date_bounds_unknown_metadata_passthrough():
    start, end, stripped = resolve_date_bounds(None, "2020-01-01", None)
    assert start == "2020-01-01"
    assert stripped is False


@resp_lib.activate
def test_get_start_on_non_temporal_warns_and_routes_to_get_local(client):
    """start= on a geo table must not block smart-routing to get_local()."""
    sentinel = pd.DataFrame({"address_id": [1], "geometry_wkt": ["POINT (0 0)"]})
    with (
        patch.object(client, "_info_cached", return_value=NON_TEMPORAL_INFO),
        patch.object(client, "get_local", return_value=sentinel) as mock_local,
        pytest.warns(UserWarning, match="start=/end= ignored"),
    ):
        result = client.get("nz_addresses", start="2020-01-01")
    mock_local.assert_called_once()
    assert result is sentinel