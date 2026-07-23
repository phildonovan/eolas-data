"""Tests for the eolas MCP server tool implementations."""
from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import pandas as pd
import pytest

from eolas_data import SyncResult
from eolas_data.exceptions import NotFoundError
from eolas_data.exceptions import BulkUpgradeRequired
from eolas_data.mcp_server import (
    _MCP_GET_ROW_CAP,
    _MCP_SEARCH_CAP,
    _set_client,
    eolas_download,
    eolas_get,
    eolas_health,
    eolas_info,
    eolas_search,
    eolas_sync,
    mcp,
)


@pytest.fixture(autouse=True)
def _reset_client():
    _set_client(None)
    yield
    _set_client(None)


def _mock_client(**methods):
    client = MagicMock()
    for name, value in methods.items():
        setattr(client, name, value)
    _set_client(client)
    return client


def test_mcp_registers_expected_tools():
    """Smoke: FastMCP exposes all six eolas tools to MCP hosts."""
    import asyncio

    async def _tool_names():
        tools = await mcp.list_tools()
        return {t.name for t in tools}

    names = asyncio.run(_tool_names())
    assert names == {
        "eolas_health",
        "eolas_search",
        "eolas_info",
        "eolas_get",
        "eolas_download",
        "eolas_sync",
    }


def test_eolas_health_ok(monkeypatch):
    class Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "ok"}

    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    out = eolas_health()
    assert out["ok"] is True
    assert out["body"]["status"] == "ok"


def test_eolas_health_failure(monkeypatch):
    import requests

    def _boom(*a, **k):
        raise requests.ConnectionError("timeout")

    monkeypatch.setattr(requests, "get", _boom)
    out = eolas_health()
    assert out["ok"] is False
    assert "error" in out


def test_eolas_search_returns_compact_rows():
    df = pd.DataFrame([
        {"name": "nz_cpi", "title": "NZ CPI", "source": "OECD", "namespace": "oecd"},
        {"name": "rbnz_m1_prices", "title": "RBNZ M1", "source": "RBNZ", "namespace": "rbnz"},
    ])
    _mock_client(search=lambda q, source=None: df)

    out = eolas_search("cpi", limit=10)
    assert out["count"] == 2
    assert len(out["datasets"]) == 2
    assert out["datasets"][0]["name"] == "nz_cpi"
    assert "description" not in out["datasets"][0]


def test_eolas_search_client_error():
    _mock_client(search=lambda q, source=None: (_ for _ in ()).throw(NotFoundError("x")))

    out = eolas_search("missing")
    assert out["datasets"] == []
    assert "error" in out


def test_eolas_search_caps_limit():
    rows = [{"name": f"d{i}", "title": f"T{i}", "source": "X"} for i in range(100)]
    _mock_client(search=lambda q, source=None: pd.DataFrame(rows))

    out = eolas_search("x", limit=999)
    assert len(out["datasets"]) == _MCP_SEARCH_CAP
    assert out["truncated"] is True


def test_eolas_info_success():
    meta = {
        "name": "nz_parcels",
        "title": "NZ Parcels",
        "source": "LINZ",
        "cdc_serving_tier": "snapshot",
        "bulk_export_class": "cc-by",
        "row_count_at_last_refresh": 5_499_508,
    }
    _mock_client(info=lambda name: meta)

    out = eolas_info("nz_parcels")
    assert out["name"] == "nz_parcels"
    assert out["cdc_serving_tier"] == "snapshot"
    assert out["row_count_at_last_refresh"] == 5_499_508


def test_eolas_info_not_found():
    _mock_client(info=lambda name: (_ for _ in ()).throw(NotFoundError("missing")))

    out = eolas_info("no_such")
    assert "error" in out
    assert out["name"] == "no_such"


def test_eolas_get_caps_rows():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=600), "value": range(600)})
    _mock_client(get=lambda name, start=None, end=None, limit=None: df.head(limit))

    out = eolas_get("nz_cpi", limit=999)
    assert out["limit_applied"] == _MCP_GET_ROW_CAP
    assert len(out["rows"]) == _MCP_GET_ROW_CAP
    assert out["truncated"] is True


def test_eolas_get_client_error():
    _mock_client(get=lambda *a, **k: (_ for _ in ()).throw(NotFoundError("nope")))

    out = eolas_get("nz_cpi")
    assert out["rows"] == []
    assert "error" in out


def test_eolas_download_writes_path(tmp_path):
    dest = tmp_path / "nz_cpi.parquet"
    dest.write_bytes(b"PAR1testdata")

    _mock_client(
        download_bulk=lambda name, path, format, freshness, progress: pathlib.Path(path),
    )

    out = eolas_download("nz_cpi", path=str(dest), format="parquet")
    assert out["path"] == str(dest.resolve())
    assert out["bytes"] == len(b"PAR1testdata")
    assert out["format"] == "parquet"


def test_eolas_download_rejects_unknown_format():
    out = eolas_download("nz_cpi", path="/tmp/x.parquet", format="xlsx")
    assert "error" in out
    assert "parquet" in out["error"]


def test_eolas_download_maps_csv_alias(tmp_path):
    dest = tmp_path / "nz_cpi.csv.gz"
    dest.write_bytes(b"gzdata")

    captured = {}

    def _dl(name, path, format, freshness, progress):
        captured["format"] = format
        return pathlib.Path(path)

    _mock_client(download_bulk=_dl)

    out = eolas_download("nz_cpi", path=str(dest), format="csv")
    assert captured["format"] == "csv_gz"
    assert out["format"] == "csv_gz"


def test_eolas_download_surfaces_client_errors(tmp_path):
    _mock_client(
        download_bulk=lambda *a, **k: (_ for _ in ()).throw(
            BulkUpgradeRequired("Pro required")
        ),
    )

    out = eolas_download("nz_cpi", path=str(tmp_path / "x.parquet"))
    assert "error" in out
    assert "Pro" in out["error"]


def test_eolas_sync_snapshot_mode(tmp_path):
    dest = tmp_path / "nz_cpi.parquet"
    result = SyncResult(
        status="unchanged",
        previous_snapshot_id="snap1",
        current_snapshot_id="snap1",
        path=dest,
        bytes_downloaded=0,
        sync_mode="snapshot",
    )
    _mock_client(
        info=lambda name: {"cdc_serving_tier": "snapshot"},
        sync=lambda name, path, format, freshness, progress: result,
    )

    out = eolas_sync("nz_cpi", path=str(dest))
    assert out["status"] == "unchanged"
    assert out["sync_mode"] == "snapshot"
    assert out["bytes_downloaded"] == 0


def test_eolas_sync_changelog_requires_parquet(tmp_path):
    dest = tmp_path / "buildings.parquet"
    _mock_client(
        info=lambda name: {"cdc_serving_tier": "changelog"},
    )

    out = eolas_sync("nz_building_outlines", path=str(dest), format="csv")
    assert "error" in out
    assert "parquet" in out["error"].lower()


def test_eolas_sync_changelog_mode(tmp_path):
    dest = tmp_path / "buildings.parquet"
    result = SyncResult(
        status="updated",
        previous_snapshot_id="snap1",
        current_snapshot_id="snap1",
        path=dest,
        bytes_downloaded=0,
        sync_mode="changelog",
        previous_seq=100,
        current_seq=142,
        ops_applied=42,
    )
    _mock_client(
        info=lambda name: {"cdc_serving_tier": "changelog"},
        sync=lambda name, path, format, freshness, progress: result,
    )

    out = eolas_sync("nz_building_outlines", path=str(dest))
    assert out["sync_mode"] == "changelog"
    assert out["ops_applied"] == 42
    assert out["current_seq"] == 142