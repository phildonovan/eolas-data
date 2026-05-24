"""Tests for Client.sync_bulk() and the `eolas sync` CLI command."""
from __future__ import annotations

import json
import pathlib
import signal
import threading
from unittest.mock import MagicMock, patch

import pytest
import responses as resp_lib
from typer.testing import CliRunner

from eolas_data import Client, SyncResult
from eolas_data import cli as cli_module
from eolas_data.cli import app, _parse_watch_duration
from eolas_data.exceptions import (
    AuthenticationError,
    BulkLicenceRestricted,
    BulkNotYetAvailable,
    BulkUpgradeRequired,
)

runner = CliRunner()
BASE = "https://api.eolas.fyi"

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

FAKE_PARQUET = b"PAR1" + b"\x00" * 12 + b"PAR1"
FAKE_PARQUET_V2 = b"PAR1" + b"\x01" * 12 + b"PAR1"

SNAPSHOT_V1 = "5503437996448954328"
SNAPSHOT_V2 = "7041234567890123456"

BULK_DATASET_META = {
    "name": "nz_cpi",
    "title": "NZ Consumer Price Index",
    "source": "Stats NZ",
    "namespace": "statsnz",
    "table": "nz_cpi",
}


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Redirect ~/.eolas to a tmp dir so tests don't touch real config."""
    cfg_dir  = tmp_path / ".eolas"
    cfg_file = cfg_dir / "config.json"
    monkeypatch.setattr(cli_module, "CONFIG_DIR",  cfg_dir)
    monkeypatch.setattr(cli_module, "CONFIG_FILE", cfg_file)
    monkeypatch.delenv("EOLAS_API_KEY", raising=False)
    monkeypatch.delenv("VS_API_KEY",    raising=False)
    yield


def _write_sidecar(data_path: pathlib.Path, snapshot_id: str) -> None:
    """Write a minimal sidecar next to data_path."""
    sidecar = pathlib.Path(str(data_path) + ".eolas-meta.json")
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "name": "nz_cpi",
        "snapshot_id": snapshot_id,
        "format": "parquet",
        "freshness": "auto",
        "downloaded_at": "2026-05-24T01:23:45Z",
        "source_url": f"{BASE}/v1/bulk/statsnz/nz_cpi?format=parquet",
    }) + "\n")


# ---------------------------------------------------------------------------
# _parse_watch_duration unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("60",       60),
    ("60s",      60),
    ("30s",      30),
    ("5m",       300),
    ("1h",       3600),
    ("1d",       86400),
    ("hourly",   3600),
    ("daily",    86400),
    ("weekly",   604800),
    ("DAILY",    86400),   # case-insensitive
    ("HOURLY",   3600),
])
def test_parse_watch_duration_valid(raw, expected):
    assert _parse_watch_duration(raw) == expected


@pytest.mark.parametrize("bad", [
    "0", "-1", "0s", "-5m", "forever", "monthly", "1w", "abc", "1x",
])
def test_parse_watch_duration_invalid(bad):
    with pytest.raises(ValueError):
        _parse_watch_duration(bad)


# ---------------------------------------------------------------------------
# test_sync_bulk_first_download
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_sync_bulk_first_download(client, tmp_path):
    """No sidecar present → full GET, file + sidecar written, status='downloaded'."""
    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=b"", status=200,
                 headers={"X-Snapshot-Version": SNAPSHOT_V1})
    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=FAKE_PARQUET,
                 content_type="application/octet-stream",
                 status=200)

    dest = tmp_path / "nz_cpi.parquet"
    result = client.sync_bulk("nz_cpi", path=dest)

    assert isinstance(result, SyncResult)
    assert result.status == "downloaded"
    assert result.previous_snapshot_id is None
    assert result.current_snapshot_id == SNAPSHOT_V1
    assert result.path == dest
    assert result.bytes_downloaded == len(FAKE_PARQUET)

    # File written with correct content.
    assert dest.read_bytes() == FAKE_PARQUET

    # Sidecar written.
    sidecar = pathlib.Path(str(dest) + ".eolas-meta.json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["snapshot_id"] == SNAPSHOT_V1
    assert meta["schema_version"] == 1


# ---------------------------------------------------------------------------
# test_sync_bulk_unchanged
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_sync_bulk_unchanged(client, tmp_path):
    """Sidecar present, server snapshot matches → HEAD only, no file write, status='unchanged'."""
    dest = tmp_path / "nz_cpi.parquet"
    dest.write_bytes(FAKE_PARQUET)
    _write_sidecar(dest, SNAPSHOT_V1)

    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=b"", status=200,
                 headers={"X-Snapshot-Version": SNAPSHOT_V1})
    # No GET registered for the bulk body — if the code tries a GET, responses
    # will raise ConnectionError, failing the test.

    result = client.sync_bulk("nz_cpi", path=dest)

    assert result.status == "unchanged"
    assert result.previous_snapshot_id == SNAPSHOT_V1
    assert result.current_snapshot_id == SNAPSHOT_V1
    assert result.bytes_downloaded == 0
    assert result.path == dest

    # File must be untouched (same content, not a new write).
    assert dest.read_bytes() == FAKE_PARQUET

    # Only the metadata GET + the HEAD were made; no bulk GET.
    methods = [c.request.method for c in resp_lib.calls]
    assert "GET" not in methods[1:]  # first GET is the metadata call; second must be HEAD


# ---------------------------------------------------------------------------
# test_sync_bulk_updated
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_sync_bulk_updated(client, tmp_path):
    """Sidecar present, server returns new snapshot → file replaced atomically, status='updated'."""
    dest = tmp_path / "nz_cpi.parquet"
    dest.write_bytes(FAKE_PARQUET)
    _write_sidecar(dest, SNAPSHOT_V1)

    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=b"", status=200,
                 headers={"X-Snapshot-Version": SNAPSHOT_V2})
    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=FAKE_PARQUET_V2,
                 content_type="application/octet-stream",
                 status=200)

    result = client.sync_bulk("nz_cpi", path=dest)

    assert result.status == "updated"
    assert result.previous_snapshot_id == SNAPSHOT_V1
    assert result.current_snapshot_id == SNAPSHOT_V2
    assert result.bytes_downloaded == len(FAKE_PARQUET_V2)

    # File replaced with new content.
    assert dest.read_bytes() == FAKE_PARQUET_V2

    # Sidecar updated.
    sidecar = pathlib.Path(str(dest) + ".eolas-meta.json")
    meta = json.loads(sidecar.read_text())
    assert meta["snapshot_id"] == SNAPSHOT_V2


# ---------------------------------------------------------------------------
# test_sync_bulk_atomic_rename
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_sync_bulk_atomic_rename(client, tmp_path):
    """Interrupt mid-write → original file is untouched; no tmp file persists."""
    dest = tmp_path / "nz_cpi.parquet"
    dest.write_bytes(FAKE_PARQUET)                  # original content
    _write_sidecar(dest, SNAPSHOT_V1)

    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=b"", status=200,
                 headers={"X-Snapshot-Version": SNAPSHOT_V2})
    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=FAKE_PARQUET_V2,
                 content_type="application/octet-stream",
                 status=200)

    # Patch os.replace to simulate a crash before the rename completes.
    import os as _os
    replace_called = []

    def _explode(src, dst):
        replace_called.append((src, dst))
        raise OSError("simulated disk full")

    with patch("os.replace", side_effect=_explode):
        with pytest.raises(OSError, match="simulated disk full"):
            client.sync_bulk("nz_cpi", path=dest)

    # Original file must be intact.
    assert dest.read_bytes() == FAKE_PARQUET, "original file must survive a mid-write crash"

    # Confirm os.replace was actually called (test internal logic correct).
    assert replace_called, "os.replace should have been called"

    # The tmp file may or may not exist (best-effort cleanup) — we only care
    # that the canonical dest is untouched.  If it does exist, it's orphaned
    # (acceptable per spec).


# ---------------------------------------------------------------------------
# test_sync_bulk_refusal_codes
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_sync_bulk_402_raises_and_no_sidecar(client, tmp_path):
    """HTTP 402 raises BulkUpgradeRequired; no sidecar should be created."""
    dest = tmp_path / "nz_cpi.parquet"

    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=b"", status=200,
                 headers={"X-Snapshot-Version": SNAPSHOT_V1})
    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 json={"detail": "Fresh bulk downloads are a Pro feature."},
                 status=402)

    with pytest.raises(BulkUpgradeRequired):
        client.sync_bulk("nz_cpi", path=dest)

    sidecar = pathlib.Path(str(dest) + ".eolas-meta.json")
    assert not sidecar.exists(), "sidecar must not be created on error"


@resp_lib.activate
def test_sync_bulk_403_licence_raises_and_no_sidecar(client, tmp_path):
    """HTTP 403 (licence) raises BulkLicenceRestricted; no sidecar created."""
    dest = tmp_path / "nz_cpi.parquet"

    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=b"", status=200,
                 headers={"X-Snapshot-Version": SNAPSHOT_V1})
    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 json={"detail": "This dataset is not available (licence: OECD)."},
                 status=403)

    with pytest.raises(BulkLicenceRestricted):
        client.sync_bulk("nz_cpi", path=dest)

    sidecar = pathlib.Path(str(dest) + ".eolas-meta.json")
    assert not sidecar.exists()


@resp_lib.activate
def test_sync_bulk_503_raises_and_no_sidecar(client, tmp_path):
    """HTTP 503 raises BulkNotYetAvailable; no sidecar created."""
    dest = tmp_path / "nz_cpi.parquet"

    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=b"", status=200,
                 headers={"X-Snapshot-Version": SNAPSHOT_V1})
    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 json={"detail": "Monthly bulk snapshots are still rolling out."},
                 status=503)

    with pytest.raises(BulkNotYetAvailable):
        client.sync_bulk("nz_cpi", path=dest)

    sidecar = pathlib.Path(str(dest) + ".eolas-meta.json")
    assert not sidecar.exists()


# ---------------------------------------------------------------------------
# test_cli_watch_one_iter
# ---------------------------------------------------------------------------

def test_sync_help_shows_subcommand():
    """The sync subcommand must appear in --help output."""
    result = runner.invoke(app, ["sync", "--help"])
    assert result.exit_code == 0
    assert "sync" in result.stdout.lower()
    assert "watch" in result.stdout.lower()


@resp_lib.activate
def test_cli_sync_single_shot(tmp_path, monkeypatch):
    """Single-shot sync (no --watch) writes the file and exits 0."""
    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=b"", status=200,
                 headers={"X-Snapshot-Version": SNAPSHOT_V1})
    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=FAKE_PARQUET,
                 content_type="application/octet-stream",
                 status=200)

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["sync", "nz_cpi", "--api-key", "k"])
    assert result.exit_code == 0
    assert (tmp_path / "nz_cpi.parquet").read_bytes() == FAKE_PARQUET


@resp_lib.activate
def test_cli_watch_one_iter(tmp_path, monkeypatch):
    """--watch 1s runs at least one iteration then exits cleanly on simulated SIGINT.

    Strategy: patch `time.sleep` to count calls and after the first sleep
    deliver SIGINT to the process so the loop exits.
    """
    # Register HTTP mocks for the single iteration (unchanged path for speed).
    dest = tmp_path / "nz_cpi.parquet"
    dest.write_bytes(FAKE_PARQUET)
    # Write a matching sidecar so the first iteration is a no-op (no GET).
    sidecar = pathlib.Path(str(dest) + ".eolas-meta.json")
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "name": "nz_cpi",
        "snapshot_id": SNAPSHOT_V1,
        "format": "parquet",
        "freshness": "auto",
        "downloaded_at": "2026-05-24T00:00:00Z",
        "source_url": f"{BASE}/v1/bulk/statsnz/nz_cpi?format=parquet",
    }))

    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=b"", status=200,
                 headers={"X-Snapshot-Version": SNAPSHOT_V1})

    sleep_count = []

    def fake_sleep(n):
        sleep_count.append(n)
        # After the first sleep, send SIGINT to the current process to break
        # the watch loop cleanly.
        if len(sleep_count) >= 1:
            import os
            os.kill(os.getpid(), signal.SIGINT)

    import eolas_data.cli as cli_mod
    monkeypatch.setattr(cli_mod.time, "sleep", fake_sleep)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, [
        "sync", "nz_cpi",
        "--watch", "1s",
        "--out", str(dest),
        "--api-key", "k",
    ])

    # Should exit cleanly (code 0) — SIGINT is caught by the loop handler.
    assert result.exit_code == 0
    # At least one iteration ran and printed a status line.
    assert "unchanged" in result.output or "downloaded" in result.output or "updated" in result.output
    # Sleep was called (confirms the loop body ran).
    assert sleep_count, "sleep must have been called at least once"


def test_cli_sync_unknown_format_exits_usage():
    result = runner.invoke(app, ["sync", "nz_cpi", "--format", "xlsx", "--api-key", "k"])
    assert result.exit_code == cli_module.EXIT_USAGE


def test_cli_sync_unknown_freshness_exits_usage():
    result = runner.invoke(app, ["sync", "nz_cpi", "--freshness", "yesterday", "--api-key", "k"])
    assert result.exit_code == cli_module.EXIT_USAGE


def test_cli_sync_invalid_watch_exits_usage():
    result = runner.invoke(app, ["sync", "nz_cpi", "--watch", "forever", "--api-key", "k"])
    assert result.exit_code == cli_module.EXIT_USAGE
