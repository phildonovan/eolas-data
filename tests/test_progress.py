"""Tests for download progress bar behaviour in Client and CLI.

Strategy:
- Mock sys.stdout.isatty() to control auto-detection.
- Mock the streaming HTTP response so no real network calls are made.
- Assert that tqdm's update() method was / was not called depending on settings.
- Use `responses` for CLI-level tests that go through the full stack.
"""
from __future__ import annotations

import pathlib
import sys
from unittest.mock import MagicMock, call, patch

import pytest
import responses as resp_lib
from typer.testing import CliRunner

from eolas_data import Client
from eolas_data import cli as cli_module
from eolas_data.cli import app

BASE = "https://api.eolas.fyi"

runner = CliRunner()

FAKE_PARQUET = b"PAR1" + b"\x00" * 12 + b"PAR1"
SNAPSHOT_V1 = "5503437996448954328"

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
    cfg_dir  = tmp_path / ".eolas"
    cfg_file = cfg_dir / "config.json"
    monkeypatch.setattr(cli_module, "CONFIG_DIR",  cfg_dir)
    monkeypatch.setattr(cli_module, "CONFIG_FILE", cfg_file)
    monkeypatch.delenv("EOLAS_API_KEY", raising=False)
    monkeypatch.delenv("VS_API_KEY",    raising=False)
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register_bulk_mocks(snapshot=SNAPSHOT_V1, body=FAKE_PARQUET):
    """Register the typical three-call sequence: metadata GET, HEAD, body GET."""
    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=b"", status=200,
                 headers={"X-Snapshot-Version": snapshot,
                           "Content-Length": str(len(body))})
    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=body,
                 content_type="application/octet-stream",
                 status=200,
                 headers={"Content-Length": str(len(body))})


def _register_download_mocks(body=FAKE_PARQUET):
    """Metadata GET + streaming body GET (no HEAD — download_bulk path)."""
    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=body,
                 content_type="application/octet-stream",
                 status=200,
                 headers={"Content-Length": str(len(body))})


# ---------------------------------------------------------------------------
# test_download_bulk_shows_progress_when_tty
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_download_bulk_shows_progress_when_tty(client, tmp_path):
    """When stdout is a TTY, tqdm.update should be called (bar shown)."""
    _register_download_mocks()

    dest = tmp_path / "nz_cpi.parquet"
    tqdm_update_calls = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            self._disabled = kwargs.get("disable", False)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n):
            tqdm_update_calls.append(n)

    with patch("sys.stdout") as mock_stdout, \
         patch("tqdm.auto.tqdm", FakeTqdm):
        mock_stdout.isatty.return_value = True
        client.download_bulk("nz_cpi", path=dest)

    assert len(tqdm_update_calls) > 0, "tqdm.update() must be called when isatty=True"
    assert dest.exists()


# ---------------------------------------------------------------------------
# test_download_bulk_silent_when_not_tty
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_download_bulk_silent_when_not_tty(client, tmp_path):
    """When stdout is NOT a TTY, tqdm should be disabled."""
    _register_download_mocks()

    dest = tmp_path / "nz_cpi.parquet"
    disabled_values = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            disabled_values.append(kwargs.get("disable", False))
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n):
            pass

    with patch("sys.stdout") as mock_stdout, \
         patch("tqdm.auto.tqdm", FakeTqdm):
        mock_stdout.isatty.return_value = False
        client.download_bulk("nz_cpi", path=dest)

    assert any(disabled_values), "tqdm should be constructed with disable=True when not a TTY"


# ---------------------------------------------------------------------------
# test_progress_kwarg_forces_show
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_progress_kwarg_forces_show(client, tmp_path):
    """progress=True forces bar even when not a TTY."""
    _register_download_mocks()

    dest = tmp_path / "nz_cpi.parquet"
    disabled_values = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            disabled_values.append(kwargs.get("disable", False))
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n):
            pass

    with patch("sys.stdout") as mock_stdout, \
         patch("tqdm.auto.tqdm", FakeTqdm):
        mock_stdout.isatty.return_value = False  # would normally suppress
        client.download_bulk("nz_cpi", path=dest, progress=True)

    assert not any(disabled_values), "tqdm must NOT be disabled when progress=True"


# ---------------------------------------------------------------------------
# test_progress_kwarg_forces_hide
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_progress_kwarg_forces_hide(client, tmp_path):
    """progress=False silences bar even when isatty() is True."""
    _register_download_mocks()

    dest = tmp_path / "nz_cpi.parquet"
    disabled_values = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            disabled_values.append(kwargs.get("disable", False))
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n):
            pass

    with patch("sys.stdout") as mock_stdout, \
         patch("tqdm.auto.tqdm", FakeTqdm):
        mock_stdout.isatty.return_value = True  # would normally show
        client.download_bulk("nz_cpi", path=dest, progress=False)

    assert all(disabled_values), "tqdm must be disabled when progress=False"


# ---------------------------------------------------------------------------
# test_download_bulk_missing_content_length
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_download_bulk_missing_content_length(client, tmp_path):
    """CDN may omit Content-Length — download must succeed with indeterminate bar."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=FAKE_PARQUET,
                 content_type="application/octet-stream",
                 status=200)
    # Deliberately no Content-Length header.

    dest = tmp_path / "nz_cpi.parquet"
    totals_seen = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            totals_seen.append(kwargs.get("total"))
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n):
            pass

    with patch("sys.stdout") as mock_stdout, \
         patch("tqdm.auto.tqdm", FakeTqdm):
        mock_stdout.isatty.return_value = True
        out = client.download_bulk("nz_cpi", path=dest, progress=True)

    assert out == dest
    assert dest.read_bytes() == FAKE_PARQUET
    assert totals_seen == [None]


# ---------------------------------------------------------------------------
# test_sync_bulk_unchanged_no_bar
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_sync_bulk_unchanged_no_bar(client, tmp_path):
    """When the sidecar matches the server snapshot, no tqdm instance is created."""
    import json as _json

    dest = tmp_path / "nz_cpi.parquet"
    dest.write_bytes(FAKE_PARQUET)

    # Write a matching sidecar.
    sidecar = pathlib.Path(str(dest) + ".eolas-meta.json")
    sidecar.write_text(_json.dumps({
        "schema_version": 1,
        "name": "nz_cpi",
        "snapshot_id": SNAPSHOT_V1,
        "format": "parquet",
        "freshness": "auto",
        "downloaded_at": "2026-05-24T00:00:00Z",
        "source_url": f"{BASE}/v1/bulk/statsnz/nz_cpi?format=parquet",
    }) + "\n")

    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=b"", status=200,
                 headers={"X-Snapshot-Version": SNAPSHOT_V1})

    tqdm_instances_created = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            tqdm_instances_created.append(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n):
            pass

    with patch("tqdm.auto.tqdm", FakeTqdm):
        result = client.sync_bulk("nz_cpi", path=dest, progress=True)

    assert result.status == "unchanged"
    assert len(tqdm_instances_created) == 0, \
        "No tqdm bar should appear when status=unchanged (no data transferred)"


# ---------------------------------------------------------------------------
# test_cli_no_progress_flag_download
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_cli_no_progress_flag_download(tmp_path, monkeypatch):
    """eolas download X --no-progress passes progress=False so no bar appears."""
    _register_download_mocks()

    monkeypatch.chdir(tmp_path)

    disabled_values = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            disabled_values.append(kwargs.get("disable", False))
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n):
            pass

    import eolas_data.client as client_mod
    with patch.object(client_mod, "sys") as mock_sys, \
         patch("tqdm.auto.tqdm", FakeTqdm):
        mock_sys.stdout.isatty.return_value = True  # would normally show
        result = runner.invoke(app, [
            "download", "nz_cpi",
            "--no-progress",
            "--api-key", "k",
        ])

    assert result.exit_code == 0
    assert all(disabled_values), "--no-progress must disable the tqdm bar"


# ---------------------------------------------------------------------------
# test_cli_no_progress_flag_sync
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_cli_no_progress_flag_sync(tmp_path, monkeypatch):
    """eolas sync X --no-progress suppresses the bar."""
    _register_bulk_mocks()
    monkeypatch.chdir(tmp_path)

    disabled_values = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            disabled_values.append(kwargs.get("disable", False))
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n):
            pass

    import eolas_data.client as client_mod
    with patch.object(client_mod, "sys") as mock_sys, \
         patch("tqdm.auto.tqdm", FakeTqdm):
        mock_sys.stdout.isatty.return_value = True
        result = runner.invoke(app, [
            "sync", "nz_cpi",
            "--no-progress",
            "--api-key", "k",
        ])

    assert result.exit_code == 0
    assert all(disabled_values), "--no-progress must disable the bar"


# ---------------------------------------------------------------------------
# test_watch_mode_always_silent
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_watch_mode_always_silent(tmp_path, monkeypatch):
    """--watch mode defaults to silent progress regardless of isatty."""
    import signal
    import time as time_mod

    dest = tmp_path / "nz_cpi.parquet"
    dest.write_bytes(FAKE_PARQUET)
    import json as _json
    sidecar = pathlib.Path(str(dest) + ".eolas-meta.json")
    sidecar.write_text(_json.dumps({
        "schema_version": 1, "name": "nz_cpi", "snapshot_id": SNAPSHOT_V1,
        "format": "parquet", "freshness": "auto",
        "downloaded_at": "2026-05-24T00:00:00Z",
        "source_url": f"{BASE}/v1/bulk/statsnz/nz_cpi?format=parquet",
    }) + "\n")

    resp_lib.add(resp_lib.GET,  f"{BASE}/v1/datasets/nz_cpi",
                 json=BULK_DATASET_META, status=200)
    resp_lib.add(resp_lib.HEAD, f"{BASE}/v1/bulk/statsnz/nz_cpi",
                 body=b"", status=200,
                 headers={"X-Snapshot-Version": SNAPSHOT_V1})

    tqdm_instances = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            tqdm_instances.append(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n):
            pass

    sleep_calls = []

    def fake_sleep(n):
        sleep_calls.append(n)
        import os
        os.kill(os.getpid(), signal.SIGINT)

    import eolas_data.cli as cli_mod
    monkeypatch.setattr(cli_mod.time, "sleep", fake_sleep)

    with patch("tqdm.auto.tqdm", FakeTqdm):
        result = runner.invoke(app, [
            "sync", "nz_cpi",
            "--watch", "1s",
            "--out", str(dest),
            "--api-key", "k",
        ])

    assert result.exit_code == 0
    # The unchanged path never creates a tqdm instance — but even if a
    # download were triggered, --watch must pass progress=False.
    assert len(tqdm_instances) == 0, "--watch must never show a progress bar"


# ---------------------------------------------------------------------------
# test_eolas_no_progress_env_suppresses_bar
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_eolas_no_progress_env_suppresses_bar(client, tmp_path, monkeypatch):
    """EOLAS_NO_PROGRESS=1 suppresses the bar even when isatty() is True."""
    _register_download_mocks()

    dest = tmp_path / "nz_cpi.parquet"
    disabled_values = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            disabled_values.append(kwargs.get("disable", False))
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n):
            pass

    monkeypatch.setenv("EOLAS_NO_PROGRESS", "1")

    with patch("sys.stdout") as mock_stdout, \
         patch("tqdm.auto.tqdm", FakeTqdm):
        mock_stdout.isatty.return_value = True  # would normally show
        client.download_bulk("nz_cpi", path=dest)

    assert all(disabled_values), "EOLAS_NO_PROGRESS=1 must disable tqdm even when isatty=True"


def test_progress_phase_selectors():
    from eolas_data.client import Client

    phases = Client._resolve_progress_phases("download")
    assert phases == {"download": True, "read": False}

    phases = Client._resolve_progress_phases("read")
    assert phases == {"download": False, "read": True}

    phases = Client._resolve_progress_phases("both")
    assert phases == {"download": True, "read": True}

    phases = Client._resolve_progress_phases("none")
    assert phases == {"download": False, "read": False}

    assert Client._resolve_show_progress("read", "read") is True
    assert Client._resolve_show_progress("download", "read") is False


def test_progress_auto_detect_uses_stderr_tty():
    """tqdm writes to stderr; progress should show when stderr is a TTY even if stdout is piped."""
    from unittest.mock import patch
    from eolas_data.client import Client

    with patch("sys.stdout.isatty", return_value=False), \
         patch("sys.stderr.isatty", return_value=True):
        assert Client._progress_auto_detect() is True

    with patch("sys.stdout.isatty", return_value=False), \
         patch("sys.stderr.isatty", return_value=False):
        assert Client._progress_auto_detect() is False


def test_progress_resolver_detects_jupyter():
    """Jupyter wraps stdout so isatty() is False, but the user IS interactive
    and tqdm.auto can render a widget. Resolver must return True when
    'ipykernel' is loaded — this was the bug causing zero progress feedback
    in VSCode/Jupyter notebooks."""
    import sys
    from unittest.mock import patch
    from eolas_data.client import Client

    # Simulate Jupyter: ipykernel imported + stdout NOT a TTY
    with patch.dict(sys.modules, {"ipykernel": __import__("os")}), \
         patch("sys.stdout.isatty", return_value=False):
        assert Client._resolve_show_progress(None) is True, \
            "ipykernel-loaded session must show progress regardless of isatty"

    # Sanity: explicit progress=False still wins
    with patch.dict(sys.modules, {"ipykernel": __import__("os")}):
        assert Client._resolve_show_progress(False) is False

    # Sanity: EOLAS_NO_PROGRESS still wins
    import os
    with patch.dict(sys.modules, {"ipykernel": __import__("os")}), \
         patch.dict(os.environ, {"EOLAS_NO_PROGRESS": "1"}):
        assert Client._resolve_show_progress(None) is False
