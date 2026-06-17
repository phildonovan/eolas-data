"""Tests for OS-keyring API-key storage (Python client + CLI).

Strategy:
- All keyring calls are patched via unittest.mock.patch so tests run without a
  real OS keyring backend (and in headless CI without the 'secure' extra).
- The CLI is exercised through Typer's CliRunner (same as test_cli.py).
- Client precedence is tested by controlling env vars + the keyring mock.
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import responses as resp_lib
from typer.testing import CliRunner

import eolas_data.client as client_module
from eolas_data import cli as cli_module
from eolas_data.cli import app
from eolas_data.client import Client, _KEYRING_SERVICE, _KEYRING_USERNAME

runner = CliRunner()
BASE = "https://api.eolas.fyi"

# Fake key used throughout
FAKE_KEY = "vs_testkey_keyring_12345"


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Redirect ~/.eolas to a tmp dir and clear env var for every test."""
    cfg_dir  = tmp_path / ".eolas"
    cfg_file = cfg_dir / "config.json"
    monkeypatch.setattr(cli_module, "CONFIG_DIR",  cfg_dir)
    monkeypatch.setattr(cli_module, "CONFIG_FILE", cfg_file)
    monkeypatch.delenv("EOLAS_API_KEY", raising=False)
    yield


# ────────────────────────────────────────────────────────────────────────────
# _keyring_get helper
# ────────────────────────────────────────────────────────────────────────────

def test_keyring_get_returns_password_when_set():
    with patch("keyring.get_password", return_value=FAKE_KEY):
        result = client_module._keyring_get()
    assert result == FAKE_KEY


def test_keyring_get_returns_empty_when_no_entry():
    with patch("keyring.get_password", return_value=None):
        result = client_module._keyring_get()
    assert result == ""


def test_keyring_get_returns_empty_when_keyring_not_installed():
    """ImportError (keyring package absent) must NOT raise — fall through."""
    with patch.dict("sys.modules", {"keyring": None}):
        result = client_module._keyring_get()
    assert result == ""


def test_keyring_get_returns_empty_on_backend_error():
    """Any exception from the keyring backend must be silently swallowed."""
    with patch("keyring.get_password", side_effect=RuntimeError("dbus not available")):
        result = client_module._keyring_get()
    assert result == ""


# ────────────────────────────────────────────────────────────────────────────
# Client precedence chain
# ────────────────────────────────────────────────────────────────────────────

def test_client_explicit_arg_beats_all(monkeypatch):
    """Explicit api_key= beats env var and keyring."""
    monkeypatch.setenv("EOLAS_API_KEY", "vs_from_env")
    with patch("keyring.get_password", return_value="vs_from_keyring"):
        c = Client(api_key="vs_explicit")
    assert c._key == "vs_explicit"


def test_client_env_beats_keyring(monkeypatch):
    """Env var beats keyring when no explicit arg given."""
    monkeypatch.setenv("EOLAS_API_KEY", "vs_from_env")
    with patch("keyring.get_password", return_value="vs_from_keyring"):
        c = Client()
    assert c._key == "vs_from_env"


def test_client_uses_keyring_when_env_absent(monkeypatch):
    """Keyring is used when env var is not set."""
    monkeypatch.delenv("EOLAS_API_KEY", raising=False)
    with patch("keyring.get_password", return_value=FAKE_KEY):
        c = Client()
    assert c._key == FAKE_KEY


def test_client_keyring_correct_service_and_username(monkeypatch):
    """keyring.get_password is called with service='eolas', username='api-key'."""
    monkeypatch.delenv("EOLAS_API_KEY", raising=False)
    with patch("keyring.get_password", return_value=FAKE_KEY) as mock_get:
        Client()
    mock_get.assert_called_once_with(_KEYRING_SERVICE, _KEYRING_USERNAME)


def test_client_falls_through_when_keyring_missing(monkeypatch, tmp_path):
    """Missing keyring package does not crash Client(); falls through to empty key."""
    monkeypatch.delenv("EOLAS_API_KEY", raising=False)
    with patch.dict("sys.modules", {"keyring": None}), \
         patch.object(client_module, "_config_file_get", return_value=""):
        c = Client()
    assert c._key == ""


def test_client_config_file_fallback(monkeypatch, tmp_path):
    """Config file is still used when env var and keyring are both absent (CLI path)."""
    # This tests the CLI's _load_api_key, not Client directly (Client has no config-file lookup).
    cfg_dir  = tmp_path / ".eolas"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.json"
    cfg_file.write_text(json.dumps({"api_key": "vs_from_config"}))
    monkeypatch.setattr(cli_module, "CONFIG_DIR",  cfg_dir)
    monkeypatch.setattr(cli_module, "CONFIG_FILE", cfg_file)
    monkeypatch.delenv("EOLAS_API_KEY", raising=False)
    with patch("keyring.get_password", return_value=None):
        key = cli_module._load_api_key()
    assert key == "vs_from_config"


# ────────────────────────────────────────────────────────────────────────────
# CLI: eolas auth save-key
# ────────────────────────────────────────────────────────────────────────────

def test_auth_save_key_interactive_prompt():
    """save-key prompts for the key when called interactively."""
    with patch("keyring.set_password") as mock_set:
        result = runner.invoke(app, ["auth", "save-key"], input=FAKE_KEY + "\n")
    assert result.exit_code == 0
    mock_set.assert_called_once_with(_KEYRING_SERVICE, _KEYRING_USERNAME, FAKE_KEY)
    assert "saved" in result.output


def test_auth_save_key_positional_arg():
    """save-key accepts the key as a positional argument (non-interactive / scripted)."""
    with patch("keyring.set_password") as mock_set:
        result = runner.invoke(app, ["auth", "save-key", FAKE_KEY])
    assert result.exit_code == 0
    mock_set.assert_called_once_with(_KEYRING_SERVICE, _KEYRING_USERNAME, FAKE_KEY)
    assert "saved" in result.output


def test_auth_save_key_fails_gracefully_without_keyring_package():
    """save-key gives a helpful error when the 'secure' extra is not installed."""
    with patch.dict("sys.modules", {"keyring": None}):
        result = runner.invoke(app, ["auth", "save-key", FAKE_KEY])
    assert result.exit_code != 0
    assert "secure" in result.output or "secure" in (result.exception or "")


def test_auth_save_key_backend_error_exits_nonzero():
    """A keyring backend exception exits non-zero with an error message."""
    with patch("keyring.set_password", side_effect=RuntimeError("backend unavailable")):
        result = runner.invoke(app, ["auth", "save-key", FAKE_KEY])
    assert result.exit_code != 0


# ────────────────────────────────────────────────────────────────────────────
# CLI: eolas auth clear-key
# ────────────────────────────────────────────────────────────────────────────

def test_auth_clear_key_removes_entry():
    """clear-key calls keyring.delete_password with the correct coords."""
    with patch("keyring.delete_password") as mock_del, \
         patch("keyring.errors.PasswordDeleteError", Exception):
        result = runner.invoke(app, ["auth", "clear-key"])
    assert result.exit_code == 0
    mock_del.assert_called_once_with(_KEYRING_SERVICE, _KEYRING_USERNAME)
    assert "cleared" in result.output


def test_auth_clear_key_nothing_to_clear():
    """clear-key is graceful when no entry exists (PasswordDeleteError)."""
    import keyring.errors

    with patch("keyring.delete_password", side_effect=keyring.errors.PasswordDeleteError):
        result = runner.invoke(app, ["auth", "clear-key"])
    assert result.exit_code == 0
    assert "nothing" in result.output or "no API key" in result.output


# ────────────────────────────────────────────────────────────────────────────
# CLI: eolas auth status — includes keyring source
# ────────────────────────────────────────────────────────────────────────────

def test_auth_status_shows_keyring_source(monkeypatch):
    """auth status reports 'OS keyring' when that's the winning source."""
    monkeypatch.delenv("EOLAS_API_KEY", raising=False)
    with patch("keyring.get_password", return_value=FAKE_KEY):
        result = runner.invoke(app, ["auth", "status"])
    assert result.exit_code == 0
    assert "keyring" in result.output.lower()
    # Key is masked — first 8 chars shown
    assert FAKE_KEY[:8] in result.output


def test_auth_status_env_beats_keyring(monkeypatch):
    """auth status reports env var source when EOLAS_API_KEY is set."""
    monkeypatch.setenv("EOLAS_API_KEY", "vs_envkey_test")
    with patch("keyring.get_password", return_value=FAKE_KEY):
        result = runner.invoke(app, ["auth", "status"])
    assert result.exit_code == 0
    assert "EOLAS_API_KEY" in result.output


# ────────────────────────────────────────────────────────────────────────────
# Save → get → clear round-trip
# ────────────────────────────────────────────────────────────────────────────

def test_save_get_clear_round_trip(monkeypatch):
    """save-key stores a key that _keyring_get can then read; clear-key removes it."""
    store: dict = {}

    def fake_set(service, username, password):
        store[(service, username)] = password

    def fake_get(service, username):
        return store.get((service, username))

    def fake_delete(service, username):
        import keyring.errors
        if (service, username) not in store:
            raise keyring.errors.PasswordDeleteError("not found")
        del store[(service, username)]

    monkeypatch.delenv("EOLAS_API_KEY", raising=False)

    with patch("keyring.set_password", side_effect=fake_set), \
         patch("keyring.get_password", side_effect=fake_get), \
         patch("keyring.delete_password", side_effect=fake_delete), \
         patch.object(client_module, "_config_file_get", return_value=""):

        # Save
        result = runner.invoke(app, ["auth", "save-key", FAKE_KEY])
        assert result.exit_code == 0

        # Read via Client
        c = Client()
        assert c._key == FAKE_KEY

        # Read via _keyring_get
        assert client_module._keyring_get() == FAKE_KEY

        # Clear
        result = runner.invoke(app, ["auth", "clear-key"])
        assert result.exit_code == 0

        # Now empty
        assert client_module._keyring_get() == ""
        c2 = Client()
        assert c2._key == ""
