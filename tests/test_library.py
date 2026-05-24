"""Tests for the library directory resolution / config persistence.

Precedence chain being tested:
  1. Explicit cache_dir= arg to get_local()      (test_get_local_explicit_cache_dir_overrides_library)
  2. EOLAS_LIBRARY env var                       (test_resolve_library_env_var)
  3. library_dir in ~/.eolas/config.json         (test_resolve_library_config_file)
  4. Interactive prompt (TTY only, once/session) (test_prompt_skipped_when_not_tty, test_prompt_writes_choice_to_config, test_prompt_only_once_per_session)
  5. ~/.cache/eolas/ fallback                    (test_resolve_library_fallback_to_cache)

CLI round-trip:                                  (test_cli_library_set_status_clear)
"""
from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Helpers to ensure a clean module-level state for each test
# ---------------------------------------------------------------------------

def _reset_library_module():
    """Reset the per-session mutable state in eolas_data.library."""
    import eolas_data.library as lib
    lib._prompt_done = False
    lib._headless_info_emitted = False


@pytest.fixture(autouse=True)
def clean_library_state():
    """Reset library module state before every test."""
    _reset_library_module()
    yield
    _reset_library_module()


# ---------------------------------------------------------------------------
# test_resolve_library_env_var
# ---------------------------------------------------------------------------

def test_resolve_library_env_var(tmp_path, monkeypatch):
    """EOLAS_LIBRARY env var is used when set, even if a config file exists."""
    import eolas_data.library as lib

    env_dir = tmp_path / "from_env"
    monkeypatch.setenv("EOLAS_LIBRARY", str(env_dir))

    # Also set a config value — env var should win
    cfg_dir = tmp_path / "from_config"
    monkeypatch.setattr(lib, "_CONFIG_FILE", tmp_path / "config.json")
    (tmp_path / "config.json").write_text(
        json.dumps({"library_dir": str(cfg_dir)}) + "\n"
    )

    result = lib.resolve_library_dir(interactive=False)
    assert result == env_dir.resolve()


# ---------------------------------------------------------------------------
# test_resolve_library_config_file
# ---------------------------------------------------------------------------

def test_resolve_library_config_file(tmp_path, monkeypatch):
    """library_dir from config.json is used when env var is not set."""
    import eolas_data.library as lib

    monkeypatch.delenv("EOLAS_LIBRARY", raising=False)
    cfg_dir = tmp_path / "from_config"
    monkeypatch.setattr(lib, "_CONFIG_FILE", tmp_path / "config.json")
    (tmp_path / "config.json").write_text(
        json.dumps({"api_key": "vs_test", "library_dir": str(cfg_dir)}) + "\n"
    )

    result = lib.resolve_library_dir(interactive=False)
    assert result == cfg_dir.resolve()


# ---------------------------------------------------------------------------
# test_resolve_library_fallback_to_cache
# ---------------------------------------------------------------------------

def test_resolve_library_fallback_to_cache(tmp_path, monkeypatch):
    """With nothing configured, resolution falls through to ~/.cache/eolas/."""
    import eolas_data.library as lib

    monkeypatch.delenv("EOLAS_LIBRARY", raising=False)
    monkeypatch.setattr(lib, "_CONFIG_FILE", tmp_path / "no_config.json")

    result = lib.resolve_library_dir(interactive=False)
    # Should be the home-based fallback (not tmp_path)
    assert result == lib._FALLBACK_DIR


# ---------------------------------------------------------------------------
# test_prompt_skipped_when_not_tty
# ---------------------------------------------------------------------------

def test_prompt_skipped_when_not_tty(tmp_path, monkeypatch, caplog):
    """Non-TTY stdin: prompt is skipped; INFO log emitted; fallback used."""
    import eolas_data.library as lib
    import logging

    monkeypatch.delenv("EOLAS_LIBRARY", raising=False)
    monkeypatch.setattr(lib, "_CONFIG_FILE", tmp_path / "no_config.json")

    # _is_tty() must return False
    with patch.object(lib, "_is_tty", return_value=False):
        with caplog.at_level(logging.INFO, logger="eolas_data"):
            result = lib.resolve_library_dir(interactive=True)

    assert result == lib._FALLBACK_DIR
    # One-time INFO log should have been emitted
    assert any("transient" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# test_prompt_writes_choice_to_config
# ---------------------------------------------------------------------------

def test_prompt_writes_choice_to_config(tmp_path, monkeypatch):
    """TTY + user picks option 2 → config written, path returned."""
    import eolas_data.library as lib

    monkeypatch.delenv("EOLAS_LIBRARY", raising=False)
    monkeypatch.setattr(lib, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(lib, "_CONFIG_FILE", tmp_path / "config.json")

    with (
        patch.object(lib, "_is_tty", return_value=True),
        patch("builtins.input", return_value="2"),  # pick ./eolas-library
        patch("builtins.print"),                    # suppress output
    ):
        result = lib.resolve_library_dir(interactive=True)

    # Config file should now contain library_dir
    assert (tmp_path / "config.json").exists()
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert "library_dir" in cfg

    # The result should match the written value
    assert result == pathlib.Path(cfg["library_dir"])

    # Second call in the same session must NOT re-prompt (_prompt_done=True)
    with (
        patch.object(lib, "_is_tty", return_value=True),
        patch("builtins.input", side_effect=AssertionError("prompt fired again")),
    ):
        result2 = lib.resolve_library_dir(interactive=True)

    # Now reads from config (written on first call)
    assert result2 == result


# ---------------------------------------------------------------------------
# test_prompt_only_once_per_session
# ---------------------------------------------------------------------------

def test_prompt_only_once_per_session(tmp_path, monkeypatch):
    """Even with repeated resolve_library_dir() calls, the prompt fires once."""
    import eolas_data.library as lib

    monkeypatch.delenv("EOLAS_LIBRARY", raising=False)
    monkeypatch.setattr(lib, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(lib, "_CONFIG_FILE", tmp_path / "config.json")

    prompt_count = {"n": 0}
    original_input = __builtins__["input"] if isinstance(__builtins__, dict) else input

    def counting_input(prompt_text=""):
        prompt_count["n"] += 1
        return "1"  # pick ~/eolas-library

    with (
        patch.object(lib, "_is_tty", return_value=True),
        patch("builtins.input", side_effect=counting_input),
        patch("builtins.print"),
    ):
        lib.resolve_library_dir(interactive=True)
        lib.resolve_library_dir(interactive=True)
        lib.resolve_library_dir(interactive=True)

    # Prompt should have fired exactly once
    assert prompt_count["n"] == 1


# ---------------------------------------------------------------------------
# test_get_local_explicit_cache_dir_overrides_library
# ---------------------------------------------------------------------------

def test_get_local_explicit_cache_dir_overrides_library(tmp_path, monkeypatch):
    """Explicit cache_dir= to get_local() wins over any library config."""
    from eolas_data import Client, SyncResult
    import eolas_data.library as lib

    monkeypatch.setenv("EOLAS_LIBRARY", str(tmp_path / "from_env"))
    explicit_dir = tmp_path / "explicit"

    client = Client("eolas_testkey", base_url="https://api.eolas.fyi")

    synced_dirs = []

    def fake_sync_bulk(name, *, path, format, freshness, progress=None):
        synced_dirs.append(path.parent)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return SyncResult(
            status="downloaded",
            previous_snapshot_id=None,
            current_snapshot_id="snap1",
            path=path,
            bytes_downloaded=1024,
        )

    with (
        patch.object(client, "info", return_value={"name": "nz_cpi"}),
        patch.object(client, "sync_bulk", side_effect=fake_sync_bulk),
        patch("pandas.read_parquet", return_value=pd.DataFrame({"v": [1]})),
    ):
        client.get_local("nz_cpi", cache_dir=str(explicit_dir))

    assert len(synced_dirs) == 1
    assert synced_dirs[0] == explicit_dir.resolve()


# ---------------------------------------------------------------------------
# test_cli_library_set_status_clear
# ---------------------------------------------------------------------------

def test_cli_library_set_status_clear(tmp_path, monkeypatch):
    """Round-trip: library set / status / clear via CLI commands."""
    from eolas_data.cli import app
    import eolas_data.library as lib

    runner = CliRunner()

    monkeypatch.delenv("EOLAS_LIBRARY", raising=False)
    monkeypatch.setattr(lib, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(lib, "_CONFIG_FILE", tmp_path / "config.json")
    # Ensure the cli module uses the same patched path
    monkeypatch.setattr("eolas_data.library._CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("eolas_data.library._CONFIG_DIR", tmp_path)

    lib_path = tmp_path / "my-eolas"

    # --- set ---
    result = runner.invoke(app, ["library", "set", str(lib_path)])
    assert result.exit_code == 0, result.output
    assert str(lib_path) in result.output

    # Config file should contain library_dir
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg.get("library_dir") == str(lib_path)

    # --- status ---
    result = runner.invoke(app, ["library", "status"])
    assert result.exit_code == 0, result.output
    assert str(lib_path) in result.output

    # --- clear ---
    result = runner.invoke(app, ["library", "clear"])
    assert result.exit_code == 0, result.output

    cfg_after = json.loads((tmp_path / "config.json").read_text())
    assert "library_dir" not in cfg_after

    # --- status after clear (should show fallback) ---
    result = runner.invoke(app, ["library", "status"])
    assert result.exit_code == 0, result.output
    assert "fallback" in result.output.lower() or ".cache" in result.output
