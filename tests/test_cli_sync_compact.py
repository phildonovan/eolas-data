"""Tests for the pipeline-sync and compact CLI commands.

Covers:
- eolas sync <name> --library DIR
- eolas sync --library DIR --datasets X Y Z
- eolas sync --library DIR --all
- eolas compact <dataset_dir>
- eolas compact --library DIR
- eolas compact --library DIR --dataset NAME
- Help text renders without error for both commands
- Exit codes on bad inputs
"""
from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest
import responses as resp_lib
from typer.testing import CliRunner

from eolas_data import cli as cli_module
from eolas_data.cli import app
from eolas_data.sync import MANIFEST_FILENAME, ManifestEntry, Manifest, write_manifest
from eolas_data.sync.sync import SyncResult
from eolas_data.sync.compact import CompactResult

runner = CliRunner()
BASE = "https://api.eolas.fyi"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


def _make_manifest(dataset_dir: pathlib.Path, name: str = "nz_cpi", snap_id: int = 1001) -> None:
    """Write a minimal manifest to dataset_dir."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    entry = ManifestEntry(
        snapshot_id=snap_id,
        kind="snapshot",
        file="snapshot-2026-05-27.parquet",
        synced_at="2026-05-27T10:00:00Z",
        rows=100,
    )
    m = Manifest(
        dataset=name,
        snapshots=[entry],
        current_snapshot=snap_id,
        format="parquet",
        schema_version=1,
    )
    write_manifest(m, dataset_dir / MANIFEST_FILENAME)


_SYNC_RESULT_FULL = SyncResult(
    status="snapshot_full",
    dataset="nz_cpi",
    library_dir=pathlib.Path("/tmp/lib"),
    bytes_downloaded=1024,
    rows_added=100,
    files_added=1,
)

_SYNC_RESULT_UNCHANGED = SyncResult(
    status="unchanged",
    dataset="nz_cpi",
    library_dir=pathlib.Path("/tmp/lib"),
    bytes_downloaded=0,
    rows_added=0,
    files_added=0,
)

_COMPACT_RESULT = CompactResult(
    dataset="nz_cpi",
    rows_before=200,
    rows_after=200,
    files_before=3,
    files_after=1,
    bytes_saved=512,
)

_COMPACT_RESULT_NOOP = CompactResult(
    dataset="nz_cpi",
    rows_before=100,
    rows_after=100,
    files_before=1,
    files_after=1,
    bytes_saved=0,
)


# ---------------------------------------------------------------------------
# Help text renders
# ---------------------------------------------------------------------------

def test_sync_help_renders():
    result = runner.invoke(app, ["sync", "--help"])
    assert result.exit_code == 0
    assert "library" in result.stdout.lower()
    assert "pipeline" in result.stdout.lower() or "dataset" in result.stdout.lower()


def test_compact_help_renders():
    result = runner.invoke(app, ["compact", "--help"])
    assert result.exit_code == 0
    assert "compact" in result.stdout.lower()
    assert "library" in result.stdout.lower()


# ---------------------------------------------------------------------------
# eolas sync <name> --library DIR
# ---------------------------------------------------------------------------

def test_sync_single_dataset_with_library(tmp_path):
    """eolas sync nz_cpi --library <dir> calls client.sync() and prints result."""
    lib = tmp_path / "lib"

    with patch.object(cli_module, "_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.sync.return_value = _SYNC_RESULT_FULL
        mock_client_factory.return_value = mock_client

        result = runner.invoke(app, [
            "sync", "nz_cpi",
            "--library", str(lib),
            "--api-key", "k",
        ])

    assert result.exit_code == 0
    mock_client.sync.assert_called_once()
    call_kwargs = mock_client.sync.call_args
    assert call_kwargs[0][0] == "nz_cpi" or call_kwargs[1].get("name") == "nz_cpi" or "nz_cpi" in str(call_kwargs)


def test_sync_single_dataset_with_library_unchanged(tmp_path):
    """Unchanged result exits 0 and prints 'unchanged'."""
    lib = tmp_path / "lib"

    with patch.object(cli_module, "_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.sync.return_value = _SYNC_RESULT_UNCHANGED
        mock_client_factory.return_value = mock_client

        result = runner.invoke(app, [
            "sync", "nz_cpi",
            "--library", str(lib),
            "--api-key", "k",
        ])

    assert result.exit_code == 0
    assert "unchanged" in result.stdout.lower()


# ---------------------------------------------------------------------------
# eolas sync --library DIR --datasets X Y Z
# ---------------------------------------------------------------------------

def test_sync_datasets_flag(tmp_path):
    """--datasets flag calls client.sync_all() with the given list.

    Typer's List[str] option requires one --datasets flag per value:
        eolas sync --library DIR --datasets nz_cpi --datasets nz_addresses
    """
    lib = tmp_path / "lib"

    _result_a = SyncResult(
        status="snapshot_full", dataset="nz_cpi",
        library_dir=lib, bytes_downloaded=512, rows_added=50, files_added=1,
    )
    _result_b = SyncResult(
        status="unchanged", dataset="nz_addresses",
        library_dir=lib, bytes_downloaded=0, rows_added=0, files_added=0,
    )

    with patch.object(cli_module, "_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.sync_all.return_value = [_result_a, _result_b]
        mock_client_factory.return_value = mock_client

        result = runner.invoke(app, [
            "sync",
            "--library", str(lib),
            "--datasets", "nz_cpi",
            "--datasets", "nz_addresses",
            "--api-key", "k",
        ])

    assert result.exit_code == 0
    mock_client.sync_all.assert_called_once()
    # Verify datasets were passed
    call_args = mock_client.sync_all.call_args
    passed_datasets = call_args[1].get("datasets")
    assert passed_datasets is not None
    assert "nz_cpi" in passed_datasets
    assert "nz_addresses" in passed_datasets


# ---------------------------------------------------------------------------
# eolas sync --library DIR --all
# ---------------------------------------------------------------------------

def test_sync_all_flag_discovers_manifests(tmp_path):
    """--all calls client.sync_all() with datasets=None to discover from manifests."""
    lib = tmp_path / "lib"
    # Create a manifest so discovery finds something
    _make_manifest(lib / "nz_cpi", "nz_cpi", 1001)

    _result = SyncResult(
        status="unchanged", dataset="nz_cpi",
        library_dir=lib, bytes_downloaded=0, rows_added=0, files_added=0,
    )

    with patch.object(cli_module, "_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.sync_all.return_value = [_result]
        mock_client_factory.return_value = mock_client

        result = runner.invoke(app, [
            "sync",
            "--library", str(lib),
            "--all",
            "--api-key", "k",
        ])

    assert result.exit_code == 0
    mock_client.sync_all.assert_called_once()
    # datasets=None means auto-discover
    call_args = mock_client.sync_all.call_args
    passed_datasets = call_args[1].get("datasets")
    assert passed_datasets is None


def test_sync_all_empty_library_exits_ok(tmp_path):
    """--all on a library with no manifests exits 0 (nothing to do)."""
    lib = tmp_path / "empty-lib"
    lib.mkdir()

    with patch.object(cli_module, "_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.sync_all.return_value = []
        mock_client_factory.return_value = mock_client

        result = runner.invoke(app, [
            "sync",
            "--library", str(lib),
            "--all",
            "--api-key", "k",
        ])

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Error / bad-input cases for sync
# ---------------------------------------------------------------------------

def test_sync_no_name_no_library_exits_usage():
    """No positional name and no --library → EXIT_USAGE."""
    result = runner.invoke(app, ["sync"])
    assert result.exit_code == cli_module.EXIT_USAGE


def test_sync_all_and_positional_exits_usage(tmp_path):
    """Combining a positional name with --all is an error."""
    lib = tmp_path / "lib"
    result = runner.invoke(app, [
        "sync", "nz_cpi",
        "--library", str(lib),
        "--all",
        "--api-key", "k",
    ])
    assert result.exit_code == cli_module.EXIT_USAGE


def test_sync_datasets_and_positional_exits_usage(tmp_path):
    """Combining a positional name with --datasets is an error."""
    lib = tmp_path / "lib"
    result = runner.invoke(app, [
        "sync", "nz_cpi",
        "--library", str(lib),
        "--datasets", "nz_addresses",  # one --datasets flag
        "--api-key", "k",
    ])
    assert result.exit_code == cli_module.EXIT_USAGE


def test_sync_all_and_datasets_without_library_exits_usage():
    """--all without --library should bail with EXIT_USAGE."""
    result = runner.invoke(app, ["sync", "--all", "--api-key", "k"])
    # Without --library, --all falls into the bulk-mode path which requires a name
    assert result.exit_code == cli_module.EXIT_USAGE


def test_sync_library_no_name_no_all_no_datasets_exits_usage(tmp_path):
    """--library but neither a name nor --datasets nor --all → EXIT_USAGE."""
    lib = tmp_path / "lib"
    result = runner.invoke(app, [
        "sync",
        "--library", str(lib),
        "--api-key", "k",
    ])
    assert result.exit_code == cli_module.EXIT_USAGE


def test_sync_error_status_exits_nonzero(tmp_path):
    """If sync_all returns a result with status='error', exit code is non-zero."""
    lib = tmp_path / "lib"
    _err_result = SyncResult(
        status="error", dataset="nz_cpi",
        library_dir=lib, bytes_downloaded=0, rows_added=0, files_added=0,
        error="ConnectionError: timeout",
    )

    with patch.object(cli_module, "_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.sync_all.return_value = [_err_result]
        mock_client_factory.return_value = mock_client

        result = runner.invoke(app, [
            "sync",
            "--library", str(lib),
            "--all",
            "--api-key", "k",
        ])

    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# eolas compact <dataset_dir>
# ---------------------------------------------------------------------------

def test_compact_single_dir(tmp_path):
    """eolas compact <dir> calls client.compact() and prints result."""
    dataset_dir = tmp_path / "nz_cpi"
    _make_manifest(dataset_dir, "nz_cpi", 1001)

    with patch.object(cli_module, "_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.compact.return_value = _COMPACT_RESULT
        mock_client_factory.return_value = mock_client

        result = runner.invoke(app, [
            "compact", str(dataset_dir),
            "--api-key", "k",
        ])

    assert result.exit_code == 0
    mock_client.compact.assert_called_once()


def test_compact_single_dir_noop(tmp_path):
    """compact with 1 file already → prints 'no-op' or similar."""
    dataset_dir = tmp_path / "nz_cpi"
    _make_manifest(dataset_dir, "nz_cpi", 1001)

    with patch.object(cli_module, "_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.compact.return_value = _COMPACT_RESULT_NOOP
        mock_client_factory.return_value = mock_client

        result = runner.invoke(app, [
            "compact", str(dataset_dir),
            "--api-key", "k",
        ])

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# eolas compact --library DIR
# ---------------------------------------------------------------------------

def test_compact_library_compacts_all_manifests(tmp_path):
    """--library with two synced datasets calls compact() twice."""
    lib = tmp_path / "lib"
    _make_manifest(lib / "nz_cpi",       "nz_cpi",       1001)
    _make_manifest(lib / "nz_addresses", "nz_addresses", 2002)

    with patch.object(cli_module, "_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.compact.return_value = _COMPACT_RESULT_NOOP
        mock_client_factory.return_value = mock_client

        result = runner.invoke(app, [
            "compact",
            "--library", str(lib),
            "--api-key", "k",
        ])

    assert result.exit_code == 0
    assert mock_client.compact.call_count == 2


def test_compact_library_with_dataset_filter(tmp_path):
    """--library --dataset NAME compacts only the named dataset."""
    lib = tmp_path / "lib"
    _make_manifest(lib / "nz_cpi",       "nz_cpi",       1001)
    _make_manifest(lib / "nz_addresses", "nz_addresses", 2002)

    with patch.object(cli_module, "_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.compact.return_value = _COMPACT_RESULT_NOOP
        mock_client_factory.return_value = mock_client

        result = runner.invoke(app, [
            "compact",
            "--library", str(lib),
            "--dataset", "nz_cpi",
            "--api-key", "k",
        ])

    assert result.exit_code == 0
    assert mock_client.compact.call_count == 1
    called_path = mock_client.compact.call_args[0][0]
    assert "nz_cpi" in str(called_path)


# ---------------------------------------------------------------------------
# Error / bad-input cases for compact
# ---------------------------------------------------------------------------

def test_compact_no_args_exits_usage():
    """compact with no arguments → EXIT_USAGE."""
    result = runner.invoke(app, ["compact"])
    assert result.exit_code == cli_module.EXIT_USAGE


def test_compact_both_positional_and_library_exits_usage(tmp_path):
    """Combining positional dir with --library is an error."""
    lib = tmp_path / "lib"
    dataset_dir = tmp_path / "nz_cpi"
    result = runner.invoke(app, [
        "compact", str(dataset_dir),
        "--library", str(lib),
        "--api-key", "k",
    ])
    assert result.exit_code == cli_module.EXIT_USAGE


def test_compact_dataset_flag_without_library_exits_usage(tmp_path):
    """--dataset without --library is an error."""
    dataset_dir = tmp_path / "nz_cpi"
    _make_manifest(dataset_dir, "nz_cpi", 1001)

    with patch.object(cli_module, "_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        result = runner.invoke(app, [
            "compact", str(dataset_dir),
            "--dataset", "nz_cpi",
            "--api-key", "k",
        ])

    assert result.exit_code == cli_module.EXIT_USAGE


def test_compact_library_not_found_exits_not_found(tmp_path):
    """--library pointing to a non-existent directory → EXIT_NOT_FOUND."""
    lib = tmp_path / "does-not-exist"
    result = runner.invoke(app, [
        "compact",
        "--library", str(lib),
        "--api-key", "k",
    ])
    assert result.exit_code == cli_module.EXIT_NOT_FOUND


def test_compact_library_no_manifests_exits_ok(tmp_path):
    """--library with no synced datasets (no manifests) → exit 0."""
    lib = tmp_path / "empty-lib"
    lib.mkdir()
    result = runner.invoke(app, [
        "compact",
        "--library", str(lib),
        "--api-key", "k",
    ])
    assert result.exit_code == 0


def test_compact_nonexistent_single_dir_exits_not_found(tmp_path):
    """compact on a directory that doesn't exist → EXIT_NOT_FOUND."""
    missing = tmp_path / "no-such-dir"

    with patch.object(cli_module, "_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.compact.side_effect = FileNotFoundError("no manifest")
        mock_client_factory.return_value = mock_client

        result = runner.invoke(app, [
            "compact", str(missing),
            "--api-key", "k",
        ])

    assert result.exit_code == cli_module.EXIT_NOT_FOUND


# ---------------------------------------------------------------------------
# Machine / JSON output
# ---------------------------------------------------------------------------

def test_sync_single_machine_mode_emits_json(tmp_path, monkeypatch):
    """In non-TTY mode the sync result is emitted as a JSON line."""
    lib = tmp_path / "lib"

    with patch.object(cli_module, "_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.sync.return_value = SyncResult(
            status="snapshot_full",
            dataset="nz_cpi",
            library_dir=lib,
            bytes_downloaded=2048,
            rows_added=200,
            files_added=1,
        )
        mock_client_factory.return_value = mock_client

        result = runner.invoke(app, [
            "sync", "nz_cpi",
            "--library", str(lib),
            "--api-key", "k",
        ])

    assert result.exit_code == 0
    # CliRunner stdout is non-TTY — should emit JSON line
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    assert lines, "Expected at least one output line"
    parsed = json.loads(lines[0])
    assert parsed["dataset"] == "nz_cpi"
    assert parsed["status"] == "snapshot_full"
    assert parsed["bytes_downloaded"] == 2048


def test_compact_machine_mode_emits_json(tmp_path):
    """In non-TTY mode compact result is emitted as a JSON line."""
    dataset_dir = tmp_path / "nz_cpi"
    _make_manifest(dataset_dir, "nz_cpi", 1001)

    with patch.object(cli_module, "_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.compact.return_value = _COMPACT_RESULT
        mock_client_factory.return_value = mock_client

        result = runner.invoke(app, [
            "compact", str(dataset_dir),
            "--api-key", "k",
        ])

    assert result.exit_code == 0
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    assert lines
    parsed = json.loads(lines[0])
    assert parsed["dataset"] == "nz_cpi"
    assert parsed["files_before"] == 3
    assert parsed["files_after"] == 1
