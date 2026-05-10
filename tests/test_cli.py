"""Tests for the eolas CLI.

Strategy: spin up the Typer app via Typer's CliRunner, mock HTTP with `responses`.
Avoids touching the real config file by patching CONFIG_FILE / CONFIG_DIR onto a
tmp_path for the auth tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import responses as resp_lib
from typer.testing import CliRunner

from eolas_data import cli as cli_module
from eolas_data.cli import app

runner = CliRunner()
BASE = "https://api.eolas.fyi"

DATASET_LIST = [
    {"name": "nz_cpi",  "title": "NZ Consumer Price Index",  "source": "Stats NZ"},
    {"name": "nz_gdp",  "title": "NZ Gross Domestic Product","source": "OECD"},
    {"name": "nz_rbnz", "title": "RBNZ data",                "source": "RBNZ"},
]

RECORDS = [
    {"date": "2023-01-01", "period": "2023Q1", "value": 100.0},
    {"date": "2023-04-01", "period": "2023Q2", "value": 101.5},
]


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Redirect ~/.eolas to a tmp dir so tests don't touch real config."""
    cfg_dir  = tmp_path / ".eolas"
    cfg_file = cfg_dir / "config.json"
    monkeypatch.setattr(cli_module, "CONFIG_DIR",  cfg_dir)
    monkeypatch.setattr(cli_module, "CONFIG_FILE", cfg_file)
    # Also unset any real env-var keys so tests start clean.
    monkeypatch.delenv("EOLAS_API_KEY", raising=False)
    monkeypatch.delenv("VS_API_KEY",    raising=False)
    yield


# ────────────────────────────────────────────────────────────────────────────
# version + health
# ────────────────────────────────────────────────────────────────────────────

def test_version_prints_semver():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip().count(".") == 2  # x.y.z


@resp_lib.activate
def test_health_ok():
    resp_lib.add(resp_lib.GET, f"{BASE}/health", json={"status": "ok"}, status=200)
    result = runner.invoke(app, ["health"])
    assert result.exit_code == 0
    # Output should mention "ok" in either tty or pipe mode
    assert "ok" in result.stdout.lower()


@resp_lib.activate
def test_health_failure_exits_nonzero():
    resp_lib.add(resp_lib.GET, f"{BASE}/health", status=500)
    result = runner.invoke(app, ["health"])
    assert result.exit_code != 0


# ────────────────────────────────────────────────────────────────────────────
# datasets list
# ────────────────────────────────────────────────────────────────────────────

@resp_lib.activate
def test_datasets_list_table_mode():
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets",
                 json={"datasets": DATASET_LIST}, status=200)
    result = runner.invoke(app, ["datasets", "list", "--api-key", "k"])
    assert result.exit_code == 0
    # Even though CliRunner makes stdout non-tty, we should at least see names
    for d in DATASET_LIST:
        assert d["name"] in result.stdout


@resp_lib.activate
def test_datasets_list_json_mode_emits_ndjson():
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets",
                 json={"datasets": DATASET_LIST}, status=200)
    result = runner.invoke(app, ["datasets", "list", "--json", "--api-key", "k"])
    assert result.exit_code == 0
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    assert len(lines) == len(DATASET_LIST)
    parsed = [json.loads(l) for l in lines]
    assert {d["name"] for d in parsed} == {d["name"] for d in DATASET_LIST}


@resp_lib.activate
def test_datasets_list_filters_by_source():
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets",
                 json={"datasets": DATASET_LIST}, status=200)
    result = runner.invoke(app, ["datasets", "list", "--source", "Stats NZ",
                                 "--json", "--api-key", "k"])
    assert result.exit_code == 0
    items = [json.loads(l) for l in result.stdout.splitlines() if l.strip()]
    assert len(items) == 1
    assert items[0]["source"] == "Stats NZ"


@resp_lib.activate
def test_datasets_list_search_substring():
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets",
                 json={"datasets": DATASET_LIST}, status=200)
    result = runner.invoke(app, ["datasets", "list", "--search", "consumer",
                                 "--json", "--api-key", "k"])
    assert result.exit_code == 0
    items = [json.loads(l) for l in result.stdout.splitlines() if l.strip()]
    assert {i["name"] for i in items} == {"nz_cpi"}


# ────────────────────────────────────────────────────────────────────────────
# datasets info
# ────────────────────────────────────────────────────────────────────────────

@resp_lib.activate
def test_datasets_info_json():
    meta = {"name": "nz_cpi", "title": "NZ CPI", "source": "Stats NZ", "n_rows": 1234}
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi", json=meta, status=200)
    result = runner.invoke(app, ["datasets", "info", "nz_cpi", "--json", "--api-key", "k"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout.strip().splitlines()[-1])
    assert parsed["name"] == "nz_cpi"


@resp_lib.activate
def test_datasets_info_not_found_returns_distinct_exit():
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/missing",
                 json={"detail": "no such dataset"}, status=404)
    result = runner.invoke(app, ["datasets", "info", "missing", "--api-key", "k"])
    assert result.exit_code == cli_module.EXIT_NOT_FOUND


@resp_lib.activate
def test_auth_error_returns_distinct_exit():
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets",
                 json={"detail": "bad key"}, status=401)
    result = runner.invoke(app, ["datasets", "list", "--api-key", "wrong"])
    assert result.exit_code == cli_module.EXIT_AUTH


@resp_lib.activate
def test_rate_limit_returns_distinct_exit():
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets",
                 json={"detail": "limit reached"}, status=429)
    result = runner.invoke(app, ["datasets", "list", "--api-key", "k"])
    assert result.exit_code == cli_module.EXIT_RATE_LIMIT


# ────────────────────────────────────────────────────────────────────────────
# get
# ────────────────────────────────────────────────────────────────────────────

@resp_lib.activate
def test_get_csv_to_stdout():
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data",
                 json={"data": RECORDS}, status=200)
    result = runner.invoke(app, ["get", "nz_cpi", "--api-key", "k"])
    assert result.exit_code == 0
    # CSV output: header + 2 data rows + possibly trailing newline
    rows = [r for r in result.stdout.splitlines() if r.strip()]
    assert rows[0].split(",")[0] == "date"
    assert len(rows) == 1 + len(RECORDS)


@resp_lib.activate
def test_get_json_format():
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/datasets/nz_cpi/data",
                 json={"data": RECORDS}, status=200)
    result = runner.invoke(app, ["get", "nz_cpi", "--format", "json", "--api-key", "k"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout.strip().splitlines()[0])
    assert isinstance(parsed, list)
    assert len(parsed) == len(RECORDS)


def test_get_parquet_without_out_fails_with_usage_exit():
    result = runner.invoke(app, ["get", "nz_cpi", "--format", "parquet", "--api-key", "k"])
    assert result.exit_code == cli_module.EXIT_USAGE


def test_get_unknown_format_fails_with_usage_exit():
    result = runner.invoke(app, ["get", "nz_cpi", "--format", "xml", "--api-key", "k"])
    assert result.exit_code == cli_module.EXIT_USAGE


# ────────────────────────────────────────────────────────────────────────────
# auth
# ────────────────────────────────────────────────────────────────────────────

def test_auth_status_no_config():
    result = runner.invoke(app, ["auth", "status"])
    assert result.exit_code == 0
    assert "no API key" in result.stdout


def test_auth_set_key_writes_config(monkeypatch):
    cfg_file = cli_module.CONFIG_FILE
    result = runner.invoke(app, ["auth", "set-key", "--key", "secrettoken123"])
    assert result.exit_code == 0
    assert cfg_file.exists()
    data = json.loads(cfg_file.read_text())
    assert data["api_key"] == "secrettoken123"
    # File should be 0600 (only owner can read/write)
    mode = oct(cfg_file.stat().st_mode)[-3:]
    assert mode == "600"


def test_auth_status_reads_config_after_set():
    runner.invoke(app, ["auth", "set-key", "--key", "verysecrettoken"])
    result = runner.invoke(app, ["auth", "status"])
    assert result.exit_code == 0
    assert "verysecr" in result.stdout
    assert str(cli_module.CONFIG_FILE) in result.stdout


def test_auth_status_env_var_wins(monkeypatch):
    runner.invoke(app, ["auth", "set-key", "--key", "from_config"])
    monkeypatch.setenv("EOLAS_API_KEY", "from_env_var")
    result = runner.invoke(app, ["auth", "status"])
    assert result.exit_code == 0
    assert "env" in result.stdout
    assert "from_env" in result.stdout


def test_auth_clear_removes_config():
    runner.invoke(app, ["auth", "set-key", "--key", "willbecleared"])
    assert cli_module.CONFIG_FILE.exists()
    result = runner.invoke(app, ["auth", "clear"])
    assert result.exit_code == 0
    assert not cli_module.CONFIG_FILE.exists()


# ────────────────────────────────────────────────────────────────────────────
# schedule
# ────────────────────────────────────────────────────────────────────────────

def _set_up_key():
    """Helper: install a fake API key so schedule add's pre-flight passes."""
    runner.invoke(app, ["auth", "set-key", "--key", "fakekey"])


def test_schedule_add_requires_api_key():
    # No key configured → pre-flight fails with EXIT_USAGE.
    result = runner.invoke(app, ["schedule", "add", "nz_cpi", "--out", "/tmp/x.csv"])
    assert result.exit_code == cli_module.EXIT_USAGE
    assert "no API key" in result.stderr


def test_schedule_add_dry_run_prints_without_calling_backend(tmp_path):
    _set_up_key()
    out_file = tmp_path / "cpi.csv"
    result = runner.invoke(app, ["schedule", "add", "nz_cpi",
                                 "--out", str(out_file), "--dry-run"])
    assert result.exit_code == 0
    assert "dry-run" in result.stdout
    assert "nz_cpi" in result.stdout
    # daily default → 06:00 cron
    assert "0 6 * * *" in result.stdout or "DAILY" in result.stdout


def test_schedule_add_mutually_exclusive_intervals_rejected():
    _set_up_key()
    result = runner.invoke(app, ["schedule", "add", "nz_cpi",
                                 "--out", "/tmp/x.csv",
                                 "--daily", "--weekly", "--dry-run"])
    assert result.exit_code == cli_module.EXIT_USAGE
    assert "only one of" in result.stderr


def test_schedule_add_cron_with_interval_rejected():
    _set_up_key()
    result = runner.invoke(app, ["schedule", "add", "nz_cpi",
                                 "--out", "/tmp/x.csv",
                                 "--cron", "0 6 * * *", "--daily", "--dry-run"])
    assert result.exit_code == cli_module.EXIT_USAGE


def test_schedule_add_invalid_cron_rejected(monkeypatch):
    _set_up_key()
    # Force POSIX path so --cron is accepted at all
    monkeypatch.setattr("eolas_data.schedule.is_windows", lambda: False)
    result = runner.invoke(app, ["schedule", "add", "nz_cpi",
                                 "--out", "/tmp/x.csv",
                                 "--cron", "every monday", "--dry-run"])
    assert result.exit_code == cli_module.EXIT_USAGE
    assert "invalid cron" in result.stderr


def test_schedule_remove_not_found_returns_distinct_exit(monkeypatch):
    monkeypatch.setattr("eolas_data.schedule.remove", lambda name: False)
    result = runner.invoke(app, ["schedule", "remove", "nz_cpi"])
    assert result.exit_code == cli_module.EXIT_NOT_FOUND


def test_schedule_list_empty_human_mode(monkeypatch):
    """In human (TTY) mode an empty list shows a friendly message."""
    monkeypatch.setattr("eolas_data.schedule.list_entries", lambda: [])
    monkeypatch.setattr(cli_module, "_machine_mode", lambda *_: False)
    result = runner.invoke(app, ["schedule", "list"])
    assert result.exit_code == 0
    assert "no eolas schedules" in result.output


def test_schedule_list_empty_machine_mode_emits_nothing(monkeypatch):
    """In machine (piped / --json) mode an empty list emits zero NDJSON lines.
    This is the correct streaming contract — no records, no output."""
    monkeypatch.setattr("eolas_data.schedule.list_entries", lambda: [])
    result = runner.invoke(app, ["schedule", "list", "--json"])
    assert result.exit_code == 0
    assert result.output.strip() == ""


# ────────────────────────────────────────────────────────────────────────────
# integrate
# ────────────────────────────────────────────────────────────────────────────

INTEGRATION_RESPONSE = {
    "platform": "meltano",
    "files": {
        "meltano.yml":  "# generated meltano config\nname: tap-eolas\n",
        "README.md":    "# eolas → Meltano\n\nrun `meltano install`\n",
        ".env.example": "EOLAS_API_KEY=your_key\n",
    },
}


@resp_lib.activate
def test_integrate_meltano_writes_files(tmp_path):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/integrations/meltano",
                 json=INTEGRATION_RESPONSE, status=200)
    out = tmp_path / "my-tap"
    result = runner.invoke(app, ["integrate", "meltano",
                                 "--datasets", "nz_cpi,nz_gdp",
                                 "--output", str(out),
                                 "--api-key", "k"])
    assert result.exit_code == 0
    assert (out / "meltano.yml").read_text().startswith("# generated meltano config")
    assert (out / "README.md").exists()
    assert (out / ".env.example").exists()


@resp_lib.activate
def test_integrate_query_param_passes_datasets(tmp_path):
    """The CLI should send the comma-separated list to the server as-is."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/integrations/meltano",
                 json=INTEGRATION_RESPONSE, status=200)
    runner.invoke(app, ["integrate", "meltano",
                        "--datasets", "nz_cpi,nz_gdp,nz_rbnz",
                        "--output", str(tmp_path),
                        "--api-key", "k"])
    sent = resp_lib.calls[0].request.url
    assert "datasets=nz_cpi%2Cnz_gdp%2Cnz_rbnz" in sent or "datasets=nz_cpi,nz_gdp,nz_rbnz" in sent


@resp_lib.activate
def test_integrate_403_shows_upgrade_message(tmp_path):
    """A non-Enterprise key triggers a 403 from the server — CLI must surface
    the server's detail (which says 'Enterprise plan feature')."""
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/integrations/meltano",
                 json={"detail": "This endpoint is an Enterprise plan feature."},
                 status=403)
    result = runner.invoke(app, ["integrate", "meltano",
                                 "--datasets", "nz_cpi",
                                 "--output", str(tmp_path),
                                 "--api-key", "wrong"])
    assert result.exit_code == cli_module.EXIT_AUTH
    assert "Enterprise" in result.stderr
    assert "pricing" in result.stderr


@resp_lib.activate
def test_integrate_refuses_to_overwrite_without_force(tmp_path):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/integrations/meltano",
                 json=INTEGRATION_RESPONSE, status=200)
    out = tmp_path / "my-tap"
    out.mkdir()
    (out / "meltano.yml").write_text("DO NOT OVERWRITE ME")
    result = runner.invoke(app, ["integrate", "meltano",
                                 "--datasets", "nz_cpi",
                                 "--output", str(out),
                                 "--api-key", "k"])
    assert result.exit_code == 0
    # existing file untouched
    assert (out / "meltano.yml").read_text() == "DO NOT OVERWRITE ME"
    # but other files written
    assert (out / "README.md").exists()
    assert "skipped" in result.output.lower()


@resp_lib.activate
def test_integrate_force_overwrites(tmp_path):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/integrations/meltano",
                 json=INTEGRATION_RESPONSE, status=200)
    out = tmp_path / "my-tap"
    out.mkdir()
    (out / "meltano.yml").write_text("OLD CONTENT")
    result = runner.invoke(app, ["integrate", "meltano",
                                 "--datasets", "nz_cpi",
                                 "--output", str(out),
                                 "--api-key", "k", "--force"])
    assert result.exit_code == 0
    assert (out / "meltano.yml").read_text().startswith("# generated meltano config")


@resp_lib.activate
def test_integrate_empty_datasets_rejected(tmp_path):
    result = runner.invoke(app, ["integrate", "meltano",
                                 "--datasets", "",
                                 "--output", str(tmp_path),
                                 "--api-key", "k"])
    assert result.exit_code == cli_module.EXIT_USAGE


@resp_lib.activate
def test_integrate_fivetran_routes_correctly(tmp_path):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/integrations/fivetran",
                 json={"platform": "fivetran",
                       "files": {"fivetran.yml": "fivetran config\n",
                                 "README.md":    "fivetran readme\n"}}, status=200)
    result = runner.invoke(app, ["integrate", "fivetran",
                                 "--datasets", "nz_cpi",
                                 "--output", str(tmp_path),
                                 "--api-key", "k"])
    assert result.exit_code == 0
    assert (tmp_path / "fivetran.yml").exists()


@resp_lib.activate
def test_integrate_adf_routes_correctly(tmp_path):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/integrations/azure-data-factory",
                 json={"platform": "azure-data-factory",
                       "files": {"linkedService.json": "{}",
                                 "pipeline.json": "{}",
                                 "README.md": "adf"}}, status=200)
    result = runner.invoke(app, ["integrate", "azure-data-factory",
                                 "--datasets", "nz_cpi",
                                 "--output", str(tmp_path),
                                 "--api-key", "k"])
    assert result.exit_code == 0
    assert (tmp_path / "linkedService.json").exists()
    assert (tmp_path / "pipeline.json").exists()


@resp_lib.activate
def test_integrate_json_mode_emits_summary(tmp_path):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/integrations/meltano",
                 json=INTEGRATION_RESPONSE, status=200)
    result = runner.invoke(app, ["integrate", "meltano",
                                 "--datasets", "nz_cpi",
                                 "--output", str(tmp_path),
                                 "--api-key", "k", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout.strip().splitlines()[0])
    assert parsed["platform"] == "meltano"
    assert len(parsed["written"]) == 3
    assert parsed["skipped"] == []


def test_schedule_list_json_output(monkeypatch):
    from eolas_data.schedule import ScheduleEntry
    monkeypatch.setattr("eolas_data.schedule.list_entries", lambda: [
        ScheduleEntry(name="foo", schedule="0 6 * * *", command="eolas get foo --out o.csv"),
        ScheduleEntry(name="bar", schedule="0 7 * * 1", command="eolas get bar --out b.csv"),
    ])
    result = runner.invoke(app, ["schedule", "list", "--json"])
    assert result.exit_code == 0
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert {p["name"] for p in parsed} == {"foo", "bar"}
