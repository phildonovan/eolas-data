"""Tests for the cron / schtasks scheduling backend.

Subprocess is mocked so tests run on any OS without touching the real crontab.
The `is_windows()` flag is patched per-test to exercise both backends from a
single test environment.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from eolas_data import schedule as sched


# ────────────────────────────────────────────────────────────────────────────
# helpers / common patches
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def posix(monkeypatch):
    monkeypatch.setattr(sched, "is_windows", lambda: False)
    monkeypatch.setattr(sched, "_crontab_available", lambda: True)


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(sched, "is_windows", lambda: True)
    monkeypatch.setattr(sched, "_schtasks_available", lambda: True)


def _mk_proc(returncode=0, stdout="", stderr=""):
    p = MagicMock()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


# ────────────────────────────────────────────────────────────────────────────
# interval helpers
# ────────────────────────────────────────────────────────────────────────────

def test_interval_to_cron_known():
    assert sched.interval_to_cron("daily")   == "0 6 * * *"
    assert sched.interval_to_cron("weekly")  == "0 6 * * 1"
    assert sched.interval_to_cron("hourly")  == "0 * * * *"
    assert sched.interval_to_cron("monthly") == "0 6 1 * *"


def test_interval_to_cron_unknown_raises():
    with pytest.raises(ValueError):
        sched.interval_to_cron("yearly")


def test_validate_cron_expr_ok():
    sched.validate_cron_expr("0 6 * * *")
    sched.validate_cron_expr("*/5 * * * 1-5")


def test_validate_cron_expr_bad_field_count():
    with pytest.raises(ValueError):
        sched.validate_cron_expr("0 6 * *")
    with pytest.raises(ValueError):
        sched.validate_cron_expr("just words")


# ────────────────────────────────────────────────────────────────────────────
# build_command
# ────────────────────────────────────────────────────────────────────────────

def test_build_command_basic():
    cmd = sched.build_command("/usr/local/bin/eolas", "nz_cpi", "/tmp/cpi.csv")
    assert cmd.startswith("/usr/local/bin/eolas get nz_cpi")
    assert "--format csv" in cmd
    assert "--out /tmp/cpi.csv" in cmd


def test_build_command_quotes_paths_with_spaces():
    cmd = sched.build_command("/usr/local/bin/eolas", "nz_cpi", "/tmp/has spaces/cpi.csv")
    # shlex.quote should wrap the path
    assert "'/tmp/has spaces/cpi.csv'" in cmd


def test_build_command_dates():
    cmd = sched.build_command("/bin/eolas", "nz_cpi", "/tmp/o.csv",
                              start="2020-01-01", end="2024-12-31")
    assert "--start 2020-01-01" in cmd
    assert "--end 2024-12-31"   in cmd


# ────────────────────────────────────────────────────────────────────────────
# POSIX cron backend
# ────────────────────────────────────────────────────────────────────────────

def test_cron_add_appends_with_sentinel(posix):
    existing = "0 7 * * * /usr/bin/something\n"
    written = {}
    def fake_run(args, **kwargs):
        if args == ["crontab", "-l"]:
            return _mk_proc(0, stdout=existing)
        if args == ["crontab", "-"]:
            written["body"] = kwargs.get("input", "")
            return _mk_proc(0)
        raise AssertionError(f"unexpected args: {args}")
    with patch("eolas_data.schedule.subprocess.run", side_effect=fake_run):
        sched.add("nz_cpi", "0 6 * * *", "/usr/local/bin/eolas get nz_cpi --out /tmp/cpi.csv")
    body = written["body"]
    assert "0 7 * * * /usr/bin/something" in body  # preserved
    assert "0 6 * * *" in body
    assert sched.SENTINEL in body
    assert "nz_cpi" in body


def test_cron_add_idempotent_replaces_same_name(posix):
    existing = (
        "0 5 * * * /usr/bin/something\n"
        f"0 6 * * * /old-eolas get nz_cpi --out /tmp/old.csv  {sched.SENTINEL} nz_cpi\n"
    )
    written = {}
    def fake_run(args, **kwargs):
        if args == ["crontab", "-l"]:
            return _mk_proc(0, stdout=existing)
        if args == ["crontab", "-"]:
            written["body"] = kwargs.get("input", "")
            return _mk_proc(0)
        raise AssertionError(args)
    with patch("eolas_data.schedule.subprocess.run", side_effect=fake_run):
        sched.add("nz_cpi", "0 7 * * *", "/new-eolas get nz_cpi --out /tmp/new.csv")
    body = written["body"]
    # exactly one managed line for that name (count by sentinel, not by string —
    # "nz_cpi" appears twice per managed line: in the command and in the tag)
    matching = [l for l in body.splitlines()
                if sched.SENTINEL in l and l.rstrip().endswith("nz_cpi")]
    assert len(matching) == 1
    assert "/old-eolas" not in body                      # old line removed
    assert "/new-eolas" in body                          # new line present
    assert "0 5 * * * /usr/bin/something" in body        # unrelated entry preserved


def test_cron_remove_returns_false_when_not_found(posix):
    existing = "0 5 * * * /usr/bin/something\n"
    def fake_run(args, **kwargs):
        if args == ["crontab", "-l"]:
            return _mk_proc(0, stdout=existing)
        return _mk_proc(0)
    with patch("eolas_data.schedule.subprocess.run", side_effect=fake_run):
        assert sched.remove("nz_cpi") is False


def test_cron_remove_strips_only_matching(posix):
    existing = (
        "0 5 * * * /usr/bin/something\n"
        f"0 6 * * * eolas get foo --out /tmp/foo.csv  {sched.SENTINEL} foo\n"
        f"0 7 * * * eolas get bar --out /tmp/bar.csv  {sched.SENTINEL} bar\n"
    )
    written = {}
    def fake_run(args, **kwargs):
        if args == ["crontab", "-l"]: return _mk_proc(0, stdout=existing)
        if args == ["crontab", "-"]:
            written["body"] = kwargs.get("input", "")
            return _mk_proc(0)
        raise AssertionError(args)
    with patch("eolas_data.schedule.subprocess.run", side_effect=fake_run):
        assert sched.remove("foo") is True
    body = written["body"]
    assert "foo" not in body
    assert "bar" in body
    assert "/usr/bin/something" in body


def test_cron_list_parses_entries(posix):
    existing = (
        "0 5 * * * /usr/bin/something\n"
        f"0 6 * * * eolas get foo --out /tmp/foo.csv  {sched.SENTINEL} foo\n"
        f"0 7 * * 1 eolas get bar --out /tmp/bar.csv  {sched.SENTINEL} bar\n"
    )
    def fake_run(args, **kwargs):
        return _mk_proc(0, stdout=existing) if args == ["crontab", "-l"] else _mk_proc(0)
    with patch("eolas_data.schedule.subprocess.run", side_effect=fake_run):
        entries = sched.list_entries()
    assert {e.name for e in entries} == {"foo", "bar"}
    foo = next(e for e in entries if e.name == "foo")
    assert foo.schedule == "0 6 * * *"
    assert "foo.csv" in foo.command


def test_cron_no_existing_crontab_treated_as_empty(posix):
    """When the user has no crontab yet, `crontab -l` exits 1 with 'no crontab'."""
    written = {}
    def fake_run(args, **kwargs):
        if args == ["crontab", "-l"]:
            return _mk_proc(1, stderr="no crontab for testuser")
        if args == ["crontab", "-"]:
            written["body"] = kwargs.get("input", "")
            return _mk_proc(0)
        raise AssertionError(args)
    with patch("eolas_data.schedule.subprocess.run", side_effect=fake_run):
        sched.add("nz_cpi", "0 6 * * *", "eolas get nz_cpi --out /tmp/o.csv")
    assert "nz_cpi" in written["body"]


def test_cron_validate_rejects_bad_expr(posix):
    with pytest.raises(ValueError):
        sched.add("nz_cpi", "every monday", "eolas get foo --out o.csv")


def test_cron_unavailable_raises(monkeypatch):
    monkeypatch.setattr(sched, "is_windows", lambda: False)
    monkeypatch.setattr(sched, "_crontab_available", lambda: False)
    with pytest.raises(RuntimeError, match="crontab is not installed"):
        sched.list_entries()


# ────────────────────────────────────────────────────────────────────────────
# Windows schtasks backend
# ────────────────────────────────────────────────────────────────────────────

def test_windows_add_calls_schtasks(windows):
    captured = {}
    def fake_run(args, **kwargs):
        captured["args"] = args
        return _mk_proc(0)
    with patch("eolas_data.schedule.subprocess.run", side_effect=fake_run):
        sched.add("nz_cpi", "daily", "C:\\eolas.exe get nz_cpi --out C:\\out.csv")
    args = captured["args"]
    assert "schtasks" in args
    assert "/create" in args
    assert "eolas-nz_cpi" in args
    assert "/sc" in args
    assert "DAILY" in args
    assert "/f" in args  # idempotent


def test_windows_add_weekly_specifies_monday(windows):
    captured = {}
    def fake_run(args, **kwargs):
        captured["args"] = args
        return _mk_proc(0)
    with patch("eolas_data.schedule.subprocess.run", side_effect=fake_run):
        sched.add("foo", "weekly", "eolas get foo --out o.csv")
    args = captured["args"]
    assert "/d" in args
    assert "MON" in args


def test_windows_add_rejects_cron_expression(windows):
    with pytest.raises(ValueError, match="interval shortcuts only"):
        sched.add("foo", "0 6 * * *", "eolas get foo --out o.csv")


def test_windows_remove_handles_missing_task(windows):
    def fake_run(args, **kwargs):
        return _mk_proc(1, stderr="ERROR: The system cannot find the file specified.")
    with patch("eolas_data.schedule.subprocess.run", side_effect=fake_run):
        assert sched.remove("nope") is False


def test_windows_list_filters_by_prefix(windows):
    csv_out = (
        '"HostName","TaskName","Schedule Type","Task To Run"\n'
        '"PC1","\\Microsoft\\OtherTask","Daily","C:\\other.exe"\n'
        '"PC1","\\eolas-foo","Daily","C:\\eolas.exe get foo --out o.csv"\n'
        '"PC1","\\eolas-bar","Weekly","C:\\eolas.exe get bar --out b.csv"\n'
    )
    def fake_run(args, **kwargs):
        return _mk_proc(0, stdout=csv_out)
    with patch("eolas_data.schedule.subprocess.run", side_effect=fake_run):
        entries = sched.list_entries()
    assert {e.name for e in entries} == {"foo", "bar"}
