"""Cross-platform scheduling backend for `eolas schedule add|list|remove`.

POSIX (Linux/macOS): edits the user's crontab via `crontab -l` / `crontab -`.
Windows:             uses `schtasks` to create per-user scheduled tasks.

Both backends only manage entries tagged with a sentinel so the user's other
cron jobs / scheduled tasks are never touched.
"""
from __future__ import annotations

import csv
import io
import platform
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

SENTINEL  = "# eolas-schedule:"
TASK_PREFIX = "eolas-"  # Windows task name prefix

# Interval shortcut → cron expression (minute hour dom month dow). Daily/weekly/
# monthly all default to 6am because datasets typically refresh in the early
# hours; running at 6am gets the freshest data without competing for resources.
INTERVALS = {
    "hourly":  "0 * * * *",
    "daily":   "0 6 * * *",
    "weekly":  "0 6 * * 1",   # Monday 6am
    "monthly": "0 6 1 * *",   # 1st of month, 6am
}

# Windows schtasks /sc value per interval. Custom cron exprs not supported on
# Windows backend — see _windows_add for the fallback message.
WIN_SCHED = {
    "hourly":  ("HOURLY",  None),
    "daily":   ("DAILY",   "06:00"),
    "weekly":  ("WEEKLY",  "06:00"),    # default day = today's weekday; we override below
    "monthly": ("MONTHLY", "06:00"),
}

CRON_EXPR_RE = re.compile(r"^\s*\S+\s+\S+\s+\S+\s+\S+\s+\S+\s*$")


@dataclass
class ScheduleEntry:
    name: str
    schedule: str   # cron expr (POSIX) or human description (Windows)
    command: str


# ────────────────────────────────────────────────────────────────────────────
# Public API — dispatches per OS
# ────────────────────────────────────────────────────────────────────────────

def is_windows() -> bool:
    return platform.system() == "Windows"


def add(name: str, schedule_expr: str, command: str) -> None:
    """Register a scheduled task. `schedule_expr` is a cron expression on POSIX
    or one of {'hourly','daily','weekly','monthly'} on Windows."""
    if is_windows():
        _windows_add(name, schedule_expr, command)
    else:
        _cron_add(name, schedule_expr, command)


def remove(name: str) -> bool:
    """Remove a managed task. Returns True if removed, False if not found."""
    if is_windows():
        return _windows_remove(name)
    return _cron_remove(name)


def list_entries() -> list[ScheduleEntry]:
    """Return all managed eolas-schedule entries."""
    if is_windows():
        return _windows_list()
    return _cron_list()


def interval_to_cron(interval: str) -> str:
    """Return the cron expression for an interval shortcut. Raises on unknown."""
    if interval not in INTERVALS:
        raise ValueError(f"unknown interval {interval!r}; expected one of {list(INTERVALS)}")
    return INTERVALS[interval]


def validate_cron_expr(expr: str) -> None:
    """Basic shape check on a 5-field cron expression. Raises on invalid."""
    if not CRON_EXPR_RE.match(expr):
        raise ValueError(
            f"invalid cron expression {expr!r}; expected 5 fields "
            "(minute hour day-of-month month day-of-week)"
        )


# ────────────────────────────────────────────────────────────────────────────
# POSIX cron backend
# ────────────────────────────────────────────────────────────────────────────

def _crontab_available() -> bool:
    return shutil.which("crontab") is not None


def _cron_read() -> list[str]:
    """Read the user's crontab. Returns [] when no crontab is set."""
    if not _crontab_available():
        raise RuntimeError(
            "crontab is not installed on this system. "
            "On Debian/Ubuntu: sudo apt-get install cron. On Alpine: apk add busybox-suid."
        )
    proc = subprocess.run(
        ["crontab", "-l"], capture_output=True, text=True
    )
    if proc.returncode == 0:
        return proc.stdout.splitlines()
    # Some implementations exit 1 with "no crontab" — treat as empty.
    if "no crontab" in (proc.stderr or "").lower():
        return []
    raise RuntimeError(f"crontab -l failed: {proc.stderr.strip() or proc.stdout.strip()}")


def _cron_write(lines: list[str]) -> None:
    payload = "\n".join(lines).rstrip() + "\n"
    proc = subprocess.run(
        ["crontab", "-"], input=payload, text=True, capture_output=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"crontab - failed: {proc.stderr.strip()}")


def _cron_format_line(name: str, cron_expr: str, command: str) -> str:
    return f"{cron_expr} {command}  {SENTINEL} {name}"


def _cron_match_name(line: str, name: str) -> bool:
    return SENTINEL in line and line.rstrip().endswith(name)


def _cron_add(name: str, cron_expr: str, command: str) -> None:
    validate_cron_expr(cron_expr)
    lines = [l for l in _cron_read() if not _cron_match_name(l, name)]  # idempotent
    lines.append(_cron_format_line(name, cron_expr, command))
    _cron_write(lines)


def _cron_remove(name: str) -> bool:
    lines = _cron_read()
    kept  = [l for l in lines if not _cron_match_name(l, name)]
    if len(kept) == len(lines):
        return False
    _cron_write(kept)
    return True


def _cron_list() -> list[ScheduleEntry]:
    out: list[ScheduleEntry] = []
    for line in _cron_read():
        if SENTINEL not in line:
            continue
        head, _, tail = line.partition(SENTINEL)
        name  = tail.strip()
        parts = head.strip().split(maxsplit=5)
        if len(parts) < 6:
            continue   # malformed; skip silently
        cron_expr = " ".join(parts[:5])
        command   = parts[5]
        out.append(ScheduleEntry(name=name, schedule=cron_expr, command=command))
    return out


# ────────────────────────────────────────────────────────────────────────────
# Windows schtasks backend
# ────────────────────────────────────────────────────────────────────────────

def _schtasks_available() -> bool:
    return shutil.which("schtasks") is not None


def _windows_add(name: str, interval: str, command: str) -> None:
    if not _schtasks_available():
        raise RuntimeError("schtasks not found — required on Windows for scheduling")
    if interval not in WIN_SCHED:
        raise ValueError(
            f"Windows backend supports interval shortcuts only "
            f"({list(WIN_SCHED)}); got {interval!r}. "
            "Custom cron expressions aren't translatable; use schtasks GUI for advanced cases."
        )
    sc, st = WIN_SCHED[interval]
    args = [
        "schtasks", "/create",
        "/tn", f"{TASK_PREFIX}{name}",
        "/tr", command,
        "/sc", sc,
        "/f",  # overwrite if exists (idempotent add)
    ]
    if st:
        args += ["/st", st]
    if interval == "weekly":
        args += ["/d", "MON"]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"schtasks /create failed: {proc.stderr.strip()}")


def _windows_remove(name: str) -> bool:
    proc = subprocess.run(
        ["schtasks", "/delete", "/tn", f"{TASK_PREFIX}{name}", "/f"],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return True
    # schtasks returns non-zero if the task doesn't exist
    if "cannot find" in (proc.stderr + proc.stdout).lower():
        return False
    raise RuntimeError(f"schtasks /delete failed: {proc.stderr.strip()}")


def _windows_list() -> list[ScheduleEntry]:
    proc = subprocess.run(
        ["schtasks", "/query", "/fo", "CSV", "/v"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"schtasks /query failed: {proc.stderr.strip()}")
    out: list[ScheduleEntry] = []
    reader = csv.DictReader(io.StringIO(proc.stdout))
    for row in reader:
        task_name = (row.get("TaskName") or "").lstrip("\\").strip()
        if not task_name.startswith(TASK_PREFIX):
            continue
        out.append(ScheduleEntry(
            name=task_name[len(TASK_PREFIX):],
            schedule=row.get("Schedule Type") or "",
            command=row.get("Task To Run") or "",
        ))
    return out


# ────────────────────────────────────────────────────────────────────────────
# Helpers used by cli.py
# ────────────────────────────────────────────────────────────────────────────

def build_command(eolas_path: str, dataset: str, out_path: str,
                  start: Optional[str] = None, end: Optional[str] = None,
                  fmt: str = "csv") -> str:
    """Construct the shell command line to put inside the cron entry."""
    parts = [shlex.quote(eolas_path), "get", shlex.quote(dataset),
             "--format", shlex.quote(fmt),
             "--out", shlex.quote(str(out_path))]
    if start:
        parts += ["--start", shlex.quote(start)]
    if end:
        parts += ["--end", shlex.quote(end)]
    return " ".join(parts)
