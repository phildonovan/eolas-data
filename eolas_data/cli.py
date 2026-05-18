"""eolas — command-line interface for the eolas.fyi data API.

Designed for two audiences:
- Humans typing in a terminal: rich tables, sensible defaults, --help everywhere.
- Shell scripts and AI agents: --json everywhere, auto-detect when stdout is
  piped (drops to NDJSON automatically), distinct exit codes per error class,
  stable output schemas.

The CLI is a thin layer over the existing `eolas_data.Client`. All HTTP, retry,
auth, and error-mapping behaviour stays in the Python client — the CLI only
formats input and output.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from . import schedule as _schedule
from .client import Client
from .exceptions import (
    APIError,
    AuthenticationError,
    EolasError,
    NotFoundError,
    RateLimitError,
)

CONFIG_DIR = Path.home() / ".eolas"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Stable, distinct exit codes — useful for shell scripts and agents that branch
# on outcome. Documented in the README.
EXIT_OK              = 0
EXIT_GENERIC         = 1
EXIT_AUTH            = 2
EXIT_RATE_LIMIT      = 3
EXIT_NOT_FOUND       = 4
EXIT_API             = 5
EXIT_USAGE           = 64  # convention from sysexits.h

app           = typer.Typer(
    name="eolas",
    help=(
        "CLI for the eolas.fyi statistical data API. Browse and fetch 1,400+ "
        "official NZ statistical & geospatial datasets, plus OECD data for "
        "international comparisons. Pipes cleanly into jq, csvkit, etc."
    ),
    no_args_is_help=True,
    add_completion=True,
)
datasets_app  = typer.Typer(help="Browse and inspect datasets.", no_args_is_help=True)
auth_app      = typer.Typer(help="Manage your API key (env var or ~/.eolas/config.json).", no_args_is_help=True)
schedule_app  = typer.Typer(help="Schedule recurring fetches via cron (POSIX) or Task Scheduler (Windows).", no_args_is_help=True)
integrate_app = typer.Typer(help="Generate connector configs for third-party data-pipeline tools (Enterprise plan).", no_args_is_help=True)
app.add_typer(datasets_app,  name="datasets")
app.add_typer(auth_app,      name="auth")
app.add_typer(schedule_app,  name="schedule")
app.add_typer(integrate_app, name="integrate")

# Errors go to stderr, data to stdout — important for piping.
err_console = Console(stderr=True)


# ────────────────────────────────────────────────────────────────────────────
# Auth resolution
# ────────────────────────────────────────────────────────────────────────────

def _load_api_key() -> str:
    """Resolve the API key. Precedence: env var → config file → empty."""
    v = os.getenv("EOLAS_API_KEY")
    if v:
        return v
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text()).get("api_key", "")
        except (json.JSONDecodeError, OSError):
            return ""
    return ""


def _client(api_key: Optional[str] = None) -> Client:
    return Client(api_key=api_key or _load_api_key())


# ────────────────────────────────────────────────────────────────────────────
# Output helpers
# ────────────────────────────────────────────────────────────────────────────

def _machine_mode(json_flag: bool) -> bool:
    """True when output should be machine-readable (NDJSON / CSV)."""
    if json_flag:
        return True
    return not sys.stdout.isatty()


def _emit_ndjson(records) -> None:
    """Write one JSON object per line to stdout (no rich formatting)."""
    for r in records:
        sys.stdout.write(json.dumps(r, default=str, ensure_ascii=False))
        sys.stdout.write("\n")


def _row_to_dict(row) -> dict:
    """Convert a pandas Series row to a JSON-friendly dict (handles NaN)."""
    try:
        import pandas as pd
        return {k: (None if pd.isna(v) else v) for k, v in row.items()}
    except ImportError:
        return dict(row.items())


def _exit_for(e: EolasError) -> int:
    """Map a client-library exception class to an exit code."""
    if isinstance(e, AuthenticationError): return EXIT_AUTH
    if isinstance(e, RateLimitError):      return EXIT_RATE_LIMIT
    if isinstance(e, NotFoundError):       return EXIT_NOT_FOUND
    if isinstance(e, APIError):            return EXIT_API
    return EXIT_GENERIC


def _bail(msg: str, code: int = EXIT_GENERIC) -> None:
    err_console.print(f"[red]error:[/red] {msg}")
    raise typer.Exit(code=code)


# ────────────────────────────────────────────────────────────────────────────
# Top-level commands
# ────────────────────────────────────────────────────────────────────────────

@app.command()
def version() -> None:
    """Print the installed eolas-data version."""
    typer.echo(__version__)


@app.command()
def health() -> None:
    """Quick reachability check against api.eolas.fyi/health."""
    import requests
    try:
        r = requests.get("https://api.eolas.fyi/health", timeout=10)
        r.raise_for_status()
    except Exception as e:
        _bail(f"health check failed: {e}", EXIT_API)
    if not sys.stdout.isatty():
        sys.stdout.write(json.dumps(r.json()))
        sys.stdout.write("\n")
    else:
        Console().print(f"[green]ok[/green] {r.json()}")


# ────────────────────────────────────────────────────────────────────────────
# datasets subcommands
# ────────────────────────────────────────────────────────────────────────────

@datasets_app.command("list")
def datasets_list(
    source:   Optional[str] = typer.Option(None, "--source", "-s", help="Filter by source, e.g. 'Stats NZ', 'OECD'."),
    search:   Optional[str] = typer.Option(None, "--search",       help="Substring match against name or title."),
    json_out: bool          = typer.Option(False, "--json",        help="Force NDJSON output."),
    api_key:  Optional[str] = typer.Option(None, "--api-key", envvar=None, help="Override resolved API key."),
) -> None:
    """List datasets, optionally filtered by source or search term."""
    try:
        items = _client(api_key).list(source=source)
    except EolasError as e:
        _bail(str(e), _exit_for(e))

    if search:
        needle = search.lower()
        items = [
            d for d in items
            if needle in (str(d.get("name", "")) + str(d.get("title", ""))).lower()
        ]

    if _machine_mode(json_out):
        _emit_ndjson(items)
        return

    table = Table(title=f"{len(items)} dataset{'' if len(items) == 1 else 's'}")
    table.add_column("name",   style="cyan",    no_wrap=True)
    table.add_column("source", style="magenta", no_wrap=True)
    table.add_column("title")
    for d in items:
        title = (d.get("title") or "")
        if len(title) > 80:
            title = title[:77] + "..."
        table.add_row(str(d.get("name", "")), str(d.get("source", "")), title)
    Console().print(table)


@datasets_app.command("info")
def datasets_info(
    name:     str,
    json_out: bool          = typer.Option(False, "--json"),
    api_key:  Optional[str] = typer.Option(None, "--api-key"),
) -> None:
    """Show metadata for a single dataset."""
    try:
        meta = _client(api_key).info(name)
    except EolasError as e:
        _bail(str(e), _exit_for(e))

    if _machine_mode(json_out):
        sys.stdout.write(json.dumps(meta, default=str, ensure_ascii=False))
        sys.stdout.write("\n")
        return

    table = Table(title=name, show_header=False, expand=False)
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value")
    for k, v in meta.items():
        if isinstance(v, (list, dict)):
            v = json.dumps(v, default=str)
        table.add_row(str(k), str(v))
    Console().print(table)


@datasets_app.command("preview")
def datasets_preview(
    name:     str,
    limit:    int           = typer.Option(10, "--limit", "-n", min=1, max=1000, help="Rows to preview."),
    json_out: bool          = typer.Option(False, "--json"),
    api_key:  Optional[str] = typer.Option(None, "--api-key"),
) -> None:
    """Preview the first N rows of a dataset."""
    try:
        df = _client(api_key).get(name, limit=limit)
    except EolasError as e:
        _bail(str(e), _exit_for(e))

    if _machine_mode(json_out):
        _emit_ndjson(_row_to_dict(row) for _, row in df.iterrows())
        return

    table = Table(title=f"{name} (showing {len(df)} rows)")
    for col in df.columns:
        table.add_column(str(col))
    for _, row in df.iterrows():
        table.add_row(*[("" if v is None else str(v)) for v in row.values])
    Console().print(table)


# ────────────────────────────────────────────────────────────────────────────
# get command — the heavy lifter (verb matches the Python client's client.get())
# ────────────────────────────────────────────────────────────────────────────

@app.command(name="get")
def get_cmd(
    name:    str,
    start:   Optional[str]  = typer.Option(None, "--start",            help="ISO date lower bound, e.g. 2020-01-01."),
    end:     Optional[str]  = typer.Option(None, "--end",              help="ISO date upper bound."),
    limit:   Optional[int]  = typer.Option(None, "--limit", "-n",      help="Max rows. Default: full dataset (Pro) or the 50,000-row Free cap."),
    fmt:     str            = typer.Option("csv", "--format", "-f",    help="Output format: csv | json | parquet."),
    out:     Optional[Path] = typer.Option(None, "--out", "-o",        help="Write to file. Default: stdout."),
    api_key: Optional[str]  = typer.Option(None, "--api-key"),
) -> None:
    """Fetch a dataset and write rows to stdout or a file.

    Examples
    --------
        eolas get nz_cpi --format csv > cpi.csv
        eolas get nz_cpi --start 2020-01-01 --format json | jq '.[].value'
        eolas get sa2_2023 --format parquet --out sa2.parquet
    """
    fmt = fmt.lower()
    if fmt not in ("csv", "json", "parquet"):
        _bail(f"unknown --format {fmt!r}; expected csv | json | parquet", EXIT_USAGE)
    if fmt == "parquet" and out is None:
        _bail("parquet requires --out FILE (binary cannot be safely streamed to stdout)", EXIT_USAGE)

    try:
        df = _client(api_key).get(name, start=start, end=end, limit=limit)
    except EolasError as e:
        _bail(str(e), _exit_for(e))

    if fmt == "csv":
        df.to_csv(out if out else sys.stdout, index=False)
    elif fmt == "json":
        text = df.to_json(orient="records", date_format="iso")
        if out:
            out.write_text(text + "\n")
        else:
            sys.stdout.write(text)
            sys.stdout.write("\n")
    elif fmt == "parquet":
        df.to_parquet(out, index=False)


# ────────────────────────────────────────────────────────────────────────────
# auth subcommands
# ────────────────────────────────────────────────────────────────────────────

def _mask(key: str) -> str:
    if not key:
        return "(none)"
    return key[:8] + "…" if len(key) > 8 else key


@auth_app.command("set-key")
def auth_set_key(
    api_key: str = typer.Option(
        ..., "--key", prompt="API key", hide_input=True,
        help="Your eolas.fyi API key. Will be saved to ~/.eolas/config.json (chmod 600).",
    ),
) -> None:
    """Save your API key to ~/.eolas/config.json."""
    CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps({"api_key": api_key}, indent=2) + "\n")
    CONFIG_FILE.chmod(0o600)
    typer.echo(f"saved {CONFIG_FILE}")


@auth_app.command("status")
def auth_status() -> None:
    """Show the resolved API key (masked) and which source supplied it."""
    v = os.getenv("EOLAS_API_KEY")
    if v:
        typer.echo(f"key:    {_mask(v)}\nsource: env EOLAS_API_KEY")
        return
    if CONFIG_FILE.exists():
        try:
            k = json.loads(CONFIG_FILE.read_text()).get("api_key", "")
        except (json.JSONDecodeError, OSError) as e:
            _bail(f"could not read {CONFIG_FILE}: {e}")
        typer.echo(f"key:    {_mask(k)}\nsource: {CONFIG_FILE}")
        return
    typer.echo("no API key configured\nset one with: eolas auth set-key")


@auth_app.command("clear")
def auth_clear() -> None:
    """Remove ~/.eolas/config.json (does not unset env vars)."""
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
        typer.echo(f"removed {CONFIG_FILE}")
    else:
        typer.echo(f"no config at {CONFIG_FILE}")


# ────────────────────────────────────────────────────────────────────────────
# schedule subcommands — cron (POSIX) / Task Scheduler (Windows)
# ────────────────────────────────────────────────────────────────────────────

def _resolve_eolas_path() -> str:
    """Find the absolute path to the `eolas` binary, for use inside cron lines.
    cron runs with a minimal PATH so we can't rely on `eolas` resolving."""
    import shutil as _shutil
    p = _shutil.which("eolas")
    if not p:
        # Fallback: invoke the python module directly (works even if the script
        # entry point isn't on PATH, e.g. inside an unusual venv layout).
        return f"{sys.executable} -m eolas_data.cli"
    return p


def _config_or_env_set() -> bool:
    """True if at least one source of API key resolution has a value."""
    return bool(_load_api_key())


@schedule_app.command("add")
def schedule_add(
    name:     str,
    out:      Path           = typer.Option(..., "--out", "-o", help="Where to write the fetched data on each run. REQUIRED — cron jobs have no terminal."),
    interval: Optional[str]  = typer.Option(None, "--interval", help="hourly | daily | weekly | monthly. Default: daily."),
    cron:     Optional[str]  = typer.Option(None, "--cron",     help="Custom cron expression, e.g. '0 6 * * 1'. POSIX only. Mutually exclusive with --interval."),
    fmt:      str            = typer.Option("csv", "--format", "-f", help="csv | json | parquet."),
    start:    Optional[str]  = typer.Option(None, "--start"),
    end:      Optional[str]  = typer.Option(None, "--end"),
    daily:    bool           = typer.Option(False, "--daily",   help="Shortcut for --interval daily."),
    weekly:   bool           = typer.Option(False, "--weekly",  help="Shortcut for --interval weekly."),
    hourly:   bool           = typer.Option(False, "--hourly",  help="Shortcut for --interval hourly."),
    monthly:  bool           = typer.Option(False, "--monthly", help="Shortcut for --interval monthly."),
    dry_run:  bool           = typer.Option(False, "--dry-run", help="Print what would be installed; don't touch crontab/Task Scheduler."),
) -> None:
    """Schedule a recurring fetch. Defaults to daily at 06:00 local time.

    The job will run as your user, with the env var search path cron provides
    by default. Make sure your API key is in ~/.eolas/config.json (run `eolas
    auth set-key` first) so the scheduled run can authenticate.
    """
    # ----- pre-flight checks ----------------------------------------------
    if not _config_or_env_set():
        _bail(
            "no API key configured. Run `eolas auth set-key` first so the "
            "scheduled job can authenticate.",
            EXIT_USAGE,
        )

    # ----- collapse interval flags ----------------------------------------
    flag_count = sum([daily, weekly, hourly, monthly, interval is not None, cron is not None])
    if flag_count > 1:
        _bail("only one of --hourly/--daily/--weekly/--monthly/--interval/--cron may be set", EXIT_USAGE)
    chosen_interval: Optional[str] = None
    if   hourly:  chosen_interval = "hourly"
    elif daily:   chosen_interval = "daily"
    elif weekly:  chosen_interval = "weekly"
    elif monthly: chosen_interval = "monthly"
    elif interval: chosen_interval = interval
    if cron and chosen_interval:
        _bail("--cron and an interval flag are mutually exclusive", EXIT_USAGE)
    if not cron and not chosen_interval:
        chosen_interval = "daily"  # default

    # ----- build the command line -----------------------------------------
    out_path = out.expanduser().resolve()
    eolas_bin = _resolve_eolas_path()
    command  = _schedule.build_command(eolas_bin, name, str(out_path),
                                       start=start, end=end, fmt=fmt)

    # ----- platform-specific schedule expression --------------------------
    if _schedule.is_windows():
        if cron:
            _bail("custom --cron expressions aren't supported on Windows; use --interval instead", EXIT_USAGE)
        schedule_expr = chosen_interval
    else:
        if cron:
            try:
                _schedule.validate_cron_expr(cron)
            except ValueError as e:
                _bail(str(e), EXIT_USAGE)
            schedule_expr = cron
        else:
            schedule_expr = _schedule.interval_to_cron(chosen_interval)

    # ----- dry run --------------------------------------------------------
    if dry_run:
        if _schedule.is_windows():
            typer.echo(f"[dry-run] would create scheduled task {_schedule.TASK_PREFIX}{name}")
            typer.echo(f"          run: {command}")
            typer.echo(f"          schedule: {schedule_expr}")
        else:
            typer.echo(f"[dry-run] would append to crontab:")
            typer.echo(f"  {schedule_expr} {command}  {_schedule.SENTINEL} {name}")
        return

    # ----- install --------------------------------------------------------
    try:
        _schedule.add(name, schedule_expr, command)
    except (RuntimeError, ValueError) as e:
        _bail(str(e), EXIT_GENERIC)

    typer.echo(f"scheduled '{name}' → {out_path}")
    typer.echo(f"  schedule: {schedule_expr}")
    typer.echo(f"  remove with: eolas schedule remove {name}")


@schedule_app.command("list")
def schedule_list(
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """List all eolas-managed scheduled tasks."""
    try:
        entries = _schedule.list_entries()
    except RuntimeError as e:
        _bail(str(e), EXIT_GENERIC)

    if _machine_mode(json_out):
        _emit_ndjson({"name": e.name, "schedule": e.schedule, "command": e.command} for e in entries)
        return

    if not entries:
        typer.echo("no eolas schedules installed")
        return

    table = Table(title=f"{len(entries)} schedule{'' if len(entries) == 1 else 's'}")
    table.add_column("name",     style="cyan", no_wrap=True)
    table.add_column("schedule", style="magenta", no_wrap=True)
    table.add_column("command")
    for e in entries:
        table.add_row(e.name, e.schedule, e.command)
    Console().print(table)


@schedule_app.command("remove")
def schedule_remove(name: str) -> None:
    """Remove a scheduled task by name."""
    try:
        removed = _schedule.remove(name)
    except RuntimeError as e:
        _bail(str(e), EXIT_GENERIC)
    if removed:
        typer.echo(f"removed schedule '{name}'")
    else:
        typer.echo(f"no schedule named '{name}' found")
        raise typer.Exit(code=EXIT_NOT_FOUND)


# ────────────────────────────────────────────────────────────────────────────
# integrate subcommands — Enterprise plan only, generates connector configs
# ────────────────────────────────────────────────────────────────────────────

def _run_integration(
    platform:    str,
    datasets:    str,
    output:      Path,
    force:       bool,
    api_key:     Optional[str],
    json_out:    bool,
) -> None:
    """Shared implementation for all `eolas integrate <platform>` commands."""
    ds_list = [d.strip() for d in datasets.split(",") if d.strip()]
    if not ds_list:
        _bail("--datasets cannot be empty", EXIT_USAGE)

    try:
        files = _client(api_key).integration(platform, ds_list)
    except AuthenticationError as e:
        # Server's 403 detail flows through — usually the "Enterprise feature"
        # upgrade message. We surface it verbatim plus a pricing link.
        err_console.print(f"[red]error:[/red] {e}")
        err_console.print("[dim]→ https://eolas.fyi/pricing[/dim]")
        raise typer.Exit(code=EXIT_AUTH)
    except EolasError as e:
        _bail(str(e), _exit_for(e))

    if not files:
        _bail(f"server returned no files for platform {platform!r}", EXIT_API)

    # Default output dir is per-platform so two integrations don't clobber each
    # other in the user's cwd.
    out_dir = output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    written:    list[Path] = []
    skipped:    list[Path] = []
    for filename, content in files.items():
        target = out_dir / filename
        if target.exists() and not force:
            skipped.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        written.append(target)

    if _machine_mode(json_out):
        sys.stdout.write(json.dumps({
            "platform": platform,
            "output_dir": str(out_dir),
            "written":   [str(p) for p in written],
            "skipped":   [str(p) for p in skipped],
        }, default=str))
        sys.stdout.write("\n")
        return

    Console().print(f"[green]✓[/green] wrote {len(written)} file(s) to {out_dir}")
    for p in written:
        Console().print(f"  [dim]·[/dim] {p.name}")
    if skipped:
        Console().print(
            f"[yellow]skipped {len(skipped)} existing file(s)[/yellow] "
            "(use --force to overwrite):"
        )
        for p in skipped:
            Console().print(f"  [dim]·[/dim] {p.name}")
    # Helpful nudge — every generator drops a README.
    if any(p.name.lower() == "readme.md" for p in written):
        Console().print(f"\nnext: open {out_dir / 'README.md'}")


def _default_output_dir(platform: str) -> Path:
    return Path.cwd() / f"eolas-{platform}"


@integrate_app.command("meltano")
def integrate_meltano(
    datasets: str           = typer.Option(..., "--datasets", "-d", help="Comma-separated dataset names, e.g. 'nz_cpi,nz_gdp'."),
    output:   Optional[Path]= typer.Option(None, "--output", "-o",   help="Output directory. Default: ./eolas-meltano/"),
    force:    bool          = typer.Option(False, "--force", "-f",   help="Overwrite existing files in the output directory."),
    json_out: bool          = typer.Option(False, "--json"),
    api_key:  Optional[str] = typer.Option(None, "--api-key"),
) -> None:
    """[verified] Generate a Meltano project (uses `tap-rest-api-msdk`) for the chosen datasets."""
    _run_integration("meltano", datasets, output or _default_output_dir("meltano"),
                     force, api_key, json_out)


@integrate_app.command("fivetran")
def integrate_fivetran(
    datasets: str           = typer.Option(..., "--datasets", "-d"),
    output:   Optional[Path]= typer.Option(None, "--output", "-o", help="Default: ./eolas-fivetran/"),
    force:    bool          = typer.Option(False, "--force", "-f"),
    json_out: bool          = typer.Option(False, "--json"),
    api_key:  Optional[str] = typer.Option(None, "--api-key"),
) -> None:
    """[experimental] Generate a Fivetran Connector Builder YAML for the chosen datasets.

    Output is structure-verified (parses as YAML, has all the fields the spec
    documents) but has not yet been end-to-end tested against a real Fivetran
    Connector Builder import. If the import rejects with a schema error,
    please share the error so the generator can be fixed.
    """
    _run_integration("fivetran", datasets, output or _default_output_dir("fivetran"),
                     force, api_key, json_out)
    if not _machine_mode(json_out):
        err_console.print(
            "[yellow]experimental:[/yellow] Fivetran output is structure-verified "
            "but not yet end-to-end tested against a real account."
        )


@integrate_app.command("azure-data-factory")
def integrate_adf(
    datasets: str           = typer.Option(..., "--datasets", "-d"),
    output:   Optional[Path]= typer.Option(None, "--output", "-o", help="Default: ./eolas-adf/"),
    force:    bool          = typer.Option(False, "--force", "-f"),
    json_out: bool          = typer.Option(False, "--json"),
    api_key:  Optional[str] = typer.Option(None, "--api-key"),
) -> None:
    """[experimental] Generate Azure Data Factory linked-service / dataset / pipeline JSON.

    Output is structure-verified (linked-service references resolve, datasets
    reference real linked services, pipeline activities reference real
    datasets) but has not yet been end-to-end tested against a real Azure
    subscription.
    """
    _run_integration("azure-data-factory", datasets,
                     output or _default_output_dir("adf"),
                     force, api_key, json_out)
    if not _machine_mode(json_out):
        err_console.print(
            "[yellow]experimental:[/yellow] Azure Data Factory output is "
            "structure-verified but not yet end-to-end tested against a real subscription."
        )


# Allow `python -m eolas_data.cli`
if __name__ == "__main__":
    app()
