"""Library directory resolution for eolas-data.

Implements the precedence chain for where ``get_local()`` / ``sync_bulk()``
cache data files on disk:

  1. Explicit ``cache_dir=`` arg passed to the method   (handled by callers)
  2. ``EOLAS_LIBRARY`` environment variable
  3. ``library_dir`` in ``~/.eolas/config.json``
  4. Interactive prompt (TTY only, first-time per session)
  5. ``~/.cache/eolas/``                                 (silent fallback)

Cross-language note: the config file ``~/.eolas/config.json`` is the same
JSON file used by the CLI for ``api_key`` storage.  ``library_dir`` is simply
an additional key.  The R client reads the same file, so a library path set
from Python is honoured in R and vice versa.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.theme import Theme

# Single shared console — writes to stderr so library status doesn't pollute
# stdout (where JSON/data output goes from CLI commands). Custom theme styles
# (`path`, `key`) are used for inline emphasis without leaking ansi codes
# to non-tty consumers.
_console = Console(
    stderr=True,
    theme=Theme({"path": "cyan", "key": "bold yellow"}),
)

_log = logging.getLogger("eolas_data")

# Per-session flag: have we already prompted (or decided not to)?
_prompt_done: bool = False

# Config file (same as CLI's auth config)
_CONFIG_DIR  = pathlib.Path.home() / ".eolas"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

# Fallback cache directory (Step 5)
_FALLBACK_DIR = pathlib.Path.home() / ".cache" / "eolas"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_library_dir(*, interactive: bool = True) -> pathlib.Path:
    """Return the resolved library directory, following the precedence chain.

    The result is always an absolute, ``~``-expanded path.  The directory is
    **not** created here — callers (``get_local``, ``_get_local_impl``) do
    that themselves immediately before writing.

    Args:
        interactive: When ``True`` (default) and the caller is running in a
            TTY session, a first-time interactive prompt may fire to ask the
            user where they want to keep their eolas files.  Pass
            ``False`` to suppress prompting (useful in tests).
    """
    # Step 2: EOLAS_LIBRARY env var
    env = os.environ.get("EOLAS_LIBRARY", "").strip()
    if env:
        return pathlib.Path(env).expanduser().resolve()

    # Step 3: config file library_dir
    cfg = _read_library_dir_from_config()
    if cfg:
        return pathlib.Path(cfg).expanduser().resolve()

    # Step 4: interactive prompt (TTY only, once per session)
    if interactive and _is_tty():
        prompted = _maybe_prompt()
        if prompted:
            return pathlib.Path(prompted).expanduser().resolve()

    # Step 5: silent fallback
    _emit_headless_info_once()
    return _FALLBACK_DIR


def library_set(path: str) -> pathlib.Path:
    """Write ``library_dir`` to ``~/.eolas/config.json`` and return the resolved path.

    Creates the config directory if needed.  Used by both the CLI
    ``eolas library set`` command and programmatically.
    """
    resolved = pathlib.Path(path).expanduser().resolve()
    _write_config_key("library_dir", str(resolved))
    return resolved


def library_clear() -> None:
    """Remove ``library_dir`` from ``~/.eolas/config.json`` (if present)."""
    _remove_config_key("library_dir")


def library_status() -> dict:
    """Return a dict describing how the library directory is currently resolved.

    Keys:
      - ``source``: one of ``"env"``, ``"config"``, ``"fallback"``
      - ``path``: the resolved path (string)
      - ``env_var``: the value of ``EOLAS_LIBRARY`` (may be empty)
      - ``config_file``: path to the config file (string)
      - ``config_value``: the ``library_dir`` value in config (may be empty)
    """
    env = os.environ.get("EOLAS_LIBRARY", "").strip()
    cfg = _read_library_dir_from_config()

    if env:
        source = "env"
        resolved = str(pathlib.Path(env).expanduser().resolve())
    elif cfg:
        source = "config"
        resolved = str(pathlib.Path(cfg).expanduser().resolve())
    else:
        source = "fallback"
        resolved = str(_FALLBACK_DIR)

    return {
        "source":       source,
        "path":         resolved,
        "env_var":      env,
        "config_file":  str(_CONFIG_FILE),
        "config_value": cfg or "",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_headless_info_emitted: bool = False


def _emit_headless_info_once() -> None:
    """Emit a one-time INFO log when the fallback is used non-interactively."""
    global _headless_info_emitted
    if _headless_info_emitted:
        return
    _headless_info_emitted = True
    _log.info(
        "eolas-data: using ~/.cache/eolas/ (transient). "
        "Set EOLAS_LIBRARY or run interactively to configure a persistent library."
    )


def _is_tty() -> bool:
    """True when stdin is a real TTY (not piped / CI)."""
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _read_library_dir_from_config() -> str:
    """Read ``library_dir`` from the config file; return ``""`` if absent/unreadable."""
    try:
        if _CONFIG_FILE.exists():
            data = json.loads(_CONFIG_FILE.read_text())
            return str(data.get("library_dir", "") or "")
    except Exception:
        pass
    return ""


def _write_config_key(key: str, value: str) -> None:
    """Write (or update) a single key in ``~/.eolas/config.json``."""
    _CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    data: dict = {}
    try:
        if _CONFIG_FILE.exists():
            data = json.loads(_CONFIG_FILE.read_text())
    except Exception:
        data = {}
    data[key] = value
    _CONFIG_FILE.write_text(json.dumps(data, indent=2) + "\n")
    _CONFIG_FILE.chmod(0o600)


def _remove_config_key(key: str) -> None:
    """Remove a key from ``~/.eolas/config.json`` (no-op if absent)."""
    try:
        if not _CONFIG_FILE.exists():
            return
        data = json.loads(_CONFIG_FILE.read_text())
        data.pop(key, None)
        _CONFIG_FILE.write_text(json.dumps(data, indent=2) + "\n")
        _CONFIG_FILE.chmod(0o600)
    except Exception:
        pass


def _maybe_prompt() -> Optional[str]:
    """Show the first-time library prompt.  Returns the chosen path, or ``None``.

    Only fires once per session (guarded by ``_prompt_done``).  Writes the
    user's choice to the config file so future sessions skip the prompt.
    """
    global _prompt_done
    if _prompt_done:
        return None
    _prompt_done = True

    home = pathlib.Path.home()
    opt1 = str(home / "eolas-library")
    opt2 = str(pathlib.Path(".").resolve() / "eolas-library")

    _console.print()
    _console.print("[bold]eolas-data:[/bold] No library configured.")
    _console.print()
    _console.print(
        "This dataset would be cached at [path]~/.cache/eolas/[/path] "
        "(transient OS cache — fine for one-off use, but cleared if your "
        "OS hits cache pressure).",
        style="dim",
        highlight=False,
    )
    _console.print()
    _console.print("For reproducible pipelines, set up a library:")
    _console.print(f"  [green]1[/green]) [path]~/eolas-library[/path]      user-wide, persistent [dim](recommended)[/dim]")
    _console.print(f"  [green]2[/green]) [path]./eolas-library[/path]      this project")
    _console.print( "  [green]3[/green]) Custom path")
    _console.print( "  [green]4[/green]) Stay with [path]~/.cache/eolas[/path] [dim](don't ask again)[/dim]")
    _console.print()

    try:
        raw = Prompt.ask("Choice", choices=["1", "2", "3", "4"], default="1", console=_console)
    except (EOFError, KeyboardInterrupt):
        _console.print()
        return None

    if raw == "1":
        chosen = opt1
    elif raw == "2":
        chosen = opt2
    elif raw == "3":
        try:
            custom = Prompt.ask("Enter path", console=_console).strip()
        except (EOFError, KeyboardInterrupt):
            _console.print()
            return None
        if not custom:
            # Fallback silently
            return None
        chosen = custom
    elif raw == "4":
        # "Stay with ~/.cache/eolas" — write it so we don't prompt again
        chosen = str(_FALLBACK_DIR)
    else:
        # Unrecognised input: fall through to cache silently
        return None

    resolved = str(pathlib.Path(chosen).expanduser().resolve())
    _write_config_key("library_dir", resolved)
    _console.print(
        f"[green]✓[/green] library set to [path]{resolved}[/path] "
        f"[dim](saved to {_CONFIG_FILE})[/dim]"
    )
    return resolved
