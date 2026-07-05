"""Library directory resolution for eolas-data.

Implements the precedence chain for where ``sync_bulk()`` / ``download_bulk()``
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
import time
from typing import Optional

from rich.prompt import Prompt

from .console import console as _console

_log = logging.getLogger("eolas_data")

# Per-session flag: have we already prompted (or decided not to)?
_prompt_done: bool = False

# Config file (same as CLI's auth config)
_CONFIG_DIR = pathlib.Path.home() / ".eolas"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

# Fallback cache directory (Step 5)
_FALLBACK_DIR = pathlib.Path.home() / ".cache" / "eolas"

# Bulk file extensions (mirrors Client._BULK_EXTENSIONS).
_BULK_EXTENSIONS = {
    "parquet": ".parquet",
    "csv_gz": ".csv.gz",
    "geoparquet": ".geo.parquet",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_library_dir(*, interactive: bool = True) -> pathlib.Path:
    """Return the resolved library directory, following the precedence chain.

    The result is always an absolute, ``~``-expanded path.  The directory is
    **not** created here — callers (``sync_bulk``, ``download_bulk``) do
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


def cache_clear(
    name: Optional[str] = None,
    *,
    cache_dir: Optional[str | pathlib.Path] = None,
    format: Optional[str] = None,
    files: bool = True,
    meta: bool = True,
    meta_cache: Optional[dict] = None,
) -> dict:
    """Clear client-side cache for a dataset (or the whole library).

    eolas-data caches at two client-only levels:

    * **On-disk bulk files** in the library directory (Parquet/GeoParquet +
      ``.eolas-meta.json`` sidecars) — cleared when ``files=True``.
    * **Session metadata** (``Client._meta_cache`` / ``info()`` per dataset) —
      cleared when ``meta=True`` and ``meta_cache`` is the client's dict.

    Does not contact the API. Use :meth:`Client.get_local` with ``force=True``
    to clear and re-download in one step.

    Args:
        name: Dataset identifier. ``None`` sweeps the whole library (``files``)
            and/or all session metadata entries (``meta``).
        cache_dir: Library directory. ``None`` uses :func:`resolve_library_dir`.
        format: ``"parquet"``, ``"csv_gz"``, or ``"geoparquet"``. ``None``
            deletes all bulk variants for ``name`` (ignored when ``name`` is
            ``None``).
        files: Delete on-disk bulk data files and sidecars.
        meta: Drop session-cached ``info()`` entries from ``meta_cache``.
        meta_cache: The client's ``_meta_cache`` dict (required for ``meta``).

    Returns:
        ``{"files": [deleted paths], "meta_cleared": int}``
    """
    deleted: list[str] = []
    meta_n = 0

    if meta and meta_cache is not None:
        if name is None:
            meta_n = len(meta_cache)
            meta_cache.clear()
        else:
            key = str(name)
            if key in meta_cache:
                del meta_cache[key]
                meta_n = 1

    if files:
        root = (
            pathlib.Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else resolve_library_dir(interactive=False)
        )
        if name is None:
            if root.is_dir():
                for p in root.iterdir():
                    if p.suffix in {".parquet", ".gz"} or p.name.endswith(
                        ".geo.parquet"
                    ):
                        p.unlink(missing_ok=True)
                        deleted.append(str(p))
                    elif p.name.endswith(".eolas-meta.json"):
                        p.unlink(missing_ok=True)
                        deleted.append(str(p))
                    elif ".eolas-tmp-" in p.name:
                        # Orphaned partial downloads — a full library clear takes
                        # them all regardless of age (PY-5).
                        p.unlink(missing_ok=True)
                        deleted.append(str(p))
        elif format is None:
            for ext in _BULK_EXTENSIONS.values():
                p = root / f"{name}{ext}"
                for candidate in (p, pathlib.Path(str(p) + ".eolas-meta.json")):
                    if candidate.exists():
                        candidate.unlink()
                        deleted.append(str(candidate))
        else:
            fmt = format.lower()
            if fmt not in _BULK_EXTENSIONS:
                raise ValueError(
                    f"Unknown format {format!r}. Expected one of: "
                    + ", ".join(_BULK_EXTENSIONS)
                )
            p = root / f"{name}{_BULK_EXTENSIONS[fmt]}"
            for candidate in (p, pathlib.Path(str(p) + ".eolas-meta.json")):
                if candidate.exists():
                    candidate.unlink()
                    deleted.append(str(candidate))

    return {"files": deleted, "meta_cleared": meta_n}


def sweep_stale_tmp_files(
    cache_dir: Optional[str | pathlib.Path] = None,
    *,
    older_than_hours: float = 24.0,
) -> list[str]:
    """Delete orphaned ``*.eolas-tmp-*`` partial-download files in the library.

    A download interrupted before the atomic rename leaves a ``<name>.eolas-tmp-
    <rand>`` file behind. These are never read (the reader only ever opens the
    final path) and are otherwise never garbage-collected (PY-5 found a 16 MB
    partial 16 days old). We sweep any older than ``older_than_hours`` so an
    in-flight concurrent download's tmp file is never touched.

    Returns the list of deleted paths. Never raises — best-effort cleanup.
    """
    deleted: list[str] = []
    try:
        root = (
            pathlib.Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else resolve_library_dir(interactive=False)
        )
        if not root.is_dir():
            return deleted
        cutoff = time.time() - older_than_hours * 3600
        for p in root.glob("*.eolas-tmp-*"):
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
                    deleted.append(str(p))
            except OSError:
                continue
    except Exception:
        pass
    return deleted


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
        "source": source,
        "path": resolved,
        "env_var": env,
        "config_file": str(_CONFIG_FILE),
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
    _console.print(
        f"  [green]1[/green]) [path]~/eolas-library[/path]      user-wide, persistent [dim](recommended)[/dim]"
    )
    _console.print(
        f"  [green]2[/green]) [path]./eolas-library[/path]      this project"
    )
    _console.print("  [green]3[/green]) Custom path")
    _console.print(
        "  [green]4[/green]) Stay with [path]~/.cache/eolas[/path] [dim](don't ask again)[/dim]"
    )
    _console.print()

    try:
        raw = Prompt.ask(
            "Choice", choices=["1", "2", "3", "4"], default="1", console=_console
        )
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
