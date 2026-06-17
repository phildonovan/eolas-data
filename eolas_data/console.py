"""Shared stderr console for programmatic user hints."""
from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

console = Console(
    stderr=True,
    theme=Theme({"path": "cyan", "key": "bold yellow"}),
)

_arrow_nagged = False


def nag_json_transport_once() -> None:
    """One-time hint when falling back to JSON instead of Arrow IPC."""
    global _arrow_nagged
    if _arrow_nagged:
        return
    _arrow_nagged = True
    console.print(
        "[dim]Using JSON transport (Arrow IPC unavailable for this server/session). "
        "For large datasets, prefer [bold]client.sync_bulk()[/bold] or "
        "[bold]client.get_local()[/bold].[/dim]"
    )