"""Regenerate ``_dataset_names.py`` from the live API.

Run before each release:

    python -m eolas_data._regen_names

Uses the same API-key resolution as :class:`~eolas_data.client.Client`
(env var → keyring → ``~/.eolas/config.json``). The catalog endpoint
requires authentication.

Writes to the same file inside the package. Commit the result.
"""
from __future__ import annotations

import datetime as _dt
import pathlib as _pathlib


def regenerate() -> None:
    from .client import Client

    df = Client().list()
    names = sorted(df["name"].tolist())
    today = _dt.date.today().isoformat()

    out = _pathlib.Path(__file__).with_name("_dataset_names.py")
    lines: list[str] = []
    lines.append('"""')
    lines.append("Type stubs for dataset names.")
    lines.append("")
    lines.append("Auto-generated from https://api.eolas.fyi/v1/datasets at release time.")
    lines.append(f"Snapshot: {today} ({len(names)} datasets).")
    lines.append("Regenerate before each release with `python -m eolas_data._regen_names`.")
    lines.append("")
    lines.append("At runtime this is just a string — `Literal[...]` only constrains static type")
    lines.append("checkers like mypy/pyright, so passing a name not in this list still works,")
    lines.append("it just doesn't autocomplete.")
    lines.append('"""')
    lines.append("from typing import Literal")
    lines.append("")
    lines.append(f'CATALOG_SNAPSHOT_DATE = "{today}"')
    lines.append(f"CATALOG_SNAPSHOT_COUNT = {len(names)}")
    lines.append("")
    lines.append("DatasetName = Literal[")
    for n in names:
        lines.append(f"    {n!r},")
    lines.append("]")
    lines.append("")
    lines.append("ALL_NAMES: tuple[str, ...] = (")
    for n in names:
        lines.append(f"    {n!r},")
    lines.append(")")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(names)} datasets to {out}")


if __name__ == "__main__":
    regenerate()