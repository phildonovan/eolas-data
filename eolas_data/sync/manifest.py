"""_eolas-manifest.json reader / writer for the multi-file sync model.

Each synced dataset directory contains one ``_eolas-manifest.json`` file that
records the full snapshot lineage.  This module owns reading and writing that
file.

The manifest is written **atomically**: the new content is serialised to a
``.tmp`` sibling, then moved over the canonical path with ``os.replace``.
Readers with the manifest open see either the old or the new version — never
a partial write.

File layout::

    ~/eolas-library/nz_parcels/
    ├── snapshot-2026-05-24.parquet
    ├── delta-2026-05-24-to-2026-05-31.parquet
    └── _eolas-manifest.json

Manifest schema (``schema_version: 1``)::

    {
      "dataset": "linz.nz_parcels",
      "snapshots": [
        {
          "snapshot_id": 5564541787213050514,   # int (Iceberg snapshot id)
          "kind": "snapshot",                    # "snapshot" | "delta"
          "file": "snapshot-2026-05-24.parquet", # relative to manifest dir
          "synced_at": "2026-05-24T11:05:00Z",  # ISO-8601 with Z
          "rows": 5431319                        # row count (int)
        },
        {
          "snapshot_id": 6789012345678901234,
          "kind": "delta",
          "parent_snapshot": 5564541787213050514,  # int
          "file": "delta-2026-05-24-to-2026-05-31.parquet",
          "synced_at": "2026-05-31T11:05:00Z",
          "rows_added": 2847                    # int (rows in this delta file only)
        }
      ],
      "current_snapshot": 6789012345678901234,  # int
      "format": "geoparquet",                   # "parquet" | "geoparquet"
      "schema_version": 1
    }
"""
from __future__ import annotations

import json
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "_eolas-manifest.json"
MANIFEST_SCHEMA_VERSION = 1

# ISO-8601 UTC timestamp pattern (YYYY-MM-DDTHH:MM:SSZ).
_ISO_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
)

# Valid relative file-name patterns:
#   snapshot-YYYY-MM-DD.parquet
#   snapshot-YYYY-MM-DD.geo.parquet
#   delta-YYYY-MM-DD-to-YYYY-MM-DD.parquet
#   delta-YYYY-MM-DD-to-YYYY-MM-DD.geo.parquet
_FILE_RE = re.compile(
    r"^(snapshot|delta)-\d{4}-\d{2}-\d{2}"
    r"(-to-\d{4}-\d{2}-\d{2})?"
    r"\.(geo\.)?parquet$"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ManifestEntry:
    """A single entry in the ``snapshots`` list.

    For ``kind="snapshot"`` entries the row count is stored in ``rows``.
    For ``kind="delta"`` entries the added-row count is in ``rows_added`` and
    the id of the base snapshot is in ``parent_snapshot``.
    """

    snapshot_id: int
    kind: Literal["snapshot", "delta"]
    file: str
    synced_at: str

    # snapshot-specific
    rows: Optional[int] = None

    # delta-specific
    parent_snapshot: Optional[int] = None
    rows_added: Optional[int] = None

    def validate(self) -> None:
        """Raise ``ValueError`` if the entry is internally inconsistent."""
        if not isinstance(self.snapshot_id, int):
            raise ValueError(
                f"ManifestEntry.snapshot_id must be int, got {type(self.snapshot_id).__name__!r}"
            )
        if self.kind not in ("snapshot", "delta"):
            raise ValueError(f"ManifestEntry.kind must be 'snapshot' or 'delta', got {self.kind!r}")
        if not _FILE_RE.match(self.file):
            raise ValueError(
                f"ManifestEntry.file {self.file!r} does not match expected naming pattern "
                "(snapshot-YYYY-MM-DD.parquet or delta-YYYY-MM-DD-to-YYYY-MM-DD.parquet, "
                ".geo.parquet variants also valid)."
            )
        if not _ISO_UTC_RE.match(self.synced_at):
            raise ValueError(
                f"ManifestEntry.synced_at {self.synced_at!r} must be ISO-8601 UTC "
                "(e.g. '2026-05-24T11:05:00Z')"
            )
        if self.kind == "snapshot":
            if self.rows is None:
                raise ValueError("ManifestEntry with kind='snapshot' must have 'rows' set")
            if not isinstance(self.rows, int) or self.rows < 0:
                raise ValueError(f"ManifestEntry.rows must be a non-negative int, got {self.rows!r}")
        if self.kind == "delta":
            if self.parent_snapshot is None:
                raise ValueError("ManifestEntry with kind='delta' must have 'parent_snapshot' set")
            if not isinstance(self.parent_snapshot, int):
                raise ValueError(
                    f"ManifestEntry.parent_snapshot must be int, got {type(self.parent_snapshot).__name__!r}"
                )
            if self.rows_added is None:
                raise ValueError("ManifestEntry with kind='delta' must have 'rows_added' set")
            if not isinstance(self.rows_added, int) or self.rows_added < 0:
                raise ValueError(
                    f"ManifestEntry.rows_added must be a non-negative int, got {self.rows_added!r}"
                )

    def to_dict(self) -> dict:
        d: dict = {
            "snapshot_id": self.snapshot_id,
            "kind": self.kind,
            "file": self.file,
            "synced_at": self.synced_at,
        }
        if self.kind == "snapshot":
            d["rows"] = self.rows
        else:
            d["parent_snapshot"] = self.parent_snapshot
            d["rows_added"] = self.rows_added
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ManifestEntry":
        return cls(
            snapshot_id=d["snapshot_id"],
            kind=d["kind"],
            file=d["file"],
            synced_at=d["synced_at"],
            rows=d.get("rows"),
            parent_snapshot=d.get("parent_snapshot"),
            rows_added=d.get("rows_added"),
        )


@dataclass
class Manifest:
    """In-memory representation of ``_eolas-manifest.json``.

    Attributes:
        dataset:          Fully-qualified dataset name (e.g. ``"linz.nz_parcels"``).
        snapshots:        Ordered list of snapshot/delta entries, oldest first.
        current_snapshot: Iceberg snapshot id of the most recent entry.
        format:           ``"parquet"`` or ``"geoparquet"``.
        schema_version:   Always 1 in this implementation.
    """

    dataset: str
    snapshots: List[ManifestEntry] = field(default_factory=list)
    current_snapshot: Optional[int] = None
    format: str = "parquet"
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def validate(self) -> None:
        """Raise ``ValueError`` if the manifest is internally inconsistent."""
        if not self.dataset:
            raise ValueError("Manifest.dataset must not be empty")
        if self.format not in ("parquet", "geoparquet"):
            raise ValueError(
                f"Manifest.format must be 'parquet' or 'geoparquet', got {self.format!r}"
            )
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Manifest.schema_version {self.schema_version!r} is not supported "
                f"(expected {MANIFEST_SCHEMA_VERSION})"
            )
        for entry in self.snapshots:
            entry.validate()
        # current_snapshot must match one of the snapshot_ids (when set).
        if self.current_snapshot is not None and self.snapshots:
            ids = {e.snapshot_id for e in self.snapshots}
            if self.current_snapshot not in ids:
                raise ValueError(
                    f"Manifest.current_snapshot {self.current_snapshot!r} "
                    f"is not found in snapshots list (ids: {ids!r})"
                )

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "snapshots": [e.to_dict() for e in self.snapshots],
            "current_snapshot": self.current_snapshot,
            "format": self.format,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        return cls(
            dataset=d["dataset"],
            snapshots=[ManifestEntry.from_dict(e) for e in d.get("snapshots", [])],
            current_snapshot=d.get("current_snapshot"),
            format=d.get("format", "parquet"),
            schema_version=d.get("schema_version", MANIFEST_SCHEMA_VERSION),
        )


# ---------------------------------------------------------------------------
# Public I/O functions
# ---------------------------------------------------------------------------

def read_manifest(manifest_path: Union[str, pathlib.Path]) -> Optional[Manifest]:
    """Read and parse ``_eolas-manifest.json`` from *manifest_path*.

    Args:
        manifest_path: Full path to the manifest file (not the directory).

    Returns:
        A :class:`Manifest` on success, or ``None`` when the file does not
        exist.  Raises ``ValueError`` on parse / validation errors so callers
        can distinguish "first sync" (``None``) from "corrupt manifest"
        (exception).
    """
    p = pathlib.Path(manifest_path)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest at {p} is not valid JSON: {exc}") from exc
    try:
        m = Manifest.from_dict(raw)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Manifest at {p} has missing or wrong-typed fields: {exc}") from exc
    m.validate()
    return m


def write_manifest(
    manifest: Manifest,
    manifest_path: Union[str, pathlib.Path],
) -> None:
    """Write *manifest* to *manifest_path* atomically.

    The content is serialised to a sibling ``.tmp`` file and then renamed over
    the canonical path with ``os.replace``.  This guarantees that readers see
    either the fully-written new content or the old content — never a partial
    write.

    Args:
        manifest:      The :class:`Manifest` to serialise.
        manifest_path: Full path to the destination manifest file.

    Raises:
        ValueError: If ``manifest.validate()`` fails.
        OSError:    If the parent directory does not exist or is not writable.
    """
    manifest.validate()
    p = pathlib.Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp-{os.urandom(4).hex()}")
    try:
        tmp.write_text(
            json.dumps(manifest.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, p)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise
