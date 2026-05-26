"""eolas_data.sync — multi-file dataset directory model for pipeline sync.

Public symbols re-exported here for convenience:

    from eolas_data.sync import SyncResult, read_manifest, write_manifest
"""
from .manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    ManifestEntry,
    Manifest,
    read_manifest,
    write_manifest,
)
from .sync import SyncResult, sync_dataset

__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestEntry",
    "Manifest",
    "read_manifest",
    "write_manifest",
    "SyncResult",
    "sync_dataset",
]
