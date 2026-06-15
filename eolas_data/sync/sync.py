"""client.sync() implementation — multi-file dataset directory model.

Decision logic
--------------
::

    sync(name, library_dir):
      manifest = read_manifest(library_dir/<name>/_eolas-manifest.json) | None
      meta     = GET /v1/datasets/{name}           # current_snapshot_id, incremental_supported

      if not manifest:
        # First sync — full bulk download
        GET /v1/bulk/{ns}/{table}?format=geoparquet|parquet
          → snapshot-<YYYY-MM-DD>.parquet
        write manifest (one snapshot entry)
        return SyncResult(status="snapshot_full", ...)

      if manifest.current_snapshot == meta.current_snapshot_id:
        return SyncResult(status="unchanged", ...)     # zero I/O

      if not meta.incremental_supported:
        # Full re-download, orphan old files
        GET /v1/bulk/{ns}/{table}?...
          → snapshot-<YYYY-MM-DD>.parquet
        rewrite manifest (new snapshot entry only)
        return SyncResult(status="snapshot_full", ...)

      # Try incremental delta
      resp = GET /v1/datasets/{name}/data/incremental
               ?since_snapshot=<manifest.current_snapshot>
               &format=geoparquet|parquet
      if resp.status == 410:
        # Lineage broken — full re-download
        GET /v1/bulk/{ns}/{table}?...
        rewrite manifest
        return SyncResult(status="snapshot_full", ...)
      if resp.status == 400:
        # Server reports incremental_supported=false at request time
        GET /v1/bulk/{ns}/{table}?...
        rewrite manifest
        return SyncResult(status="snapshot_full", ...)
      if resp.status == 200 and X-Eolas-Row-Count == 0:
        # Empty delta (since == current; shouldn't reach here but handle it)
        return SyncResult(status="unchanged", ...)
      # 200 with body → save delta file, update manifest
      save body → delta-<from>-to-<to>.parquet
      append entry to manifest.snapshots
      update manifest.current_snapshot
      return SyncResult(status="snapshot_delta", ...)

All file writes are atomic (tmp + os.replace).

Public surface
--------------
``sync_dataset(client, name, library_dir, ...)`` is the implementation
function. ``Client.sync()`` is a thin wrapper method in ``client.py`` that
delegates here, keeping this module free of the ``Client`` circular import.
"""
from __future__ import annotations

import datetime
import logging
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

from .manifest import (
    MANIFEST_FILENAME,
    Manifest,
    ManifestEntry,
    read_manifest,
    write_manifest,
)

if TYPE_CHECKING:
    from ..client import Client

_log = logging.getLogger("eolas_data")


# ---------------------------------------------------------------------------
# SyncResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyncResult:
    """Result of a :meth:`Client.sync` call.

    Attributes:
        status:
            ``"snapshot_full"`` — a full bulk snapshot was downloaded (first
            sync, or a fallback when incremental was unavailable / lineage
            broken).

            ``"snapshot_delta"`` — an incremental delta was downloaded and
            appended to the dataset directory.

            ``"unchanged"`` — the server snapshot id matches the local
            manifest; no I/O was performed on the data files.

            ``"error"`` — an exception was raised while syncing this dataset
            (only set by :func:`sync_all` when it catches a per-dataset
            failure; ``error`` carries the exception string repr).

        dataset:
            The dataset name passed to ``sync()``.

        library_dir:
            The resolved library directory (absolute path).

        bytes_downloaded:
            Total bytes written to disk in this call.  ``0`` for
            ``"unchanged"`` and ``"error"``.

        rows_added:
            Number of new rows brought in.

            - For ``"snapshot_full"``: total rows in the snapshot file
              (from the ``x-eolas-row-count`` header if available, otherwise
              counted from the parquet metadata).
            - For ``"snapshot_delta"``: rows in the delta file (from the
              ``X-Eolas-Row-Count`` response header, or counted if absent).
            - For ``"unchanged"`` or ``"error"``: ``0``.

        files_added:
            Number of new parquet files written.  ``0`` for ``"unchanged"``
            and ``"error"``.

        error:
            ``None`` for normal statuses.  When ``status="error"`` this holds
            the string representation of the exception that was raised.
    """

    status: str
    dataset: str
    library_dir: pathlib.Path
    bytes_downloaded: int
    rows_added: int
    files_added: int
    error: Optional[str] = field(default=None, compare=False)

    def __repr__(self) -> str:
        if self.error:
            return (
                f"<SyncResult dataset={self.dataset!r} status={self.status!r} "
                f"error={self.error!r}>"
            )
        return (
            f"<SyncResult dataset={self.dataset!r} status={self.status!r} "
            f"rows_added={self.rows_added} bytes_downloaded={self.bytes_downloaded}>"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string with Z suffix."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_str() -> str:
    """Return today's UTC date as YYYY-MM-DD."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _count_parquet_rows(path: pathlib.Path) -> int:
    """Return the row count stored in the parquet file footer metadata.

    Falls back to ``0`` on any error (pyarrow not installed, corrupt file, etc.)
    so the caller can continue; row counts are informational.
    """
    try:
        import pyarrow.parquet as pq
        meta = pq.read_metadata(path)
        return meta.num_rows
    except Exception:
        return 0


def _atomic_write(dest: pathlib.Path, data: bytes) -> int:
    """Write *data* to *dest* atomically (tmp + os.replace).

    Returns the number of bytes written.
    Raises on any I/O error (the original file is never corrupted).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".eolas-tmp-{os.urandom(4).hex()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return len(data)


def _stream_atomic_write(
    resp,                     # requests.Response (streaming)
    dest: pathlib.Path,
    *,
    total_bytes: Optional[int],
    label: str,
    show_progress: bool,
) -> int:
    """Stream *resp* body to *dest* atomically.  Returns bytes written."""
    import tqdm.auto

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".eolas-tmp-{os.urandom(4).hex()}")
    bytes_written = 0
    try:
        with tqdm.auto.tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=label,
            disable=not show_progress,
            leave=False,
        ) as bar:
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
                        bar.update(len(chunk))
                        bytes_written += len(chunk)
        os.replace(tmp, dest)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return bytes_written


# ---------------------------------------------------------------------------
# Public implementation function
# ---------------------------------------------------------------------------

def sync_dataset(
    client: "Client",
    name: str,
    library_dir: Union[str, pathlib.Path],
    *,
    progress: Optional[bool] = None,
) -> SyncResult:
    """Implement ``client.sync(name, library_dir=...)``.

    Parameters
    ----------
    client:
        The :class:`~eolas_data.Client` instance (provides auth + HTTP helpers).
    name:
        Dataset identifier, e.g. ``"doc_huts"`` or ``"nz_parcels"``.
    library_dir:
        Root directory of the local data library.  A subdirectory named
        ``<name>`` is created inside it (e.g. ``library_dir/doc_huts/``).
        Accepts ``~``-prefixed paths.
    progress:
        Tri-state progress bar override.  ``None`` (default) auto-detects.
        ``True`` forces on; ``False`` forces off.

    Returns
    -------
    SyncResult
    """
    lib = pathlib.Path(library_dir).expanduser().resolve()
    dataset_dir = lib / name
    manifest_path = dataset_dir / MANIFEST_FILENAME

    # ------------------------------------------------------------------
    # 1. Read local manifest (None → first sync)
    # ------------------------------------------------------------------
    manifest = read_manifest(manifest_path)

    # ------------------------------------------------------------------
    # 2. Fetch dataset metadata (includes current_snapshot_id,
    #    incremental_supported, namespace, table, has_geometry)
    # ------------------------------------------------------------------
    meta = client._get(f"/v1/datasets/{name}")
    namespace = meta.get("namespace") or ""
    table = meta.get("table") or meta.get("name") or name
    if not namespace:
        from ..exceptions import NotFoundError
        raise NotFoundError(
            f"Dataset {name!r} metadata did not include a namespace field. "
            "Cannot construct bulk URL."
        )

    current_snapshot_id_raw = meta.get("current_snapshot_id")
    # The server may return snapshot IDs as ints or strings — normalise to int.
    try:
        current_snapshot_id: Optional[int] = (
            int(current_snapshot_id_raw)
            if current_snapshot_id_raw is not None
            else None
        )
    except (TypeError, ValueError):
        current_snapshot_id = None

    # Fail-safe default False (matches server catalog default): incremental is opt-in.
    # If metadata omits the field, do NOT attempt a delta — fall back to full bulk.
    incremental_supported: bool = bool(meta.get("incremental_supported", False))

    # Determine format from metadata.
    gt  = meta.get("geometry_type")
    wkt = meta.get("geometry_wkt")
    gt_truthy  = bool(gt)  and gt  != "none"
    wkt_truthy = bool(wkt) and wkt != "none"
    is_geo = gt_truthy or wkt_truthy or bool(meta.get("has_geometry"))
    fmt = "geoparquet" if is_geo else "parquet"

    bulk_path = f"/v1/bulk/{namespace}/{table}"
    bulk_params: dict = {"format": fmt}

    show = client._resolve_show_progress(progress)

    # ------------------------------------------------------------------
    # 3. No local manifest → first sync (full bulk download)
    # ------------------------------------------------------------------
    if manifest is None:
        return _do_full_download(
            client=client,
            name=name,
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            bulk_path=bulk_path,
            bulk_params=bulk_params,
            fmt=fmt,
            current_snapshot_id=current_snapshot_id,
            show=show,
        )

    # ------------------------------------------------------------------
    # 4. Manifest exists but we don't know the server snapshot — treat
    #    as changed (conservative; shouldn't happen normally).
    # ------------------------------------------------------------------
    if current_snapshot_id is None:
        _log.warning(
            "sync(%r): server did not return current_snapshot_id; "
            "falling back to full download.",
            name,
        )
        return _do_full_download(
            client=client,
            name=name,
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            bulk_path=bulk_path,
            bulk_params=bulk_params,
            fmt=fmt,
            current_snapshot_id=current_snapshot_id,
            show=show,
            existing_manifest=manifest,
        )

    # ------------------------------------------------------------------
    # 5. Snapshot unchanged → return immediately
    # ------------------------------------------------------------------
    if manifest.current_snapshot == current_snapshot_id:
        return SyncResult(
            status="unchanged",
            dataset=name,
            library_dir=lib,
            bytes_downloaded=0,
            rows_added=0,
            files_added=0,
        )

    # ------------------------------------------------------------------
    # 6. Snapshot changed — decide incremental vs full
    # ------------------------------------------------------------------
    if not incremental_supported:
        _log.debug(
            "sync(%r): incremental_supported=False → full re-download", name
        )
        return _do_full_download(
            client=client,
            name=name,
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            bulk_path=bulk_path,
            bulk_params=bulk_params,
            fmt=fmt,
            current_snapshot_id=current_snapshot_id,
            show=show,
            existing_manifest=manifest,
        )

    # ------------------------------------------------------------------
    # 7. Attempt incremental fetch
    # ------------------------------------------------------------------
    since_id = manifest.current_snapshot
    incremental_url = f"/v1/datasets/{name}/data/incremental"
    incremental_params = {
        "since_snapshot": since_id,
        "format": fmt,
    }

    raw_resp = client._session.get(
        f"{client._base}{incremental_url}",
        params=incremental_params,
        stream=True,
    )

    # 410 or 400 → fall back to full download
    if raw_resp.status_code in (400, 410):
        _log.debug(
            "sync(%r): incremental returned %d → full re-download",
            name,
            raw_resp.status_code,
        )
        return _do_full_download(
            client=client,
            name=name,
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            bulk_path=bulk_path,
            bulk_params=bulk_params,
            fmt=fmt,
            current_snapshot_id=current_snapshot_id,
            show=show,
            existing_manifest=manifest,
        )

    # Raise on any other non-200
    if raw_resp.status_code != 200:
        from ..client import Client as _Client
        _Client._raise_for_status(raw_resp)

    # Inspect row count header.
    row_count_hdr = raw_resp.headers.get("X-Eolas-Row-Count", "")
    try:
        server_row_count = int(row_count_hdr)
    except (ValueError, TypeError):
        server_row_count = None

    # Server says 200 but zero rows added (since == current edge case)
    if server_row_count == 0:
        # Consume the (empty) body to be a good HTTP citizen.
        raw_resp.content  # noqa: B018
        return SyncResult(
            status="unchanged",
            dataset=name,
            library_dir=lib,
            bytes_downloaded=0,
            rows_added=0,
            files_added=0,
        )

    # ------------------------------------------------------------------
    # 8. Save the delta file
    # ------------------------------------------------------------------
    # Derive date range from the snapshot ids we know.
    # We use today's date for the 'to' side and read the 'from' side
    # from the synced_at of the current manifest tail entry.
    from_date = _date_from_manifest_tail(manifest)
    to_date = _date_from_header(raw_resp, current_snapshot_id) or _today_str()
    ext = ".geo.parquet" if fmt == "geoparquet" else ".parquet"
    delta_filename = f"delta-{from_date}-to-{to_date}{ext}"
    delta_path = dataset_dir / delta_filename

    total_bytes = int(raw_resp.headers.get("Content-Length", 0)) or None
    bytes_dl = _stream_atomic_write(
        raw_resp,
        delta_path,
        total_bytes=total_bytes,
        label=delta_filename,
        show_progress=show,
    )

    # Read row count from the file if the header was missing.
    if server_row_count is None:
        server_row_count = _count_parquet_rows(delta_path)

    # ------------------------------------------------------------------
    # 9. Update manifest — append delta entry, preserve old snapshots
    # ------------------------------------------------------------------
    new_entry = ManifestEntry(
        snapshot_id=current_snapshot_id,
        kind="delta",
        parent_snapshot=since_id,
        file=delta_filename,
        synced_at=_utc_now(),
        rows_added=server_row_count,
    )
    manifest.snapshots.append(new_entry)
    manifest.current_snapshot = current_snapshot_id
    write_manifest(manifest, manifest_path)

    return SyncResult(
        status="snapshot_delta",
        dataset=name,
        library_dir=lib,
        bytes_downloaded=bytes_dl,
        rows_added=server_row_count,
        files_added=1,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _do_full_download(
    *,
    client: "Client",
    name: str,
    dataset_dir: pathlib.Path,
    manifest_path: pathlib.Path,
    bulk_path: str,
    bulk_params: dict,
    fmt: str,
    current_snapshot_id: Optional[int],
    show: bool,
    existing_manifest: Optional[Manifest] = None,
) -> SyncResult:
    """Download the full bulk snapshot and write a fresh manifest.

    Old snapshot / delta files are left in place as orphans; ``compact()``
    (Day 2) will roll them up.  This is intentional — we never delete files
    here so that a concurrent reader isn't interrupted.
    """
    today = _today_str()
    ext = ".geo.parquet" if fmt == "geoparquet" else ".parquet"
    snapshot_filename = f"snapshot-{today}{ext}"
    snapshot_path = dataset_dir / snapshot_filename

    resp = client._raw_bulk_get(bulk_path, params=bulk_params, stream=True)
    total_bytes = int(resp.headers.get("Content-Length", 0)) or None

    bytes_dl = _stream_atomic_write(
        resp,
        snapshot_path,
        total_bytes=total_bytes,
        label=snapshot_filename,
        show_progress=show,
    )

    if bytes_dl == 0:
        from ..exceptions import APIError
        raise APIError(
            200,
            f"Bulk download for {name!r} returned an empty body (0 bytes). "
            "The snapshot may not exist for this format. "
            "Try format='parquet' for non-geo datasets.",
        )

    row_count = _count_parquet_rows(snapshot_path)

    # Derive snapshot id: use what the server told us, or a placeholder.
    snap_id = current_snapshot_id if current_snapshot_id is not None else 0

    new_entry = ManifestEntry(
        snapshot_id=snap_id,
        kind="snapshot",
        file=snapshot_filename,
        synced_at=_utc_now(),
        rows=row_count,
    )
    new_manifest = Manifest(
        dataset=name,
        snapshots=[new_entry],
        current_snapshot=snap_id,
        format=fmt,
        schema_version=1,
    )
    write_manifest(new_manifest, manifest_path)

    return SyncResult(
        status="snapshot_full",
        dataset=name,
        library_dir=dataset_dir.parent,
        bytes_downloaded=bytes_dl,
        rows_added=row_count,
        files_added=1,
    )


def _date_from_manifest_tail(manifest: Manifest) -> str:
    """Return the synced_at date of the last entry as YYYY-MM-DD.

    Falls back to today if the manifest is empty or synced_at is unparseable.
    """
    if manifest.snapshots:
        tail = manifest.snapshots[-1]
        # synced_at is "YYYY-MM-DDTHH:MM:SSZ" — the date part is the first 10 chars.
        return tail.synced_at[:10]
    return _today_str()


def _date_from_header(resp, snapshot_id: Optional[int]) -> Optional[str]:
    """Try to extract a YYYY-MM-DD date string from the response.

    The incremental endpoint doesn't directly return a date, but
    ``X-Eolas-Current-Snapshot`` carries the snapshot id which is purely
    numeric.  We use today's date for the 'to' side of the delta filename
    because the date a snapshot was created isn't in the response headers.
    Keeping the filename deterministic (by snapshot id or today) is the goal.
    """
    return _today_str()


# ---------------------------------------------------------------------------
# sync_all
# ---------------------------------------------------------------------------

def sync_all(
    client: "Client",
    library_dir: Union[str, pathlib.Path],
    *,
    datasets: Optional[List[str]] = None,
    max_concurrent: int = 4,
    progress: Optional[bool] = None,
) -> List[SyncResult]:
    """Implement ``client.sync_all(library_dir, datasets=..., max_concurrent=...)``.

    Parameters
    ----------
    client:
        The :class:`~eolas_data.Client` instance.
    library_dir:
        Root directory of the local data library.
    datasets:
        List of dataset names to sync.  If ``None``, all sub-directories
        that contain a ``_eolas-manifest.json`` are synced.
    max_concurrent:
        Maximum number of parallel sync operations.  Each
        :func:`sync_dataset` call is mostly I/O-bound (HTTP wait dominates),
        so a thread pool is used rather than asyncio.
    progress:
        Tri-state progress bar override passed through to each
        :func:`sync_dataset` call.

    Returns
    -------
    list[SyncResult]
        One :class:`SyncResult` per dataset, in the same order as *datasets*
        (or discovery order when *datasets* is ``None``).  On a per-dataset
        failure the corresponding entry has ``status="error"`` and ``error``
        set to the string repr of the exception; other datasets still
        complete normally.
    """
    lib = pathlib.Path(library_dir).expanduser().resolve()

    # ------------------------------------------------------------------
    # Determine the dataset list
    # ------------------------------------------------------------------
    if datasets is None:
        # Discover all sub-directories that have a manifest.
        names: List[str] = []
        if lib.is_dir():
            for sub in sorted(lib.iterdir()):
                if sub.is_dir() and (sub / MANIFEST_FILENAME).exists():
                    names.append(sub.name)
        if not names:
            return []
    else:
        names = list(datasets)

    total = len(names)

    # ------------------------------------------------------------------
    # Run up to max_concurrent syncs in parallel via a thread pool.
    # Each sync is mostly I/O-bound (HTTP), so threads beat asyncio here.
    # ------------------------------------------------------------------
    # Map future → (index, name) so we can preserve order in the output.
    results: List[Optional[SyncResult]] = [None] * total

    def _sync_one(idx: int, name: str) -> tuple[int, SyncResult]:
        try:
            result = sync_dataset(client, name, library_dir=lib, progress=progress)
        except Exception as exc:
            result = SyncResult(
                status="error",
                dataset=name,
                library_dir=lib,
                bytes_downloaded=0,
                rows_added=0,
                files_added=0,
                error=repr(exc),
            )
        _log.info("[%d/%d] %s: %s", idx + 1, total, name, result.status)
        return idx, result

    with ThreadPoolExecutor(max_workers=min(max_concurrent, total)) as pool:
        futures = {pool.submit(_sync_one, i, n): i for i, n in enumerate(names)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results  # type: ignore[return-value]  # all slots filled
