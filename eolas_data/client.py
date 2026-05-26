from __future__ import annotations

import datetime
import json
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
import requests

from .dataset import Dataset
from .exceptions import (
    APIError,
    AuthenticationError,
    BulkLicenceRestricted,
    BulkNotYetAvailable,
    BulkUpgradeRequired,
    NotFoundError,
    RateLimitError,
)
from .library import resolve_library_dir
from .sync.sync import SyncResult as _SyncResult  # noqa: F401 — re-exported below
from .sync.sync import sync_dataset as _sync_dataset

_log = logging.getLogger("eolas_data")

# Per-session set: tracks which dataset names we have already emitted the
# auto-routing INFO message for, so we don't spam on repeated calls.
_auto_route_notified: set[str] = set()

# Imported separately so the names module is also re-exportable for users who
# want IDE autocomplete on dataset names without instantiating a Client.
from ._dataset_names import DatasetName  # noqa: F401  (public re-export)


BASE_URL = "https://api.eolas.fyi"

_SIDECAR_SCHEMA_VERSION = 1

# OS-keyring constants — service name must match the R client so a key set
# from one language is readable from the other.
_KEYRING_SERVICE  = "eolas"
_KEYRING_USERNAME = "api-key"


def _keyring_get() -> str:
    """Return the API key from the OS keyring, or an empty string.

    Silently returns ``""`` when:
    - the ``keyring`` package is not installed
    - no entry exists under service="eolas", username="api-key"
    - the keyring backend is locked / unavailable (headless CI)

    Never raises — the caller treats a falsy return as "not found" and falls
    through to the next lookup step (config file or error).
    """
    try:
        import keyring as _kr
        value = _kr.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
        return value or ""
    except Exception:
        return ""


@dataclass
class SyncResult:
    """Result of a :meth:`Client.sync_bulk` call.

    Attributes:
        status: One of ``"downloaded"`` (first time), ``"updated"``
            (new snapshot available and written), or ``"unchanged"``
            (local file is already current — no I/O performed).
        previous_snapshot_id: The snapshot id recorded in the local sidecar
            before the sync, or ``None`` when no sidecar existed.
        current_snapshot_id: The snapshot id reported by the server's
            ``X-Snapshot-Version`` response header.
        path: The local file path that was written (or preserved unchanged).
        bytes_downloaded: Bytes written in this call. ``0`` when unchanged.
    """

    status: str
    previous_snapshot_id: Optional[str]
    current_snapshot_id: str
    path: pathlib.Path
    bytes_downloaded: int


def _to_geodataframe(df: "pd.DataFrame", force: bool = False):
    """Convert a DataFrame with a ``geometry_wkt`` column to a GeoDataFrame (CRS WGS84).

    Returns the GeoDataFrame on success, or ``None`` when geopandas isn't installed
    (and ``force`` is False) so the caller can fall back to the plain DataFrame.
    Raises ImportError when ``force=True`` but geopandas is missing.
    """
    try:
        import geopandas as gpd
        from shapely import wkt as _wkt
    except ImportError:
        if force:
            raise ImportError(
                "geopandas + shapely are required to return geospatial datasets "
                "as GeoDataFrames. Install with: pip install eolas-data[geo]"
            )
        return None

    geom = df["geometry_wkt"].apply(lambda s: _wkt.loads(s) if isinstance(s, str) and s else None)
    gdf = gpd.GeoDataFrame(df.drop(columns=["geometry_wkt"]), geometry=geom, crs="EPSG:4326")
    for attr in ("eolas_name", "eolas_source"):
        if hasattr(df, attr):
            try:
                setattr(gdf, attr, getattr(df, attr))
            except Exception:
                pass
    return gdf


class Client:
    """Client for the eolas.fyi statistical data API.

    Args:
        api_key:  Your API key. Falls back to the ``EOLAS_API_KEY`` env var.
        base_url: Override the API base URL (useful for testing).
        cache:    Cache responses in memory for the lifetime of the client.
                  Useful in notebooks to avoid re-fetching on re-runs.

    Examples::

        from eolas_data import Client
        client = Client("your_api_key")

        # Source-specific helpers
        df = client.statsnz("nz_cpi", start="2020-01-01")
        df = client.oecd("nz_gdp")

        # Generic
        df = client.get("nz_cpi")

        # Discovery
        all_datasets = client.list()
        nz_datasets  = client.list("Stats NZ")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        cache: bool = False,
    ):
        # Precedence: explicit arg → EOLAS_API_KEY env var → OS keyring → ""
        self._key = (
            api_key
            or os.getenv("EOLAS_API_KEY")
            or _keyring_get()
            or ""
        )
        self._base  = base_url.rstrip("/")
        self._cache: dict | None = {} if cache else None
        self._session = requests.Session()
        # Explicit User-Agent: good API-client hygiene, and insulation against
        # the Cloudflare edge tightening bot rules (raw default UAs can be
        # 403'd by managed rulesets — a custom UA is always allowed).
        try:
            import importlib.metadata as _md
            _ver = _md.version("eolas-data")
        except Exception:
            _ver = "1.0.0"
        self._session.headers.update({
            "X-API-Key": self._key,
            "User-Agent": f"eolas-data/{_ver} (python; +https://eolas.fyi)",
        })
        # Tri-state Arrow capability memo: None=unknown (try it), True=server
        # speaks Arrow (keep using it), False=server ignored format=arrow
        # (old server — go straight to JSON, don't waste a round-trip retrying
        # every call).
        self._arrow_supported: Optional[bool] = None

    def __repr__(self) -> str:
        masked = self._key[:8] + "..." if len(self._key) > 8 else self._key
        cache  = " cache=on" if self._cache is not None else ""
        return f"<eolas_data.Client key={masked!r}{cache}>"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list(self, source: Optional[str] = None) -> list[dict]:
        """Return metadata for all available datasets.

        Args:
            source: Optional filter, e.g. ``"Stats NZ"``, ``"OECD"``.
        """
        data = self._get("/v1/datasets")
        items = data.get("datasets", data) if isinstance(data, dict) else data
        if source:
            items = [s for s in items if s.get("source") == source]
        return items

    def info(self, name: Union[str, "DatasetName"]) -> dict:
        """Return metadata for a single dataset."""
        return self._get(f"/v1/datasets/{name}")

    # ------------------------------------------------------------------
    # Integrations (Enterprise plan only)
    # ------------------------------------------------------------------

    def integration(self, platform: str, datasets: list[str]) -> dict[str, str]:
        """Generate connector config files for a third-party data-pipeline tool.

        Enterprise plan only. Other plans receive an
        :class:`AuthenticationError` with the upgrade message in the detail.

        Args:
            platform: One of ``"meltano"``, ``"fivetran"``, ``"azure-data-factory"``.
            datasets: Dataset names to include in the generated config.

        Returns:
            ``{filename: file_contents}`` ready to write to disk.

        Examples::

            files = client.integration("meltano", ["nz_cpi", "nz_gdp"])
            for filename, content in files.items():
                Path("./tap-eolas") / filename).write_text(content)
        """
        if not datasets:
            raise ValueError("datasets cannot be empty")
        resp = self._get(
            f"/v1/integrations/{platform}",
            params={"datasets": ",".join(datasets)},
        )
        return resp.get("files", {})

    # ------------------------------------------------------------------
    # Bulk download
    # ------------------------------------------------------------------

    _BULK_EXTENSIONS = {
        "parquet":    ".parquet",
        "csv_gz":     ".csv.gz",
        "geoparquet": ".geo.parquet",
    }

    def download_bulk(
        self,
        name: Union[str, "DatasetName"],
        *,
        freshness: str = "auto",
        format: str = "parquet",
        path: Optional[Union[str, "pathlib.Path"]] = None,
        progress: Optional[bool] = None,
    ) -> "Union[pathlib.Path, bytes]":
        """Download a complete dataset as a single binary file.

        Wraps ``GET /v1/bulk/{namespace}/{table}`` which streams a Parquet,
        gzipped-CSV, or GeoParquet snapshot. Monthly snapshots are served from
        Cloudflare's edge cache and are typically delivered in milliseconds.
        Current snapshots are lazy-generated for Pro users on first request.

        The endpoint requires both ``namespace`` and ``table``. These are
        resolved automatically by first calling ``GET /v1/datasets/{name}`` and
        reading ``namespace`` and ``table`` off the metadata response.

        Args:
            name: Dataset identifier, e.g. ``"nz_cpi"``.
            freshness: ``"auto"`` (default) — omit the query param so the
                server picks the right level for your plan (Free → monthly,
                Pro → current). ``"monthly"`` or ``"current"`` override
                explicitly.
            format: ``"parquet"`` (default), ``"csv_gz"``, or ``"geoparquet"``.
                GeoParquet is only available on geospatial datasets.
            path: Where to write the file. ``None`` (default) returns the raw
                bytes. Pass a ``str`` or ``pathlib.Path`` to write to disk and
                return the resolved path. Parent directories are created if
                needed.
            progress: Control the download progress bar.
                ``None`` (default) auto-detects: bar shown when
                ``sys.stdout.isatty()`` is True (interactive terminal or
                notebook), hidden when piped or in CI.
                ``True`` forces the bar on regardless (useful in log-tailing
                scenarios).  ``False`` forces it off.  When ``path`` is
                ``None`` (bytes mode) progress is always disabled — there is
                no file path to label.

        Returns:
            ``pathlib.Path`` when ``path`` is set (the resolved, written path).
            ``bytes`` when ``path`` is ``None``.

        Raises:
            BulkUpgradeRequired: HTTP 402 — ``freshness="current"`` requires Pro.
            BulkLicenceRestricted: HTTP 403 with a licence body — dataset is
                excluded from bulk (e.g. OECD). Use ``client.get()`` instead.
            BulkNotYetAvailable: HTTP 503 — monthly snapshot not yet generated.
            NotFoundError: Dataset or namespace/table not found.
            AuthenticationError: Invalid or missing API key.

        Examples::

            # Return bytes (e.g. hand to pd.read_parquet)
            import io, pandas as pd
            raw = client.download_bulk("nz_cpi")
            df = pd.read_parquet(io.BytesIO(raw))

            # Write to a file, get the path back
            p = client.download_bulk("nz_cpi", path="nz_cpi.parquet")
            df = pd.read_parquet(p)

            # Gzipped CSV for spreadsheet users
            client.download_bulk("nz_cpi", format="csv_gz", path="nz_cpi.csv.gz")

            # Force monthly freshness (works on any plan — useful for reproducibility)
            client.download_bulk("nz_cpi", freshness="monthly", path="nz_cpi.parquet")

            # Silence the bar in a script even when run interactively
            client.download_bulk("nz_cpi", path="nz_cpi.parquet", progress=False)

        See Also:
            https://docs.eolas.fyi/bulk-downloads/
        """
        fmt = format.lower()
        if fmt not in self._BULK_EXTENSIONS:
            raise ValueError(
                f"Unknown format {format!r}. Expected one of: "
                + ", ".join(self._BULK_EXTENSIONS)
            )
        if freshness not in ("auto", "monthly", "current"):
            raise ValueError(
                f"Unknown freshness {freshness!r}. Expected 'auto', 'monthly', or 'current'."
            )

        # Resolve name → namespace + table via the datasets metadata endpoint.
        meta = self._get(f"/v1/datasets/{name}")
        namespace = meta.get("namespace") or ""
        table     = meta.get("table") or meta.get("name") or name
        if not namespace:
            raise NotFoundError(
                f"Dataset {name!r} metadata did not include a namespace field. "
                "Cannot construct bulk URL."
            )

        params: dict = {"format": fmt}
        if freshness != "auto":
            params["freshness"] = freshness

        bulk_path = f"/v1/bulk/{namespace}/{table}"

        if path is None:
            # Bytes mode: no progress bar (no file label to show).
            resp = self._raw_bulk_get(bulk_path, params=params)
            return resp.content

        out = pathlib.Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        show = self._resolve_show_progress(progress)
        resp = self._raw_bulk_get(bulk_path, params=params, stream=True)
        total = int(resp.headers.get("Content-Length", 0)) or None
        self._stream_to_file_with_progress(
            resp, out,
            total_bytes=total,
            label=out.name,
            show_progress=show,
        )
        return out

    def sync_bulk(
        self,
        name: Union[str, "DatasetName"],
        *,
        path: Union[str, "pathlib.Path"],
        format: str = "parquet",
        freshness: str = "auto",
        progress: Optional[bool] = None,
    ) -> SyncResult:
        """Incrementally sync a bulk dataset file — only re-download when the snapshot changes.

        Wraps the same ``/v1/bulk/{namespace}/{table}`` endpoint as
        :meth:`download_bulk`, but adds a lightweight HEAD optimisation: a
        single ~200-byte HEAD request checks the ``X-Snapshot-Version``
        response header on the canonical redirect URL.  If the snapshot id
        matches what's recorded in the local sidecar, the function returns
        immediately with ``status="unchanged"`` and zero I/O on the data file.

        A sidecar file ``<path>.eolas-meta.json`` is written next to the data
        file and contains the snapshot id, download timestamp, and source URL.
        It is used on the next call to perform the no-op check cheaply.

        The replacement is **atomic**: the new file is downloaded to a
        temporary path ``<path>.eolas-tmp-<rand>`` and then renamed over the
        original with :func:`os.replace`.  Readers with the file open see no
        partial bytes.

        Args:
            name: Dataset identifier, e.g. ``"nz_cpi"``.
            path: **Required.** Where to write the data file.  The sidecar
                lives at ``str(path) + ".eolas-meta.json"``.  Parent
                directories are created if needed.
            format: ``"parquet"`` (default), ``"csv_gz"``, or
                ``"geoparquet"``.
            freshness: ``"auto"`` (default), ``"monthly"``, or ``"current"``.
                Passed verbatim to the bulk endpoint.
            progress: Control the download progress bar.
                ``None`` (default) auto-detects via ``sys.stdout.isatty()``.
                ``True`` forces the bar on; ``False`` forces it off.
                When ``status="unchanged"`` no data is transferred so no bar
                is shown regardless of this setting.

        Returns:
            A :class:`SyncResult` dataclass with ``status``,
            ``previous_snapshot_id``, ``current_snapshot_id``, ``path``, and
            ``bytes_downloaded``.

        Raises:
            BulkUpgradeRequired: HTTP 402.
            BulkLicenceRestricted: HTTP 403 (licence body).
            BulkNotYetAvailable: HTTP 503.
            NotFoundError: Dataset not found.
            AuthenticationError: Invalid or missing API key.

        Examples::

            from eolas_data import Client, SyncResult

            client = Client("your_api_key")

            # First call: full download
            r = client.sync_bulk("nz_cpi", path="nz_cpi.parquet")
            print(r.status)           # "downloaded"
            print(r.bytes_downloaded) # e.g. 2_100_000

            # Second call (same snapshot): no-op
            r = client.sync_bulk("nz_cpi", path="nz_cpi.parquet")
            print(r.status)           # "unchanged"
            print(r.bytes_downloaded) # 0

            # After a new ETL run: new snapshot → file replaced in-place
            r = client.sync_bulk("nz_cpi", path="nz_cpi.parquet")
            print(r.status)           # "updated"

        See Also:
            https://docs.eolas.fyi/bulk-downloads/
        """
        fmt = format.lower()
        if fmt not in self._BULK_EXTENSIONS:
            raise ValueError(
                f"Unknown format {format!r}. Expected one of: "
                + ", ".join(self._BULK_EXTENSIONS)
            )
        if freshness not in ("auto", "monthly", "current"):
            raise ValueError(
                f"Unknown freshness {freshness!r}. Expected 'auto', 'monthly', or 'current'."
            )

        out = pathlib.Path(path).expanduser().resolve()
        sidecar = pathlib.Path(str(out) + ".eolas-meta.json")

        # Read local sidecar if present.
        prev: Optional[dict] = None
        if sidecar.exists():
            try:
                prev = json.loads(sidecar.read_text())
            except Exception:
                prev = None

        # Resolve name → namespace + table (needed to construct the bulk URL).
        meta = self._get(f"/v1/datasets/{name}")
        namespace = meta.get("namespace") or ""
        table     = meta.get("table") or meta.get("name") or name
        if not namespace:
            raise NotFoundError(
                f"Dataset {name!r} metadata did not include a namespace field. "
                "Cannot construct bulk URL."
            )

        params: dict = {"format": fmt}
        if freshness != "auto":
            params["freshness"] = freshness

        bulk_path = f"/v1/bulk/{namespace}/{table}"
        canonical_url = f"{self._base}{bulk_path}"

        # HEAD the canonical URL to read X-Snapshot-Version cheaply.
        # We follow redirects on the HEAD so we land on the versioned CDN URL
        # that carries the header.
        current_sid = self._head_snapshot_version(canonical_url, params=params)

        # No-op fast path: snapshot hasn't changed AND file exists on disk.
        if (
            prev is not None
            and prev.get("snapshot_id") == current_sid
            and out.exists()
        ):
            return SyncResult(
                status="unchanged",
                previous_snapshot_id=prev.get("snapshot_id"),
                current_snapshot_id=current_sid,
                path=out,
                bytes_downloaded=0,
            )

        # Download (atomic replace).
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + f".eolas-tmp-{os.urandom(4).hex()}")
        show = self._resolve_show_progress(progress)
        try:
            resp = self._raw_bulk_get(bulk_path, params=params, stream=True)
            total = int(resp.headers.get("Content-Length", 0)) or None
            bytes_dl = self._stream_to_file_with_progress(
                resp, tmp,
                total_bytes=total,
                label=out.name,
                show_progress=show,
            )
            if bytes_dl == 0:
                raise APIError(
                    200,
                    f"Bulk download for {name!r} returned an empty body "
                    "(0 bytes). The snapshot may not exist for this dataset "
                    "or format. Use format='parquet' for non-geo datasets.",
                )
            os.replace(tmp, out)
        except Exception:
            # Best-effort cleanup of the tmp file; the original is NEVER touched
            # on failure so no 0-byte file is left at the final path.
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        # Write sidecar next to the data file.
        sidecar_data = {
            "schema_version": _SIDECAR_SCHEMA_VERSION,
            "name": str(name),
            "snapshot_id": current_sid,
            "format": fmt,
            "freshness": freshness,
            "downloaded_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_url": canonical_url + "?" + "&".join(f"{k}={v}" for k, v in params.items()),
        }
        sidecar.write_text(json.dumps(sidecar_data, indent=2) + "\n")

        return SyncResult(
            status="downloaded" if prev is None else "updated",
            previous_snapshot_id=prev.get("snapshot_id") if prev else None,
            current_snapshot_id=current_sid,
            path=out,
            bytes_downloaded=bytes_dl,
        )

    # ------------------------------------------------------------------
    # Multi-file directory sync (pipeline verb)
    # ------------------------------------------------------------------

    def sync(
        self,
        name: Union[str, "DatasetName"],
        *,
        library_dir: Optional[Union[str, "pathlib.Path"]] = None,
        progress: Optional[bool] = None,
    ) -> "_SyncResult":
        """Incrementally sync a dataset into a multi-file local directory.

        This is the **pipeline verb** for eolas: it keeps a local copy of a
        dataset up to date with minimal bandwidth.  On the first call a full
        bulk snapshot is downloaded.  On subsequent calls only the delta since
        the last sync is fetched (when the server supports incremental
        delivery); if incremental delivery is unavailable a fresh full
        snapshot is downloaded instead.

        The synced dataset lives in ``library_dir/<name>/`` as a collection
        of parquet files plus a ``_eolas-manifest.json`` lineage file.
        Read the directory as a single logical table with::

            import pyarrow.dataset as ds
            table = ds.dataset("library_dir/nz_parcels").to_table()

            # or with DuckDB:
            import duckdb
            df = duckdb.query("SELECT * FROM read_parquet('library_dir/nz_parcels/*.parquet')").df()

        Args:
            name:        Dataset identifier, e.g. ``"doc_huts"`` or
                         ``"nz_parcels"``.
            library_dir: Root directory of the local data library.  A
                         sub-directory named ``<name>`` is created inside
                         (e.g. ``library_dir/doc_huts/``).  ``None``
                         (default) resolves via the library precedence
                         chain — see :func:`eolas_data.library.resolve_library_dir`.
            progress:    Tri-state progress bar override.  ``None`` (default)
                         auto-detects via ``sys.stdout.isatty()``.  ``True``
                         forces the bar on; ``False`` forces it off.

        Returns:
            A :class:`~eolas_data.sync.sync.SyncResult` with:

            - ``status``: ``"snapshot_full"``, ``"snapshot_delta"``, or
              ``"unchanged"``
            - ``bytes_downloaded``: bytes written to disk (``0`` if unchanged)
            - ``rows_added``: new rows in this sync (``0`` if unchanged)
            - ``files_added``: new parquet files written (``0`` if unchanged)
            - ``dataset``: the dataset name
            - ``library_dir``: the resolved library root path

        Raises:
            BulkUpgradeRequired: HTTP 402 — snapshot requires Pro plan.
            BulkLicenceRestricted: HTTP 403 — dataset excluded from bulk.
            BulkNotYetAvailable: HTTP 503 — monthly snapshot not yet generated.
            NotFoundError: Dataset not found.
            AuthenticationError: Invalid or missing API key.

        Examples::

            from eolas_data import Client
            client = Client("your_api_key")

            # First sync: full download (~seconds from CDN)
            r = client.sync("doc_huts", library_dir="/data/nz-warehouse")
            print(r)  # <SyncResult dataset='doc_huts' status='snapshot_full' ...>

            # Second sync: unchanged → zero I/O
            r = client.sync("doc_huts", library_dir="/data/nz-warehouse")
            print(r.status)  # 'unchanged'

            # Inspect files written
            import pathlib
            for f in sorted(pathlib.Path("/data/nz-warehouse/doc_huts").iterdir()):
                print(f.name)
            # _eolas-manifest.json
            # snapshot-2026-05-27.parquet  (or .geo.parquet for geo datasets)
        """
        if library_dir is None:
            resolved_lib = resolve_library_dir(interactive=False)
        else:
            resolved_lib = pathlib.Path(library_dir).expanduser().resolve()

        return _sync_dataset(
            self,
            str(name),
            library_dir=resolved_lib,
            progress=progress,
        )

    # ------------------------------------------------------------------
    # Streaming helper
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_show_progress(progress: Optional[bool]) -> bool:
        """Resolve the tri-state ``progress`` kwarg to a concrete bool.

        Priority (highest first):
        1. Explicit ``progress=True/False`` kwarg from the caller.
        2. ``EOLAS_NO_PROGRESS=1`` environment variable — always suppresses.
        3. Jupyter / IPython kernel detection — Jupyter wraps stdout in an
           ``OutStream``, so ``sys.stdout.isatty()`` returns False even when
           the user is clearly in an interactive notebook session. If
           ``ipykernel`` is loaded (the standard signal used by tqdm itself),
           we're in a notebook → show the bar; ``tqdm.auto`` then renders it
           as an ipywidget (or text bar as fallback).
        4. ``sys.stdout.isatty()`` auto-detection for plain terminals.
        """
        if progress is not None:
            return bool(progress)
        if os.getenv("EOLAS_NO_PROGRESS", "").strip() in ("1", "true", "yes"):
            return False
        if "ipykernel" in sys.modules:
            return True
        return sys.stdout.isatty()

    @staticmethod
    def _stream_to_file_with_progress(
        resp: "requests.Response",
        dest: "pathlib.Path",
        *,
        total_bytes: Optional[int],
        label: str,
        show_progress: bool,
    ) -> int:
        """Stream *resp* body to *dest*, optionally displaying a tqdm progress bar.

        Parameters
        ----------
        resp:
            A streaming ``requests.Response`` object (caller must have
            passed ``stream=True`` to the ``requests.get`` / ``Session.get``
            call).
        dest:
            File path to write into.  The file is opened in ``"wb"`` mode;
            the caller is responsible for ensuring the parent directory
            exists.
        total_bytes:
            Expected file size for the bar's ``total`` parameter.
            ``None`` disables the percentage / ETA display but keeps the
            bytes-transferred counter (tqdm handles ``total=None`` cleanly).
        label:
            Short description shown left of the bar (e.g. the filename).
        show_progress:
            ``True`` → show bar.  ``False`` → silent (tqdm ``disable=True``).

        Returns
        -------
        int
            Actual bytes written.
        """
        import tqdm.auto

        bytes_written = 0
        with tqdm.auto.tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=label,
            disable=not show_progress,
            leave=False,
        ) as bar:
            with dest.open("wb") as fh:
                # 1 MiB chunks: responsive bar updates (bar refreshes ~once per
                # MiB) without excessive syscall overhead on large files.
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
                        bar.update(len(chunk))
                        bytes_written += len(chunk)
        return bytes_written

    def _head_snapshot_version(self, url: str, params: Optional[dict] = None) -> str:
        """Issue a HEAD to ``url`` (following redirects) and return ``X-Snapshot-Version``.

        The header is set by the eolas CDN/server on the canonical versioned
        bulk URL.  If the header is absent (e.g. in tests against a stub
        server), we fall back to an empty string, which will never match a
        real sidecar snapshot_id, so the full GET always fires.

        Raises the same bulk-refusal exceptions as :meth:`_raw_bulk_get` so
        that the caller doesn't need to handle HEAD errors differently.
        """
        full_url = url if url.startswith("http") else f"{self._base}{url}"
        resp = self._session.head(full_url, params=params, allow_redirects=True)
        self._raise_for_bulk_status(resp)
        return resp.headers.get("X-Snapshot-Version", "")

    def _raw_bulk_get(
        self,
        path: str,
        params: Optional[dict] = None,
        stream: bool = False,
    ) -> requests.Response:
        """Issue a GET that may 302-redirect to a canonical CDN URL.

        ``requests.Session`` follows redirects by default, which is exactly
        what we want: the bare ``/v1/bulk/{ns}/{table}`` URL redirects to the
        canonical versioned URL, and the session fetches that transparently.
        We only need special handling for the bulk-specific HTTP status codes
        (402, 503) that ``_raise_for_status`` doesn't know about.

        When ``stream=True`` the response body is not eagerly downloaded —
        callers use :meth:`_stream_to_file_with_progress` to consume it.
        """
        url  = f"{self._base}{path}"
        resp = self._session.get(url, params=params, stream=stream)
        self._raise_for_bulk_status(resp)
        return resp

    @staticmethod
    def _raise_for_bulk_status(resp: requests.Response) -> None:
        """Like ``_raise_for_status`` but handles the extra bulk refusal codes."""
        if resp.status_code == 200:
            return
        if resp.status_code == 402:
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = ""
            raise BulkUpgradeRequired(detail) if detail else BulkUpgradeRequired()
        if resp.status_code == 403:
            try:
                body   = resp.json()
                detail = body.get("detail", "")
            except Exception:
                detail = ""
            # Distinguish licence-restriction 403 from auth 403 by the presence
            # of "licence" in the server detail. A key-auth 403 goes to AuthenticationError.
            if detail and "licence" in detail.lower():
                raise BulkLicenceRestricted(detail)
            raise AuthenticationError(detail or "API key is inactive.")
        if resp.status_code == 503:
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = ""
            raise BulkNotYetAvailable(detail) if detail else BulkNotYetAvailable()
        # Delegate everything else to the standard handler.
        Client._raise_for_status(resp)

    # ------------------------------------------------------------------
    # Source-specific helpers
    # ------------------------------------------------------------------

    def statsnz(self, name, **kwargs) -> Dataset:
        """Fetch a Stats NZ dataset."""
        return self._get_source(name, "Stats NZ", **kwargs)

    def oecd(self, name, **kwargs) -> Dataset:
        """Fetch an OECD dataset."""
        return self._get_source(name, "OECD", **kwargs)

    def rbnz(self, name, **kwargs) -> Dataset:
        """Fetch an RBNZ dataset."""
        return self._get_source(name, "RBNZ", **kwargs)

    def treasury(self, name, **kwargs) -> Dataset:
        """Fetch an NZ Treasury dataset."""
        return self._get_source(name, "NZ Treasury", **kwargs)

    def linz(self, name, **kwargs) -> Dataset:
        """Fetch a LINZ dataset."""
        return self._get_source(name, "LINZ", **kwargs)

    def statsnz_geo(self, name, **kwargs) -> Dataset:
        """Fetch a Stats NZ geospatial dataset (boundaries, census meshblocks, etc.).

        Kept as a convenience helper for discoverability — the server returns
        ``source = "Stats NZ"`` for both SDMX time series and Datafinder
        geospatial datasets, so the metadata on the returned Dataset reads
        ``"Stats NZ"`` (not ``"Stats NZ Geospatial"``).
        """
        return self._get_source(name, "Stats NZ", **kwargs)

    def mbie(self, name, **kwargs) -> Dataset:
        """Fetch an MBIE dataset."""
        return self._get_source(name, "MBIE", **kwargs)

    def nzta(self, name, **kwargs) -> Dataset:
        """Fetch a Waka Kotahi (NZTA) dataset."""
        return self._get_source(name, "Waka Kotahi", **kwargs)

    def msd(self, name, **kwargs) -> Dataset:
        """Fetch an MSD dataset."""
        return self._get_source(name, "MSD", **kwargs)

    def police(self, name, **kwargs) -> Dataset:
        """Fetch an NZ Police / MoJ dataset."""
        return self._get_source(name, "NZ Police / MoJ", **kwargs)

    def acc(self, name, **kwargs) -> Dataset:
        """Fetch an ACC dataset."""
        return self._get_source(name, "ACC", **kwargs)

    def edcounts(self, name, **kwargs) -> Dataset:
        """Fetch an Education Counts dataset."""
        return self._get_source(name, "Education Counts", **kwargs)

    def eeca(self, name, **kwargs) -> Dataset:
        """Fetch an EECA dataset (NZ energy use, EV chargers, regional heat demand).

        Examples::

            client.eeca("eeca_energy_end_use")           # NZ energy by sector x fuel x end-use x year
            client.eeca("eeca_ev_chargers_public")       # public EV charging network (point geometry)
            client.eeca("eeca_ev_metrics_district")      # EV penetration by territorial authority
            client.eeca("eeca_regional_heat_demand")     # industrial process heat by region x sector

        Notes:
            From the Energy Efficiency and Conservation Authority. EV charger streams
            refresh quarterly; Energy End Use Database is annual; Regional Heat Demand
            is an Aug 2024 snapshot. CC-BY 4.0 NZ.
        """
        return self._get_source(name, "EECA", **kwargs)

    def worksafe(self, name, **kwargs) -> Dataset:
        """Fetch a WorkSafe NZ dataset."""
        return self._get_source(name, "WorkSafe NZ", **kwargs)

    def immigration(self, name, **kwargs) -> Dataset:
        """Fetch an Immigration NZ dataset."""
        return self._get_source(name, "Immigration NZ", **kwargs)

    def geonet(self, name, **kwargs) -> Dataset:
        """Fetch a GeoNet dataset (NZ earthquakes, volcanic alert levels, strong-motion sensors).

        Examples::

            client.geonet("geonet_quakes_recent")              # rolling ~100 recent MMI>=3 quakes
            client.geonet("geonet_volcanic_alert_levels")      # 12 monitored NZ volcanoes
            client.geonet("geonet_strong_motion_sensors")      # 25 strong-motion stations

        Notes:
            Refreshed every 6 hours from api.geonet.org.nz. Earthquake catalogue is a
            rolling window of recent events, not a historical archive. CC-BY 3.0 NZ
            (Earth Sciences New Zealand, formerly GNS Science).
        """
        return self._get_source(name, "GeoNet", **kwargs)

    def lris(self, name, **kwargs) -> Dataset:
        """Fetch a Manaaki Whenua LRIS dataset (land cover, soil, protected areas).

        Examples::

            client.lris("lcdb_v6_mainland")   # current NZ land cover (~543k polygons)
            client.lris("nzlum_v03")           # NZ Land Use Management v0.3
            client.lris("pan_nz_2025_draft")   # protected areas (Draft, 2025)

        Notes:
            LCDB v3.0–v4.1 are deprecated vintages, retained for longitudinal
            analysis. LCDB v5 is superseded by v6 but still served.
            PAN-NZ 2025 was marked Draft at the time of ingestion (2026-05-12).
            Source: https://lris.scinfo.org.nz
            Licence: CC-BY 4.0 International (LCDB v5/v6, NZLUM, PBC, PAN-NZ);
            CC-BY 3.0 NZ (LCDB v3/v4 vintages). Attribution: Manaaki Whenua.
        """
        return self._get_source(name, "Manaaki Whenua / LRIS", **kwargs)

    def doc(self, name, **kwargs) -> Dataset:
        """Fetch a DOC (Department of Conservation) dataset.

        Examples::

            client.doc("doc_public_conservation_land")   # ~11k polygons of NZ public conservation land
            client.doc("doc_huts")                        # 1,429 DOC huts (Point geometry)
            client.doc("doc_tracks")                      # 3,248 DOC tracks (Polyline)

        Notes:
            Refreshed weekly from DOC's ArcGIS hub. Operational alert streams
            (track closures, hazard notices) are wired but currently blocked on
            an API key issue; they will appear automatically once resolved.
            CC-BY 4.0 International (Crown / Department of Conservation).
        """
        return self._get_source(name, "DOC", **kwargs)

    def akl_council(self, name, **kwargs) -> Dataset:
        """Fetch an Auckland Council dataset (overlays, heritage, hazards, zoning).

        Examples::

            client.akl_council("akc_notable_trees_overlay")
            client.akl_council("akc_significant_ecological_areas_overlay")
            client.akl_council("akc_historic_heritage_overlay_place")

        Notes:
            Open data from the Auckland Council ArcGIS hub. Covers district
            plan overlays, heritage areas, ecological areas, stormwater
            management zones, and more. CC-BY 4.0 (Auckland Council).
            Source: https://data-aucklandcouncil.opendata.arcgis.com
        """
        return self._get_source(name, "Auckland Council", **kwargs)

    def akl_transport(self, name, **kwargs) -> Dataset:
        """Fetch an Auckland Transport dataset (roads, public transport, cycling).

        Examples::

            client.akl_transport("akt_bus_stop")
            client.akl_transport("akt_bus_route")
            client.akl_transport("akt_cycle_facility_network")

        Notes:
            Open data from Auckland Transport (AT). Covers bus stops,
            bus routes, bridges, cycle infrastructure, and more.
            CC-BY 4.0 (Auckland Transport).
            Source: https://data-atgis.opendata.arcgis.com
        """
        return self._get_source(name, "Auckland Transport", **kwargs)

    def bay_of_plenty(self, name, **kwargs) -> Dataset:
        """Fetch a Bay of Plenty Councils dataset (hazards, resource consents, planning).

        Examples::

            client.bay_of_plenty("boprc_historic_flood_extents")
            client.bay_of_plenty("boprc_liquefaction_level_b")
            client.bay_of_plenty("boprc_rcep_ascv")

        Notes:
            Open data from Bay of Plenty Regional Council and its territorial
            authorities. Covers flood extents, liquefaction, coastal hazards,
            resource consents, and planning layers. CC-BY 4.0.
            Source: https://www.boprc.govt.nz
        """
        return self._get_source(name, "Bay of Plenty Councils", **kwargs)

    def charities(self, name, **kwargs) -> Dataset:
        """Fetch a Charities Services dataset (registered NZ charities).

        Examples::

            client.charities("charities_organisations")
            client.charities("charities_annual_returns")
            client.charities("charities_activities")

        Notes:
            Data from Charities Services (a business unit of the Department
            of Internal Affairs). Covers registered charities, officers,
            beneficiary groups, and annual financial returns.
            Open Government Licence v3.0.
            Source: https://www.charities.govt.nz
        """
        return self._get_source(name, "Charities Services", **kwargs)

    def colab_waikato(self, name, **kwargs) -> Dataset:
        """Fetch a Co-Lab Waikato dataset (planning, hazards, heritage across Waikato councils).

        Examples::

            client.colab_waikato("wmkdc_buildings")
            client.colab_waikato("tcdc_dp_coastal_environment")
            client.colab_waikato("wbopdc_coastal_erosion")

        Notes:
            Data aggregated via the Co-Lab Waikato open data hub. Covers
            district plan zones, coastal hazards, heritage, and building
            footprints across Waikato-region territorial authorities.
            CC-BY 4.0 (respective councils).
            Source: https://data-waikatolass.opendata.arcgis.com
        """
        return self._get_source(name, "Co-Lab Waikato", **kwargs)

    def ecan_canterbury(self, name, **kwargs) -> Dataset:
        """Fetch an ECan / Canterbury dataset (environment, hazards, resource consents).

        Examples::

            client.ecan_canterbury("ecan_liquefaction_susceptibility_final")
            client.ecan_canterbury("ecan_tsunami_evacuation_zones")
            client.ecan_canterbury("ecan_resource_consents_active_all")

        Notes:
            Open data from Environment Canterbury (ECan) and Canterbury-region
            councils. Covers liquefaction, earthquake faults, tsunami zones,
            water allocation, resource consents, and planning layers.
            CC-BY 4.0 (Environment Canterbury / respective councils).
            Source: https://opendata.canterburymaps.govt.nz
        """
        return self._get_source(name, "ECan / Canterbury", **kwargs)

    def hawkes_bay(self, name, **kwargs) -> Dataset:
        """Fetch a Hawke's Bay Councils dataset (hazards, planning, coastal management).

        Examples::

            client.hawkes_bay("hbrc_coastal_erosion_likely_66")
            client.hawkes_bay("hbrc_coastal_erosion_possible_33")
            client.hawkes_bay("hbrc_chb_hdc_wdc_liquefaction_severity")

        Notes:
            Open data from Hawke's Bay Regional Council and its territorial
            authorities. Covers coastal erosion, liquefaction, flood hazards,
            and district planning layers. CC-BY 4.0.
            Source: https://www.hbrc.govt.nz
        """
        return self._get_source(name, "Hawke's Bay Councils", **kwargs)

    def manawatu_whanganui(self, name, **kwargs) -> Dataset:
        """Fetch a Manawatu-Whanganui Councils dataset (airsheds, coastal, freshwater).

        Examples::

            client.manawatu_whanganui("horizons_coastal_marine_area")
            client.manawatu_whanganui("horizons_airshed_taihape")
            client.manawatu_whanganui("horizons_airshed_taumarunui")

        Notes:
            Open data from Horizons Regional Council (Manawatu-Whanganui) and
            its territorial authorities. Covers airsheds, coastal marine areas,
            freshwater, and planning layers. CC-BY 4.0.
            Source: https://www.horizons.govt.nz
        """
        return self._get_source(name, "Manawatu-Whanganui Councils", **kwargs)

    def napier_whanganui(self, name, **kwargs) -> Dataset:
        """Fetch a Napier or Whanganui city dataset (district plan, heritage, infrastructure).

        Examples::

            client.napier_whanganui("napier_heritage_buildings")
            client.napier_whanganui("napier_address_points")
            client.napier_whanganui("napier_parcels")

        Notes:
            Open data from Napier City Council and Whanganui District Council.
            Covers district plan precincts, heritage buildings and areas,
            address points, road centrelines, and parcels. CC-BY 4.0.
            Source: https://www.napier.govt.nz / https://www.whanganui.govt.nz
        """
        return self._get_source(name, "Napier + Whanganui", **kwargs)

    def northland(self, name, **kwargs) -> Dataset:
        """Fetch a Northland Councils dataset (district plans, designations, heritage).

        Examples::

            client.northland("fndc_district_plan_zones")
            client.northland("fndc_heritage_areas")
            client.northland("fndc_designations")

        Notes:
            Open data from Northland Regional Council and its territorial
            authorities (Far North, Whangarei, Kaipara). Covers district plan
            zones, designations, heritage, and environmental layers. CC-BY 4.0.
            Source: https://www.nrc.govt.nz
        """
        return self._get_source(name, "Northland Councils", **kwargs)

    def otago(self, name, **kwargs) -> Dataset:
        """Fetch an Otago Councils dataset (land use, water, planning, hazards).

        Examples::

            client.otago("orc_otago_irrigated_areas")
            client.otago("orc_otago_land_use_2024")
            client.otago("orc_floodbanks")

        Notes:
            Open data from Otago Regional Council and its territorial
            authorities (Dunedin, Queenstown-Lakes, Central Otago, Clutha,
            Waitaki). Covers land use, floodbanks, groundwater protection,
            and planning layers. CC-BY 4.0.
            Source: https://www.orc.govt.nz
        """
        return self._get_source(name, "Otago Councils", **kwargs)

    def pharmac(self, name, **kwargs) -> Dataset:
        """Fetch a PHARMAC dataset (NZ pharmaceutical subsidy schedule + hospital medicines).

        Examples::

            client.pharmac("pharmac_schedule")             # current month's funded medicines
            client.pharmac("pharmac_schedule_history")     # 2006-present subsidy archive
            client.pharmac("pharmac_hospital_medicines_list")  # current HML
            client.pharmac("pharmac_hml_history")          # 2011-present HML archive

        Notes:
            Monthly snapshots of NZ's national drug funding schedule + hospital
            formulary, from PHARMAC (Pharmaceutical Management Agency).
            Historical archives are append-mode — each month's snapshot is tagged
            with ``time_frame`` (YYYY-MM). CC-BY 3.0 NZ.
        """
        return self._get_source(name, "PHARMAC", **kwargs)

    def southland(self, name, **kwargs) -> Dataset:
        """Fetch a Southland Councils dataset (district plans, coastal, natural hazards).

        Examples::

            client.southland("sdc_southland_dp_zones")
            client.southland("sdc_southland_dp_heritage_items")
            client.southland("es_southland_land_use_2025")

        Notes:
            Open data from Environment Southland and its territorial
            authorities (Southland District, Gore, Invercargill). Covers
            district plan zones, coastal hazards, heritage, and land use.
            CC-BY 4.0.
            Source: https://www.es.govt.nz
        """
        return self._get_source(name, "Southland Councils", **kwargs)

    def taranaki(self, name, **kwargs) -> Dataset:
        """Fetch a Taranaki Councils dataset (coastal, biodiversity, district plans).

        Examples::

            client.taranaki("trc_biodiversity_coastal_mgmt_areas")
            client.taranaki("npdc_dp_operative_coastal_flooding")
            client.taranaki("npdc_dp_operative_archaeological")

        Notes:
            Open data from Taranaki Regional Council and its territorial
            authorities (New Plymouth, Stratford, South Taranaki). Covers
            biodiversity, coastal management, and district planning layers.
            CC-BY 4.0.
            Source: https://www.trc.govt.nz
        """
        return self._get_source(name, "Taranaki Councils", **kwargs)

    def top_of_south(self, name, **kwargs) -> Dataset:
        """Fetch a Gisborne / Top of South Councils dataset (coastal, planning, heritage).

        Examples::

            client.top_of_south("gdc_coastal_environment")
            client.top_of_south("gdc_coastal_erosion")
            client.top_of_south("gdc_coastal_flooding")

        Notes:
            Open data from Gisborne District Council, Marlborough District
            Council, Nelson City Council, and Tasman District Council.
            Covers coastal hazards, planning zones, and heritage layers.
            CC-BY 4.0.
            Source: https://www.gdc.govt.nz
        """
        return self._get_source(name, "Gisborne / Top of South Councils", **kwargs)

    def wellington(self, name, **kwargs) -> Dataset:
        """Fetch a Wellington Region Councils dataset (hazards, planning, infrastructure).

        Examples::

            client.wellington("wcc_district_plan_zones_2024")
            client.wellington("wcc_flood_hazard_operative")
            client.wellington("gwrc_flood_hazard_extents")

        Notes:
            Open data from Greater Wellington Regional Council and its
            territorial authorities (Wellington, Hutt, Upper Hutt, Porirua,
            Kapiti Coast). Covers flood and earthquake hazards, district plan
            zones, and coastal inundation. CC-BY 4.0.
            Source: https://www.gw.govt.nz
        """
        return self._get_source(name, "Wellington Region Councils", **kwargs)

    def west_coast(self, name, **kwargs) -> Dataset:
        """Fetch a West Coast (Te Tai o Poutini) dataset (faults, landslides, planning).

        Examples::

            client.west_coast("wcrc_active_faults")
            client.west_coast("wcrc_alpine_fault_traces")
            client.west_coast("wcrc_landslide_catalog")

        Notes:
            Open data from West Coast Regional Council (Te Tai o Poutini) and
            its territorial authorities (Buller, Grey, Westland). Covers
            active faults, the Alpine Fault, landslide catalogs, and
            significant natural areas. CC-BY 4.0.
            Source: https://www.ttpp.nz
        """
        return self._get_source(name, "West Coast (Te Tai o Poutini)", **kwargs)

    def _get_source(self, name, source: str, **kwargs) -> Dataset:
        result = self.get(name, **kwargs)
        # When as_arrow=True the result is a pyarrow.Table which has no
        # eolas_source attribute — skip the metadata tag silently.
        try:
            result.eolas_source = source
        except (AttributeError, TypeError):
            pass
        return result

    # ------------------------------------------------------------------
    # Local-file convenience
    # ------------------------------------------------------------------

    _DEFAULT_CACHE_DIR = pathlib.Path.home() / ".cache" / "eolas"

    def get_local(
        self,
        name: Union[str, "DatasetName"],
        *,
        cache_dir: Optional[Union[str, "pathlib.Path"]] = None,
        format: Optional[str] = None,
        freshness: str = "auto",
        as_geo: Optional[bool] = None,
        as_arrow: bool = False,
        progress: Optional[bool] = None,
    ) -> "pd.DataFrame":
        """Force the cache+sync path.  Equivalent to ``get(name, mode='cached')``.

        This is a convenience alias for users who want to be explicit about the
        data path.  ``client.get(name)`` with ``mode='auto'`` (the default) will
        route through here automatically for large or geospatial datasets.

        ``as_geo`` defaults to ``None`` (auto: convert to GeoDataFrame when the
        dataset is geo and geopandas is installed) rather than ``True`` when
        ``as_arrow=True`` is passed, avoiding a conflict between the two flags.

        For the full parameter documentation see :meth:`_get_local_impl`.
        """
        # Resolve as_geo: None → True (auto) unless as_arrow overrides.
        resolved_as_geo = as_geo if as_geo is not None else (not as_arrow)
        return self._get_local_impl(
            name,
            cache_dir=cache_dir,
            format=format,
            freshness=freshness,
            as_geo=resolved_as_geo,
            as_arrow=as_arrow,
            progress=progress,
        )

    def _get_local_impl(
        self,
        name: Union[str, "DatasetName"],
        *,
        cache_dir: Optional[Union[str, "pathlib.Path"]] = None,
        format: Optional[str] = None,
        freshness: str = "auto",
        as_geo: Optional[bool] = True,
        as_arrow: bool = False,
        progress: Optional[bool] = None,
    ) -> "pd.DataFrame":
        """Download (or serve from cache) a whole dataset as a local DataFrame.

        This is the recommended path for large or geospatial datasets in a
        notebook workflow.  On the first call it fetches the bulk file from
        CDN (milliseconds for monthly snapshots) and writes it to the
        configured library directory (see :func:`eolas_data.library.resolve_library_dir`).
        On subsequent calls in the same or future sessions a lightweight HEAD
        request checks whether the file is still current; if so the local copy
        is read directly with zero network I/O on the data payload.

        If you have been calling ``client.get("nz_parcels")`` on a 3-million-
        row geospatial dataset and it takes 15+ minutes, use ``get_local``
        instead — it serves a pre-materialised GeoParquet from CDN, not a live
        Iceberg scan through the row-oriented data endpoint.

        Args:
            name: Dataset identifier, e.g. ``"nz_parcels"``.
            cache_dir: Local directory for cached files.  Accepts ``~``-prefixed
                strings, ``str``, or ``pathlib.Path``.  The directory is
                created if it does not exist.  ``None`` (default) resolves via
                the library precedence chain (``EOLAS_LIBRARY`` env var →
                ``library_dir`` in ``~/.eolas/config.json`` → interactive
                prompt on first TTY call → ``~/.cache/eolas/`` fallback).
                An explicit value here always wins (highest priority).
            format: ``"parquet"``, ``"csv_gz"``, or ``"geoparquet"``.  ``None``
                (default) auto-detects: calls ``self.info(name)`` and checks
                whether the dataset metadata indicates geometry; geo datasets
                use ``"geoparquet"``, everything else uses ``"parquet"``.
            freshness: ``"auto"`` (default), ``"monthly"``, or ``"current"``.
                Passed verbatim to :meth:`sync_bulk`.
            as_geo: When ``True`` (default) and the file is GeoParquet and
                ``geopandas`` is installed, the returned object is a
                ``geopandas.GeoDataFrame``.  When ``False`` (or geopandas is
                not installed) the raw WKB binary column is returned in a plain
                ``pd.DataFrame``.  Ignored when ``as_arrow=True``.
            as_arrow: When ``True``, skip all native geometry materialisation
                and return a ``pyarrow.Table`` directly.  Geometry stays as
                Arrow buffers — zero-copy, suitable for DuckDB / polars
                pipelines that work on a sample before converting to
                GeoDataFrame.  Works for both geo and non-geo datasets.
                Cannot be combined with ``as_geo=True`` (raises
                ``ValueError``).
            progress: Control the download progress bar shown during the
                sync phase.  Forwarded verbatim to :meth:`sync_bulk`.
                ``None`` (default) auto-detects via ``sys.stdout.isatty()``.

        Returns:
            ``pd.DataFrame`` for tabular datasets.  ``geopandas.GeoDataFrame``
            for geospatial datasets when ``as_geo=True`` and geopandas is
            installed.

        Raises:
            BulkUpgradeRequired: Passes through from :meth:`sync_bulk` — the
                dataset requires a Pro plan for the requested freshness.
            BulkLicenceRestricted: Passes through — this dataset cannot be
                bulk-downloaded (e.g. OECD).  Use ``client.get(name)`` instead.
            BulkNotYetAvailable: Passes through — the monthly snapshot has not
                been generated yet.
            NotFoundError: Dataset not found.
            AuthenticationError: Invalid or missing API key.

        Examples::

            from eolas_data import Client
            client = Client("your_api_key")

            # 3-million-row geospatial dataset — first call downloads ~1 GB
            # GeoParquet from CDN; subsequent calls return in <1 s.
            gdf = client.get_local("nz_parcels")

            # Non-geo dataset
            df = client.get_local("nz_cpi")

            # Explicit cache dir (overrides library config — highest priority)
            df = client.get_local("nz_cpi", cache_dir="/tmp/eolas-cache")

            # Force CSV format
            df = client.get_local("nz_cpi", format="csv_gz")

            # Keep raw WKB column instead of converting to GeoDataFrame
            df = client.get_local("nz_parcels", as_geo=False)

        See Also:
            :meth:`sync_bulk` — for advanced control over the sync lifecycle.
            https://docs.eolas.fyi/bulk-downloads/
        """
        # ---- as_arrow / as_geo conflict guard ------------------------------------
        # Both flags being True is contradictory — raise early so the error is
        # surfaced regardless of whether the dataset has geometry.
        if as_arrow and as_geo is True:
            raise ValueError(
                "as_arrow=True and as_geo=True are mutually exclusive. "
                "as_arrow returns a pyarrow.Table (no geometry materialisation); "
                "as_geo materialises geometry as shapely objects in a GeoDataFrame. "
                "Choose one."
            )

        # ---- resolve cache_dir -----------------------------------------------
        # Explicit cache_dir= arg wins outright (Step 1 of the precedence chain).
        # None triggers the library resolver (Steps 2-5).
        if cache_dir is not None:
            cache_path = pathlib.Path(cache_dir).expanduser().resolve()
        else:
            cache_path = resolve_library_dir()
        cache_path.mkdir(parents=True, exist_ok=True)

        # ---- auto-detect format if not specified -----------------------------
        if format is None:
            try:
                meta = self.info(name)
                gt  = meta.get("geometry_type")
                wkt = meta.get("geometry_wkt")
                gt_truthy  = bool(gt)  and gt  != "none"
                wkt_truthy = bool(wkt) and wkt != "none"
                is_geo = gt_truthy or wkt_truthy or bool(meta.get("has_geometry"))
            except Exception:
                is_geo = False
            fmt = "geoparquet" if is_geo else "parquet"
        else:
            fmt = format.lower()
            if fmt not in self._BULK_EXTENSIONS:
                raise ValueError(
                    f"Unknown format {format!r}. Expected one of: "
                    + ", ".join(self._BULK_EXTENSIONS)
                )

        # ---- compute local file path -----------------------------------------
        ext = self._BULK_EXTENSIONS[fmt]  # e.g. ".parquet", ".csv.gz", ".geo.parquet"
        file_path = cache_path / f"{name}{ext}"

        # ---- sync (download if needed, HEAD check if cached) -----------------
        # Bulk exceptions (BulkLicenceRestricted, BulkUpgradeRequired,
        # BulkNotYetAvailable) propagate unchanged — their messages already
        # tell the user what to do.
        self.sync_bulk(name, path=file_path, format=fmt, freshness=freshness, progress=progress)

        # ---- read the local file into a DataFrame ----------------------------
        # as_arrow=True: return a pyarrow.Table directly, skipping all
        # geometry materialisation.  Works for every format.
        if as_arrow:
            import pyarrow.parquet as _pq
            if fmt == "csv_gz":
                # CSV-GZ: read via pandas then convert to Arrow — small cost,
                # still avoids shapely / WKB overhead.
                import pyarrow as _pa
                return _pa.Table.from_pandas(
                    pd.read_csv(file_path),
                    preserve_index=False,
                )
            else:
                # parquet or geoparquet — pyarrow.parquet reads both natively.
                return _pq.read_table(file_path)

        if fmt == "geoparquet":
            if as_geo:
                try:
                    import geopandas as gpd
                    return gpd.read_parquet(file_path)
                except ImportError:
                    pass
            # geopandas not installed or as_geo=False — read as plain parquet
            return pd.read_parquet(file_path)
        elif fmt == "csv_gz":
            return pd.read_csv(file_path)  # pandas handles .gz automatically
        else:  # parquet
            return pd.read_parquet(file_path)

    # ------------------------------------------------------------------
    # Core data fetch
    # ------------------------------------------------------------------

    # Slice kwargs that, when present, force the live-API path in auto mode.
    # These indicate the caller wants a filtered or size-capped subset that
    # a whole-file bulk cache cannot serve.
    _SLICE_KWARGS = frozenset({"start", "end", "limit", "dimensions"})

    # Row-count threshold above which a bulk-eligible dataset is auto-routed
    # through the cache+sync path instead of the live Iceberg scan.
    _AUTO_ROUTE_ROW_THRESHOLD = 100_000

    def get(
        self,
        name: Union[str, "DatasetName"],
        start: Optional[str] = None,
        end: Optional[str] = None,
        format: str = "json",
        engine: str = "pandas",
        limit: Optional[int] = None,
        as_geo: Optional[bool] = None,
        as_arrow: bool = False,
        *,
        mode: str = "auto",
    ) -> Dataset:
        """Fetch dataset rows as a pandas (or polars / geopandas) DataFrame.

        The ``mode`` parameter controls which data path is used:

        - ``"auto"`` (default): smart-routes based on dataset metadata.
          If any slice kwarg (``start``, ``end``, ``limit``) is set the live
          API is always used. Otherwise ``info(name)`` is called and the
          result routed through :meth:`get_local` (cache+sync) when the
          dataset is both bulk-eligible and large (>100k rows) or geospatial.
          OECD and other licence-restricted datasets always fall through to
          live regardless of size.
        - ``"live"``: hit ``/v1/datasets/{name}/data`` directly, bypassing the
          cache.  Useful for the freshest data, OECD-licence-restricted sources,
          or small slices of big datasets (e.g. with a ``limit=``, ``start=``,
          or ``end=`` filter).  **The server returns 413** if you request all
          rows of a large (>100 k rows) or geometry dataset without a filter —
          use ``mode="cached"`` (or omit ``mode=``) for whole-dataset pulls.
        - ``"cached"``: always use the cache+sync path (same as calling
          :meth:`get_local`). ``start``, ``end``, ``limit``, and ``format``
          are ignored in this mode; ``as_geo`` is forwarded.

        Args:
            name:   Dataset identifier, e.g. ``"nz_cpi"``. Type-checked against
                    the ``DatasetName`` Literal at static-analysis time so
                    IDEs autocomplete the catalog.
            start:  ISO date lower bound, e.g. ``"2020-01-01"``.
            end:    ISO date upper bound, e.g. ``"2024-12-31"``.
            format: ``"json"`` (default) or ``"csv"``.
            engine: ``"pandas"`` (default) or ``"polars"``.
            limit:  Max rows to return. Default ``None`` requests the full dataset
                    (server enforces a 50,000-row cap on Free/Starter plans; Pro is
                    unlimited). Pass an explicit integer to request fewer rows.
            as_geo: Convert geospatial datasets to a ``GeoDataFrame``.
                    ``None`` (default) auto-converts when the dataset has a
                    ``geometry_wkt`` column AND ``geopandas`` is importable.
                    ``True`` forces the conversion (raises if geopandas missing).
                    ``False`` keeps the raw WKT string column.
                    Install with ``pip install eolas-data[geo]``.
                    Cannot be combined with ``as_arrow=True``.
            as_arrow: When ``True``, return a ``pyarrow.Table`` instead of a
                    ``DataFrame`` / ``GeoDataFrame``.  Geometry stays as Arrow
                    buffers — zero-copy, no shapely allocation.  Works on every
                    dataset (geo or non-geo) and every routing mode (live,
                    cached, auto).  On the live path the Arrow IPC response is
                    returned directly when the server supports it; otherwise a
                    JSON response is converted via ``pa.Table.from_pandas``.
                    Cannot be combined with ``as_geo=True`` (raises
                    ``ValueError``).
            mode:   ``"auto"`` (default), ``"live"``, or ``"cached"``. Controls
                    smart-routing behaviour; see above.  ``"live"`` returns 413
                    from the server when the dataset is large/geo and no filter
                    is set — pass ``limit=``, ``start=``, or ``end=`` to slice,
                    or use ``mode="cached"`` for whole-dataset access.

        Returns:
            A :class:`Dataset` (pandas DataFrame subclass), a polars DataFrame
            when ``engine="polars"``, or a ``geopandas.GeoDataFrame`` when
            geometry is present and conversion is enabled.

        Examples::

            # Smart-routing (default) — nz_parcels auto-routes to cache+sync;
            # nz_cpi (145 rows, no geo) stays on the live path.
            gdf = client.get("nz_parcels")    # auto → cache path in ~seconds
            df  = client.get("nz_cpi")        # auto → live path (small dataset)

            # Force live even for large datasets
            gdf = client.get("nz_parcels", mode="live")

            # Force cache+sync explicitly (same as get_local)
            gdf = client.get("nz_parcels", mode="cached")

            # Slice queries always go live regardless of size
            gdf = client.get("nz_parcels", limit=10)
        """
        if mode not in ("auto", "live", "cached"):
            raise ValueError(
                f"Unknown mode {mode!r}. Expected 'auto', 'live', or 'cached'."
            )

        # as_arrow + as_geo conflict — check early so we fail fast regardless of mode.
        if as_arrow and as_geo:
            raise ValueError(
                "as_arrow=True and as_geo=True are mutually exclusive. "
                "as_arrow returns a pyarrow.Table (no geometry materialisation); "
                "as_geo materialises geometry as shapely objects in a GeoDataFrame. "
                "Choose one."
            )

        # ---- mode="cached": delegate entirely to the cache+sync path ----------
        if mode == "cached":
            as_geo_for_local = (not as_arrow) if as_geo is None else bool(as_geo)
            return self._get_local_impl(name, as_geo=as_geo_for_local, as_arrow=as_arrow)

        # ---- early in-memory cache check (before routing / network calls) -----
        # The cache key uses the parameters that affect the live-API result.
        # If we have a cached result, return it immediately without spending a
        # metadata round-trip on the routing decision.
        _early_cache_key = f"{name}:{start}:{end}:{format}:{0 if limit is None else int(limit)}:{as_geo}"
        if self._cache is not None and _early_cache_key in self._cache:
            return self._cache[_early_cache_key]

        # ---- mode="auto": decide based on slice kwargs + dataset metadata -----
        if mode == "auto":
            has_slice = (start is not None) or (end is not None) or (limit is not None)
            if not has_slice:
                # One /v1/datasets/{name} call to read metadata.
                try:
                    meta = self.info(name)
                except Exception:
                    meta = {}

                bulk_ok = (meta.get("bulk_export_class", "none") or "none") != "none"
                gt  = meta.get("geometry_type")
                wkt = meta.get("geometry_wkt")
                gt_truthy  = bool(gt)  and gt  != "none"
                wkt_truthy = bool(wkt) and wkt != "none"
                is_geo = gt_truthy or wkt_truthy or bool(meta.get("has_geometry"))
                row_count = meta.get("row_count_at_last_refresh") or 0
                is_large = (row_count > self._AUTO_ROUTE_ROW_THRESHOLD)

                if bulk_ok and (is_geo or is_large):
                    # Emit a one-time per-dataset INFO so notebook users see why
                    # a disk cache is silently appearing.
                    name_str = str(name)
                    if name_str not in _auto_route_notified:
                        _auto_route_notified.add(name_str)
                        _log.info(
                            "auto-routing %r through cache+sync (large/geo dataset). "
                            "Use mode='live' to override, or `eolas library status` to see cache location.",
                            name_str,
                        )
                    as_geo_for_local = (not as_arrow) if as_geo is None else bool(as_geo)
                    return self._get_local_impl(name, as_geo=as_geo_for_local, as_arrow=as_arrow)
                # Fall through to the live path.

        # ---- live path (mode="live", or auto fell through) -------------------
        params: dict = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        # Server-side: limit=0 means "as much as the plan allows" (full dataset for Pro,
        # 50K cap for Free/Starter). limit=None on the client maps to limit=0.
        params["limit"] = 0 if limit is None else int(limit)

        cache_key = f"{name}:{start}:{end}:{format}:{params['limit']}:{as_geo}"
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]

        if format == "csv":
            from io import StringIO
            resp = self._raw_get(f"/v1/datasets/{name}/data", params={"format": "csv", **params})
            df   = pd.read_csv(StringIO(resp.text))
        else:
            df = self._fetch_dataframe(name, params)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

        # API streams from Iceberg in file order, not chronological — sort here so
        # callers can `df.plot(x="date", y="value")` without seeing zigzag lines.
        if "date" in df.columns:
            df = df.sort_values("date", kind="stable").reset_index(drop=True)

        # as_arrow on the live path: convert the pandas DataFrame to an Arrow
        # Table, avoiding any shapely allocation.  We convert before the
        # geo step so geometry_wkt stays as a string column in the Arrow table.
        if as_arrow:
            import pyarrow as _pa
            tbl = _pa.Table.from_pandas(df, preserve_index=False)
            if self._cache is not None:
                self._cache[cache_key] = tbl
            return tbl

        result = Dataset(df)
        result.eolas_name   = name
        result.eolas_source = ""

        if engine == "polars":
            try:
                import polars as pl
                return pl.from_pandas(result)
            except ImportError:
                raise ImportError(
                    "polars is required for engine='polars'. "
                    "Install with: pip install eolas-data[polars]"
                )

        # Optional geopandas conversion. When as_geo=None we auto-convert if both
        # (a) the dataset has a geometry_wkt column AND (b) geopandas is importable.
        if as_geo is not False and "geometry_wkt" in result.columns:
            converted = _to_geodataframe(result, force=as_geo is True)
            if converted is not None:
                result = converted

        if self._cache is not None:
            self._cache[cache_key] = result

        return result

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _fetch_dataframe(self, name, params: dict) -> pd.DataFrame:
        """Fetch dataset rows as a DataFrame, negotiating Arrow IPC over the
        wire (≈5x faster end-to-end, ≈82x faster parse than JSON on large
        pulls — benchmarked 2026-05-18). Transparently falls back to JSON for
        older servers, unexpected content-types, or any pyarrow issue, so the
        returned DataFrame is identical either way.
        """
        if self._arrow_supported is not False:
            try:
                import io
                import pyarrow as pa  # hard dependency; guarded for resilience

                resp = self._raw_get(
                    f"/v1/datasets/{name}/data",
                    params={**params, "format": "arrow"},
                )
                if "arrow" in resp.headers.get("content-type", ""):
                    self._arrow_supported = True
                    tbl = pa.ipc.open_stream(io.BytesIO(resp.content)).read_all()
                    return tbl.to_pandas()
                # Old server ignored format=arrow and returned JSON. Remember
                # so we don't pay the failed round-trip on every future call.
                self._arrow_supported = False
            except Exception:
                self._arrow_supported = False

        data = self._get(f"/v1/datasets/{name}/data", params=params)
        records = data.get("data", data) if isinstance(data, dict) else data
        return pd.DataFrame(records)

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        return self._raw_get(path, params=params).json()

    def _raw_get(self, path: str, params: Optional[dict] = None) -> requests.Response:
        url  = f"{self._base}{path}"
        resp = self._session.get(url, params=params)
        self._raise_for_status(resp)
        return resp

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code == 200:
            return
        if resp.status_code == 401:
            raise AuthenticationError("Invalid or missing API key.")
        if resp.status_code == 403:
            try:
                detail = resp.json().get("detail", "API key is inactive.")
            except Exception:
                detail = "API key is inactive."
            raise AuthenticationError(detail)
        if resp.status_code == 429:
            raise RateLimitError(
                "Monthly request limit reached. Upgrade for higher limits."
            )
        if resp.status_code == 404:
            try:
                detail = resp.json().get("detail", "Not found.")
            except Exception:
                detail = "Not found."
            raise NotFoundError(detail)
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise APIError(resp.status_code, detail)
