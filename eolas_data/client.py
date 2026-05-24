from __future__ import annotations

import datetime
import json
import os
import pathlib
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

# Imported separately so the names module is also re-exportable for users who
# want IDE autocomplete on dataset names without instantiating a Client.
from ._dataset_names import DatasetName  # noqa: F401  (public re-export)


BASE_URL = "https://api.eolas.fyi"

_SIDECAR_SCHEMA_VERSION = 1


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
        self._key   = api_key or os.getenv("EOLAS_API_KEY") or ""
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
        resp = self._raw_bulk_get(bulk_path, params=params)

        data = resp.content
        if path is None:
            return data

        out = pathlib.Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out

    def sync_bulk(
        self,
        name: Union[str, "DatasetName"],
        *,
        path: Union[str, "pathlib.Path"],
        format: str = "parquet",
        freshness: str = "auto",
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
        try:
            resp = self._raw_bulk_get(bulk_path, params=params)
            data = resp.content
            tmp.write_bytes(data)
            os.replace(tmp, out)
            bytes_dl = len(data)
        except Exception:
            # Best-effort cleanup of the tmp file; the original is untouched.
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

    def _raw_bulk_get(self, path: str, params: Optional[dict] = None) -> requests.Response:
        """Issue a GET that may 302-redirect to a canonical CDN URL.

        ``requests.Session`` follows redirects by default, which is exactly
        what we want: the bare ``/v1/bulk/{ns}/{table}`` URL redirects to the
        canonical versioned URL, and the session fetches that transparently.
        We only need special handling for the bulk-specific HTTP status codes
        (402, 503) that ``_raise_for_status`` doesn't know about.
        """
        url  = f"{self._base}{path}"
        resp = self._session.get(url, params=params)
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
        df = self.get(name, **kwargs)
        df.eolas_source = source
        return df

    # ------------------------------------------------------------------
    # Core data fetch
    # ------------------------------------------------------------------

    def get(
        self,
        name: Union[str, "DatasetName"],
        start: Optional[str] = None,
        end: Optional[str] = None,
        format: str = "json",
        engine: str = "pandas",
        limit: Optional[int] = None,
        as_geo: Optional[bool] = None,
    ) -> Dataset:
        """Fetch dataset rows as a pandas (or polars / geopandas) DataFrame.

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

        Returns:
            A :class:`Dataset` (pandas DataFrame subclass), a polars DataFrame
            when ``engine="polars"``, or a ``geopandas.GeoDataFrame`` when
            geometry is present and conversion is enabled.
        """
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
