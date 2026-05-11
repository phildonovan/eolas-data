from __future__ import annotations

import os
from typing import Optional, Union

import pandas as pd
import requests

from .dataset import Dataset
from .exceptions import APIError, AuthenticationError, NotFoundError, RateLimitError

# Imported separately so the names module is also re-exportable for users who
# want IDE autocomplete on dataset names without instantiating a Client.
from ._dataset_names import DatasetName  # noqa: F401  (public re-export)


BASE_URL = "https://api.eolas.fyi"


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
        api_key:  Your API key. Falls back to the ``EOLAS_API_KEY`` env var
                  (or ``VS_API_KEY`` for back-compat with the legacy library).
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
        self._key   = api_key or os.getenv("EOLAS_API_KEY") or os.getenv("VS_API_KEY") or ""
        self._base  = base_url.rstrip("/")
        self._cache: dict | None = {} if cache else None
        self._session = requests.Session()
        self._session.headers.update({"X-API-Key": self._key})

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

    def worksafe(self, name, **kwargs) -> Dataset:
        """Fetch a WorkSafe NZ dataset."""
        return self._get_source(name, "WorkSafe NZ", **kwargs)

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
            data    = self._get(f"/v1/datasets/{name}/data", params=params)
            records = data.get("data", data) if isinstance(data, dict) else data
            df      = pd.DataFrame(records)
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
