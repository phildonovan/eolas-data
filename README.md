# eolas-data

Python client for the [eolas.fyi](https://eolas.fyi) statistical data API — 717+ datasets across NZ, Australia, OECD, and more, served as tidy `pandas` DataFrames (or `polars` / `geopandas` if you prefer).

```bash
pip install eolas-data
```

## Quickstart

```python
from eolas_data import Client

client = Client("your_api_key")   # or set EOLAS_API_KEY in env

# Generic
df = client.get("nz_cpi", start="2020-01-01")

# Source-specific (sets the `eolas_source` metadata)
df = client.statsnz("nz_cpi")
df = client.oecd("nz_gdp_production_annual")

# Discovery
all_datasets = client.list()
nz_only      = client.list("Stats NZ")
meta         = client.info("nz_cpi")
```

Get an API key at <https://eolas.fyi/signup>. Free plan is 10 requests/month; Starter is 100; Pro is unlimited.

## Geospatial

Datasets with a `geometry_wkt` column auto-convert to `geopandas.GeoDataFrame` if `geopandas` is installed:

```bash
pip install eolas-data[geo]
```

```python
gdf = client.get("nz_addresses")                  # GeoDataFrame
df  = client.get("nz_addresses", as_geo=False)    # plain DataFrame, WKT preserved
```

## Polars

```bash
pip install eolas-data[polars]
```

```python
df = client.get("nz_cpi", engine="polars")
```

## Plotting

```bash
pip install eolas-data[plot]
```

```python
df = client.statsnz("nz_cpi")
df.plot_dataset()
```

## Type stubs

Dataset names are exposed as a `Literal` so IDEs autocomplete the catalog:

```python
from eolas_data import Client

client = Client()
client.get("nz_")    # autocomplete shows nz_cpi, nz_gdp_production_annual, ...
```

The list is regenerated from the live API at release time. Passing a name not in the snapshot still works at runtime — the type hint just won't autocomplete it. Catalog snapshot date is exposed as `eolas_data._dataset_names.CATALOG_SNAPSHOT_DATE`.

## Migrating from `vswarehouse`

The previous package name was `vswarehouse`. Direct equivalents:

| `vswarehouse` | `eolas_data` |
|---|---|
| `from vswarehouse import Client, VSeries` | `from eolas_data import Client, Dataset` |
| `df.vs_name`, `df.vs_source` | `df.eolas_name`, `df.eolas_source` |
| `df.plot_series()` | `df.plot_dataset()` |
| `VS_API_KEY` env var | `EOLAS_API_KEY` (legacy `VS_API_KEY` still honoured) |

The API surface is otherwise identical. The default base URL is now `https://api.eolas.fyi` (the old `https://api.virtus-solutions.io` still 301-redirects and works fine — but uses the legacy endpoint shape).

## Releasing

See [`docs/clients.md`](https://github.com/phildonovan/eolas/blob/master/docs/clients.md) in the eolas data repo for the tagged-release flow and PyPI token rotation.

Before each release: `python -m eolas_data._regen_names` to refresh the dataset name stubs from the live API, commit the change, then tag and push.

## License

MIT
