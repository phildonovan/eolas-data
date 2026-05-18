# eolas-data

Python client for the [eolas.fyi](https://eolas.fyi) statistical data API — 1,400+ official New Zealand statistical & geospatial datasets, plus OECD data for international comparisons, served as tidy `pandas` DataFrames (or `polars` / `geopandas` if you prefer).

_Coverage is New Zealand + OECD today. Australian sources are on the roadmap — not yet available; OECD data already includes Australia (and other OECD members) for cross-country comparisons._

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

Get an API key at <https://eolas.fyi/signup>. Free plan is 10 requests/month; Pro ($49/month) is unlimited.

## Command-line interface

`pip install eolas-data[cli]` adds an `eolas` command for browsing, fetching, and
scheduling — useful for shell scripts, cron jobs, and AI-agent workflows. Output
auto-detects piping: rich tables in a terminal, newline-delimited JSON when
stdout is piped.

```bash
# one-time setup
eolas auth set-key
eolas health

# discover
eolas datasets list --source "Stats NZ"
eolas datasets list --search cpi --json | jq '.[].name'
eolas datasets info nz_cpi
eolas datasets preview nz_cpi --limit 5

# fetch (verb matches the Python lib's client.get())
eolas get nz_cpi --format csv > cpi.csv
eolas get nz_cpi --start 2020-01-01 --format json | jq '.[].value'
eolas get sa2_2023 --format parquet --out sa2.parquet
```

### Scheduling

Set up recurring fetches without touching crontab/Task Scheduler syntax. Works
on Linux, macOS (cron), and Windows (Task Scheduler).

```bash
eolas schedule add nz_cpi --daily   --out ~/data/cpi.csv
eolas schedule add nz_gdp --weekly  --out ~/data/gdp.csv
eolas schedule add nzd_usd --cron "0 */6 * * *" --out ~/data/fx.csv   # POSIX only

eolas schedule list
eolas schedule remove nz_cpi
```

Daily is the default. Pre-flight check refuses to install a schedule unless
your API key is configured (otherwise the job would fail silently forever).

### Integrations (Enterprise plan)

Generate ready-to-run connector configs for popular data-pipeline tools — eolas
becomes a one-command source for Meltano, Fivetran, or Azure Data Factory.

```bash
eolas integrate meltano             --datasets nz_cpi,nz_gdp --output ./my-pipeline/
eolas integrate fivetran            --datasets nz_cpi
eolas integrate azure-data-factory  --datasets nz_cpi,nz_gdp
```

The generated directory has everything needed to plug into your destination
warehouse: `meltano.yml`, `fivetran.yml`, or ADF JSON resources, plus a `README.md`
walking through the rest of the setup. Non-Enterprise users see a clear
upgrade pointer; the gating lives server-side so the capability is bypass-proof.

### Exit codes

Distinct exit codes per error class, for shell scripts and agents:

| Code | Meaning |
|---|---|
| `0`  | Success |
| `1`  | Generic error |
| `2`  | Auth (`AuthenticationError`, including Enterprise-gate 403) |
| `3`  | Rate limit hit |
| `4`  | Dataset / resource not found |
| `5`  | Other API error |
| `64` | Bad usage (mirrors `sysexits.h`) |

## Performance (Arrow)

`client.get()` transparently negotiates **Apache Arrow** over the wire — same
`DataFrame` back, typically **5–10× faster end-to-end** on large pulls, with
an automatic JSON fallback. No setup: `pyarrow` ships with `eolas-data`, so
this is on by default; `format=` (`"json"`/`"csv"`) is only for the rare case
you want the raw text payload.

For a columnar file (CLI), use `--format parquet --out FILE`; via the REST
API directly, `?format=parquet`. Full benchmark: [docs.eolas.fyi → Python
reference → Performance](https://docs.eolas.fyi/python/reference/).

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

`Dataset` is a `pandas.DataFrame` subclass — use matplotlib / seaborn / plotly
directly. No bundled plot helper, because there's no universal "right" plot for
a tidy dataset (single-series time series vs. wide multi-measure vs. WKT
geometry all need different code).

```python
import matplotlib.pyplot as plt

df = client.statsnz("nz_cpi")
df.plot(x="date", y="value")
plt.show()
```

## Type stubs

Dataset names are exposed as a `Literal` so IDEs autocomplete the catalog:

```python
from eolas_data import Client

client = Client()
client.get("nz_")    # autocomplete shows nz_cpi, nz_gdp_production_annual, ...
```

The list is regenerated from the live API at release time. Passing a name not in the snapshot still works at runtime — the type hint just won't autocomplete it. Catalog snapshot date is exposed as `eolas_data._dataset_names.CATALOG_SNAPSHOT_DATE`.

## Releasing

See [`docs/clients.md`](https://github.com/phildonovan/eolas/blob/master/docs/clients.md) in the eolas data repo for the tagged-release flow and PyPI token rotation.

Before each release: `python -m eolas_data._regen_names` to refresh the dataset name stubs from the live API, commit the change, then tag and push.

## License

MIT
