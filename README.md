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
df = client.oecd("nz_gdp_growth")

# Discovery
all_datasets = client.list()
nz_only      = client.list("Stats NZ")
meta         = client.info("nz_cpi")
```

Get an API key at <https://eolas.fyi/signup>. Free plan is 10 requests/month; Pro ($49/month) is unlimited.

## Quick setup (workstation)

Two one-off commands make every future session frictionless:

**1. Save your API key** to the OS keyring (macOS Keychain / Windows Credential Manager / Linux Secret Service) so `Client()` finds it automatically — no env var, no pasting:

```bash
pip install 'eolas-data[secure]'   # adds the keyring package
eolas auth save-key                # interactive prompt
```

```python
from eolas_data import Client
client = Client()   # key read from OS keyring automatically
```

**2. Set a library directory** so downloaded bulk files land somewhere permanent instead of the transient `~/.cache/eolas/` OS cache:

```bash
eolas library set ~/eolas-library  # writes to ~/.eolas/config.json
```

Or set the env var instead (useful for CI / Docker):

```bash
export EOLAS_LIBRARY=~/eolas-library
```

After setting the library, `client.get_local("nz_parcels")` and the smart-routing in `client.get("nz_parcels")` will use `~/eolas-library/` automatically.

The keyring slot and config file are shared with the R `eolas` client — a key saved from Python is immediately readable from R and vice versa (see the [R client README](https://github.com/phildonovan/eolas-r)).

---

## Command-line interface

`pip install eolas-data[cli]` adds an `eolas` command for browsing, fetching, and
scheduling — useful for shell scripts, cron jobs, and AI-agent workflows. Output
auto-detects piping: rich tables in a terminal, newline-delimited JSON when
stdout is piped.

```bash
# one-time setup (OS keyring — recommended)
pip install 'eolas-data[secure]'
eolas auth save-key

# or config file (no extra install)
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
eolas get nz_meshblock_2023 --format parquet --out sa2.parquet
```

### Scheduling

Set up recurring fetches without touching crontab/Task Scheduler syntax. Works
on Linux, macOS (cron), and Windows (Task Scheduler).

```bash
eolas schedule add nz_cpi --daily   --out ~/data/cpi.csv
eolas schedule add nz_gdp_growth --weekly  --out ~/data/gdp.csv
eolas schedule add rbnz_b1_exchange_rates_monthly --cron "0 */6 * * *" --out ~/data/fx.csv   # POSIX only

eolas schedule list
eolas schedule remove nz_cpi
```

Daily is the default. Pre-flight check refuses to install a schedule unless
your API key is configured (otherwise the job would fail silently forever).

### Integrations (Enterprise plan)

Generate ready-to-run connector configs for popular data-pipeline tools — eolas
becomes a one-command source for Meltano, Fivetran, or Azure Data Factory.

```bash
eolas integrate meltano             --datasets nz_cpi,nz_gdp_growth --output ./my-pipeline/
eolas integrate fivetran            --datasets nz_cpi
eolas integrate azure-data-factory  --datasets nz_cpi,nz_gdp_growth
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

## Bulk downloads — `client.get()` is now smart

`client.get()` auto-routes large or geospatial datasets through the cache+sync path — no code change needed. `client.linz("nz_parcels")` used to take 15 minutes (live Iceberg scan through the row-oriented endpoint); it now returns a GeoDataFrame in seconds.

```python
# Smart default: nz_parcels auto-routes to CDN-cached GeoParquet, no limit needed
gdf = client.linz("nz_parcels")   # geopandas.GeoDataFrame in seconds
df  = client.get("nz_cpi")        # small dataset → stays on live path

# Escape hatches when you need explicit control:
gdf = client.get("nz_parcels", mode="live")      # force live Iceberg scan
gdf = client.get("nz_parcels", mode="cached")    # force cache+sync (= get_local)
```

**Routing rules (mode="auto", the default):**
1. If `start=`, `end=`, or `limit=` is set → always live (slice queries can't use a whole-file cache).
2. If the dataset is licence-restricted (`bulk_export_class="none"`, e.g. OECD) → always live.
3. If bulk-eligible AND (has geometry OR >100k rows) → cache+sync path.
4. Otherwise → live.

`get_local()` is the explicit alias for `mode="cached"` — use it when you need to control `cache_dir`, `format`, or `freshness`:

```python
# Explicit cache+sync with extra options
gdf = client.get_local("nz_parcels")
gdf = client.get_local("nz_parcels", cache_dir="/data/eolas", freshness="monthly")
df  = client.get_local("nz_cpi", format="csv_gz")
```

For advanced control over the sync lifecycle (sidecar tracking, atomic replace), use `sync_bulk()` directly. For one-shot bytes-or-path downloads, use `download_bulk()`:

```python
r    = client.sync_bulk("nz_cpi", path="nz_cpi.parquet")
# r.status ∈ {"downloaded", "unchanged", "updated"}; r.bytes_downloaded == 0 when unchanged.
path = client.download_bulk("treasury_fiscal_spending", path="t.parquet")
```

**Progress bars:** `download_bulk`, `sync_bulk`, `get_local`, and the smart-routing path in `get()` all show a `tqdm` progress bar automatically in interactive terminals and VSCode notebooks, so 1+ GB files are never silent. Pass `progress=False` to suppress in scripts, or set `EOLAS_NO_PROGRESS=1` in the environment for a CI-wide escape hatch. The `--no-progress` flag does the same from the CLI.

CLI mirror: `eolas download <name>` for one-shot, `eolas sync <name> [--watch hourly]` for an incremental check. Full docs: [docs.eolas.fyi/bulk-downloads/](https://docs.eolas.fyi/bulk-downloads/).

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
client.get("nz_")    # autocomplete shows nz_cpi, nz_gdp_growth, ...
```

The list is regenerated from the live API at release time. Passing a name not in the snapshot still works at runtime — the type hint just won't autocomplete it. Catalog snapshot date is exposed as `eolas_data._dataset_names.CATALOG_SNAPSHOT_DATE`.

## Releasing

See [`docs/clients.md`](https://github.com/phildonovan/eolas/blob/master/docs/clients.md) in the eolas data repo for the tagged-release flow and PyPI token rotation.

Before each release: `python -m eolas_data._regen_names` to refresh the dataset name stubs from the live API, commit the change, then tag and push.

## License

MIT
