# eolas-data

[![Tests](https://github.com/phildonovan/eolas-data/actions/workflows/test.yml/badge.svg)](https://github.com/phildonovan/eolas-data/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/phildonovan/eolas-data/graph/badge.svg)](https://codecov.io/gh/phildonovan/eolas-data)
[![PyPI](https://img.shields.io/pypi/v/eolas-data)](https://pypi.org/project/eolas-data/)
[![Python](https://img.shields.io/pypi/pyversions/eolas-data)](https://pypi.org/project/eolas-data/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python client for the [eolas.fyi](https://eolas.fyi) statistical data API — 1,500+ official New Zealand statistical & geospatial datasets, plus OECD data for international comparisons, served as tidy `pandas` DataFrames (or `polars` / `geopandas` if you prefer).

_Coverage is New Zealand + OECD today. Australian sources are on the roadmap — not yet available; OECD data already includes Australia (and other OECD members) for cross-country comparisons._

```bash
pip install eolas-data
```

## Quickstart

```python
from eolas_data import Client

client = Client("your_api_key")   # or set EOLAS_API_KEY in env

# CPI index (monthly, RBNZ M1) — the usual Treasury/analyst choice
cpi = client.rbnz("rbnz_m1_prices", start="2020-01-01")

# OECD macro indicators (quarterly YoY % — not CPI index levels)
inflation = client.oecd("nz_cpi", start="2020-01-01")
gdp       = client.oecd("nz_gdp_growth")

# Discovery
all_datasets = client.list()
nz_only      = client.list("Stats NZ")
client.search("cpi")   # expands aliases; surfaces rbnz_m1_prices before nz_cpi
meta         = client.info("rbnz_m1_prices")
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

After setting the library, `client.get_local("nz_parcels")` will use `~/eolas-library/` automatically.

The keyring slot and config file are shared with the R `eolas` client — a key saved from Python is immediately readable from R and vice versa (see the [R client README](https://github.com/phildonovan/eolas-r)).

---

## Command-line interface

`pip install eolas-data` includes the `eolas` CLI for browsing, fetching, and
scheduling — useful for shell scripts, cron jobs, and AI-agent workflows. Rich
tables by default; pass ``--json`` for newline-delimited JSON in scripts.

```bash
# one-time setup (OS keyring — recommended)
pip install 'eolas-data[secure]'
eolas auth save-key

# or config file (no extra install)
eolas auth set-key
eolas health

# discover
eolas datasets list --source "Stats NZ"
eolas datasets list --search cpi          # table + CPI guidance note
eolas datasets list --search cpi --json | jq '.[].name'
eolas datasets info rbnz_m1_prices
eolas datasets preview rbnz_m1_prices --limit 5

# fetch (verb matches the Python lib's client.get())
eolas get rbnz_m1_prices --format csv > cpi.csv
eolas get nz_cpi --start 2020-01-01 --format json | jq '.[].value'   # OECD YoY %
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

## Bulk downloads — use `get_local()` for whole datasets

`client.get()` hits the live `/data` endpoint (good for slices and small pulls). For whole datasets — especially large or geospatial layers — use `get_local()`. It syncs a CDN-cached Parquet/GeoParquet file to your library directory and reads from disk on subsequent calls.

```python
# Whole-dataset path: nz_parcels from CDN-cached GeoParquet (seconds, not a 15-min Iceberg scan)
gdf = client.get_local("nz_parcels")   # geopandas.GeoDataFrame when [geo] is installed
df  = client.get_local("nz_cpi")       # tidy DataFrame from cached Parquet

# Live path: date slices, row limits, licence-restricted sources (e.g. OECD)
df  = client.get("nz_cpi", start="2020-01-01")
df  = client.get("nz_cpi", limit=100)
```

Use `get_local()` when you need to control `cache_dir`, `format`, or `freshness`:

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

**Progress bars:** `get_local()` shows two phases in interactive sessions — a **download** byte bar while fetching from CDN, then a **read** spinner while Parquet/GeoParquet is loaded (often the slow part on multi-million-row geo datasets). Control with `progress=True` (both), `False` (neither), `"download"`, or `"read"`. Set `EOLAS_NO_PROGRESS=1` to suppress both in batch scripts. Cached files skip the download bar and print an informative message instead.

CLI mirror: `eolas download <name>` for one-shot, `eolas sync <name> [--watch hourly]` for an incremental check. Full docs: [docs.eolas.fyi/bulk-downloads/](https://docs.eolas.fyi/bulk-downloads/).

## Sync — always-fresh local copy

`client.sync(name, path)` keeps a local file current, automatically choosing *how* based on the dataset's CDC serving tier — you make the same call either way:

- **snapshot-tier** datasets → full-snapshot download, re-fetched only when the server snapshot changes (`sync_bulk()`).
- **changelog-tier** datasets (e.g. the LINZ SCD2 layers) → incremental: the first call downloads a baseline, then later calls fetch *only what changed* from the `/changes` feed and pk-merge it into your file (`sync_changes()`).

```python
# Same call regardless of tier:
r = client.sync("nz_building_outlines", path="buildings.parquet")
r.status        # "downloaded" (baseline) | "updated" | "unchanged"
r.sync_mode     # "changelog" for changelog-tier datasets
r.ops_applied   # number of change rows applied this run
r.current_seq   # feed watermark after this sync

# First call baselines; subsequent calls apply only new changes:
r = client.sync("nz_building_outlines", path="buildings.parquet")
r.ops_applied   # e.g. 1240
```

A sidecar at `str(path) + ".eolas-meta.json"` records the snapshot id / feed watermark so the next call fetches only new data. For SCD2 datasets the merge keeps only the current rows (`is_current = true`), so `buildings.parquet` is always a clean current-state snapshot — the SCD2 history is handled for you. A `410` (watermark expired) self-heals by re-baselining.

The changelog sidecar (`schema_version` 2) is byte-compatible with the R `eolas` client: a file synced from Python can be resumed from R and vice versa.

## Geospatial

Datasets with a `geometry_wkt` column auto-convert to `geopandas.GeoDataFrame` if `geopandas` is installed:

```bash
pip install eolas-data[geo]
```

```python
gdf = client.get("nz_addresses")                  # GeoDataFrame
df  = client.get("nz_addresses", as_geo=False)    # plain DataFrame, WKT preserved
```

## Working with large geo datasets

The 5.4M-row `linz.nz_parcels` table allocates ~10 GB when materialised as a GeoDataFrame. Pass `as_arrow=True` to skip all shapely allocation and get a zero-copy `pyarrow.Table` instead — geometry stays as Arrow buffers until you need it:

```python
# Zero-copy Arrow table — no shapely allocation
tbl = client.linz("nz_parcels", as_arrow=True)

# Filter before materialising — dramatically cheaper than loading the full GeoDataFrame
import duckdb
result = duckdb.sql("""
    SELECT parcel_id, geometry_wkt
    FROM tbl
    WHERE ST_Within(ST_GeomFromText(geometry_wkt),
                    ST_GeomFromText('POLYGON((174.7 -41.3, 174.8 -41.3, 174.8 -41.4, 174.7 -41.4, 174.7 -41.3))'))
""").df()
```

`as_arrow=True` works on all datasets (geo or non-geo), all routing modes (live, cached, auto), and all source helpers. It cannot be combined with `as_geo=True`.

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

## Testing

```bash
# unit tests (mocked HTTP — no API key needed)
pytest -q -m "not integration"

# live smoke (requires EOLAS_API_KEY)
EOLAS_API_KEY=vs_... pytest -q -m integration tests/test_smoke_live.py
```

CI runs the unit suite on Python 3.10, 3.12, and 3.13 on every push/PR, with [coverage](https://codecov.io/gh/phildonovan/eolas-data) uploaded to Codecov. A weekly workflow optionally runs live smoke tests when `EOLAS_API_KEY` is configured as a repository secret.

## Releasing

See [`docs/clients.md`](https://github.com/phildonovan/eolas/blob/master/docs/clients.md) in the eolas data repo for the tagged-release flow and PyPI token rotation.

Before each release: `python -m eolas_data._regen_names` to refresh the dataset name stubs from the live API, commit the change, then tag and push.

## License

MIT — applies to this client software only. Dataset use is subject to each
source's licence and your [eolas API plan](https://eolas.fyi/#pricing).
