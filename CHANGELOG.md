# Changelog

All notable changes to `eolas-data` are recorded here. This project follows
[Semantic Versioning](https://semver.org/).

## 1.9.0

> Version jumps 1.4.0 -> 1.9.0. PyPI versions 1.5.0-1.8.0 were uploaded in May
> 2026 and later yanked; yanked filenames are permanently reserved, so those
> numbers can never be published again. The 1.9.0 jump clears the band for good.
> The R client moves to 1.9.0 at the same time so the two stay aligned.


### Added

- **`geometry=False` on `get()`** — omit the `geometry_wkt` column. Two-thirds of
  eolas datasets (1017/1536) carry geometry, and on TA/RC boundary tables the WKT
  dwarfs the attributes. The column is projected away at the API's storage layer,
  so it is never read from S3 or transferred — this cuts I/O, not just payload.
  Responses carry `X-Eolas-Geometry-Omitted: true`.

### Changed

- **Whole-dataset pulls of small spatial tables stay on the live path when
  `geometry=False`.** Geometry was one of the two triggers for the API's
  large-dataset (413) guard, so `get("some_boundary_table")` was transparently
  diverted to a bulk download. The client now mirrors the server's relaxed guard.
  The row-count trigger is unchanged — dropping a column doesn't reduce row count.

### Fixed

- **`geometry=False` is now honoured on the bulk-routed path.** A spatial table
  over the 100k-row threshold stays "blocked" even with `geometry=False`, so
  `get()` routes it to the bulk cache — which has no server-side projection. The
  flag was dropped at that hand-off, so the caller silently received the full
  geometry-bearing file and, with `as_geo=None`, an auto-converted GeoDataFrame.
  Found in peer review.
- **The in-memory response cache now keys on `geometry`.** Without it,
  `get(x)` and `get(x, geometry=False)` shared a cache key, so whichever ran
  second silently returned the other's shape (a whole column different). Only
  reachable with `Client(cache=True)`.

## 1.4.0

### Added

- **`Client.preview(name)`** — up to 10 sample rows via the unauthenticated
  `/preview` endpoint. No API key required, no rate-limit cost, geometry hidden.
  `eolas datasets preview` now uses it instead of burning an authenticated,
  rate-limited `/data` call (DRIFT-5).
- **`dimensions=` filter on `get()`** and `eolas get --dimensions` — a
  case-insensitive substring filter on dimension columns, applied server-side on
  the live `/data` path (the filter was live in prod but exposed by no client;
  DRIFT-4).

## 1.3.22

Network-hardening + correctness release, addressing findings from the
2026-07-05 client-library audit.

### Fixed

- **Request timeouts everywhere (EH-1).** Every HTTP call now carries a default
  `(10, 300)` second (connect, read) timeout via a session subclass. A
  black-holed connection raises instead of hanging the caller forever. Override
  with `Client(timeout=...)`.
- **Clean errors on transport failures (EH-2).** Connection refused / DNS /
  timeout / reset now surface as `EolasError("Network error talking to
  api.eolas.fyi: ...")` instead of a raw urllib3 traceback. The CLI's existing
  error handler renders them cleanly.
- **Atomic downloads (EH-5).** `download()` / `download_bulk()` write to a temp
  file and atomically rename on success, and verify bytes received against
  `Content-Length`. An interrupted or truncated download now leaves **no file**
  at the final path instead of a silent partial.
- **`get()` no longer degrades a bulk failure to a misleading 413 (EH-3).** The
  auto-route swallow is narrowed to the routing decision; a real failure in the
  bulk path now propagates instead of falling through to "use bulk download".
- **Arrow negotiation no longer masks real errors (EH-7).** An auth/404/429
  during the Arrow attempt now propagates immediately, without the bogus "Arrow
  IPC unavailable" nag and without re-sending the failed request as JSON. Only a
  non-Arrow 200 or a genuine pyarrow/IPC decode error downgrades the session.
- **Watermark floor parsed correctly (DRIFT-2).** The HTTP 410 re-baseline
  response nests its fields under `detail`; `WatermarkExpired.min_available_seq`
  was always 0. Now read from the correct level with a top-level fallback.
- **No more pandas-3 warning on every `get()` (PY-3).** Metadata accessors are
  set via `object.__setattr__`, bypassing the pandas 3 `UserWarning`.
- **Orphaned partial downloads are swept (PY-5).** `*.eolas-tmp-*` files >24h
  old are cleaned on download start; a full `cache_clear()` removes them all.
- **`download()` docstring corrected (PY-1).** It no longer claims to work for
  "all datasets" — it is the live path and 413s on whole-dataset pulls of
  large/geo tables; use `get()` or `download_bulk()` there.
- **`eolas datasets list` no longer renders a blank ghost column (CLI-1).** On
  narrow (<100 col) terminals the title column is dropped with a hint instead of
  being squeezed to zero width.
- **Bulk-download docs corrected (DRIFT-3).** `download_bulk()` / the CLI
  `--freshness` help no longer claim Free plans get monthly bulk — bulk is
  Pro/Enterprise and Free keys receive HTTP 402.

### Changed

- **Single-sourced version (REL-1).** `__version__` is read from installed
  distribution metadata, so it can never drift from `pyproject.toml` again
  (shipped 1.3.21 previously self-reported 1.3.20). The publish workflow asserts
  the installed `__version__` equals the release tag.
- **`merge_changes` is now a public export (DRIFT-6)** — added to `__all__`.
