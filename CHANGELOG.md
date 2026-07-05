# Changelog

All notable changes to `eolas-data` are recorded here. This project follows
[Semantic Versioning](https://semver.org/).

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

### Changed

- **Single-sourced version (REL-1).** `__version__` is read from installed
  distribution metadata, so it can never drift from `pyproject.toml` again
  (shipped 1.3.21 previously self-reported 1.3.20). The publish workflow asserts
  the installed `__version__` equals the release tag.
- **`merge_changes` is now a public export (DRIFT-6)** — added to `__all__`.
