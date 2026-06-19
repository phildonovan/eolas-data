#!/usr/bin/env python3
"""Verify parity scenario tests from eolas/docs/client-contract.md exist locally.

Run in CI and before release. Cross-repo version parity is checked separately
in .github/workflows/version-parity.yml on tag push.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

# filename -> substrings that must appear in that file
REQUIRED: dict[str, list[str]] = {
    "test_progress.py": [
        "test_download_bulk_missing_content_length",
        "Deliberately no Content-Length",
    ],
    "test_sync_bulk.py": [
        "test_sync_bulk_force_redownloads_when_unchanged",
        "test_cache_clear_removes_files_and_session_meta",
    ],
    "test_client.py": [
        "test_get_whole_dataset_defers_to_get_local_for_large_geo",
    ],
    "test_meta.py": [
        "test_attach_meta_geodataframe_head_does_not_break_repr",
    ],
    "test_smoke_live.py": [
        "test_linz_nz_addresses_bulk_route_and_head",
        "test_client_exports_cache_clear",
    ],
}


def main() -> int:
    errors: list[str] = []
    for filename, patterns in REQUIRED.items():
        path = TESTS / filename
        if not path.exists():
            errors.append(f"missing test file: {filename}")
            continue
        text = path.read_text(encoding="utf-8")
        for pat in patterns:
            if pat not in text:
                errors.append(f"{filename}: missing required pattern {pat!r}")

    if errors:
        print("Client contract check FAILED (see eolas/docs/client-contract.md):")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Client contract check OK (Python)")
    return 0


if __name__ == "__main__":
    sys.exit(main())