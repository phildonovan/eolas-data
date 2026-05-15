#!/usr/bin/env bash
# Pre-release gate for the `eolas-data` Python client + `eolas` CLI.
#
# CRITICAL CONTEXT: .github/workflows/publish.yml triggers on `v*` tags and runs
# `python -m build` then publishes STRAIGHT TO PyPI with **no pytest step**.
# There is no automated test gate. This script IS the gate — run it GREEN
# before creating any release tag.
#
# Usage:  ./scripts/preflight.sh
# Exit:   0 = safe to release, non-zero = do NOT release.
#
# The CLI smoke step needs a live API key in EOLAS_API_KEY. If unset, that step
# is SKIPPED (loudly) so the script still works in CI without secrets — but a
# real pre-release run should set it.

set -euo pipefail
cd "$(dirname "$0")/.."

fail() { echo "❌ PREFLIGHT FAILED: $*" >&2; exit 1; }

echo "==> 1/4  install dev deps"
pip install -q -e ".[dev]" || fail "pip install -e .[dev] failed"

echo "==> 2/4  pytest (the gate publish.yml does NOT have)"
pytest -q || fail "pytest failed"

echo "==> 3/4  build sdist + wheel"
python -m build >/dev/null 2>&1 || fail "python -m build failed"

echo "==> 4/4  CLI smoke against the live API"
if [[ -z "${EOLAS_API_KEY:-}" ]]; then
  echo "    ⚠️  EOLAS_API_KEY unset — SKIPPING live CLI smoke."
  echo "    ⚠️  A real pre-release run MUST exercise this. Set EOLAS_API_KEY and re-run."
else
  eolas datasets >/dev/null            || fail "'eolas datasets' failed against live API"
  echo "    OK — 'eolas datasets'"
  eolas get nz_cpi --limit 5 >/dev/null || fail "'eolas get nz_cpi' failed against live API"
  echo "    OK — 'eolas get nz_cpi --limit 5'"
fi

echo "✅ PREFLIGHT OK — safe to release the Python client / CLI"
echo "   (reminder: clients are frozen at v1.0.0 until launch — see"
echo "    feedback_clients_freeze_until_launch.md before tagging)"
