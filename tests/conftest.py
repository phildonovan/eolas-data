import pytest


@pytest.fixture(autouse=True)
def _no_plan_notice_lookup(monkeypatch):
    """`eolas auth set-key` looks up /v1/me for a plan notice; keep tests offline."""
    monkeypatch.setenv("EOLAS_NO_NOTICE", "1")
