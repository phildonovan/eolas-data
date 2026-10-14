"""Trial / plan notices and PlanError (eolas-data 1.14.0)."""
import warnings

import pytest
import responses as resp_lib
from typer.testing import CliRunner

from eolas_data import Client, PlanError, PlanNotice
from eolas_data.exceptions import APIError

BASE = "https://api.eolas.nz"
NOTICE = ("Pro trial: 50,000 rows/request, no bulk downloads; ends 2026-10-28. "
          "Large pulls are paginated and slow on trial - upgrade: https://eolas.nz/pricing")

ME = {"plan": "trial", "trial_ends_at": "2026-10-28T00:00:00Z", "row_cap": 50000,
      "bulk": False, "changes": False, "notice": NOTICE,
      "upgrade_url": "https://eolas.nz/pricing"}


@resp_lib.activate
def test_me_returns_plan():
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/me", json=ME, headers={"X-Eolas-Notice": NOTICE})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert Client(api_key="vs_t").me()["plan"] == "trial"


@resp_lib.activate
def test_notice_header_warns_once_per_client():
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/me", json=ME, headers={"X-Eolas-Notice": NOTICE})
    c = Client(api_key="vs_t")
    with pytest.warns(PlanNotice, match="Pro trial"):
        c.me()
    with warnings.catch_warnings():
        warnings.simplefilter("error", PlanNotice)
        c.me()  # same notice: not repeated


@resp_lib.activate
def test_no_notice_no_warning():
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/me", json={**ME, "plan": "pro", "notice": None})
    with warnings.catch_warnings():
        warnings.simplefilter("error", PlanNotice)
        Client(api_key="vs_t").me()


@pytest.mark.parametrize("status,code", [
    (402, "trial_expired"), (403, "bulk_not_in_plan"), (429, "trial_soft_cap"),
])
@resp_lib.activate
def test_plan_refusals_raise_plan_error(status, code):
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/me", status=status,
                 json={"detail": "Not in your plan. https://eolas.nz/pricing"},
                 headers={"X-Eolas-Error-Code": code})
    with pytest.raises(PlanError) as ei:
        Client(api_key="vs_t").me()
    assert ei.value.code == code and ei.value.status_code == status
    assert isinstance(ei.value, APIError)  # existing `except APIError` still catches it


def test_bulk_status_raises_plan_error():
    import requests

    r = requests.Response()
    r.status_code = 403
    r.headers["X-Eolas-Error-Code"] = "bulk_not_in_plan"
    r._content = b'{"detail": "Bulk not included in the eolas trial."}'
    with pytest.raises(PlanError, match="pricing"):
        Client._raise_for_bulk_status(r)


def test_plain_403_is_not_plan_error():
    import requests
    from eolas_data.exceptions import AuthenticationError

    r = requests.Response()
    r.status_code = 403
    r._content = b'{"detail": "API key is inactive."}'
    with pytest.raises(AuthenticationError):
        Client._raise_for_status(r)


@resp_lib.activate
def test_set_key_prints_plan_notice(monkeypatch, tmp_path):
    from eolas_data import cli

    monkeypatch.delenv("EOLAS_NO_NOTICE", raising=False)
    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "config.json")
    resp_lib.add(resp_lib.GET, f"{BASE}/v1/me", json=ME)
    out = CliRunner().invoke(cli.app, ["auth", "set-key", "--key", "vs_trialkey"])
    assert out.exit_code == 0
    assert "plan: trial" in out.output and "paginated" in out.output


def test_set_key_survives_offline(monkeypatch, tmp_path):
    from eolas_data import cli

    monkeypatch.delenv("EOLAS_NO_NOTICE", raising=False)
    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "config.json")
    with resp_lib.RequestsMock():  # no registered URL -> connection error
        out = CliRunner().invoke(cli.app, ["auth", "set-key", "--key", "vs_k"])
    assert out.exit_code == 0 and "saved" in out.output
