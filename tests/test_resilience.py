"""Resilience regressions from the 2026-07-05 client-library audit.

Covers the network-hardening fixes in the 1.3.22 bundle:
  * EH-1 — a default request timeout is applied to every call.
  * EH-2 — transport failures surface as EolasError, not raw urllib3 tracebacks.
  * EH-5 — an interrupted download leaves NO file at the final path.
"""

from __future__ import annotations

import pathlib

import pytest
import requests
import responses as resp_lib

from eolas_data import Client
from eolas_data.client import _DEFAULT_TIMEOUT, _TimeoutSession
from eolas_data.exceptions import EolasError

BASE = "https://api.eolas.fyi"


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


# ---- EH-1: timeouts --------------------------------------------------------


def test_session_is_timeout_session_with_default(client):
    assert isinstance(client._session, _TimeoutSession)
    assert client._session._default_timeout == _DEFAULT_TIMEOUT


def test_custom_timeout_is_honoured():
    c = Client("k", base_url=BASE, timeout=7)
    assert c._session._default_timeout == 7


def test_timeout_injected_into_every_request(monkeypatch):
    # The session sets timeout via setdefault so callers never have to thread it.
    captured = {}
    sess = _TimeoutSession((3, 30))

    def fake_super_request(self, *args, **kwargs):
        captured.update(kwargs)

        class _R:
            status_code = 200

        return _R()

    monkeypatch.setattr(requests.Session, "request", fake_super_request)
    sess.request("GET", f"{BASE}/ping")
    assert captured["timeout"] == (3, 30)


# ---- EH-2: transport errors become EolasError ------------------------------


@resp_lib.activate
def test_connection_error_wrapped_as_eolas_error(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/v1/datasets/nz_cpi/data",
        body=requests.exceptions.ConnectionError("connection refused"),
    )
    with pytest.raises(EolasError) as exc:
        client._raw_get("/v1/datasets/nz_cpi/data", params={})
    assert "Network error" in str(exc.value)
    # The raw requests exception must not leak to the caller.
    assert not isinstance(exc.value, requests.exceptions.RequestException)


# ---- EH-5: interrupted downloads are atomic --------------------------------


@resp_lib.activate
def test_truncated_download_leaves_no_file(client, tmp_path):
    # Server promises 1000 bytes (Content-Length) but the connection drops
    # mid-stream. The final path must not exist and no error must be silent.
    dest = tmp_path / "nz_cpi.csv"
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/v1/datasets/nz_cpi/data",
        body=requests.exceptions.ChunkedEncodingError("peer reset"),
        headers={"Content-Length": "1000"},
    )
    with pytest.raises(EolasError):
        client.download("nz_cpi", path=str(dest))
    assert not dest.exists()
    # No orphaned tmp file at the final location either.
    assert list(pathlib.Path(tmp_path).glob("*.eolas-tmp-*")) == []
