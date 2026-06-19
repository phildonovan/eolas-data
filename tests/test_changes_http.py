"""HTTP error-path tests for Client._raw_changes_get (changelog endpoint)."""
from __future__ import annotations

import pytest
import responses as resp_lib

from eolas_data import Client
from eolas_data.exceptions import (
    AuthenticationError,
    ChangesLicenceRestricted,
    ChangesUpgradeRequired,
    WatermarkExpired,
)

BASE = "https://api.eolas.fyi"


@pytest.fixture()
def client():
    return Client("eolas_testkey123", base_url=BASE)


@resp_lib.activate
@pytest.mark.parametrize(
    "status,body,exc_cls,match",
    [
        (402, {"detail": "Pro required"}, ChangesUpgradeRequired, "Pro"),
        (
            403,
            {"detail": "licence: OECD prohibits export"},
            ChangesLicenceRestricted,
            "licence",
        ),
        (403, {"detail": "API key is inactive"}, AuthenticationError, "inactive"),
        (
            410,
            {"error": "watermark_expired", "min_available_seq": 42},
            WatermarkExpired,
            "watermark",
        ),
    ],
)
def test_raw_changes_get_typed_errors(client, status, body, exc_cls, match):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/v1/datasets/nz_parcels/changes",
        json=body,
        status=status,
    )
    with pytest.raises(exc_cls, match=match):
        client._raw_changes_get("/v1/datasets/nz_parcels/changes", params={"since_seq": 0})