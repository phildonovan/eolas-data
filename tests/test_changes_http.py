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
            # Real server shape: FastAPI nests HTTPException(detail={...}) under
            # "detail" (DRIFT-2). The flat shape the mock used before never
            # exercised the nesting the client has to parse.
            410,
            {"detail": {"error": "watermark_expired", "min_available_seq": 42}},
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
        client._raw_changes_get(
            "/v1/datasets/nz_parcels/changes", params={"since_seq": 0}
        )


@resp_lib.activate
def test_raw_changes_get_410_parses_nested_min_available_seq(client):
    # The watermark floor drives the client's re-baseline decision; parsing it
    # from the wrong nesting level silently pinned it to 0 (DRIFT-2).
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/v1/datasets/nz_parcels/changes",
        json={"detail": {"error": "watermark_expired", "min_available_seq": 514000}},
        status=410,
    )
    with pytest.raises(WatermarkExpired) as excinfo:
        client._raw_changes_get(
            "/v1/datasets/nz_parcels/changes", params={"since_seq": 0}
        )
    assert excinfo.value.min_available_seq == 514000
