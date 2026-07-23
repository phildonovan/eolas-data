"""Live MCP integration — stdio session against api.eolas.fyi with a real API key.

Skipped in the default unit CI job (``pytest -m 'not integration'``).
Enable locally or in the weekly smoke workflow::

    EOLAS_API_KEY=vs_... pytest -m integration tests/test_mcp_integration.py -q
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.integration

_SKIP = not os.environ.get("EOLAS_API_KEY")


def _server_params() -> StdioServerParameters:
    exe = shutil.which("eolas-mcp")
    if exe:
        return StdioServerParameters(command=exe, args=[], env=os.environ.copy())
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "eolas_data.mcp_server"],
        env=os.environ.copy(),
    )


async def _call_tool(name: str, arguments: dict) -> dict:
    if _SKIP:
        pytest.skip("EOLAS_API_KEY not set — live MCP integration skipped")

    params = _server_params()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            text = result.content[0].text
            return json.loads(text)


def test_mcp_search_cpi_live():
    payload = asyncio.run(_call_tool("eolas_search", {"query": "cpi", "limit": 10}))
    assert "error" not in payload
    assert payload["count"] >= 1
    names = {row["name"] for row in payload["datasets"]}
    assert "nz_cpi" in names or "rbnz_m1_prices" in names


def test_mcp_info_nz_cpi_live():
    payload = asyncio.run(_call_tool("eolas_info", {"name": "nz_cpi"}))
    assert "error" not in payload
    assert payload["name"] == "nz_cpi"
    assert payload.get("source")


def test_mcp_get_nz_cpi_slice_live():
    payload = asyncio.run(_call_tool("eolas_get", {"name": "nz_cpi", "limit": 5}))
    assert "error" not in payload
    assert payload["row_count"] >= 1
    assert len(payload["rows"]) >= 1