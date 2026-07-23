"""MCP protocol tests — spawn eolas-mcp over stdio and exercise the wire format.

These run in the default unit CI job (no EOLAS_API_KEY required).
``eolas_health`` hits the public /health endpoint only.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import sys
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _server_params() -> StdioServerParameters:
    """Spawn the server hermetically as ``<this interpreter> -m eolas_data.mcp_server``.

    Deliberately does NOT use ``shutil.which("eolas-mcp")``: the console script
    resolves against PATH, which on a dev box (or a machine with a global
    anaconda install) can pick up an ``eolas-mcp`` from a *different*, older
    eolas-data — one that predates this module — and the subprocess then dies with
    ``ModuleNotFoundError: No module named 'eolas_data.mcp_server'``, failing these
    tests for a reason that has nothing to do with the code under test.

    ``sys.executable -m`` always runs the interpreter these tests are running
    under (the venv/tox env with the code being tested), and exercises the exact
    same ``main()`` the console script calls. Entry-point wiring is covered
    separately by the wheel-install CI job.
    """
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "eolas_data.mcp_server"],
    )


async def _with_session(coro):
    params = _server_params()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await coro(session)


def _tool_json(result) -> Any:
    assert result.content, "tool result must include content"
    block = result.content[0]
    text = getattr(block, "text", None)
    assert text is not None, f"expected text content block, got {type(block)}"
    return json.loads(text)


def test_stdio_lists_six_tools():
    async def _run(session: ClientSession):
        listed = await session.list_tools()
        names = {t.name for t in listed.tools}
        assert names == {
            "eolas_health",
            "eolas_search",
            "eolas_info",
            "eolas_get",
            "eolas_download",
            "eolas_sync",
        }

    asyncio.run(_with_session(_run))


def test_stdio_eolas_health_over_wire():
    async def _run(session: ClientSession):
        result = await session.call_tool("eolas_health", {})
        payload = _tool_json(result)
        assert payload["ok"] is True
        assert payload["status_code"] == 200
        assert payload["body"]["status"] == "ok"

    asyncio.run(_with_session(_run))


def test_eolas_mcp_entry_point_installed():
    """Console script from pyproject [project.scripts] must be on PATH in dev/CI."""
    assert shutil.which("eolas-mcp") or importlib.util.find_spec(
        "eolas_data.mcp_server"
    )
