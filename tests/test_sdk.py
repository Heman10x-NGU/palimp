"""Tests for the GraphCtx Python async SDK."""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

from palimp.sdk import PalimpClient
from palimp.server import app


@pytest.fixture()
async def sdk(tmp_path):
    """Yield a PalimpClient pointed at the test ASGI app.

    Manually runs the lifespan so app.state is populated.
    """
    db_path = str(tmp_path / "test_sdk.db")
    old_val = os.environ.get("PALIMP_DB")
    os.environ["PALIMP_DB"] = db_path

    # Run lifespan to populate app.state
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        client = PalimpClient.__new__(PalimpClient)
        client._base = "http://testserver"
        client._client = AsyncClient(transport=transport, base_url="http://testserver")
        yield client
        await client._client.aclose()

    if old_val is None:
        os.environ.pop("PALIMP_DB", None)
    else:
        os.environ["PALIMP_DB"] = old_val


@pytest.mark.asyncio
async def test_health(sdk):
    result = await sdk.health()
    assert result["status"] == "ok"
    assert "version" in result


@pytest.mark.asyncio
async def test_add_memory(sdk):
    result = await sdk.add_memory("demo", "Alice prefers concise answers.")
    assert result["memory_id"].startswith("mem_")
    assert result["episode_id"].startswith("eps_")


@pytest.mark.asyncio
async def test_recall(sdk):
    # Seed data
    await sdk.add_memory("demo", "Alice prefers concise answers.", extract=False)
    await sdk.add_knowledge("demo", "Architecture", "GraphCtx uses SQLite.", extract=False)

    # Recall
    result = await sdk.recall("demo", "What does Alice prefer?")
    assert "results" in result
    assert len(result["results"]) >= 1
    for r in result["results"]:
        assert "id" in r
        assert "kind" in r
        assert r["kind"] in ("memory", "knowledge")
        assert "score" in r


@pytest.mark.asyncio
async def test_session_lifecycle(sdk):
    # Create session
    session = await sdk.create_session("demo", user_ref="alice")
    assert session["session_id"].startswith("ses_")
    assert session["namespace"] == "demo"
    assert session["user_ref"] == "alice"
    assert session["closed_at"] is None

    # Close session
    close_result = await sdk.close_session(session["session_id"], summary="Done.")
    assert close_result["status"] == "closed"
    assert close_result["session_id"] == session["session_id"]
