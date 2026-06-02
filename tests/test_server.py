"""Tests for GraphCtx REST API server."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from graphctx.server import app


@pytest.fixture()
def client(tmp_path):
    """Yield a TestClient backed by a temp SQLite database."""
    db_path = str(tmp_path / "test_server.db")
    old_val = os.environ.get("GRAPHCTX_DB")
    os.environ["GRAPHCTX_DB"] = db_path
    with TestClient(app) as c:
        yield c
    # Restore env
    if old_val is None:
        os.environ.pop("GRAPHCTX_DB", None)
    else:
        os.environ["GRAPHCTX_DB"] = old_val


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.3.0"
    assert body["storage"] == "sqlite"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats(client):
    resp = client.get("/v1/stats", params={"namespace": "demo"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["namespace"] == "demo"
    assert body["memories"] == 0
    assert body["knowledge_items"] == 0
    assert body["entities"] == 0
    assert body["edges"] == 0
    assert body["claims"] == 0


# ---------------------------------------------------------------------------
# Add memory
# ---------------------------------------------------------------------------


def test_add_memory(client):
    resp = client.post(
        "/v1/memories",
        json={
            "namespace": "demo",
            "content": "Alice prefers concise technical answers.",
            "source_ref": "chat:2026-06-02",
            "metadata": {"user_id": "local-user"},
            "extract": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["memory_id"].startswith("mem_")
    assert body["episode_id"].startswith("eps_")
    # RuleBasedExtractor should extract Alice as entity
    assert isinstance(body["entities"], list)
    assert isinstance(body["claims"], list)
    assert isinstance(body["warnings"], list)

    # Verify stats updated
    stats = client.get("/v1/stats", params={"namespace": "demo"}).json()
    assert stats["memories"] >= 1


# ---------------------------------------------------------------------------
# Add knowledge
# ---------------------------------------------------------------------------


def test_add_knowledge(client):
    resp = client.post(
        "/v1/knowledge",
        json={
            "namespace": "demo",
            "title": "GraphCtx Architecture",
            "content": "GraphCtx uses SQLite, FTS, embeddings, and provenance.",
            "source_ref": "docs/architecture.md",
            "extract": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["knowledge_id"].startswith("knw_")
    assert body["episode_id"].startswith("eps_")
    assert isinstance(body["entities"], list)
    assert isinstance(body["claims"], list)

    stats = client.get("/v1/stats", params={"namespace": "demo"}).json()
    assert stats["knowledge_items"] >= 1


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


def test_recall(client):
    # Seed a memory
    client.post(
        "/v1/memories",
        json={"namespace": "demo", "content": "Alice prefers concise technical answers.", "extract": False},
    )
    # Seed knowledge
    client.post(
        "/v1/knowledge",
        json={
            "namespace": "demo",
            "title": "Architecture",
            "content": "GraphCtx uses SQLite and provenance.",
            "extract": False,
        },
    )

    resp = client.post(
        "/v1/recall",
        json={
            "namespace": "demo",
            "query": "What does Alice prefer and what storage does GraphCtx use?",
            "mode": "hybrid",
            "limit": 8,
            "include_provenance": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "query" in body
    assert "results" in body
    assert isinstance(body["results"], list)
    # We should get at least one result
    assert len(body["results"]) >= 1
    for result in body["results"]:
        assert "id" in result
        assert "kind" in result
        assert result["kind"] in ("memory", "knowledge")
        assert "score" in result
        assert "safety" in result
        assert result["safety"]["treat_as_instruction"] is False


# ---------------------------------------------------------------------------
# Context entity
# ---------------------------------------------------------------------------


def test_context_entity(client):
    # Add memory with extraction to create entities
    resp = client.post(
        "/v1/memories",
        json={
            "namespace": "demo",
            "content": "Alice prefers concise technical answers.",
            "extract": True,
        },
    )
    body = resp.json()
    entities = body.get("entities", [])
    if entities:
        entity_id = entities[0]["id"]
        ctx_resp = client.get(
            f"/v1/context/{entity_id}",
            params={"namespace": "demo"},
        )
        assert ctx_resp.status_code == 200
        ctx = ctx_resp.json()
        assert ctx["entity"]["id"] == entity_id
        assert "claims" in ctx
        assert "edges" in ctx
        assert "provenance" in ctx


def test_context_entity_not_found(client):
    resp = client.get("/v1/context/ent_nonexistent", params={"namespace": "demo"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Delete source
# ---------------------------------------------------------------------------


def test_delete_source(client):
    # Add knowledge
    resp = client.post(
        "/v1/knowledge",
        json={
            "namespace": "demo",
            "title": "Temp Doc",
            "content": "Temporary document for deletion test.",
            "extract": False,
        },
    )
    episode_id = resp.json()["episode_id"]

    # Delete it
    del_resp = client.delete(
        f"/v1/sources/{episode_id}",
        params={"namespace": "demo"},
    )
    assert del_resp.status_code == 200
    del_body = del_resp.json()
    assert del_body["status"] == "deleted"
    assert del_body["episode_id"] == episode_id

    # Recall should no longer return it
    recall_resp = client.post(
        "/v1/recall",
        json={"namespace": "demo", "query": "Temporary document", "mode": "fast"},
    )
    result_ids = [r["id"] for r in recall_resp.json()["results"]]
    assert episode_id not in result_ids


def test_delete_source_not_found(client):
    resp = client.delete("/v1/sources/eps_nonexistent", params={"namespace": "demo"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Namespace validation (422)
# ---------------------------------------------------------------------------


def test_missing_namespace_memory(client):
    resp = client.post("/v1/memories", json={"content": "test"})
    assert resp.status_code == 422


def test_missing_namespace_knowledge(client):
    resp = client.post(
        "/v1/knowledge",
        json={"title": "t", "content": "c"},
    )
    assert resp.status_code == 422


def test_missing_namespace_recall(client):
    resp = client.post("/v1/recall", json={"query": "test"})
    assert resp.status_code == 422


def test_invalid_namespace_characters(client):
    resp = client.post(
        "/v1/memories",
        json={"namespace": "bad namespace!", "content": "test"},
    )
    assert resp.status_code == 422


def test_namespace_too_long(client):
    resp = client.post(
        "/v1/memories",
        json={"namespace": "a" * 65, "content": "test"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------


def test_namespace_isolation(client):
    # Insert secret memory in "private"
    client.post(
        "/v1/memories",
        json={
            "namespace": "private",
            "content": "Secret information only for private namespace.",
            "extract": False,
        },
    )

    # Search in "public" should return zero matching results
    recall_resp = client.post(
        "/v1/recall",
        json={
            "namespace": "public",
            "query": "Secret information",
            "mode": "fast",
        },
    )
    assert recall_resp.status_code == 200
    results = recall_resp.json()["results"]
    # No results from private namespace should leak
    for r in results:
        assert "Secret information" not in r.get("content", "")

    # Stats for public should not include private data
    stats = client.get("/v1/stats", params={"namespace": "public"}).json()
    assert stats["memories"] == 0

    stats_private = client.get("/v1/stats", params={"namespace": "private"}).json()
    assert stats_private["memories"] >= 1


# ---------------------------------------------------------------------------
# Batch memory
# ---------------------------------------------------------------------------


def test_memory_batch(client):
    """10 items, all succeed."""
    items = [{"content": f"Memory item {i}", "extract": False} for i in range(10)]
    resp = client.post(
        "/v1/memories/batch",
        json={"namespace": "demo", "items": items, "extract": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 10
    assert body["successful"] == 10
    assert body["failed"] == 0
    assert len(body["results"]) == 10
    assert len(body["errors"]) == 0

    # Verify stats
    stats = client.get("/v1/stats", params={"namespace": "demo"}).json()
    assert stats["memories"] == 10


# ---------------------------------------------------------------------------
# Batch knowledge
# ---------------------------------------------------------------------------


def test_knowledge_batch(client):
    """10 items, all succeed."""
    items = [{"title": f"Doc {i}", "content": f"Knowledge content {i}", "extract": False} for i in range(10)]
    resp = client.post(
        "/v1/knowledge/batch",
        json={"namespace": "demo", "items": items, "extract": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 10
    assert body["successful"] == 10
    assert body["failed"] == 0
    assert len(body["results"]) == 10
    assert len(body["errors"]) == 0

    stats = client.get("/v1/stats", params={"namespace": "demo"}).json()
    assert stats["knowledge_items"] == 10


# ---------------------------------------------------------------------------
# Batch max size
# ---------------------------------------------------------------------------


def test_batch_max_size(client):
    """51 items -> 422 error."""
    items = [{"content": f"Item {i}", "extract": False} for i in range(51)]
    resp = client.post(
        "/v1/memories/batch",
        json={"namespace": "demo", "items": items, "extract": False},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Batch partial failure
# ---------------------------------------------------------------------------


def test_batch_partial_failure(client):
    """Some items fail (missing content), others succeed."""
    items = [
        {"content": "Valid memory 0", "extract": False},
        {"content": "", "extract": False},  # empty content - may still succeed but tests the path
        {"content": "Valid memory 2", "extract": False},
    ]
    # Valid items should succeed; empty content items may or may not fail
    # depending on schema constraints, but the batch should not abort
    resp = client.post(
        "/v1/memories/batch",
        json={"namespace": "demo", "items": items, "extract": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    # At least the valid items succeed
    assert body["successful"] >= 2
    assert body["successful"] + body["failed"] == body["total"]


# ---------------------------------------------------------------------------
# Batch timing
# ---------------------------------------------------------------------------


def test_batch_timing(client):
    """elapsed_ms is populated and positive."""
    items = [{"content": f"Timing test {i}", "extract": False} for i in range(5)]
    resp = client.post(
        "/v1/memories/batch",
        json={"namespace": "demo", "items": items, "extract": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "elapsed_ms" in body
    assert body["elapsed_ms"] > 0.0


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------


def test_create_session(client):
    resp = client.post(
        "/v1/sessions",
        json={"namespace": "demo", "user_ref": "alice", "metadata": {"role": "tester"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"].startswith("ses_")
    assert body["namespace"] == "demo"
    assert body["user_ref"] == "alice"
    assert body["closed_at"] is None
    assert "created_at" in body


def test_close_session(client):
    # Create
    resp = client.post("/v1/sessions", json={"namespace": "demo"})
    session_id = resp.json()["session_id"]

    # Close
    resp = client.post(
        f"/v1/sessions/{session_id}/close",
        json={"summary": "Task completed."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "closed"
    assert body["session_id"] == session_id


def test_close_session_not_found(client):
    resp = client.post("/v1/sessions/ses_nonexistent/close", json={})
    assert resp.status_code == 404


def test_close_session_already_closed(client):
    resp = client.post("/v1/sessions", json={"namespace": "demo"})
    session_id = resp.json()["session_id"]

    # Close once
    client.post(f"/v1/sessions/{session_id}/close", json={})
    # Close again -> 409
    resp = client.post(f"/v1/sessions/{session_id}/close", json={})
    assert resp.status_code == 409


def test_list_sessions(client):
    # Create two sessions
    client.post("/v1/sessions", json={"namespace": "demo", "user_ref": "alice"})
    client.post("/v1/sessions", json={"namespace": "demo", "user_ref": "bob"})

    resp = client.get("/v1/sessions", params={"namespace": "demo"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert all(s["namespace"] == "demo" for s in body)


def test_list_sessions_empty(client):
    resp = client.get("/v1/sessions", params={"namespace": "empty"})
    assert resp.status_code == 200
    assert resp.json() == []
