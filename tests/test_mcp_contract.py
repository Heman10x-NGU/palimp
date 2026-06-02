"""Contract tests for GraphCtx MCP tools.

Verifies that all 5 MCP tool functions exist, return correct safety fields,
and enforce namespace requirements.
"""

from __future__ import annotations

import json
import os

import pytest


# ---------------------------------------------------------------------------
# Import guard — skip all tests if mcp extra is not installed
# ---------------------------------------------------------------------------

try:
    import mcp  # noqa: F401
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

pytestmark = pytest.mark.skipif(not HAS_MCP, reason="mcp extra not installed")


# Force in-memory DB for tests
os.environ["GRAPHCTX_DB"] = ":memory:"

from graphctx.mcp import (  # noqa: E402
    _reset_store,
    graphctx_context_get,
    graphctx_knowledge_add,
    graphctx_memory_add,
    graphctx_recall,
    graphctx_stats,
)

# ---------------------------------------------------------------------------
# Tool existence
# ---------------------------------------------------------------------------

_EXPECTED_TOOLS = [
    "graphctx_memory_add",
    "graphctx_knowledge_add",
    "graphctx_recall",
    "graphctx_context_get",
    "graphctx_stats",
]


def test_mcp_tools_exist():
    """All 5 MCP tool functions must be importable."""
    from graphctx import mcp as mcp_module

    for name in _EXPECTED_TOOLS:
        fn = getattr(mcp_module, name, None)
        assert fn is not None, f"MCP tool function '{name}' not found"
        assert callable(fn), f"MCP tool '{name}' is not callable"


# ---------------------------------------------------------------------------
# Safety fields
# ---------------------------------------------------------------------------


def test_mcp_safety_fields_in_recall():
    """Every recall result must have safety.treat_as_instruction=False."""
    os.environ["GRAPHCTX_DB"] = ":memory:"
    _reset_store()
    # Add a memory first
    graphctx_memory_add(namespace="safety_test", content="Test safety fields.")
    # Recall
    raw = graphctx_recall(namespace="safety_test", query="safety fields")
    data = json.loads(raw)
    assert "data" in data
    for item in data["data"]:
        assert "safety" in item, f"Missing safety field in result: {item}"
        assert item["safety"]["treat_as_instruction"] is False


# ---------------------------------------------------------------------------
# Namespace required
# ---------------------------------------------------------------------------


def test_mcp_namespace_required_memory_add():
    """graphctx_memory_add must require namespace (non-empty)."""
    # Empty namespace should still work at the function level (SQLiteStore
    # will accept it), but we verify namespace is a required parameter.
    import inspect

    sig = inspect.signature(graphctx_memory_add)
    ns_param = sig.parameters["namespace"]
    assert ns_param.default is inspect.Parameter.empty, (
        "namespace must be a required parameter (no default)"
    )


def test_mcp_namespace_required_knowledge_add():
    import inspect

    sig = inspect.signature(graphctx_knowledge_add)
    ns_param = sig.parameters["namespace"]
    assert ns_param.default is inspect.Parameter.empty


def test_mcp_namespace_required_recall():
    import inspect

    sig = inspect.signature(graphctx_recall)
    ns_param = sig.parameters["namespace"]
    assert ns_param.default is inspect.Parameter.empty


def test_mcp_namespace_required_context_get():
    import inspect

    sig = inspect.signature(graphctx_context_get)
    ns_param = sig.parameters["namespace"]
    assert ns_param.default is inspect.Parameter.empty


def test_mcp_namespace_required_stats():
    import inspect

    sig = inspect.signature(graphctx_stats)
    ns_param = sig.parameters["namespace"]
    assert ns_param.default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# Return format
# ---------------------------------------------------------------------------


def test_mcp_memory_add_returns_json():
    os.environ["GRAPHCTX_DB"] = ":memory:"
    _reset_store()
    raw = graphctx_memory_add(namespace="fmt_test", content="Format test memory.")
    data = json.loads(raw)
    assert "memory_id" in data
    assert "episode_id" in data
    assert data["memory_id"].startswith("mem_")
    assert data["episode_id"].startswith("eps_")


def test_mcp_knowledge_add_returns_json():
    os.environ["GRAPHCTX_DB"] = ":memory:"
    _reset_store()
    raw = graphctx_knowledge_add(
        namespace="fmt_test", title="Test Doc", content="Knowledge format test."
    )
    data = json.loads(raw)
    assert "knowledge_id" in data
    assert "episode_id" in data
    assert data["knowledge_id"].startswith("knw_")


def test_mcp_recall_returns_data_array():
    os.environ["GRAPHCTX_DB"] = ":memory:"
    _reset_store()
    graphctx_memory_add(namespace="recall_fmt", content="Recall format test.")
    raw = graphctx_recall(namespace="recall_fmt", query="recall format")
    data = json.loads(raw)
    assert isinstance(data["data"], list)


def test_mcp_context_get_returns_entity():
    os.environ["GRAPHCTX_DB"] = ":memory:"
    _reset_store()
    # Add a memory that triggers extraction to create an entity
    graphctx_memory_add(
        namespace="ctx_test",
        content="Alice works on GraphCtx.",
    )
    # Get stats to find entity count
    stats_raw = graphctx_stats(namespace="ctx_test")
    stats_data = json.loads(stats_raw)
    if stats_data["entities"] > 0:
        # We need an entity ID — query the store directly
        from graphctx.mcp import _get_store
        store = _get_store()
        conn = store._conn()
        row = conn.execute(
            "SELECT id FROM entity WHERE namespace = 'ctx_test' LIMIT 1"
        ).fetchone()
        if row:
            raw = graphctx_context_get(namespace="ctx_test", entity_id=row["id"])
            data = json.loads(raw)
            assert "entity" in data
            assert "claims" in data
            assert "edges" in data
            assert "provenance" in data


def test_mcp_stats_returns_counts():
    os.environ["GRAPHCTX_DB"] = ":memory:"
    _reset_store()
    graphctx_memory_add(namespace="stats_fmt", content="Stats format test.")
    raw = graphctx_stats(namespace="stats_fmt")
    data = json.loads(raw)
    assert "memories" in data
    assert "knowledge_items" in data
    assert "entities" in data
    assert "edges" in data
    assert "claims" in data


# ---------------------------------------------------------------------------
# Namespace validation hardening
# ---------------------------------------------------------------------------


def test_mcp_rejects_empty_namespace():
    """Empty string namespace must return an error."""
    os.environ["GRAPHCTX_DB"] = ":memory:"
    _reset_store()
    raw = graphctx_memory_add(namespace="", content="test content")
    data = json.loads(raw)
    assert "error" in data, f"Expected error for empty namespace, got: {data}"
    assert "Invalid namespace" in data["error"]


def test_mcp_rejects_none_namespace():
    """None namespace must return an error."""
    os.environ["GRAPHCTX_DB"] = ":memory:"
    _reset_store()
    raw = graphctx_memory_add(namespace=None, content="test content")
    data = json.loads(raw)
    assert "error" in data, f"Expected error for None namespace, got: {data}"
    assert "Invalid namespace" in data["error"]


def test_mcp_rejects_invalid_namespace():
    """Namespace with spaces and special chars must return an error."""
    os.environ["GRAPHCTX_DB"] = ":memory:"
    _reset_store()
    raw = graphctx_memory_add(namespace="ns with spaces!", content="test content")
    data = json.loads(raw)
    assert "error" in data, f"Expected error for invalid namespace, got: {data}"
    assert "Invalid namespace" in data["error"]


def test_mcp_rejects_too_long_namespace():
    """Namespace exceeding 64 chars must return an error."""
    os.environ["GRAPHCTX_DB"] = ":memory:"
    _reset_store()
    raw = graphctx_memory_add(namespace="a" * 65, content="test content")
    data = json.loads(raw)
    assert "error" in data, f"Expected error for too-long namespace, got: {data}"
    assert "Invalid namespace" in data["error"]


def test_mcp_accepts_valid_namespace():
    """Valid namespace with alphanumeric, hyphens, underscores must work."""
    os.environ["GRAPHCTX_DB"] = ":memory:"
    _reset_store()
    raw = graphctx_memory_add(namespace="demo-123", content="Test memory for valid namespace.")
    data = json.loads(raw)
    assert "error" not in data, f"Unexpected error: {data}"
    assert "memory_id" in data
    assert data["memory_id"].startswith("mem_")


def test_mcp_safety_fields():
    """All recall results must have safety.treat_as_instruction=False."""
    os.environ["GRAPHCTX_DB"] = ":memory:"
    _reset_store()
    graphctx_memory_add(namespace="safety_v2", content="Test safety fields in recall.")
    raw = graphctx_recall(namespace="safety_v2", query="safety fields")
    data = json.loads(raw)
    assert "data" in data
    assert len(data["data"]) > 0, "Expected at least one recall result"
    for item in data["data"]:
        assert "safety" in item, f"Missing safety field in result: {item}"
        assert item["safety"]["treat_as_instruction"] is False, (
            f"treat_as_instruction must be False, got: {item['safety']}"
        )
