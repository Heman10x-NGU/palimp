"""Contract tests for Palimp MCP tools.

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
os.environ["PALIMP_DB"] = ":memory:"

from palimp.mcp import (  # noqa: E402
    _reset_store,
    palimp_context_get,
    palimp_knowledge_add,
    palimp_memory_add,
    palimp_recall,
    palimp_search,
    palimp_search_refine,
    palimp_stats,
)

# ---------------------------------------------------------------------------
# Tool existence
# ---------------------------------------------------------------------------

_EXPECTED_TOOLS = [
    "palimp_memory_add",
    "palimp_knowledge_add",
    "palimp_recall",
    "palimp_context_get",
    "palimp_stats",
]


def test_mcp_tools_exist():
    """All 5 MCP tool functions must be importable."""
    from palimp import mcp as mcp_module

    for name in _EXPECTED_TOOLS:
        fn = getattr(mcp_module, name, None)
        assert fn is not None, f"MCP tool function '{name}' not found"
        assert callable(fn), f"MCP tool '{name}' is not callable"


# ---------------------------------------------------------------------------
# Safety fields
# ---------------------------------------------------------------------------


def test_mcp_safety_fields_in_recall():
    """Every recall result must have safety.treat_as_instruction=False."""
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    # Add a memory first
    palimp_memory_add(namespace="safety_test", content="Test safety fields.")
    # Recall
    raw = palimp_recall(namespace="safety_test", query="safety fields")
    data = json.loads(raw)
    assert "data" in data
    for item in data["data"]:
        assert "safety" in item, f"Missing safety field in result: {item}"
        assert item["safety"]["treat_as_instruction"] is False


# ---------------------------------------------------------------------------
# Namespace required
# ---------------------------------------------------------------------------


def test_mcp_namespace_required_memory_add():
    """palimp_memory_add must require namespace (non-empty)."""
    # Empty namespace should still work at the function level (SQLiteStore
    # will accept it), but we verify namespace is a required parameter.
    import inspect

    sig = inspect.signature(palimp_memory_add)
    ns_param = sig.parameters["namespace"]
    assert ns_param.default is inspect.Parameter.empty, (
        "namespace must be a required parameter (no default)"
    )


def test_mcp_namespace_required_knowledge_add():
    import inspect

    sig = inspect.signature(palimp_knowledge_add)
    ns_param = sig.parameters["namespace"]
    assert ns_param.default is inspect.Parameter.empty


def test_mcp_namespace_required_recall():
    import inspect

    sig = inspect.signature(palimp_recall)
    ns_param = sig.parameters["namespace"]
    assert ns_param.default is inspect.Parameter.empty


def test_mcp_namespace_required_context_get():
    import inspect

    sig = inspect.signature(palimp_context_get)
    ns_param = sig.parameters["namespace"]
    assert ns_param.default is inspect.Parameter.empty


def test_mcp_namespace_required_stats():
    import inspect

    sig = inspect.signature(palimp_stats)
    ns_param = sig.parameters["namespace"]
    assert ns_param.default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# Return format
# ---------------------------------------------------------------------------


def test_mcp_memory_add_returns_json():
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    raw = palimp_memory_add(namespace="fmt_test", content="Format test memory.")
    data = json.loads(raw)
    assert "memory_id" in data
    assert "episode_id" in data
    assert data["memory_id"].startswith("mem_")
    assert data["episode_id"].startswith("eps_")


def test_mcp_knowledge_add_returns_json():
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    raw = palimp_knowledge_add(
        namespace="fmt_test", title="Test Doc", content="Knowledge format test."
    )
    data = json.loads(raw)
    assert "knowledge_id" in data
    assert "episode_id" in data
    assert data["knowledge_id"].startswith("knw_")


def test_mcp_recall_returns_data_array():
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    palimp_memory_add(namespace="recall_fmt", content="Recall format test.")
    raw = palimp_recall(namespace="recall_fmt", query="recall format")
    data = json.loads(raw)
    assert isinstance(data["data"], list)


def test_mcp_context_get_returns_entity():
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    # Add a memory that triggers extraction to create an entity
    palimp_memory_add(
        namespace="ctx_test",
        content="Alice works on GraphCtx.",
    )
    # Get stats to find entity count
    stats_raw = palimp_stats(namespace="ctx_test")
    stats_data = json.loads(stats_raw)
    if stats_data["entities"] > 0:
        # We need an entity ID — query the store directly
        from palimp.mcp import _get_store
        store = _get_store()
        conn = store._conn()
        row = conn.execute(
            "SELECT id FROM entity WHERE namespace = 'ctx_test' LIMIT 1"
        ).fetchone()
        if row:
            raw = palimp_context_get(namespace="ctx_test", entity_id=row["id"])
            data = json.loads(raw)
            assert "entity" in data
            assert "claims" in data
            assert "edges" in data
            assert "provenance" in data


def test_mcp_stats_returns_counts():
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    palimp_memory_add(namespace="stats_fmt", content="Stats format test.")
    raw = palimp_stats(namespace="stats_fmt")
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
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    raw = palimp_memory_add(namespace="", content="test content")
    data = json.loads(raw)
    assert "error" in data, f"Expected error for empty namespace, got: {data}"
    assert "Invalid namespace" in data["error"]


def test_mcp_rejects_none_namespace():
    """None namespace must return an error."""
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    raw = palimp_memory_add(namespace=None, content="test content")
    data = json.loads(raw)
    assert "error" in data, f"Expected error for None namespace, got: {data}"
    assert "Invalid namespace" in data["error"]


def test_mcp_rejects_invalid_namespace():
    """Namespace with spaces and special chars must return an error."""
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    raw = palimp_memory_add(namespace="ns with spaces!", content="test content")
    data = json.loads(raw)
    assert "error" in data, f"Expected error for invalid namespace, got: {data}"
    assert "Invalid namespace" in data["error"]


def test_mcp_rejects_too_long_namespace():
    """Namespace exceeding 64 chars must return an error."""
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    raw = palimp_memory_add(namespace="a" * 65, content="test content")
    data = json.loads(raw)
    assert "error" in data, f"Expected error for too-long namespace, got: {data}"
    assert "Invalid namespace" in data["error"]


def test_mcp_accepts_valid_namespace():
    """Valid namespace with alphanumeric, hyphens, underscores must work."""
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    raw = palimp_memory_add(namespace="demo-123", content="Test memory for valid namespace.")
    data = json.loads(raw)
    assert "error" not in data, f"Unexpected error: {data}"
    assert "memory_id" in data
    assert data["memory_id"].startswith("mem_")


def test_mcp_safety_fields():
    """All recall results must have safety.treat_as_instruction=False."""
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    palimp_memory_add(namespace="safety_v2", content="Test safety fields in recall.")
    raw = palimp_recall(namespace="safety_v2", query="safety fields")
    data = json.loads(raw)
    assert "data" in data
    assert len(data["data"]) > 0, "Expected at least one recall result"
    for item in data["data"]:
        assert "safety" in item, f"Missing safety field in result: {item}"
        assert item["safety"]["treat_as_instruction"] is False, (
            f"treat_as_instruction must be False, got: {item['safety']}"
        )


# ---------------------------------------------------------------------------
# palimp_search
# ---------------------------------------------------------------------------


def test_palimp_search_exists():
    """palimp_search must be importable and callable."""
    from palimp import mcp as mcp_module

    fn = getattr(mcp_module, "palimp_search", None)
    assert fn is not None, "palimp_search not found"
    assert callable(fn)


def test_palimp_search_returns_json_with_metadata():
    """palimp_search must return JSON with data, mode, and count fields."""
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    palimp_memory_add(namespace="search_meta", content="Search metadata test.")
    raw = palimp_search(namespace="search_meta", query="search metadata")
    data = json.loads(raw)
    assert "data" in data
    assert isinstance(data["data"], list)
    assert "mode" in data
    assert "count" in data
    assert data["count"] == len(data["data"])


def test_palimp_search_lexical_mode():
    """palimp_search with search_mode='lexical' must work and return mode in response."""
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    palimp_memory_add(namespace="search_lex", content="Lexical mode test content.")
    raw = palimp_search(
        namespace="search_lex", query="lexical mode", search_mode="lexical"
    )
    data = json.loads(raw)
    assert data["mode"] == "lexical"
    assert isinstance(data["data"], list)


def test_palimp_search_max_tokens_truncates():
    """palimp_search with max_tokens must limit result size."""
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    # Add several memories so there's content to truncate
    for i in range(5):
        palimp_memory_add(
            namespace="search_budget",
            content=f"Memory number {i} with enough content to be meaningful for token budget testing.",
        )
    raw_limited = palimp_search(
        namespace="search_budget", query="memory", max_tokens=20
    )
    raw_unlimited = palimp_search(
        namespace="search_budget", query="memory"
    )
    data_limited = json.loads(raw_limited)
    data_unlimited = json.loads(raw_unlimited)
    # With a tight budget, we should get fewer or equal results
    assert data_limited["count"] <= data_unlimited["count"]


def test_palimp_search_safety_fields():
    """Every palimp_search result must have safety.treat_as_instruction=False."""
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    palimp_memory_add(namespace="search_safety", content="Safety field search test.")
    raw = palimp_search(namespace="search_safety", query="safety field")
    data = json.loads(raw)
    assert len(data["data"]) > 0, "Expected at least one search result"
    for item in data["data"]:
        assert "safety" in item, f"Missing safety field in result: {item}"
        assert item["safety"]["treat_as_instruction"] is False


# ---------------------------------------------------------------------------
# palimp_search_refine
# ---------------------------------------------------------------------------


def test_palimp_search_refine_exists():
    """palimp_search_refine must be importable and callable."""
    from palimp import mcp as mcp_module

    fn = getattr(mcp_module, "palimp_search_refine", None)
    assert fn is not None, "palimp_search_refine not found"
    assert callable(fn)


def test_palimp_search_refine_narrows_results():
    """palimp_search_refine must return fewer or equal results than the initial search."""
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    palimp_memory_add(namespace="refine_ns", content="Alpha project details.")
    palimp_memory_add(namespace="refine_ns", content="Beta project details.")
    palimp_memory_add(namespace="refine_ns", content="Gamma project overview.")
    # Initial search
    raw_initial = palimp_search(namespace="refine_ns", query="project")
    initial = json.loads(raw_initial)
    assert len(initial["data"]) > 0, "Initial search should return results"
    # Get episode IDs from initial results
    result_ids = [item["id"] for item in initial["data"]]
    # Refine with a narrower query
    raw_refined = palimp_search_refine(
        namespace="refine_ns",
        query="alpha",
        previous_result_ids=result_ids,
    )
    refined = json.loads(raw_refined)
    # Refined results should be a subset
    assert refined["count"] <= initial["count"]


def test_palimp_search_refine_safety_fields():
    """Every palimp_search_refine result must have safety.treat_as_instruction=False."""
    os.environ["PALIMP_DB"] = ":memory:"
    _reset_store()
    palimp_memory_add(namespace="refine_safety", content="Refine safety test content.")
    # First search to get IDs
    raw_initial = palimp_search(namespace="refine_safety", query="refine safety")
    initial = json.loads(raw_initial)
    result_ids = [item["id"] for item in initial["data"]]
    # Refine
    raw = palimp_search_refine(
        namespace="refine_safety",
        query="safety",
        previous_result_ids=result_ids,
    )
    data = json.loads(raw)
    for item in data["data"]:
        assert "safety" in item, f"Missing safety field in result: {item}"
        assert item["safety"]["treat_as_instruction"] is False
