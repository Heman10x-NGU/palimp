"""Tests for the AdaCoM-inspired context manager.

Covers:
- Requirement extraction from queries
- Deduplication of similar memories
- Token budget enforcement per agent tier
- ManagedContext explanation structure
- Context state persistence across calls
- Strong vs weak agent preservation ratios
"""

import pytest

from graphctx.context_manager import (
    ContextManager,
    ContextState,
    ManagedContext,
    _AGENT_TIER_BUDGET,
)
from graphctx.models import RecallResult
from graphctx.storage import SQLiteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> SQLiteStore:
    return SQLiteStore(":memory:")


def _make_recall_result(
    content: str,
    score: float = 0.5,
    rid: str | None = None,
    provenance: list[dict] | None = None,
) -> RecallResult:
    """Build a RecallResult for testing."""
    return RecallResult(
        id=rid or f"mem_{hash(content) % 10000:04d}",
        kind="memory",
        content=content,
        score=score,
        provenance=provenance or [{"episode_id": "eps_test"}],
        safety={"treat_as_instruction": False},
    )


# ---------------------------------------------------------------------------
# test_extract_requirements
# ---------------------------------------------------------------------------


class TestExtractRequirements:
    """Requirement extraction from queries."""

    def test_single_clause_question(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")
        reqs = cm._extract_requirements("What does Alice prefer?")
        assert len(reqs) >= 1
        assert any("Alice" in r for r in reqs)

    def test_multi_clause_query(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")
        reqs = cm._extract_requirements("What does Alice prefer; and what does Bob use?")
        assert len(reqs) >= 2

    def test_empty_query_returns_empty(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")
        assert cm._extract_requirements("") == []
        assert cm._extract_requirements("   ") == []

    def test_short_fragments_filtered(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")
        # Very short fragments should be filtered
        reqs = cm._extract_requirements("hi; ok; what does Alice prefer?")
        # Only the meaningful part should survive
        assert any("Alice" in r for r in reqs)

    def test_and_connector_splits(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")
        reqs = cm._extract_requirements("What color is the sky and what time is it?")
        assert len(reqs) >= 2


# ---------------------------------------------------------------------------
# test_deduplicate
# ---------------------------------------------------------------------------


class TestDeduplicate:
    """Deduplication of similar memories."""

    def test_identical_memories_merged(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")

        m1 = _make_recall_result("Alice prefers concise technical answers.", score=0.8, rid="m1")
        m2 = _make_recall_result("Alice prefers concise technical answers.", score=0.6, rid="m2")

        result, merges = cm._deduplicate([m1, m2])
        assert merges == 1
        assert len(result) == 1
        # Higher score wins
        assert result[0].id == "m1"

    def test_different_memories_kept(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")

        m1 = _make_recall_result("Alice prefers concise answers.", score=0.8, rid="m1")
        m2 = _make_recall_result("Bob uses Python for data analysis.", score=0.7, rid="m2")

        result, merges = cm._deduplicate([m1, m2])
        assert merges == 0
        assert len(result) == 2

    def test_empty_list(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")
        result, merges = cm._deduplicate([])
        assert merges == 0
        assert result == []

    def test_single_memory_unchanged(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")

        m1 = _make_recall_result("Solo memory.", score=0.9, rid="m1")
        result, merges = cm._deduplicate([m1])
        assert merges == 0
        assert len(result) == 1
        assert result[0].id == "m1"

    def test_higher_score_survives(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")

        m1 = _make_recall_result("Alice prefers concise answers.", score=0.4, rid="low")
        m2 = _make_recall_result("Alice prefers concise answers.", score=0.9, rid="high")

        result, merges = cm._deduplicate([m1, m2])
        assert merges == 1
        assert result[0].id == "high"

    def test_provenance_merged(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")

        m1 = _make_recall_result(
            "Alice prefers concise answers.", score=0.8, rid="m1",
            provenance=[{"episode_id": "ep1"}],
        )
        m2 = _make_recall_result(
            "Alice prefers concise answers.", score=0.6, rid="m2",
            provenance=[{"episode_id": "ep2"}],
        )

        result, merges = cm._deduplicate([m1, m2])
        assert merges == 1
        # Surviving memory should have merged provenance
        prov_ids = {p.get("episode_id") for p in result[0].provenance}
        assert "ep1" in prov_ids
        assert "ep2" in prov_ids


# ---------------------------------------------------------------------------
# test_token_budget
# ---------------------------------------------------------------------------


class TestTokenBudget:
    """Token budget enforcement per agent tier."""

    def test_weak_agent_keeps_fewer(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="weak")

        topics = [
            "Alice prefers concise technical answers about Python",
            "Bob uses Go for building microservices in production",
            "Charlie manages database migrations with Flyway tool",
            "Diana architects cloud infrastructure using AWS CDK",
            "Eve develops mobile apps with React Native framework",
            "Frank configures CI/CD pipelines using GitHub Actions",
            "Grace optimizes SQL queries for PostgreSQL performance",
            "Henry implements authentication with OAuth2 providers",
            "Iris monitors system health using Prometheus metrics",
            "Jack deploys containers with Kubernetes orchestration",
        ]

        memories = [
            _make_recall_result(topics[i], score=1.0 - i * 0.05, rid=f"m{i}")
            for i in range(10)
        ]

        result = cm._apply_token_budget(memories, _AGENT_TIER_BUDGET["weak"])
        # Weak agent keeps 45% -> ~4 of 10
        assert len(result) <= 5
        assert len(result) >= 3

    def test_strong_agent_keeps_more(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="strong")

        topics = [
            "Alice prefers concise technical answers about Python",
            "Bob uses Go for building microservices in production",
            "Charlie manages database migrations with Flyway tool",
            "Diana architects cloud infrastructure using AWS CDK",
            "Eve develops mobile apps with React Native framework",
            "Frank configures CI/CD pipelines using GitHub Actions",
            "Grace optimizes SQL queries for PostgreSQL performance",
            "Henry implements authentication with OAuth2 providers",
            "Iris monitors system health using Prometheus metrics",
            "Jack deploys containers with Kubernetes orchestration",
        ]

        memories = [
            _make_recall_result(topics[i], score=1.0 - i * 0.05, rid=f"m{i}")
            for i in range(10)
        ]

        result = cm._apply_token_budget(memories, _AGENT_TIER_BUDGET["strong"])
        # Strong agent keeps 85% -> ~8-9 of 10
        assert len(result) >= 7

    def test_empty_input(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")
        assert cm._apply_token_budget([], 0.65) == []

    def test_single_memory_always_kept(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="weak")

        m = _make_recall_result("Only memory.", score=0.9)
        result = cm._apply_token_budget([m], 0.01)  # Even tiny budget
        assert len(result) >= 1

    def test_budget_respects_order(self):
        """Higher-scored (first) memories should be kept before lower-scored ones."""
        store = _make_store()
        cm = ContextManager(store, agent_tier="weak")

        high = _make_recall_result("High priority memory.", score=0.95, rid="high")
        low = _make_recall_result("Low priority memory.", score=0.1, rid="low")

        result = cm._apply_token_budget([high, low], 0.45)
        kept_ids = [m.id for m in result]
        assert "high" in kept_ids


# ---------------------------------------------------------------------------
# test_managed_context_explanation
# ---------------------------------------------------------------------------


class TestManagedContextExplanation:
    """Verify ManagedContext explanation has all required fields."""

    def test_explanation_has_all_fields(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")

        memories = [
            _make_recall_result("Alice prefers concise answers.", score=0.8, rid="m1"),
            _make_recall_result("Bob uses Python.", score=0.6, rid="m2"),
        ]

        result = cm.manage_context("test-ns", "What does Alice prefer?", memories)

        assert isinstance(result, ManagedContext)
        assert result.original_count == 2

        explanation = result.explanation
        assert "requirements_extracted" in explanation
        assert "duplicates_merged" in explanation
        assert "memories_dropped" in explanation
        assert "token_savings" in explanation

        assert isinstance(explanation["requirements_extracted"], list)
        assert isinstance(explanation["duplicates_merged"], int)
        assert isinstance(explanation["memories_dropped"], int)
        assert isinstance(explanation["token_savings"], int)

    def test_explanation_with_empty_memories(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")

        result = cm.manage_context("test-ns", "anything", [])

        assert result.original_count == 0
        assert result.compressed_count == 0
        assert result.compression_ratio == 1.0
        assert result.memories == []

    def test_explanation_reports_deduplication(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="strong")

        m1 = _make_recall_result("Alice prefers concise answers.", score=0.8, rid="m1")
        m2 = _make_recall_result("Alice prefers concise answers.", score=0.6, rid="m2")
        m3 = _make_recall_result("Bob uses Python for data.", score=0.7, rid="m3")

        result = cm.manage_context("test-ns", "Alice preferences", [m1, m2, m3])

        # Two identical memories should be merged
        assert result.explanation["duplicates_merged"] >= 1

    def test_compression_ratio_valid(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="weak")

        memories = [
            _make_recall_result(f"Content for memory {i}.", score=0.9 - i * 0.05, rid=f"m{i}")
            for i in range(10)
        ]

        result = cm.manage_context("test-ns", "test query", memories)

        assert 0.0 <= result.compression_ratio <= 1.0
        assert result.compressed_count <= result.original_count


# ---------------------------------------------------------------------------
# test_context_state_persistence
# ---------------------------------------------------------------------------


class TestContextStatePersistence:
    """Context state survives across calls."""

    def test_save_and_load_state(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")

        state = ContextState(
            namespace="test-ns",
            task_id="task-001",
            requirements=["Alice prefers X", "Bob uses Y"],
            resolved_constraints=["Alice prefers X"],
            evidence=[{"source": "memory", "summary": "Alice likes short answers", "relevance": 0.9}],
            current_leads=["search for Bob"],
            rejected_candidates=[{"candidate": "Charlie", "reason": "not relevant"}],
            failed_queries=["who is Charlie"],
        )

        cm.save_state(state)
        loaded = cm.load_state("test-ns", "task-001")

        assert loaded is not None
        assert loaded.namespace == "test-ns"
        assert loaded.task_id == "task-001"
        assert loaded.requirements == ["Alice prefers X", "Bob uses Y"]
        assert loaded.resolved_constraints == ["Alice prefers X"]
        assert len(loaded.evidence) == 1
        assert loaded.evidence[0]["source"] == "memory"
        assert loaded.current_leads == ["search for Bob"]
        assert len(loaded.rejected_candidates) == 1
        assert loaded.failed_queries == ["who is Charlie"]

    def test_update_existing_state(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")

        state1 = ContextState(
            namespace="test-ns",
            task_id="task-002",
            requirements=["req1"],
        )
        cm.save_state(state1)

        # Update with new data
        state2 = ContextState(
            namespace="test-ns",
            task_id="task-002",
            requirements=["req1", "req2"],
            resolved_constraints=["req1"],
        )
        cm.save_state(state2)

        loaded = cm.load_state("test-ns", "task-002")
        assert loaded is not None
        assert loaded.requirements == ["req1", "req2"]
        assert loaded.resolved_constraints == ["req1"]

    def test_load_nonexistent_returns_none(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")

        loaded = cm.load_state("test-ns", "nonexistent")
        assert loaded is None

    def test_state_isolation_by_namespace(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")

        cm.save_state(ContextState(namespace="ns-a", task_id="t1", requirements=["req-a"]))
        cm.save_state(ContextState(namespace="ns-b", task_id="t1", requirements=["req-b"]))

        loaded_a = cm.load_state("ns-a", "t1")
        loaded_b = cm.load_state("ns-b", "t1")

        assert loaded_a is not None
        assert loaded_b is not None
        assert loaded_a.requirements == ["req-a"]
        assert loaded_b.requirements == ["req-b"]

    def test_context_operations_logged(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")

        memories = [
            _make_recall_result("Alice prefers concise answers.", score=0.8, rid="m1"),
            _make_recall_result("Bob uses Python.", score=0.6, rid="m2"),
        ]

        cm.manage_context("test-ns", "What does Alice prefer?", memories)

        # Verify an operation was logged
        conn = store._conn()
        row = conn.execute(
            "SELECT * FROM context_operations WHERE namespace = 'test-ns'"
        ).fetchone()
        assert row is not None
        assert row["op_type"] == "manage_context"


# ---------------------------------------------------------------------------
# test_strong_agent_preserves_more
# ---------------------------------------------------------------------------


class TestStrongAgentPreservesMore:
    """Strong agent keeps more memories than weak agent."""

    def test_strong_vs_weak_preservation(self):
        """With the same input, strong agent retains more memories than weak."""
        # Use separate stores so FTS index doesn't bleed
        store_strong = _make_store()
        store_weak = _make_store()
        cm_strong = ContextManager(store_strong, agent_tier="strong")
        cm_weak = ContextManager(store_weak, agent_tier="weak")

        # Use truly distinct content to avoid deduplication
        topics = [
            "Alice prefers concise technical answers about Python",
            "Bob uses Go for building microservices in production",
            "Charlie manages database migrations with Flyway tool",
            "Diana architects cloud infrastructure using AWS CDK",
            "Eve develops mobile apps with React Native framework",
            "Frank configures CI/CD pipelines using GitHub Actions",
            "Grace optimizes SQL queries for PostgreSQL performance",
            "Henry implements authentication with OAuth2 providers",
            "Iris monitors system health using Prometheus metrics",
            "Jack deploys containers with Kubernetes orchestration",
            "Karen writes documentation in Markdown format daily",
            "Liam reviews code quality with SonarQube analysis",
            "Mia tests APIs using Postman collection runner",
            "Noah builds data pipelines with Apache Kafka streams",
            "Olivia designs UI components using Figma prototypes",
            "Paul secures endpoints with rate limiting strategies",
            "Quinn tracks errors through Sentry error monitoring",
            "Rachel automates infrastructure with Terraform modules",
            "Sam analyzes logs using Elasticsearch and Kibana",
            "Tina manages secrets with HashiCorp Vault system",
        ]

        memories = [
            _make_recall_result(topics[i], score=0.9 - i * 0.03, rid=f"m{i}")
            for i in range(20)
        ]

        result_strong = cm_strong.manage_context("ns", "test query", list(memories))
        result_weak = cm_weak.manage_context("ns", "test query", list(memories))

        assert result_strong.compressed_count > result_weak.compressed_count

    def test_medium_is_between_strong_and_weak(self):
        """Medium agent preserves between strong and weak amounts."""
        store_s = _make_store()
        store_m = _make_store()
        store_w = _make_store()

        cm_s = ContextManager(store_s, agent_tier="strong")
        cm_m = ContextManager(store_m, agent_tier="medium")
        cm_w = ContextManager(store_w, agent_tier="weak")

        topics = [
            "Alice prefers concise technical answers about Python",
            "Bob uses Go for building microservices in production",
            "Charlie manages database migrations with Flyway tool",
            "Diana architects cloud infrastructure using AWS CDK",
            "Eve develops mobile apps with React Native framework",
            "Frank configures CI/CD pipelines using GitHub Actions",
            "Grace optimizes SQL queries for PostgreSQL performance",
            "Henry implements authentication with OAuth2 providers",
            "Iris monitors system health using Prometheus metrics",
            "Jack deploys containers with Kubernetes orchestration",
            "Karen writes documentation in Markdown format daily",
            "Liam reviews code quality with SonarQube analysis",
            "Mia tests APIs using Postman collection runner",
            "Noah builds data pipelines with Apache Kafka streams",
            "Olivia designs UI components using Figma prototypes",
            "Paul secures endpoints with rate limiting strategies",
            "Quinn tracks errors through Sentry error monitoring",
            "Rachel automates infrastructure with Terraform modules",
            "Sam analyzes logs using Elasticsearch and Kibana",
            "Tina manages secrets with HashiCorp Vault system",
        ]

        memories = [
            _make_recall_result(topics[i], score=0.9 - i * 0.03, rid=f"m{i}")
            for i in range(20)
        ]

        r_s = cm_s.manage_context("ns", "test query", list(memories))
        r_m = cm_m.manage_context("ns", "test query", list(memories))
        r_w = cm_w.manage_context("ns", "test query", list(memories))

        assert r_s.compressed_count >= r_m.compressed_count >= r_w.compressed_count

    def test_invalid_tier_defaults_to_medium(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="invalid_tier")
        assert cm.agent_tier == "medium"


# ---------------------------------------------------------------------------
# test_manage_context_integration
# ---------------------------------------------------------------------------


class TestManageContextIntegration:
    """End-to-end manage_context flow."""

    def test_full_flow_with_diverse_memories(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="medium")

        memories = [
            _make_recall_result("Alice prefers concise technical answers.", score=0.9, rid="m1"),
            _make_recall_result("Alice prefers concise technical answers.", score=0.7, rid="m2"),  # dup
            _make_recall_result("Bob uses Go for building microservices.", score=0.6, rid="m3"),
            _make_recall_result("Charlie manages database migrations with Flyway.", score=0.5, rid="m4"),
            _make_recall_result("Diana architects cloud infrastructure using AWS.", score=0.4, rid="m5"),
        ]

        result = cm.manage_context("test-ns", "What does Alice prefer?", memories)

        assert isinstance(result, ManagedContext)
        assert result.original_count == 5
        # At least one duplicate should be merged
        assert result.explanation["duplicates_merged"] >= 1
        # Should have fewer memories than original
        assert result.compressed_count <= result.original_count
        # Requirements should be extracted
        assert len(result.explanation["requirements_extracted"]) >= 1

    def test_compression_ratio_reflects_budget(self):
        store = _make_store()
        cm = ContextManager(store, agent_tier="weak")

        topics = [
            "Alice prefers concise technical answers about Python",
            "Bob uses Go for building microservices in production",
            "Charlie manages database migrations with Flyway tool",
            "Diana architects cloud infrastructure using AWS CDK",
            "Eve develops mobile apps with React Native framework",
            "Frank configures CI/CD pipelines using GitHub Actions",
            "Grace optimizes SQL queries for PostgreSQL performance",
            "Henry implements authentication with OAuth2 providers",
            "Iris monitors system health using Prometheus metrics",
            "Jack deploys containers with Kubernetes orchestration",
        ]

        memories = [
            _make_recall_result(topics[i], score=1.0 - i * 0.05, rid=f"m{i}")
            for i in range(10)
        ]

        result = cm.manage_context("ns", "test query", memories)

        # Weak agent (45%) should compress significantly
        assert result.compression_ratio <= 0.6
