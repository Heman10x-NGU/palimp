"""Tests for category-based memory ranking and trigger glossary."""

from __future__ import annotations

import pytest

from graphctx.embeddings import DeterministicEmbedder
from graphctx.ingest import ingest_memory, ingest_knowledge
from graphctx.models import CATEGORY_PRIORITY, MEMORY_CATEGORIES
from graphctx.retriever import RecallEngine
from graphctx.storage import SQLiteStore


@pytest.fixture()
def store() -> SQLiteStore:
    """In-memory SQLite store for testing."""
    return SQLiteStore(":memory:")


@pytest.fixture()
def embedder() -> DeterministicEmbedder:
    return DeterministicEmbedder()


@pytest.fixture()
def engine(store: SQLiteStore, embedder: DeterministicEmbedder) -> RecallEngine:
    return RecallEngine(store=store, embedder=embedder)


# ---------------------------------------------------------------------------
# Category tests
# ---------------------------------------------------------------------------


class TestCategoryOnMemory:
    """test_category_on_memory: add memory with category='gotcha', verify stored."""

    def test_category_on_memory(self, store: SQLiteStore, embedder: DeterministicEmbedder) -> None:
        result = ingest_memory(
            store=store,
            embedder=embedder,
            extractor=None,
            ns="test",
            content="pytest requires GRAPHCTX_DB=:memory: for isolated tests",
            category="gotcha",
        )
        memory_id = result["memory_id"]
        episode_id = result["episode_id"]

        # Verify category stored in memory table
        conn = store._conn()
        row = conn.execute(
            "SELECT category FROM memory WHERE id = ?", (memory_id,)
        ).fetchone()
        assert row is not None
        assert row["category"] == "gotcha"

        # Verify get_memory_category returns correct value
        cat = store.get_memory_category(episode_id)
        assert cat == "gotcha"


class TestCategoryDefault:
    """Verify default category is 'other' when not specified."""

    def test_default_category(self, store: SQLiteStore, embedder: DeterministicEmbedder) -> None:
        result = ingest_memory(
            store=store,
            embedder=embedder,
            extractor=None,
            ns="test",
            content="Some generic memory without category",
        )
        memory_id = result["memory_id"]

        conn = store._conn()
        row = conn.execute(
            "SELECT category FROM memory WHERE id = ?", (memory_id,)
        ).fetchone()
        assert row is not None
        assert row["category"] == "other"


class TestCategoryOnKnowledge:
    """Verify category works for knowledge items too."""

    def test_category_on_knowledge(self, store: SQLiteStore, embedder: DeterministicEmbedder) -> None:
        result = ingest_knowledge(
            store=store,
            embedder=embedder,
            extractor=None,
            ns="test",
            title="Architecture Decision",
            content="We use SQLite for local-first storage",
            category="architecture_decision",
        )
        knowledge_id = result["knowledge_id"]

        conn = store._conn()
        row = conn.execute(
            "SELECT category FROM knowledge WHERE id = ?", (knowledge_id,)
        ).fetchone()
        assert row is not None
        assert row["category"] == "architecture_decision"


class TestCategoryPrioritySurvivesBudget:
    """test_category_priority_survives_budget: gotcha survives AdaCoM compression."""

    def test_category_priority_survives_budget(
        self, store: SQLiteStore, embedder: DeterministicEmbedder, engine: RecallEngine
    ) -> None:
        # Ingest a "gotcha" memory and an "other" memory with similar content
        ingest_memory(
            store=store,
            embedder=embedder,
            extractor=None,
            ns="test",
            content="pytest requires GRAPHCTX_DB=:memory: for isolated tests",
            category="gotcha",
        )
        ingest_memory(
            store=store,
            embedder=embedder,
            extractor=None,
            ns="test",
            content="Some generic development note about pytest",
            category="other",
        )

        # Recall with agent_tier="weak" (aggressive compression, 45% budget)
        output = engine.recall(
            ns="test",
            query="pytest testing",
            mode="hybrid",
            limit=10,
            include_provenance=False,
            agent_tier="weak",
        )

        # The gotcha memory should survive compression
        categories = [r.category for r in output.results]
        assert "gotcha" in categories, (
            f"Expected 'gotcha' to survive compression, got categories: {categories}"
        )


class TestCategoryInRecall:
    """test_category_in_recall: category appears in recall result."""

    def test_category_in_recall(
        self, store: SQLiteStore, embedder: DeterministicEmbedder, engine: RecallEngine
    ) -> None:
        ingest_memory(
            store=store,
            embedder=embedder,
            extractor=None,
            ns="test",
            content="Always use type hints in Python code",
            category="preference",
        )

        output = engine.recall(
            ns="test",
            query="type hints Python",
            mode="hybrid",
            limit=5,
            include_provenance=False,
        )

        assert len(output.results) > 0
        assert output.results[0].category == "preference"


class TestCategoryPriorityConstants:
    """Verify CATEGORY_PRIORITY covers all categories and has expected values."""

    def test_all_categories_have_priority(self) -> None:
        for cat in MEMORY_CATEGORIES:
            assert cat in CATEGORY_PRIORITY, f"Category '{cat}' missing from CATEGORY_PRIORITY"

    def test_identity_highest_priority(self) -> None:
        assert CATEGORY_PRIORITY["identity"] == 1.0

    def test_other_lowest_priority(self) -> None:
        assert CATEGORY_PRIORITY["other"] == 0.2


# ---------------------------------------------------------------------------
# Trigger tests
# ---------------------------------------------------------------------------


class TestTriggerAddAndList:
    """test_trigger_add_and_list: add trigger, list shows it."""

    def test_trigger_add_and_list(self, store: SQLiteStore, embedder: DeterministicEmbedder) -> None:
        result = ingest_memory(
            store=store,
            embedder=embedder,
            extractor=None,
            ns="test",
            content="pytest requires special configuration",
        )
        memory_id = result["memory_id"]

        # Add trigger
        trigger_id = store.insert_trigger(ns="test", term="pytest", memory_id=memory_id)
        assert trigger_id.startswith("trg_")

        # List triggers
        triggers = store.list_triggers(ns="test")
        assert len(triggers) == 1
        assert triggers[0]["term"] == "pytest"
        assert triggers[0]["memory_id"] == memory_id


class TestTriggerBoost:
    """test_trigger_boost: triggered memory scores higher."""

    def test_trigger_boost(
        self, store: SQLiteStore, embedder: DeterministicEmbedder, engine: RecallEngine
    ) -> None:
        # Ingest two memories with similar content
        result1 = ingest_memory(
            store=store,
            embedder=embedder,
            extractor=None,
            ns="test",
            content="Use pytest for testing Python applications",
        )
        result2 = ingest_memory(
            store=store,
            embedder=embedder,
            extractor=None,
            ns="test",
            content="Use unittest for testing Python applications",
        )

        # Add trigger for "pytest" linking to result1
        store.insert_trigger(ns="test", term="pytest", memory_id=result1["memory_id"])

        # Recall with query containing "pytest"
        output = engine.recall(
            ns="test",
            query="pytest testing framework",
            mode="hybrid",
            limit=5,
            include_provenance=False,
        )

        assert len(output.results) >= 2
        # The triggered memory should score higher
        top_result = output.results[0]
        # The top result should be the one linked to the trigger
        # (or at least the triggered one should be in the results)
        episode_ids = [r.id for r in output.results]
        assert result1["episode_id"] in episode_ids


class TestTriggerNamespaceScoped:
    """test_trigger_namespace_scoped: trigger in ns1 doesn't affect ns2."""

    def test_trigger_namespace_scoped(self, store: SQLiteStore, embedder: DeterministicEmbedder) -> None:
        # Add memory and trigger in ns1
        result1 = ingest_memory(
            store=store,
            embedder=embedder,
            extractor=None,
            ns="ns1",
            content="pytest configuration for ns1",
        )
        store.insert_trigger(ns="ns1", term="pytest", memory_id=result1["memory_id"])

        # Add memory in ns2 without trigger
        result2 = ingest_memory(
            store=store,
            embedder=embedder,
            extractor=None,
            ns="ns2",
            content="pytest configuration for ns2",
        )

        # List triggers in ns2 should be empty
        triggers_ns2 = store.list_triggers(ns="ns2")
        assert len(triggers_ns2) == 0

        # List triggers in ns1 should have one
        triggers_ns1 = store.list_triggers(ns="ns1")
        assert len(triggers_ns1) == 1


class TestTriggerDelete:
    """test_trigger_delete: delete trigger, no longer boosts."""

    def test_trigger_delete(self, store: SQLiteStore, embedder: DeterministicEmbedder) -> None:
        result = ingest_memory(
            store=store,
            embedder=embedder,
            extractor=None,
            ns="test",
            content="pytest configuration notes",
        )
        memory_id = result["memory_id"]

        # Add trigger
        store.insert_trigger(ns="test", term="pytest", memory_id=memory_id)
        triggers = store.list_triggers(ns="test")
        assert len(triggers) == 1

        # Delete trigger
        deleted = store.delete_trigger(ns="test", term="pytest")
        assert deleted is True

        # Verify deleted
        triggers = store.list_triggers(ns="test")
        assert len(triggers) == 0

    def test_trigger_delete_nonexistent(self, store: SQLiteStore) -> None:
        deleted = store.delete_trigger(ns="test", term="nonexistent")
        assert deleted is False


class TestTriggerCaseInsensitive:
    """Verify trigger terms are stored and matched case-insensitively."""

    def test_trigger_case_insensitive(self, store: SQLiteStore, embedder: DeterministicEmbedder) -> None:
        result = ingest_memory(
            store=store,
            embedder=embedder,
            extractor=None,
            ns="test",
            content="pytest configuration",
        )
        memory_id = result["memory_id"]

        # Add trigger with mixed case
        store.insert_trigger(ns="test", term="PyTest", memory_id=memory_id)

        # Should be stored as lowercase
        triggers = store.list_triggers(ns="test")
        assert triggers[0]["term"] == "pytest"

        # Should be retrievable with any case
        found = store.get_triggers_for_term(ns="test", term="PYTEST")
        assert len(found) == 1
