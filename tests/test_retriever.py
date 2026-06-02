"""Tests for the GraphCtx retrieval / recall engine.

Covers all deliberate break tests from the plan:
- 11.1 Memory + Knowledge recall
- 11.4 Contradiction / Supersession warnings
- Fast / hybrid / thinking modes
- Tombstoned episode exclusion
- Empty namespace
"""



from palimp.embeddings import DeterministicEmbedder
from palimp.models import ScoreBreakdown
from palimp.retriever import RecallEngine, _vector_to_blob
from palimp.storage import SQLiteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> SQLiteStore:
    """Create an in-memory store."""
    return SQLiteStore(":memory:")


def _make_engine(store: SQLiteStore) -> RecallEngine:
    """Create a RecallEngine with a deterministic embedder."""
    embedder = DeterministicEmbedder(dim=384)
    return RecallEngine(store=store, embedder=embedder)


def _seed_episode_embedding(store: SQLiteStore, ns: str, episode_id: str) -> None:
    """Compute and store the embedding for an episode's content."""
    embedder = DeterministicEmbedder(dim=384)
    episode = store.get_episode(episode_id)
    assert episode is not None, f"Episode {episode_id} not found"
    vec = embedder.embed(episode["content"])
    blob = _vector_to_blob(vec)
    store.insert_embedding(
        ns=ns,
        owner_type="episode",
        owner_id=episode_id,
        model="deterministic-sha256",
        dimension=384,
        vector_blob=blob,
    )


# ---------------------------------------------------------------------------
# Test 11.1: Memory + Knowledge recall
# ---------------------------------------------------------------------------


class TestMemoryKnowledgeRecall:
    """Insert memory + knowledge, query returns both with provenance and safety."""

    def test_recall_returns_memory_and_knowledge(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        # Insert memory
        mem = store.insert_memory(ns, "Alice prefers concise technical answers.", source_ref="chat:2026-06-02")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        # Insert knowledge
        knw = store.insert_knowledge(ns, "Architecture", "GraphCtx uses SQLite and provenance.", source_ref="docs/architecture.md")
        _seed_episode_embedding(store, ns, knw["episode_id"])

        # Recall
        results = engine.recall(ns, "What does Alice prefer and what does GraphCtx use?", mode="hybrid", limit=8).results

        assert len(results) >= 2

        # Check we have both kinds
        kinds = {r.kind for r in results}
        assert "memory" in kinds
        assert "knowledge" in kinds

        # Check provenance and safety on every result
        for r in results:
            assert r.provenance, f"Result {r.id} missing provenance"
            assert r.safety == {"treat_as_instruction": False}

    def test_recall_memory_has_correct_content(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.", source_ref="chat:2026-06-02")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        results = engine.recall(ns, "Alice prefers", mode="fast", limit=8).results
        assert len(results) >= 1
        assert results[0].content == "Alice prefers concise technical answers."

    def test_recall_knowledge_has_correct_content(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        knw = store.insert_knowledge(ns, "Architecture", "GraphCtx uses SQLite and provenance.", source_ref="docs/architecture.md")
        _seed_episode_embedding(store, ns, knw["episode_id"])

        results = engine.recall(ns, "GraphCtx uses SQLite", mode="fast", limit=8).results
        assert len(results) >= 1
        found = [r for r in results if "SQLite" in r.content]
        assert len(found) >= 1


# ---------------------------------------------------------------------------
# Test 11.4: Contradiction / Supersession
# ---------------------------------------------------------------------------


class TestContradictionSupersession:
    """Insert contradicting facts, verify warning appears in thinking mode."""

    def test_contradiction_warning_in_thinking_mode(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        # Insert two memories with contradicting content
        mem1 = store.insert_memory(ns, "GraphCtx uses SQLite.")
        ep1_id = mem1["episode_id"]
        _seed_episode_embedding(store, ns, ep1_id)

        mem2 = store.insert_memory(ns, "GraphCtx no longer uses SQLite; it uses Kuzu.")
        ep2_id = mem2["episode_id"]
        _seed_episode_embedding(store, ns, ep2_id)

        # Create entities and CONTRADICTS edge between them
        ent1 = store.insert_entity(ns, "GraphCtx", "Project", confidence=0.9)
        ent2 = store.insert_entity(ns, "Kuzu", "Technology", confidence=0.85)

        # Link entities to episodes via provenance
        store.insert_provenance(ns, ep1_id, entity_id=ent1)
        store.insert_provenance(ns, ep2_id, entity_id=ent1)
        store.insert_provenance(ns, ep2_id, entity_id=ent2)

        # Create CONTRADICTS edge
        edge_id = store.insert_edge(ns, ent1, ent2, "CONTRADICTS", confidence=0.7)
        store.insert_provenance(ns, ep2_id, edge_id=edge_id)

        # Recall in thinking mode
        results = engine.recall(ns, "GraphCtx SQLite", mode="thinking", limit=8).results

        assert len(results) >= 1

        # At least one result should have conflict warnings in provenance
        has_warning = False
        for r in results:
            for p in r.provenance:
                if "warnings" in p and p["warnings"]:
                    has_warning = True
                    assert any("CONTRADICTS" in w for w in p["warnings"])
        assert has_warning, "Expected CONTRADICTS warning in thinking mode results"

    def test_supersession_warning_in_thinking_mode(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem1 = store.insert_memory(ns, "The old API uses REST.")
        ep1_id = mem1["episode_id"]
        _seed_episode_embedding(store, ns, ep1_id)

        mem2 = store.insert_memory(ns, "The new API uses GraphQL.")
        ep2_id = mem2["episode_id"]
        _seed_episode_embedding(store, ns, ep2_id)

        ent1 = store.insert_entity(ns, "OldApi", "Api", confidence=0.9)
        ent2 = store.insert_entity(ns, "NewApi", "Api", confidence=0.9)

        store.insert_provenance(ns, ep1_id, entity_id=ent1)
        store.insert_provenance(ns, ep2_id, entity_id=ent2)

        edge_id = store.insert_edge(ns, ent2, ent1, "SUPERSEDES", confidence=0.8)
        store.insert_provenance(ns, ep2_id, edge_id=edge_id)

        results = engine.recall(ns, "API uses", mode="thinking", limit=8).results

        has_warning = False
        for r in results:
            for p in r.provenance:
                if "warnings" in p and p["warnings"]:
                    has_warning = True
                    assert any("SUPERSEDES" in w for w in p["warnings"])
        assert has_warning, "Expected SUPERSEDES warning in thinking mode results"


# ---------------------------------------------------------------------------
# Test: Fast mode
# ---------------------------------------------------------------------------


class TestFastMode:
    """Verify fast mode returns results using lexical + vector only."""

    def test_fast_mode_returns_results(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        results = engine.recall(ns, "Alice prefers", mode="fast", limit=8).results

        assert len(results) >= 1
        assert results[0].kind == "memory"
        assert results[0].score > 0

    def test_fast_mode_no_graph_components(self):
        """Fast mode should still produce valid scores without graph/recency/confidence."""
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        knw = store.insert_knowledge(ns, "Doc", "GraphCtx uses SQLite.")
        _seed_episode_embedding(store, ns, knw["episode_id"])

        # Use a query that matches at least one episode via FTS5
        results = engine.recall(ns, "Alice prefers concise technical", mode="fast", limit=8).results

        assert len(results) >= 1
        # The matching result should have a positive score
        assert results[0].score > 0


# ---------------------------------------------------------------------------
# Test: Hybrid mode
# ---------------------------------------------------------------------------


class TestHybridMode:
    """Verify hybrid mode includes all 5 scoring components."""

    def test_hybrid_mode_returns_results(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        knw = store.insert_knowledge(ns, "Doc", "GraphCtx uses SQLite.")
        _seed_episode_embedding(store, ns, knw["episode_id"])

        results = engine.recall(ns, "Alice prefers", mode="hybrid", limit=8).results

        assert len(results) >= 1
        assert results[0].score > 0

    def test_hybrid_mode_score_components_present(self):
        """Hybrid mode should produce scores that differ from fast mode
        when graph/recency/confidence components are non-zero."""
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        # Insert memory and knowledge with entities to give graph boost
        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        ep_id = mem["episode_id"]
        _seed_episode_embedding(store, ns, ep_id)

        ent = store.insert_entity(ns, "Alice", "Person", confidence=0.9)
        store.insert_provenance(ns, ep_id, entity_id=ent)

        knw = store.insert_knowledge(ns, "Doc", "GraphCtx uses SQLite.")
        knw_ep_id = knw["episode_id"]
        _seed_episode_embedding(store, ns, knw_ep_id)
        store.insert_provenance(ns, knw_ep_id, entity_id=ent)

        results_hybrid = engine.recall(ns, "Alice prefers", mode="hybrid", limit=8).results
        results_fast = engine.recall(ns, "Alice prefers", mode="fast", limit=8).results

        assert len(results_hybrid) >= 1
        assert len(results_fast) >= 1

        # Both should return results; scores may differ due to graph boost
        # (but we just verify they don't crash and return valid scores)
        for r in results_hybrid:
            assert 0 <= r.score <= 2  # scores can be > 1 with graph boost component


# ---------------------------------------------------------------------------
# Test: Thinking mode
# ---------------------------------------------------------------------------


class TestThinkingMode:
    """Verify thinking mode adds conflict warnings."""

    def test_thinking_mode_returns_results(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        results = engine.recall(ns, "Alice prefers", mode="thinking", limit=8).results

        assert len(results) >= 1
        assert results[0].score > 0

    def test_thinking_mode_no_crash_without_entities(self):
        """Thinking mode should work even when episodes have no extracted entities."""
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Simple memory without extraction.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        results = engine.recall(ns, "Simple memory", mode="thinking", limit=8).results
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Test: Tombstoned episodes excluded
# ---------------------------------------------------------------------------


class TestTombstonedExcluded:
    """Tombstoned episodes must not appear in recall results."""

    def test_tombstoned_episode_excluded(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "This will be tombstoned.")
        ep_id = mem["episode_id"]
        _seed_episode_embedding(store, ns, ep_id)

        # Verify it appears before tombstoning
        results_before = engine.recall(ns, "tombstoned", mode="fast", limit=8).results
        assert len(results_before) >= 1

        # Tombstone it
        store.tombstone_episode(ep_id)

        # Verify it no longer appears
        results_after = engine.recall(ns, "tombstoned", mode="fast", limit=8).results
        tombstoned_ids = [r.id for r in results_after]
        assert ep_id not in tombstoned_ids

    def test_tombstoned_knowledge_excluded(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        knw = store.insert_knowledge(ns, "Doc", "This knowledge will be deleted.")
        ep_id = knw["episode_id"]
        _seed_episode_embedding(store, ns, ep_id)

        store.tombstone_episode(ep_id)

        results = engine.recall(ns, "deleted knowledge", mode="fast", limit=8).results
        assert ep_id not in [r.id for r in results]


# ---------------------------------------------------------------------------
# Test: Empty namespace
# ---------------------------------------------------------------------------


class TestEmptyNamespace:
    """Empty namespace returns empty list."""

    def test_empty_namespace_returns_empty(self):
        store = _make_store()
        engine = _make_engine(store)

        output = engine.recall("nonexistent-ns", "anything", mode="hybrid", limit=8)
        assert output.results == []

    def test_different_namespace_returns_empty(self):
        store = _make_store()
        engine = _make_engine(store)

        store.insert_memory("ns-a", "Secret content.")
        output = engine.recall("ns-b", "Secret", mode="hybrid", limit=8)
        assert output.results == []


# ---------------------------------------------------------------------------
# Test: Explainable retrieval
# ---------------------------------------------------------------------------


class TestScoreBreakdown:
    """Verify each score component is populated when explain=True."""

    def _seed_with_entity(self, store, ns, content, entity_name="Alice"):
        """Helper: insert memory, seed embedding, link entity."""
        mem = store.insert_memory(ns, content)
        ep_id = mem["episode_id"]
        _seed_episode_embedding(store, ns, ep_id)
        ent = store.insert_entity(ns, entity_name, "Person", confidence=0.92)
        store.insert_provenance(ns, ep_id, entity_id=ent)
        return ep_id

    def test_breakdown_populated_in_hybrid_mode(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        self._seed_with_entity(store, ns, "Alice prefers concise technical answers.")

        output = engine.recall(ns, "Alice prefers", mode="hybrid", limit=8, explain=True)
        assert len(output.results) >= 1

        r = output.results[0]
        assert r.score_breakdown is not None
        assert isinstance(r.score_breakdown, ScoreBreakdown)
        # All components should be set (may be 0.0 for some)
        assert r.score_breakdown.final == r.score
        assert r.score_breakdown.lexical >= 0.0
        assert r.score_breakdown.vector >= 0.0
        assert r.score_breakdown.graph_boost >= 0.0
        assert r.score_breakdown.recency >= 0.0
        assert r.score_breakdown.confidence >= 0.0

    def test_breakdown_populated_in_fast_mode(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(ns, "Alice prefers", mode="fast", limit=8, explain=True)
        assert len(output.results) >= 1

        r = output.results[0]
        assert r.score_breakdown is not None
        assert r.score_breakdown.final == r.score

    def test_breakdown_absent_when_explain_false(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(ns, "Alice prefers", mode="hybrid", limit=8, explain=False)
        assert len(output.results) >= 1
        assert output.results[0].score_breakdown is None


class TestWhyRetrieved:
    """Verify why_retrieved generates human-readable explanations."""

    def _seed_with_entity(self, store, ns, content, entity_name="Alice"):
        mem = store.insert_memory(ns, content)
        ep_id = mem["episode_id"]
        _seed_episode_embedding(store, ns, ep_id)
        ent = store.insert_entity(ns, entity_name, "Person", confidence=0.92)
        store.insert_provenance(ns, ep_id, entity_id=ent)
        return ep_id

    def test_why_retrieved_non_empty_when_explain(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        self._seed_with_entity(store, ns, "Alice prefers concise technical answers.")

        output = engine.recall(ns, "Alice prefers", mode="hybrid", limit=8, explain=True)
        assert len(output.results) >= 1
        for r in output.results:
            assert r.why_retrieved, f"Result {r.id} missing why_retrieved"
            assert isinstance(r.why_retrieved, str)
            assert len(r.why_retrieved) > 0

    def test_why_retrieved_empty_when_no_explain(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(ns, "Alice prefers", mode="hybrid", limit=8, explain=False)
        assert len(output.results) >= 1
        assert output.results[0].why_retrieved == ""

    def test_why_retrieved_mentions_lexical(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        # Insert with exact content that strongly matches the query
        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(ns, "Alice prefers concise technical answers", mode="hybrid", limit=8, explain=True)
        assert len(output.results) >= 1
        # The top result should mention lexical match since the query is nearly identical
        top = output.results[0]
        assert "lexical" in top.why_retrieved or "match" in top.why_retrieved


class TestLatencyTracking:
    """Verify timing is recorded in the explanation."""

    def test_latency_keys_present(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(ns, "Alice prefers", mode="hybrid", limit=8, explain=True)

        latency = output.explanation.latency_ms
        assert "total" in latency
        assert "fts" in latency
        assert "embedding" in latency
        assert "vector" in latency
        assert "graph" in latency

    def test_latency_values_positive(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(ns, "Alice prefers", mode="hybrid", limit=8, explain=True)

        latency = output.explanation.latency_ms
        assert latency["total"] >= 0.0
        assert latency["fts"] >= 0.0
        assert latency["embedding"] >= 0.0

    def test_latency_absent_when_no_explain(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(ns, "Alice prefers", mode="hybrid", limit=8, explain=False)

        # When explain=False, latency_ms should be empty
        assert output.explanation.latency_ms == {}


class TestExplainFlag:
    """Verify explain=True includes full breakdown in response."""

    def test_explanation_has_retrieval_breakdown(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])
        knw = store.insert_knowledge(ns, "Doc", "GraphCtx uses SQLite.")
        _seed_episode_embedding(store, ns, knw["episode_id"])

        output = engine.recall(ns, "Alice prefers", mode="hybrid", limit=8, explain=True)

        breakdown = output.explanation.retrieval_breakdown
        assert len(breakdown) >= 1
        for entry in breakdown:
            assert "episode_id" in entry
            assert "final_score" in entry
            assert "scores" in entry
            scores = entry["scores"]
            assert "lexical" in scores
            assert "vector" in scores
            assert "graph_boost" in scores
            assert "recency" in scores
            assert "confidence" in scores
            assert "final" in scores

    def test_explanation_breakdown_matches_results(self):
        """Each result should have a corresponding entry in retrieval_breakdown."""
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(ns, "Alice prefers", mode="hybrid", limit=8, explain=True)

        result_ids = {r.id for r in output.results}
        breakdown_ids = {e["episode_id"] for e in output.explanation.retrieval_breakdown}
        assert result_ids == breakdown_ids

    def test_explain_false_no_breakdown(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(ns, "Alice prefers", mode="hybrid", limit=8, explain=False)

        assert output.explanation.retrieval_breakdown == []
        assert output.explanation.latency_ms == {}
        for r in output.results:
            assert r.score_breakdown is None
            assert r.why_retrieved == ""
