"""Tests for iterative recall (recall_refine).

Covers:
- Narrowing results to a subset of previous IDs
- Dropping non-matching IDs
- Fallback to full recall when previous_result_ids is empty
- Tombstoned episode exclusion
- Empty result when no match
- Independent scoring (no score accumulation)
- Limit enforcement
- Provenance preservation
"""


from palimp.embeddings import DeterministicEmbedder
from palimp.retriever import RecallEngine, _vector_to_blob
from palimp.storage import SQLiteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> SQLiteStore:
    return SQLiteStore(":memory:")


def _make_engine(store: SQLiteStore) -> RecallEngine:
    embedder = DeterministicEmbedder(dim=384)
    return RecallEngine(store=store, embedder=embedder)


def _seed_episode_embedding(store: SQLiteStore, ns: str, episode_id: str) -> None:
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


def _insert_and_seed(store, ns, content, source_ref=None):
    """Insert a memory, seed its embedding, return the episode ID."""
    mem = store.insert_memory(ns, content, source_ref=source_ref)
    ep_id = mem["episode_id"]
    _seed_episode_embedding(store, ns, ep_id)
    return ep_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRefineNarrowsResults:
    """recall_refine returns only results whose IDs are in previous_result_ids."""

    def test_refine_narrows_to_subset(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        ep1 = _insert_and_seed(store, ns, "Alice prefers concise technical answers.")
        ep2 = _insert_and_seed(store, ns, "Bob likes long explanations with examples.")
        ep3 = _insert_and_seed(store, ns, "Charlie uses Python for data science.")

        # Refine with only ep1 and ep3
        output = engine.recall_refine(
            ns, "Alice prefers", previous_result_ids=[ep1, ep3], mode="fast", limit=8,
        )

        result_ids = {r.id for r in output.results}
        assert ep1 in result_ids
        # ep2 was not in previous_result_ids, must not appear
        assert ep2 not in result_ids

    def test_refine_returns_only_allowed_ids(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        ep1 = _insert_and_seed(store, ns, "GraphCtx uses SQLite for storage.")
        ep2 = _insert_and_seed(store, ns, "GraphCtx uses Kuzu for graph queries.")

        # Only pass ep1
        output = engine.recall_refine(
            ns, "GraphCtx", previous_result_ids=[ep1], mode="fast", limit=8,
        )

        result_ids = {r.id for r in output.results}
        assert ep1 in result_ids
        assert ep2 not in result_ids


class TestRefineDropsNonMatching:
    """Episodes in previous_result_ids that don't match the new query still
    participate (they may score via decay/confidence), but IDs not in
    previous_result_ids are excluded."""

    def test_non_matching_id_not_in_previous_excluded(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        ep1 = _insert_and_seed(store, ns, "Alice prefers concise technical answers.")
        ep2 = _insert_and_seed(store, ns, "Bob likes long explanations.")

        # Refine with only ep2, but query about Alice
        output = engine.recall_refine(
            ns, "Alice prefers", previous_result_ids=[ep2], mode="fast", limit=8,
        )

        result_ids = {r.id for r in output.results}
        # ep1 was not in previous_result_ids — must not appear
        assert ep1 not in result_ids


class TestRefineEmptyPreviousIdsFallsBack:
    """Empty previous_result_ids falls back to full recall."""

    def test_empty_ids_falls_back_to_recall(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        ep1 = _insert_and_seed(store, ns, "Alice prefers concise technical answers.")
        _insert_and_seed(store, ns, "Bob likes long explanations.")

        # Empty list should fall back to full recall
        output_refine = engine.recall_refine(
            ns, "Alice prefers", previous_result_ids=[], mode="fast", limit=8,
        )
        output_recall = engine.recall(
            ns, "Alice prefers", mode="fast", limit=8,
        )

        # Both should return results (full recall fallback)
        assert len(output_refine.results) > 0
        refine_ids = {r.id for r in output_refine.results}
        recall_ids = {r.id for r in output_recall.results}
        assert refine_ids == recall_ids


class TestRefineWithTombstonedEpisodes:
    """Tombstoned episodes must not appear in refine results."""

    def test_tombstoned_excluded_from_refine(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        ep1 = _insert_and_seed(store, ns, "Alice prefers concise technical answers.")
        ep2 = _insert_and_seed(store, ns, "Bob likes long explanations.")

        # Tombstone ep1
        store.tombstone_episode(ep1)

        # Try to refine with both IDs
        output = engine.recall_refine(
            ns, "Alice prefers", previous_result_ids=[ep1, ep2], mode="fast", limit=8,
        )

        result_ids = {r.id for r in output.results}
        assert ep1 not in result_ids, "Tombstoned episode must not appear"


class TestRefineReturnsEmptyWhenNoMatch:
    """When all previous episodes are tombstoned/deleted, return empty."""

    def test_all_tombstoned_returns_empty(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        ep1 = _insert_and_seed(store, ns, "Alice prefers concise technical answers.")
        store.tombstone_episode(ep1)

        output = engine.recall_refine(
            ns, "Alice prefers", previous_result_ids=[ep1], mode="fast", limit=8,
        )

        assert output.results == []

    def test_empty_namespace_returns_empty(self):
        store = _make_store()
        engine = _make_engine(store)

        output = engine.recall_refine(
            "nonexistent", "anything", previous_result_ids=["fake-id"], mode="fast",
        )

        assert output.results == []


class TestRefineScoresAreIndependent:
    """Refine scores episodes from scratch — no accumulation from prior recall."""

    def test_refine_scores_match_fresh_recall(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        ep1 = _insert_and_seed(store, ns, "Alice prefers concise technical answers.")

        # Get fresh recall score
        fresh = engine.recall(ns, "Alice prefers", mode="fast", limit=8)
        fresh_score = next(r.score for r in fresh.results if r.id == ep1)

        # Get refine score with the same episode
        refined = engine.recall_refine(
            ns, "Alice prefers", previous_result_ids=[ep1], mode="fast", limit=8,
        )
        refined_score = next(r.score for r in refined.results if r.id == ep1)

        # Scores should be identical (independent recomputation)
        assert abs(fresh_score - refined_score) < 0.0001

    def test_refine_explain_shows_independent_breakdown(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        ep1 = _insert_and_seed(store, ns, "Alice prefers concise technical answers.")

        refined = engine.recall_refine(
            ns, "Alice prefers", previous_result_ids=[ep1], mode="hybrid",
            limit=8, explain=True,
        )

        assert len(refined.results) == 1
        r = refined.results[0]
        assert r.score_breakdown is not None
        assert r.score_breakdown.final == r.score
        assert refined.explanation.refined_from_count == 1


class TestRefineRespectsLimit:
    """Refine must respect the limit parameter."""

    def test_limit_caps_results(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        ids = []
        for i in range(10):
            ep_id = _insert_and_seed(store, ns, f"Episode number {i} about technology and science.")
            ids.append(ep_id)

        # Refine with limit=3
        output = engine.recall_refine(
            ns, "technology", previous_result_ids=ids, mode="fast", limit=3,
        )

        assert len(output.results) <= 3


class TestRefinePreservesProvenance:
    """Refine results should have provenance when include_provenance=True."""

    def test_provenance_present(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        ep1 = _insert_and_seed(store, ns, "Alice prefers concise technical answers.", source_ref="chat:123")

        output = engine.recall_refine(
            ns, "Alice prefers", previous_result_ids=[ep1], mode="fast",
            limit=8, include_provenance=True,
        )

        assert len(output.results) == 1
        r = output.results[0]
        assert r.provenance, "Provenance must be present"
        assert r.provenance[0]["episode_id"] == ep1

    def test_provenance_absent_when_disabled(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        ep1 = _insert_and_seed(store, ns, "Alice prefers concise technical answers.")

        output = engine.recall_refine(
            ns, "Alice prefers", previous_result_ids=[ep1], mode="fast",
            limit=8, include_provenance=False,
        )

        assert len(output.results) == 1
        r = output.results[0]
        assert r.provenance == []
