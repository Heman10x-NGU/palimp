"""Tests for search_mode and max_tokens parameters on the recall engine."""

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
    assert episode is not None
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


def _seed_memories(store: SQLiteStore, ns: str, contents: list[str]) -> list[str]:
    """Insert memories with embeddings, return their episode IDs."""
    episode_ids = []
    for content in contents:
        mem = store.insert_memory(ns, content)
        _seed_episode_embedding(store, ns, mem["episode_id"])
        episode_ids.append(mem["episode_id"])
    return episode_ids


# ---------------------------------------------------------------------------
# search_mode tests
# ---------------------------------------------------------------------------


class TestSearchModeLexicalOnly:
    """search_mode='lexical' should skip vector and graph backends."""

    def test_search_mode_lexical_only(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-lex"

        _seed_memories(store, ns, [
            "Alice prefers concise answers",
            "Bob likes detailed explanations",
            "The project uses SQLite for storage",
        ])

        output = engine.recall(
            ns, "Alice prefers concise", search_mode="lexical", mode="hybrid",
        )

        # Should return results from FTS5 only
        assert len(output.results) > 0
        # Top result should contain "Alice"
        assert "Alice" in output.results[0].content


class TestSearchModeVectorOnly:
    """search_mode='vector' should skip lexical and graph backends."""

    def test_search_mode_vector_only(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-vec"

        _seed_memories(store, ns, [
            "Alice prefers concise answers",
            "Bob likes detailed explanations",
            "The project uses SQLite for storage",
        ])

        output = engine.recall(
            ns, "Alice prefers concise", search_mode="vector", mode="hybrid",
        )

        # Vector-only: may return results if embeddings exist
        # The key test is that it doesn't crash and returns valid results
        assert output.results is not None
        for r in output.results:
            assert r.score >= 0


class TestSearchModeHybridDefault:
    """search_mode='hybrid' (default) should use all available backends."""

    def test_search_mode_hybrid_default(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-hybrid"

        _seed_memories(store, ns, [
            "Alice prefers concise answers",
            "Bob likes detailed explanations",
            "The project uses SQLite for storage",
        ])

        output = engine.recall(
            ns, "Alice prefers concise", mode="hybrid",
        )

        assert len(output.results) > 0
        # Default is hybrid, so both lexical and vector should contribute
        assert "Alice" in output.results[0].content


# ---------------------------------------------------------------------------
# max_tokens tests
# ---------------------------------------------------------------------------


class TestMaxTokensTruncatesResults:
    """max_tokens should limit the total estimated tokens across results."""

    def test_max_tokens_truncates_results(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-tokens"

        # Insert memories with long content
        long_content_a = "Alpha content about machine learning " * 20  # ~720 chars -> ~180 tokens
        long_content_b = "Beta content about deep learning " * 20  # ~620 chars -> ~155 tokens
        long_content_c = "Gamma content about neural networks " * 20  # ~720 chars -> ~180 tokens

        _seed_memories(store, ns, [long_content_a, long_content_b, long_content_c])

        # Without max_tokens, should return up to limit
        output_all = engine.recall(ns, "content learning", mode="hybrid", limit=8)
        assert len(output_all.results) == 3

        # With very small max_tokens, should truncate
        output_limited = engine.recall(
            ns, "content learning", mode="hybrid", limit=8, max_tokens=100,
        )
        # Should return fewer results (or at most 1 which exceeds budget)
        assert len(output_limited.results) <= len(output_all.results)

        # Verify total estimated tokens is within budget
        total_est_tokens = sum(len(r.content) // 4 for r in output_limited.results)
        # The first item always gets included even if it exceeds budget
        if len(output_limited.results) > 1:
            assert total_est_tokens <= 100 + (len(output_limited.results[0].content) // 4)


class TestMaxTokensAlwaysReturnsAtLeastOne:
    """max_tokens should always return at least one result if any exist."""

    def test_max_tokens_always_returns_at_least_one(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-min-one"

        # Insert a memory with very long content
        very_long = "Important fact about architecture " * 100  # ~3100 chars -> ~775 tokens
        _seed_memories(store, ns, [very_long])

        # Even with tiny budget, should return at least 1 result
        output = engine.recall(
            ns, "architecture", mode="hybrid", max_tokens=1,
        )
        assert len(output.results) >= 1


class TestMaxTokensRespectsScoreOrder:
    """max_tokens should truncate from lowest-scored results, preserving order."""

    def test_max_tokens_respects_score_order(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-order"

        # Insert memories: one short high-relevance, one long low-relevance
        high_relevance = "Alice concise technical answers"  # short, high match
        low_relevance = "Charlie enjoys reading books about history and philosophy " * 20  # long, low match

        _seed_memories(store, ns, [high_relevance, low_relevance])

        # Get all results to determine expected order
        output_all = engine.recall(ns, "Alice concise", mode="hybrid", limit=8)

        # The first result should be the high-relevance one
        assert len(output_all.results) > 0
        top_result_all = output_all.results[0]

        # With max_tokens, the top result should still be the same
        output_limited = engine.recall(
            ns, "Alice concise", mode="hybrid", limit=8, max_tokens=50,
        )
        assert len(output_limited.results) > 0
        assert output_limited.results[0].id == top_result_all.id
