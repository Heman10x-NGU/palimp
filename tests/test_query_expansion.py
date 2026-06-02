"""Tests for query expansion and reranking (plan section 3.5).

Covers:
- Query splitting on conjunctions
- Temporal expansion (removing temporal cues)
- Entity extraction (capitalized words)
- Merge deduplication across sub-queries
- Heuristic reranker
- HTTP reranker fallback
- Query expansions in explanation
"""

from dataclasses import dataclass
from typing import Optional


from palimp.embeddings import DeterministicEmbedder
from palimp.models import RecallResult
from palimp.query_expansion import expand_query, merge_expansion_results
from palimp.reranker import rerank_results
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


@dataclass
class FakeResult:
    """Lightweight result object for reranker unit tests."""
    id: str
    content: str
    score: float = 0.0
    kind: str = "memory"
    temporal_status: Optional[str] = None


# ---------------------------------------------------------------------------
# test_split_conjunctions
# ---------------------------------------------------------------------------


class TestSplitConjunctions:
    """Complex queries with 'and', 'or', 'but' split into sub-queries."""

    def test_and_split(self):
        subs = expand_query("Alice prefers concise answers and Bob likes verbose output")
        assert len(subs) >= 2
        lower_subs = [s.lower() for s in subs]
        assert any("alice" in s for s in lower_subs)
        assert any("bob" in s for s in lower_subs)

    def test_or_split(self):
        subs = expand_query("use PostgreSQL or use SQLite for storage")
        assert len(subs) >= 2

    def test_but_split(self):
        subs = expand_query("the API is fast but the CLI is slow")
        assert len(subs) >= 2

    def test_semicolon_split(self):
        subs = expand_query("run the tests; check the logs")
        assert len(subs) >= 2

    def test_single_clause_unchanged(self):
        subs = expand_query("What is GraphCtx?")
        # Single clause should return at least the original query
        assert len(subs) >= 1
        assert any("graphctx" in s.lower() for s in subs)


# ---------------------------------------------------------------------------
# test_temporal_expansion
# ---------------------------------------------------------------------------


class TestTemporalExpansion:
    """Temporal cue words produce an expansion without those words."""

    def test_now_expansion(self):
        subs = expand_query("what is the API key now")
        # Should include a version without "now"
        assert any("now" not in s.lower() and "api" in s.lower() for s in subs)

    def test_currently_expansion(self):
        subs = expand_query("what version is currently deployed")
        assert any("currently" not in s.lower() for s in subs)

    def test_today_expansion(self):
        subs = expand_query("what happened today in the pipeline")
        assert any("today" not in s.lower() and "pipeline" in s.lower() for s in subs)

    def test_before_preserved(self):
        subs = expand_query("what was the config before the migration")
        # "before" queries should be preserved (not stripped)
        assert any("before" in s.lower() for s in subs)


# ---------------------------------------------------------------------------
# test_entity_extraction
# ---------------------------------------------------------------------------


class TestEntityExtraction:
    """Capitalized words are extracted as targeted entity sub-queries."""

    def test_capitalized_entity(self):
        subs = expand_query("How does GraphCtx work")
        assert any("graphctx" in s.lower() for s in subs)

    def test_multiple_entities(self):
        subs = expand_query("compare GraphCtx and Mem0 features")
        lower_subs = [s.lower() for s in subs]
        assert any("graphctx" in s for s in lower_subs)
        assert any("mem0" in s for s in lower_subs)

    def test_short_caps_ignored(self):
        subs = expand_query("what is X in the system")
        # Single-letter caps like "X" (len <= 2) should be excluded from entity extraction
        # The query may still be returned as a whole, but "X" alone shouldn't be an entity query
        entity_only = [s for s in subs if s.strip().lower() == "x"]
        assert len(entity_only) == 0


# ---------------------------------------------------------------------------
# test_merge_dedup
# ---------------------------------------------------------------------------


class TestMergeDedup:
    """Same episode appearing in multiple sub-queries is kept once with best score."""

    def test_dedup_keeps_best_score(self):
        r1 = RecallResult(id="ep-1", kind="memory", content="hello", score=0.5)
        r2 = RecallResult(id="ep-1", kind="memory", content="hello", score=0.8)
        r3 = RecallResult(id="ep-2", kind="memory", content="world", score=0.6)

        merged = merge_expansion_results([[r1, r3], [r2]])
        ids = [r.id for r in merged]
        assert "ep-1" in ids
        assert "ep-2" in ids
        # ep-1 should have the best score
        ep1 = next(r for r in merged if r.id == "ep-1")
        assert ep1.score == 0.8

    def test_dedup_preserves_unique_episodes(self):
        r1 = RecallResult(id="ep-1", kind="memory", content="a", score=0.3)
        r2 = RecallResult(id="ep-2", kind="memory", content="b", score=0.4)
        r3 = RecallResult(id="ep-3", kind="memory", content="c", score=0.5)

        merged = merge_expansion_results([[r1, r2], [r3]])
        assert len(merged) == 3

    def test_empty_inputs(self):
        merged = merge_expansion_results([[], []])
        assert merged == []


# ---------------------------------------------------------------------------
# test_reranker_missing_endpoint
# ---------------------------------------------------------------------------


class TestRerankerMissingEndpoint:
    """No endpoint falls back to heuristic reranking."""

    def test_heuristic_rerank_with_no_endpoint(self):
        results = [
            RecallResult(id="ep-1", kind="memory", content="the quick brown fox", score=0.3),
            RecallResult(id="ep-2", kind="memory", content="quick brown fox jumps", score=0.5),
        ]
        reranked = rerank_results(results, "quick fox", endpoint=None, top_k=20)
        assert len(reranked) == 2
        # Both should still be present
        ids = {r.id for r in reranked}
        assert ids == {"ep-1", "ep-2"}

    def test_empty_results_no_error(self):
        reranked = rerank_results([], "query", endpoint=None)
        assert reranked == []


# ---------------------------------------------------------------------------
# test_reranker_heuristic
# ---------------------------------------------------------------------------


class TestRerankerHeuristic:
    """Results are reranked by term overlap, entity bonus, temporal bonus."""

    def test_term_overlap_boosts_rank(self):
        # ep-2 has more term overlap with the query
        r1 = RecallResult(id="ep-1", kind="memory", content="alpha beta gamma", score=0.9)
        r2 = RecallResult(id="ep-2", kind="memory", content="database migration postgresql", score=0.1)
        reranked = rerank_results([r1, r2], "database migration", endpoint=None, top_k=20)
        # ep-2 should rank higher because it has better term overlap
        assert reranked[0].id == "ep-2"

    def test_knowledge_kind_gets_bonus(self):
        r1 = RecallResult(id="ep-1", kind="memory", content="database migration", score=0.5)
        r2 = RecallResult(id="ep-2", kind="knowledge", content="database migration", score=0.5)
        reranked = rerank_results([r1, r2], "database migration", endpoint=None, top_k=20)
        # Knowledge kind gets entity_bonus=0.1, so ep-2 should rank first
        assert reranked[0].id == "ep-2"

    def test_current_temporal_gets_bonus(self):
        r1 = RecallResult(id="ep-1", kind="memory", content="api key", score=0.5, temporal_status="historical")
        r2 = RecallResult(id="ep-2", kind="memory", content="api key", score=0.5, temporal_status="current")
        reranked = rerank_results([r1, r2], "api key", endpoint=None, top_k=20)
        assert reranked[0].id == "ep-2"


# ---------------------------------------------------------------------------
# test_expansion_in_explanation
# ---------------------------------------------------------------------------


class TestExpansionInExplanation:
    """Thinking mode explanation includes query_expansions."""

    def test_explanation_has_query_expansions(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        # Insert episodes that will match different parts of the query
        store.insert_memory(ns, "Alice prefers concise technical answers.", source_ref="chat:1")
        store.insert_memory(ns, "Bob likes verbose documentation.", source_ref="chat:2")

        output = engine.recall(
            ns,
            "Alice prefers concise answers and Bob likes verbose docs",
            mode="thinking",
            limit=8,
            explain=True,
        )

        # Explanation should include query_expansions (non-empty because "and" split)
        assert len(output.explanation.query_expansions) > 0

    def test_hybrid_mode_no_expansions(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        store.insert_memory(ns, "Alice prefers concise answers and Bob likes verbose docs.", source_ref="chat:1")

        output = engine.recall(
            ns,
            "Alice prefers concise answers and Bob likes verbose docs",
            mode="hybrid",
            limit=8,
            explain=True,
        )

        # Hybrid mode should NOT produce query expansions
        assert output.explanation.query_expansions == []

    def test_thinking_expands_and_retrieves_broadly(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        # Two episodes matching different sub-clauses
        mem1 = store.insert_memory(ns, "GraphCtx uses SQLite for local storage.", source_ref="docs:1")
        _seed_episode_embedding(store, ns, mem1["episode_id"])
        mem2 = store.insert_memory(ns, "Mem0 uses hosted vector DB for cloud storage.", source_ref="docs:2")
        _seed_episode_embedding(store, ns, mem2["episode_id"])

        output = engine.recall(
            ns,
            "GraphCtx uses SQLite and Mem0 uses vector DB",
            mode="thinking",
            limit=8,
            explain=True,
        )

        # Thinking mode with expansion should retrieve both episodes
        result_ids = {r.id for r in output.results}
        assert mem1["episode_id"] in result_ids
        assert mem2["episode_id"] in result_ids

        # Explanation should show the expansions
        assert len(output.explanation.query_expansions) >= 2

        # Latency should include rerank key
        assert "rerank" in output.explanation.latency_ms
