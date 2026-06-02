"""Tests for retrieval improvements (plan section 4.8).

Covers:
- source_ref in provenance for every recall result
- warnings for contradicted/superseded results in ALL modes
- query_terms and matched_terms in explanation
- kind consistency (always "memory" or "knowledge")
- CLI --explain flag
- CLI --limit and --mode flags
"""

from __future__ import annotations

import struct

import pytest
from typer.testing import CliRunner

from graphctx.cli import app
from graphctx.embeddings import DeterministicEmbedder
from graphctx.models import RecallResult, RecallExplanation
from graphctx.retriever import RecallEngine, _vector_to_blob, _extract_query_terms
from graphctx.storage import SQLiteStore

runner = CliRunner()


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
# Test: source_ref in provenance
# ---------------------------------------------------------------------------


class TestSourceRefInProvenance:
    """Verify every recall result includes source_ref in provenance when available."""

    def test_memory_source_ref_in_provenance(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(
            ns, "Alice prefers concise answers.", source_ref="chat:2026-06-02"
        )
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(ns, "Alice prefers", mode="hybrid", limit=8)
        assert len(output.results) >= 1

        r = output.results[0]
        assert r.provenance, "Result missing provenance"
        assert any(
            p.get("source_ref") == "chat:2026-06-02" for p in r.provenance
        ), f"source_ref not found in provenance: {r.provenance}"

    def test_knowledge_source_ref_in_provenance(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        knw = store.insert_knowledge(
            ns, "Architecture", "GraphCtx uses SQLite.", source_ref="docs/arch.md"
        )
        _seed_episode_embedding(store, ns, knw["episode_id"])

        output = engine.recall(ns, "GraphCtx SQLite", mode="hybrid", limit=8)
        assert len(output.results) >= 1

        r = output.results[0]
        assert r.provenance
        assert any(
            p.get("source_ref") == "docs/arch.md" for p in r.provenance
        ), f"source_ref not found in provenance: {r.provenance}"

    def test_no_source_ref_still_has_provenance(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "No source ref here.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(ns, "No source ref", mode="hybrid", limit=8)
        assert len(output.results) >= 1

        r = output.results[0]
        assert r.provenance, "Result missing provenance"
        # source_ref should not be present when not set
        for p in r.provenance:
            assert "source_ref" not in p or p["source_ref"] is None


# ---------------------------------------------------------------------------
# Test: warnings for contradictions
# ---------------------------------------------------------------------------


class TestWarningsForContradictions:
    """Verify contradicted/superseded results have warnings in ALL modes."""

    def test_contradiction_warning_in_fast_mode(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem1 = store.insert_memory(ns, "GraphCtx uses SQLite.")
        ep1_id = mem1["episode_id"]
        _seed_episode_embedding(store, ns, ep1_id)

        mem2 = store.insert_memory(ns, "GraphCtx no longer uses SQLite; it uses Kuzu.")
        ep2_id = mem2["episode_id"]
        _seed_episode_embedding(store, ns, ep2_id)

        ent1 = store.insert_entity(ns, "GraphCtx", "Project", confidence=0.9)
        ent2 = store.insert_entity(ns, "Kuzu", "Technology", confidence=0.85)

        store.insert_provenance(ns, ep1_id, entity_id=ent1)
        store.insert_provenance(ns, ep2_id, entity_id=ent1)
        store.insert_provenance(ns, ep2_id, entity_id=ent2)

        edge_id = store.insert_edge(ns, ent1, ent2, "CONTRADICTS", confidence=0.7)
        store.insert_provenance(ns, ep2_id, edge_id=edge_id)

        results = engine.recall(ns, "GraphCtx SQLite", mode="fast", limit=8).results
        assert len(results) >= 1

        has_warning = False
        for r in results:
            if r.warnings:
                has_warning = True
                assert any("CONTRADICTS" in w for w in r.warnings)
        assert has_warning, "Expected CONTRADICTS warning in fast mode results"

    def test_contradiction_warning_in_hybrid_mode(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem1 = store.insert_memory(ns, "GraphCtx uses SQLite.")
        ep1_id = mem1["episode_id"]
        _seed_episode_embedding(store, ns, ep1_id)

        mem2 = store.insert_memory(ns, "GraphCtx no longer uses SQLite; it uses Kuzu.")
        ep2_id = mem2["episode_id"]
        _seed_episode_embedding(store, ns, ep2_id)

        ent1 = store.insert_entity(ns, "GraphCtx", "Project", confidence=0.9)
        ent2 = store.insert_entity(ns, "Kuzu", "Technology", confidence=0.85)

        store.insert_provenance(ns, ep1_id, entity_id=ent1)
        store.insert_provenance(ns, ep2_id, entity_id=ent1)
        store.insert_provenance(ns, ep2_id, entity_id=ent2)

        edge_id = store.insert_edge(ns, ent1, ent2, "CONTRADICTS", confidence=0.7)
        store.insert_provenance(ns, ep2_id, edge_id=edge_id)

        results = engine.recall(ns, "GraphCtx SQLite", mode="hybrid", limit=8).results
        assert len(results) >= 1

        has_warning = False
        for r in results:
            if r.warnings:
                has_warning = True
                assert any("CONTRADICTS" in w for w in r.warnings)
        assert has_warning, "Expected CONTRADICTS warning in hybrid mode results"

    def test_contradiction_warning_in_thinking_mode(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem1 = store.insert_memory(ns, "GraphCtx uses SQLite.")
        ep1_id = mem1["episode_id"]
        _seed_episode_embedding(store, ns, ep1_id)

        mem2 = store.insert_memory(ns, "GraphCtx no longer uses SQLite; it uses Kuzu.")
        ep2_id = mem2["episode_id"]
        _seed_episode_embedding(store, ns, ep2_id)

        ent1 = store.insert_entity(ns, "GraphCtx", "Project", confidence=0.9)
        ent2 = store.insert_entity(ns, "Kuzu", "Technology", confidence=0.85)

        store.insert_provenance(ns, ep1_id, entity_id=ent1)
        store.insert_provenance(ns, ep2_id, entity_id=ent1)
        store.insert_provenance(ns, ep2_id, entity_id=ent2)

        edge_id = store.insert_edge(ns, ent1, ent2, "CONTRADICTS", confidence=0.7)
        store.insert_provenance(ns, ep2_id, edge_id=edge_id)

        results = engine.recall(ns, "GraphCtx SQLite", mode="thinking", limit=8).results
        assert len(results) >= 1

        has_warning = False
        for r in results:
            if r.warnings:
                has_warning = True
                assert any("CONTRADICTS" in w for w in r.warnings)
        assert has_warning, "Expected CONTRADICTS warning in thinking mode results"

    def test_supersession_warning_in_fast_mode(self):
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

        results = engine.recall(ns, "API uses", mode="fast", limit=8).results
        assert len(results) >= 1

        has_warning = False
        for r in results:
            if r.warnings:
                has_warning = True
                assert any("SUPERSEDES" in w for w in r.warnings)
        assert has_warning, "Expected SUPERSEDES warning in fast mode results"

    def test_no_warnings_without_conflict(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        results = engine.recall(ns, "Alice prefers", mode="hybrid", limit=8).results
        assert len(results) >= 1

        for r in results:
            assert r.warnings == [], f"Unexpected warnings: {r.warnings}"


# ---------------------------------------------------------------------------
# Test: query_terms in explanation
# ---------------------------------------------------------------------------


class TestQueryTermsInExplanation:
    """Verify explanation includes query_terms."""

    def test_query_terms_populated_when_explain(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(
            ns, "Alice prefers concise answers", mode="hybrid", limit=8, explain=True
        )

        assert output.explanation.query_terms, "query_terms should be populated"
        # Should contain meaningful tokens
        terms = output.explanation.query_terms
        assert "alice" in terms
        assert "prefers" in terms
        assert "concise" in terms
        assert "answers" in terms

    def test_query_terms_empty_when_no_explain(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(
            ns, "Alice prefers", mode="hybrid", limit=8, explain=False
        )

        assert output.explanation.query_terms == []


# ---------------------------------------------------------------------------
# Test: matched_terms in explanation
# ---------------------------------------------------------------------------


class TestMatchedTermsInExplanation:
    """Verify explanation includes matched_terms per result."""

    def test_matched_terms_in_retrieval_breakdown(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise technical answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(
            ns, "Alice prefers concise answers", mode="hybrid", limit=8, explain=True
        )

        assert output.explanation.retrieval_breakdown
        entry = output.explanation.retrieval_breakdown[0]
        assert "matched_terms" in entry, "matched_terms should be in breakdown entry"
        matched = entry["matched_terms"]
        assert len(matched) > 0
        # "alice" and "prefers" should match
        assert "alice" in matched
        assert "prefers" in matched

    def test_matched_terms_reflect_content(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        # Insert memory that only matches some query terms
        mem = store.insert_memory(ns, "Alice likes Python programming.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(
            ns, "Alice prefers concise answers", mode="hybrid", limit=8, explain=True
        )

        assert output.explanation.retrieval_breakdown
        entry = output.explanation.retrieval_breakdown[0]
        matched = entry.get("matched_terms", [])
        # "alice" should match, but "prefers", "concise", "answers" should not
        assert "alice" in matched
        assert "prefers" not in matched
        assert "concise" not in matched


# ---------------------------------------------------------------------------
# Test: kind consistency
# ---------------------------------------------------------------------------


class TestKindConsistency:
    """Verify all results have kind='memory' or 'knowledge'."""

    def test_kind_is_memory_or_knowledge(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        knw = store.insert_knowledge(ns, "Doc", "GraphCtx uses SQLite.")
        _seed_episode_embedding(store, ns, knw["episode_id"])

        output = engine.recall(ns, "Alice GraphCtx", mode="hybrid", limit=8)

        for r in output.results:
            assert r.kind in ("memory", "knowledge"), f"Invalid kind: {r.kind}"

    def test_memory_result_has_kind_memory(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        output = engine.recall(ns, "Alice prefers", mode="fast", limit=8)
        assert len(output.results) >= 1
        assert output.results[0].kind == "memory"

    def test_knowledge_result_has_kind_knowledge(self):
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        knw = store.insert_knowledge(ns, "Doc", "GraphCtx uses SQLite.")
        _seed_episode_embedding(store, ns, knw["episode_id"])

        output = engine.recall(ns, "GraphCtx SQLite", mode="fast", limit=8)
        assert len(output.results) >= 1
        assert output.results[0].kind == "knowledge"

    def test_kind_consistency_across_modes(self):
        """All modes should return consistent kind values."""
        store = _make_store()
        engine = _make_engine(store)
        ns = "test-ns"

        mem = store.insert_memory(ns, "Alice prefers concise answers.")
        _seed_episode_embedding(store, ns, mem["episode_id"])

        for mode in ("fast", "hybrid", "thinking"):
            output = engine.recall(ns, "Alice prefers", mode=mode, limit=8)
            for r in output.results:
                assert r.kind in ("memory", "knowledge"), f"Invalid kind in {mode}: {r.kind}"


# ---------------------------------------------------------------------------
# Test: CLI --explain flag
# ---------------------------------------------------------------------------


class TestCliRecallExplain:
    """Verify --explain flag works in CLI recall."""

    def test_explain_flag_prints_breakdown(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        # Add a memory first
        runner.invoke(
            app,
            [
                "memory", "add",
                "Alice prefers concise technical answers.",
                "--namespace", "test",
                "--db", db_path,
            ],
        )

        # Recall with --explain
        result = runner.invoke(
            app,
            [
                "recall",
                "Alice prefers",
                "--namespace", "test",
                "--explain",
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "Score Breakdown" in result.stdout
        assert "lexical:" in result.stdout
        assert "vector:" in result.stdout
        assert "final:" in result.stdout

    def test_explain_flag_prints_why(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        runner.invoke(
            app,
            [
                "memory", "add",
                "Alice prefers concise technical answers.",
                "--namespace", "test",
                "--db", db_path,
            ],
        )

        result = runner.invoke(
            app,
            [
                "recall",
                "Alice prefers concise technical answers",
                "--namespace", "test",
                "--explain",
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "why:" in result.stdout

    def test_explain_flag_prints_query_terms(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        runner.invoke(
            app,
            [
                "memory", "add",
                "Alice prefers concise technical answers.",
                "--namespace", "test",
                "--db", db_path,
            ],
        )

        result = runner.invoke(
            app,
            [
                "recall",
                "Alice prefers concise",
                "--namespace", "test",
                "--explain",
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "Query terms:" in result.stdout

    def test_no_explain_flag_omits_breakdown(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        runner.invoke(
            app,
            [
                "memory", "add",
                "Alice prefers concise technical answers.",
                "--namespace", "test",
                "--db", db_path,
            ],
        )

        result = runner.invoke(
            app,
            [
                "recall",
                "Alice prefers",
                "--namespace", "test",
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "Score Breakdown" not in result.stdout


# ---------------------------------------------------------------------------
# Test: CLI --limit and --mode flags
# ---------------------------------------------------------------------------


class TestCliRecallLimitMode:
    """Verify --limit and --mode flags work in CLI recall."""

    def test_limit_flag(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        # Add multiple memories
        for i in range(5):
            runner.invoke(
                app,
                [
                    "memory", "add",
                    f"Memory number {i} about testing.",
                    "--namespace", "test",
                    "--db", db_path,
                ],
            )

        # Recall with limit=2
        result = runner.invoke(
            app,
            [
                "recall",
                "testing",
                "--namespace", "test",
                "--limit", "2",
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0, result.stdout
        # Count "--- Result N ---" occurrences
        result_count = result.stdout.count("--- Result")
        assert result_count <= 2, f"Expected at most 2 results, got {result_count}"

    def test_mode_flag_fast(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        runner.invoke(
            app,
            [
                "memory", "add",
                "Alice prefers concise technical answers.",
                "--namespace", "test",
                "--db", db_path,
            ],
        )

        result = runner.invoke(
            app,
            [
                "recall",
                "Alice prefers",
                "--namespace", "test",
                "--mode", "fast",
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "kind:" in result.stdout

    def test_mode_flag_thinking(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        runner.invoke(
            app,
            [
                "memory", "add",
                "Alice prefers concise technical answers.",
                "--namespace", "test",
                "--db", db_path,
            ],
        )

        result = runner.invoke(
            app,
            [
                "recall",
                "Alice prefers",
                "--namespace", "test",
                "--mode", "thinking",
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "kind:" in result.stdout


# ---------------------------------------------------------------------------
# Test: source_ref in provenance via CLI
# ---------------------------------------------------------------------------


class TestCliSourceRefInProvenance:
    """Verify source_ref appears in CLI recall output."""

    def test_source_ref_in_cli_output(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        runner.invoke(
            app,
            [
                "memory", "add",
                "Alice prefers concise answers.",
                "--namespace", "test",
                "--source-ref", "chat:2026-06-02",
                "--db", db_path,
            ],
        )

        result = runner.invoke(
            app,
            [
                "recall",
                "Alice prefers",
                "--namespace", "test",
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "source: chat:2026-06-02" in result.stdout


# ---------------------------------------------------------------------------
# Test: warnings in CLI output
# ---------------------------------------------------------------------------


class TestCliWarningsOutput:
    """Verify warnings appear in CLI recall output."""

    def test_warnings_in_cli_output(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        # Add two memories
        runner.invoke(
            app,
            [
                "memory", "add",
                "GraphCtx uses SQLite.",
                "--namespace", "test",
                "--db", db_path,
            ],
        )
        runner.invoke(
            app,
            [
                "memory", "add",
                "GraphCtx no longer uses SQLite; it uses Kuzu.",
                "--namespace", "test",
                "--db", db_path,
            ],
        )

        # We need to manually add entities and CONTRADICTS edge
        # since CLI add doesn't do full extraction by default in memory add
        # Use the store directly
        store = SQLiteStore(db_path)
        ns = "test"

        # Get episodes
        episodes = store.search_fts(ns, "GraphCtx", limit=10)
        assert len(episodes) >= 2

        ep1_id = episodes[0]["episode_id"]
        ep2_id = episodes[1]["episode_id"]

        ent1 = store.insert_entity(ns, "GraphCtx", "Project", confidence=0.9)
        ent2 = store.insert_entity(ns, "Kuzu", "Technology", confidence=0.85)

        store.insert_provenance(ns, ep1_id, entity_id=ent1)
        store.insert_provenance(ns, ep2_id, entity_id=ent1)
        store.insert_provenance(ns, ep2_id, entity_id=ent2)

        edge_id = store.insert_edge(ns, ent1, ent2, "CONTRADICTS", confidence=0.7)
        store.insert_provenance(ns, ep2_id, edge_id=edge_id)

        # Recall
        result = runner.invoke(
            app,
            [
                "recall",
                "GraphCtx SQLite",
                "--namespace", "test",
                "--db", db_path,
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "WARNING:" in result.stdout
        assert "CONTRADICTS" in result.stdout
