"""Tests for multi-hop BFS graph traversal integration."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from graphctx.embeddings import DeterministicEmbedder
from graphctx.graph_traversal import bfs_graph_traversal, get_episodes_for_entities
from graphctx.retriever import RecallEngine
from graphctx.storage import SQLiteStore


def _make_engine(store: SQLiteStore) -> RecallEngine:
    """Create a RecallEngine with deterministic embedder."""
    return RecallEngine(store, DeterministicEmbedder())


def _build_chain(store: SQLiteStore, ns: str, n: int = 3) -> list[str]:
    """Build a linear entity chain A->B->C->... of length n.

    Returns list of episode_ids [eps_A, eps_B, eps_C, ...].
    """
    entity_ids: list[str] = []
    episode_ids: list[str] = []
    names = ["Alice", "Bob", "Carol", "Dave", "Eve"]

    for i in range(n):
        name = names[i % len(names)]
        eps = store.insert_episode(ns, f"{name} fact number {i}", "memory")
        ent = store.insert_entity(ns, name, "person", confidence=1.0)
        store.insert_provenance(ns, eps, entity_id=ent)
        entity_ids.append(ent)
        episode_ids.append(eps)

    # Create edges: A->B, B->C, ...
    for i in range(n - 1):
        store.insert_edge(ns, entity_ids[i], entity_ids[i + 1], "related_to")

    return episode_ids


class TestBfsGraphTraversal:
    """Unit tests for the bfs_graph_traversal function directly."""

    def test_2hop_from_seeds(self) -> None:
        """BFS from A discovers B (hop 1) and C (hop 2)."""
        store = SQLiteStore(":memory:")
        ns = "test"

        ent_a = store.insert_entity(ns, "A", "thing")
        ent_b = store.insert_entity(ns, "B", "thing")
        ent_c = store.insert_entity(ns, "C", "thing")
        store.insert_edge(ns, ent_a, ent_b, "link")
        store.insert_edge(ns, ent_b, ent_c, "link")

        result = bfs_graph_traversal(store, ns, [ent_a], max_hops=2, depth_decay=0.55)

        assert ent_a in result
        assert result[ent_a]["hop"] == 0
        assert result[ent_a]["score"] == 1.0

        assert ent_b in result
        assert result[ent_b]["hop"] == 1
        assert abs(result[ent_b]["score"] - 0.55) < 1e-6

        assert ent_c in result
        assert result[ent_c]["hop"] == 2
        assert abs(result[ent_c]["score"] - 0.55**2) < 1e-6

    def test_cycle_no_infinite_loop(self) -> None:
        """A->B->A cycle does not cause infinite loop."""
        store = SQLiteStore(":memory:")
        ns = "test"

        ent_a = store.insert_entity(ns, "A", "thing")
        ent_b = store.insert_entity(ns, "B", "thing")
        store.insert_edge(ns, ent_a, ent_b, "link")
        store.insert_edge(ns, ent_b, ent_a, "link_back")

        # Should complete without hanging
        result = bfs_graph_traversal(store, ns, [ent_a], max_hops=3, max_expansions=100)

        assert ent_a in result
        assert ent_b in result
        # Only 2 entities total (cycle stops at visited)
        assert len(result) == 2

    def test_tombstoned_entity_skipped(self) -> None:
        """Tombstoned (deleted) entities are excluded from BFS."""
        store = SQLiteStore(":memory:")
        ns = "test"

        ent_a = store.insert_entity(ns, "A", "thing")
        ent_b = store.insert_entity(ns, "B", "thing")
        ent_c = store.insert_entity(ns, "C", "thing")
        store.insert_edge(ns, ent_a, ent_b, "link")
        store.insert_edge(ns, ent_b, ent_c, "link")

        # Tombstone B
        conn = store._conn()
        conn.execute("UPDATE entity SET deleted_at = '2025-01-01' WHERE id = ?", (ent_b,))
        conn.commit()

        result = bfs_graph_traversal(store, ns, [ent_a], max_hops=2)

        # A is seed, B is deleted so skipped, C unreachable
        assert ent_a in result
        assert ent_b not in result
        assert ent_c not in result

    def test_deleted_edge_skipped(self) -> None:
        """Deleted edges are excluded from BFS."""
        store = SQLiteStore(":memory:")
        ns = "test"

        ent_a = store.insert_entity(ns, "A", "thing")
        ent_b = store.insert_entity(ns, "B", "thing")
        edge_id = store.insert_edge(ns, ent_a, ent_b, "link")

        # Delete the edge
        conn = store._conn()
        conn.execute("UPDATE edge SET deleted_at = '2025-01-01' WHERE id = ?", (edge_id,))
        conn.commit()

        result = bfs_graph_traversal(store, ns, [ent_a], max_hops=2)

        assert ent_a in result
        assert ent_b not in result

    def test_max_expansions_respected(self) -> None:
        """BFS stops after max_expansions."""
        store = SQLiteStore(":memory:")
        ns = "test"

        # Create a star: center -> 10 leaves
        center = store.insert_entity(ns, "center", "thing")
        leaf_ids = []
        for i in range(10):
            leaf = store.insert_entity(ns, f"leaf_{i}", "thing")
            store.insert_edge(ns, center, leaf, "link")
            leaf_ids.append(leaf)

        # Only allow 3 expansions
        result = bfs_graph_traversal(store, ns, [center], max_hops=1, max_expansions=3)

        # center (seed) + 3 expanded leaves = 4 total
        assert len(result) == 4
        assert center in result

    def test_depth_decay_formula(self) -> None:
        """hop-2 score = depth_decay * hop-1 score."""
        store = SQLiteStore(":memory:")
        ns = "test"

        ent_a = store.insert_entity(ns, "A", "thing")
        ent_b = store.insert_entity(ns, "B", "thing")
        ent_c = store.insert_entity(ns, "C", "thing")
        store.insert_edge(ns, ent_a, ent_b, "link")
        store.insert_edge(ns, ent_b, ent_c, "link")

        decay = 0.55
        result = bfs_graph_traversal(store, ns, [ent_a], max_hops=2, depth_decay=decay)

        hop1_score = result[ent_b]["score"]
        hop2_score = result[ent_c]["score"]

        assert abs(hop1_score - decay) < 1e-6
        assert abs(hop2_score - decay**2) < 1e-6
        assert abs(hop2_score - decay * hop1_score) < 1e-6

    def test_get_episodes_for_entities(self) -> None:
        """get_episodes_for_entities maps entities to their source episodes."""
        store = SQLiteStore(":memory:")
        ns = "test"

        eps_a = store.insert_episode(ns, "content A", "memory")
        ent_a = store.insert_entity(ns, "A", "thing")
        store.insert_provenance(ns, eps_a, entity_id=ent_a)

        eps_b = store.insert_episode(ns, "content B", "memory")
        ent_b = store.insert_entity(ns, "B", "thing")
        store.insert_provenance(ns, eps_b, entity_id=ent_b)

        result = get_episodes_for_entities(store, [ent_a, ent_b])

        assert ent_a in result
        assert eps_a in result[ent_a]
        assert ent_b in result
        assert eps_b in result[ent_b]


class TestMultiHopRecall:
    """Integration tests for multi-hop BFS in the recall engine."""

    def test_2hop_retrieval(self, memory_store: SQLiteStore) -> None:
        """A->B->C: query about A retrieves C via 2-hop even without lexical match."""
        ns = "test"
        store = memory_store

        # Episode A: content about Alice
        eps_a = store.insert_episode(ns, "Alice works on ProjectX daily", "memory")
        ent_a = store.insert_entity(ns, "Alice", "person")
        store.insert_provenance(ns, eps_a, entity_id=ent_a)

        # Episode B: content about ProjectX
        eps_b = store.insert_episode(ns, "ProjectX uses Python extensively", "memory")
        ent_b = store.insert_entity(ns, "ProjectX", "project")
        store.insert_provenance(ns, eps_b, entity_id=ent_b)

        # Episode C: content about Python (no lexical overlap with "Alice")
        eps_c = store.insert_episode(ns, "Python requires pip for installation", "memory")
        ent_c = store.insert_entity(ns, "Python", "language")
        store.insert_provenance(ns, eps_c, entity_id=ent_c)

        # Graph edges: Alice -> ProjectX -> Python
        store.insert_edge(ns, ent_a, ent_b, "works_on")
        store.insert_edge(ns, ent_b, ent_c, "uses")

        engine = _make_engine(store)
        output = engine.recall(ns, "Alice", mode="hybrid", limit=8, explain=True)

        result_ids = [r.id for r in output.results]
        # eps_a matches lexically, eps_b and eps_c should be discovered via graph
        assert eps_a in result_ids, "Seed episode should be in results"
        assert eps_c in result_ids, (
            f"eps_c should be discovered via 2-hop graph traversal, got {result_ids}"
        )

    def test_3hop_with_env(self, memory_store: SQLiteStore) -> None:
        """With GRAPHCTX_GRAPH_MAX_HOPS=3, a 3-hop path A->B->C->D works."""
        ns = "test"
        store = memory_store

        # Build 4-node chain
        names = ["Alice", "ProjectX", "Python", "PipTool"]
        contents = [
            "Alice leads the ProjectX initiative",
            "ProjectX is built with Python framework",
            "Python ecosystem includes PipTool",
            "PipTool manages dependencies efficiently",
        ]
        ent_ids = []
        eps_ids = []
        for i, (name, content) in enumerate(zip(names, contents)):
            eps = store.insert_episode(ns, content, "memory")
            ent = store.insert_entity(ns, name, "thing")
            store.insert_provenance(ns, eps, entity_id=ent)
            ent_ids.append(ent)
            eps_ids.append(eps)

        for i in range(3):
            store.insert_edge(ns, ent_ids[i], ent_ids[i + 1], "related")

        # Set max_hops=3 via env
        with patch.dict(os.environ, {"GRAPHCTX_GRAPH_MAX_HOPS": "3"}):
            from graphctx.config import get_config
            config = get_config()
            engine = RecallEngine(store, DeterministicEmbedder(), config=config)
            output = engine.recall(ns, "Alice", mode="hybrid", limit=8, explain=True)

        result_ids = [r.id for r in output.results]
        # 3-hop: Alice(0) -> ProjectX(1) -> Python(2) -> PipTool(3)
        assert eps_ids[3] in result_ids, (
            f"3-hop episode should be reachable with max_hops=3, got {result_ids}"
        )

    def test_3hop_blocked_at_default(self, memory_store: SQLiteStore) -> None:
        """With default max_hops=2, a 3-hop target is NOT reachable."""
        ns = "test"
        store = memory_store

        names = ["Alice", "ProjectX", "Python", "PipTool"]
        contents = [
            "Alice leads the ProjectX initiative",
            "ProjectX is built with Python framework",
            "Python ecosystem includes PipTool",
            "PipTool manages dependencies efficiently",
        ]
        ent_ids = []
        eps_ids = []
        for i, (name, content) in enumerate(zip(names, contents)):
            eps = store.insert_episode(ns, content, "memory")
            ent = store.insert_entity(ns, name, "thing")
            store.insert_provenance(ns, eps, entity_id=ent)
            ent_ids.append(ent)
            eps_ids.append(eps)

        for i in range(3):
            store.insert_edge(ns, ent_ids[i], ent_ids[i + 1], "related")

        engine = _make_engine(store)
        output = engine.recall(ns, "Alice", mode="hybrid", limit=8, explain=True)

        result_ids = [r.id for r in output.results]
        # With default max_hops=2, eps_ids[3] (hop 3) should NOT be reachable
        assert eps_ids[3] not in result_ids, (
            f"3-hop episode should NOT be reachable with default max_hops=2"
        )

    def test_cycle_protection(self, memory_store: SQLiteStore) -> None:
        """A->B->A cycle does not cause infinite loop in recall."""
        ns = "test"
        store = memory_store

        eps_a = store.insert_episode(ns, "Alpha connects to Beta", "memory")
        ent_a = store.insert_entity(ns, "Alpha", "thing")
        store.insert_provenance(ns, eps_a, entity_id=ent_a)

        eps_b = store.insert_episode(ns, "Beta connects back to Alpha", "memory")
        ent_b = store.insert_entity(ns, "Beta", "thing")
        store.insert_provenance(ns, eps_b, entity_id=ent_b)

        # Bidirectional edges create a cycle
        store.insert_edge(ns, ent_a, ent_b, "link")
        store.insert_edge(ns, ent_b, ent_a, "link_back")

        engine = _make_engine(store)
        # Should complete without hanging
        output = engine.recall(ns, "Alpha", mode="hybrid", limit=8, explain=True)

        result_ids = [r.id for r in output.results]
        assert eps_a in result_ids
        assert eps_b in result_ids

    def test_tombstoned_excluded(self, memory_store: SQLiteStore) -> None:
        """Tombstoned entity in path blocks traversal beyond it."""
        ns = "test"
        store = memory_store

        eps_a = store.insert_episode(ns, "Alice works on ProjectX", "memory")
        ent_a = store.insert_entity(ns, "Alice", "person")
        store.insert_provenance(ns, eps_a, entity_id=ent_a)

        eps_b = store.insert_episode(ns, "ProjectX uses Python", "memory")
        ent_b = store.insert_entity(ns, "ProjectX", "project")
        store.insert_provenance(ns, eps_b, entity_id=ent_b)

        eps_c = store.insert_episode(ns, "Python requires pip for install", "memory")
        ent_c = store.insert_entity(ns, "Python", "language")
        store.insert_provenance(ns, eps_c, entity_id=ent_c)

        store.insert_edge(ns, ent_a, ent_b, "works_on")
        store.insert_edge(ns, ent_b, ent_c, "uses")

        # Tombstone entity B
        conn = store._conn()
        conn.execute("UPDATE entity SET deleted_at = '2025-01-01' WHERE id = ?", (ent_b,))
        conn.commit()

        engine = _make_engine(store)
        output = engine.recall(ns, "Alice", mode="hybrid", limit=8, explain=True)

        result_ids = [r.id for r in output.results]
        assert eps_a in result_ids, "Seed episode should still be found"
        # B is tombstoned, so C should not be reachable
        assert eps_c not in result_ids, (
            "eps_c should not be reachable when intermediate entity is tombstoned"
        )

    def test_depth_decay_scores(self, memory_store: SQLiteStore) -> None:
        """Hop-2 episode gets lower graph score than hop-1 episode."""
        ns = "test"
        store = memory_store

        eps_a = store.insert_episode(ns, "Alice works on ProjectX", "memory")
        ent_a = store.insert_entity(ns, "Alice", "person")
        store.insert_provenance(ns, eps_a, entity_id=ent_a)

        eps_b = store.insert_episode(ns, "ProjectX uses Python", "memory")
        ent_b = store.insert_entity(ns, "ProjectX", "project")
        store.insert_provenance(ns, eps_b, entity_id=ent_b)

        eps_c = store.insert_episode(ns, "Python requires pip for install", "memory")
        ent_c = store.insert_entity(ns, "Python", "language")
        store.insert_provenance(ns, eps_c, entity_id=ent_c)

        store.insert_edge(ns, ent_a, ent_b, "works_on")
        store.insert_edge(ns, ent_b, ent_c, "uses")

        engine = _make_engine(store)
        output = engine.recall(ns, "Alice", mode="hybrid", limit=8, explain=True)

        result_map = {r.id: r for r in output.results}
        assert eps_b in result_map
        assert eps_c in result_map

        # eps_b (hop 1) should have higher score than eps_c (hop 2)
        # because eps_c has no lexical match and lower graph path score
        assert result_map[eps_b].score >= result_map[eps_c].score, (
            f"hop-1 score {result_map[eps_b].score} should be >= hop-2 score {result_map[eps_c].score}"
        )

    def test_max_expansions_limit(self, memory_store: SQLiteStore) -> None:
        """BFS respects max_expansions config."""
        ns = "test"
        store = memory_store

        # Create seed episode
        eps_seed = store.insert_episode(ns, "Center node connects to many", "memory")
        ent_seed = store.insert_entity(ns, "Center", "thing")
        store.insert_provenance(ns, eps_seed, entity_id=ent_seed)

        # Create 10 leaf episodes connected to seed
        leaf_eps_ids = []
        for i in range(10):
            eps = store.insert_episode(ns, f"Leaf node number {i} data", "memory")
            ent = store.insert_entity(ns, f"Leaf{i}", "thing")
            store.insert_provenance(ns, eps, entity_id=ent)
            store.insert_edge(ns, ent_seed, ent, "connects")
            leaf_eps_ids.append(eps)

        # Set max_expansions=2
        with patch.dict(os.environ, {"GRAPHCTX_GRAPH_MAX_EXPANSIONS": "2"}):
            from graphctx.config import get_config
            config = get_config()
            engine = RecallEngine(store, DeterministicEmbedder(), config=config)
            output = engine.recall(ns, "Center", mode="hybrid", limit=20, explain=True)

        # With max_expansions=2, only 2 leaf entities should be discovered
        graph_discovered = [
            r.id for r in output.results
            if r.id != eps_seed and r.id in leaf_eps_ids
        ]
        assert len(graph_discovered) <= 2, (
            f"Expected at most 2 graph-discovered results, got {len(graph_discovered)}"
        )

    def test_graph_paths_in_explanation(self, memory_store: SQLiteStore) -> None:
        """Explanation includes graph_paths, hop_count, and path_score."""
        ns = "test"
        store = memory_store

        eps_a = store.insert_episode(ns, "Alice works on ProjectX", "memory")
        ent_a = store.insert_entity(ns, "Alice", "person")
        store.insert_provenance(ns, eps_a, entity_id=ent_a)

        eps_b = store.insert_episode(ns, "ProjectX uses Python", "memory")
        ent_b = store.insert_entity(ns, "ProjectX", "project")
        store.insert_provenance(ns, eps_b, entity_id=ent_b)

        eps_c = store.insert_episode(ns, "Python requires pip for install", "memory")
        ent_c = store.insert_entity(ns, "Python", "language")
        store.insert_provenance(ns, eps_c, entity_id=ent_c)

        store.insert_edge(ns, ent_a, ent_b, "works_on")
        store.insert_edge(ns, ent_b, ent_c, "uses")

        engine = _make_engine(store)
        output = engine.recall(ns, "Alice", mode="hybrid", limit=8, explain=True)

        explanation = output.explanation

        # hop_count should reflect max hops traversed
        assert explanation.hop_count >= 2, (
            f"Expected hop_count >= 2, got {explanation.hop_count}"
        )

        # path_score should be positive
        assert explanation.path_score > 0, (
            f"Expected positive path_score, got {explanation.path_score}"
        )

        # graph_paths should contain path entries
        assert len(explanation.graph_paths) > 0, (
            "Expected non-empty graph_paths in explanation"
        )

        # Each graph path entry should have expected fields
        for gp in explanation.graph_paths:
            assert "entity_id" in gp
            assert "hop" in gp
            assert "score" in gp

        # retrieval_breakdown entries should include graph_paths for BFS-discovered episodes
        breakdown = explanation.retrieval_breakdown
        graph_episodes = [e for e in breakdown if e.get("graph_paths")]
        assert len(graph_episodes) > 0, (
            "Expected at least one retrieval_breakdown entry with graph_paths"
        )

    def test_fast_mode_no_graph(self, memory_store: SQLiteStore) -> None:
        """Fast mode does not run BFS traversal."""
        ns = "test"
        store = memory_store

        eps_a = store.insert_episode(ns, "Alice works on ProjectX", "memory")
        ent_a = store.insert_entity(ns, "Alice", "person")
        store.insert_provenance(ns, eps_a, entity_id=ent_a)

        eps_b = store.insert_episode(ns, "ProjectX uses Python", "memory")
        ent_b = store.insert_entity(ns, "ProjectX", "project")
        store.insert_provenance(ns, eps_b, entity_id=ent_b)

        store.insert_edge(ns, ent_a, ent_b, "works_on")

        engine = _make_engine(store)
        output = engine.recall(ns, "Alice", mode="fast", limit=8, explain=True)

        # In fast mode, graph_paths should be empty
        assert output.explanation.graph_paths == []
        assert output.explanation.hop_count == 0

    def test_custom_depth_decay(self, memory_store: SQLiteStore) -> None:
        """Custom GRAPHCTX_GRAPH_DEPTH_DECAY is applied."""
        ns = "test"
        store = memory_store

        eps_a = store.insert_episode(ns, "Alice works on ProjectX", "memory")
        ent_a = store.insert_entity(ns, "Alice", "person")
        store.insert_provenance(ns, eps_a, entity_id=ent_a)

        eps_b = store.insert_episode(ns, "ProjectX uses Python", "memory")
        ent_b = store.insert_entity(ns, "ProjectX", "project")
        store.insert_provenance(ns, eps_b, entity_id=ent_b)

        eps_c = store.insert_episode(ns, "Python requires pip for install", "memory")
        ent_c = store.insert_entity(ns, "Python", "language")
        store.insert_provenance(ns, eps_c, entity_id=ent_c)

        store.insert_edge(ns, ent_a, ent_b, "works_on")
        store.insert_edge(ns, ent_b, ent_c, "uses")

        # Use a very steep decay so hop-2 score is much lower
        with patch.dict(os.environ, {"GRAPHCTX_GRAPH_DEPTH_DECAY": "0.3"}):
            from graphctx.config import get_config
            config = get_config()
            assert config.depth_decay == 0.3
            engine = RecallEngine(store, DeterministicEmbedder(), config=config)
            output = engine.recall(ns, "Alice", mode="hybrid", limit=8, explain=True)

        # With decay=0.3, hop-2 score = 0.09, which is very low
        # eps_c may or may not appear depending on other score components,
        # but the graph path score for hop-2 should be 0.3^2 = 0.09
        result_map = {r.id: r for r in output.results}
        if eps_c in result_map:
            # The score breakdown should reflect the steep decay
            breakdown = result_map[eps_c].score_breakdown
            assert breakdown is not None
            assert breakdown.graph_boost < 0.1, (
                f"Expected very low graph_boost with decay=0.3, got {breakdown.graph_boost}"
            )
