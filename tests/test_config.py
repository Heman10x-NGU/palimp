"""Tests for palimp.config — configurable search weights and settings."""

from __future__ import annotations

import pytest

from palimp.config import GraphConfig, SearchWeights, get_config


# ---------------------------------------------------------------------------
# Default weights match plan section 3.3
# ---------------------------------------------------------------------------


class TestDefaultWeights:
    """Default weights must match plan values: 0.35/0.30/0.15/0.05/0.05/0.05/0.05."""

    def test_default_weights(self):
        w = SearchWeights()
        assert w.lexical == pytest.approx(0.35)
        assert w.vector == pytest.approx(0.30)
        assert w.graph == pytest.approx(0.15)
        assert w.recency == pytest.approx(0.05)
        assert w.confidence == pytest.approx(0.05)
        assert w.temporal == pytest.approx(0.05)
        assert w.category == pytest.approx(0.05)

    def test_default_total(self):
        w = SearchWeights()
        assert w.total() == pytest.approx(1.0)

    def test_to_dict_keys(self):
        w = SearchWeights()
        d = w.to_dict()
        assert set(d.keys()) == {
            "lexical", "vector", "graph", "recency",
            "confidence", "temporal", "category",
        }

    def test_default_graph_config(self):
        cfg = GraphConfig()
        assert cfg.max_hops == 2
        assert cfg.depth_decay == pytest.approx(0.55)
        assert cfg.max_expansions == 50
        assert cfg.reranker_endpoint is None
        assert cfg.rerank_top_k == 20

    def test_get_config_returns_fresh_instance(self):
        a = get_config()
        b = get_config()
        assert a is not b  # fresh instance each call


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvOverride:
    """Setting PALIMP_WEIGHT_* env vars must override defaults."""

    def test_env_override_lexical(self, monkeypatch):
        monkeypatch.setenv("PALIMP_WEIGHT_LEXICAL", "0.50")
        w = SearchWeights()
        assert w.lexical == pytest.approx(0.50)
        # Others unchanged
        assert w.vector == pytest.approx(0.30)

    def test_env_override_vector(self, monkeypatch):
        monkeypatch.setenv("PALIMP_WEIGHT_VECTOR", "0.60")
        w = SearchWeights()
        assert w.vector == pytest.approx(0.60)

    def test_env_override_graph_config(self, monkeypatch):
        monkeypatch.setenv("PALIMP_GRAPH_MAX_HOPS", "3")
        monkeypatch.setenv("PALIMP_GRAPH_DEPTH_DECAY", "0.40")
        cfg = GraphConfig()
        assert cfg.max_hops == 3
        assert cfg.depth_decay == pytest.approx(0.40)

    def test_env_override_reranker(self, monkeypatch):
        monkeypatch.setenv("PALIMP_RERANKER_ENDPOINT", "http://localhost:9999")
        cfg = GraphConfig()
        assert cfg.reranker_endpoint == "http://localhost:9999"


# ---------------------------------------------------------------------------
# Invalid values raise clear errors
# ---------------------------------------------------------------------------


class TestInvalidWeights:
    """Negative or non-numeric env values must raise ValueError."""

    def test_negative_weight_raises(self, monkeypatch):
        monkeypatch.setenv("PALIMP_WEIGHT_LEXICAL", "-0.10")
        with pytest.raises(ValueError, match="must be non-negative"):
            SearchWeights()

    def test_non_numeric_weight_raises(self, monkeypatch):
        monkeypatch.setenv("PALIMP_WEIGHT_LEXICAL", "abc")
        with pytest.raises(ValueError, match="Invalid"):
            SearchWeights()

    def test_negative_int_raises(self, monkeypatch):
        monkeypatch.setenv("PALIMP_GRAPH_MAX_HOPS", "-1")
        with pytest.raises(ValueError, match="must be non-negative"):
            GraphConfig()

    def test_non_numeric_int_raises(self, monkeypatch):
        monkeypatch.setenv("PALIMP_GRAPH_MAX_HOPS", "xyz")
        with pytest.raises(ValueError, match="Invalid"):
            GraphConfig()


# ---------------------------------------------------------------------------
# Weights appear in explanation.scoring_config
# ---------------------------------------------------------------------------


class TestWeightsInExplanation:
    """The recall explanation must include scoring_config with active weights."""

    def test_weights_in_explanation(self, memory_store):
        """explanation.scoring_config must match the configured weights."""
        from palimp.embeddings import DeterministicEmbedder
        from palimp.retriever import RecallEngine

        embedder = DeterministicEmbedder()
        engine = RecallEngine(store=memory_store, embedder=embedder)

        # Ingest a memory so there's something to recall
        memory_store.insert_episode(
            ns="test",
            content="Python is a programming language",
            source_type="memory",
        )

        output = engine.recall(
            ns="test",
            query="Python",
            mode="hybrid",
            explain=True,
        )

        sc = output.explanation.scoring_config
        assert sc, "scoring_config should not be empty"
        assert sc["lexical"] == pytest.approx(0.35)
        assert sc["vector"] == pytest.approx(0.30)
        assert sc["graph"] == pytest.approx(0.15)
        assert sc["recency"] == pytest.approx(0.05)
        assert sc["confidence"] == pytest.approx(0.05)
        assert sc["temporal"] == pytest.approx(0.05)
        assert sc["category"] == pytest.approx(0.05)

    def test_env_override_shows_in_explanation(self, memory_store, monkeypatch):
        """When env weights are changed, explanation reflects the new values."""
        from palimp.embeddings import DeterministicEmbedder
        from palimp.retriever import RecallEngine

        monkeypatch.setenv("PALIMP_WEIGHT_LEXICAL", "0.50")

        embedder = DeterministicEmbedder()
        # Config is created fresh per RecallEngine instantiation
        engine = RecallEngine(store=memory_store, embedder=embedder)

        memory_store.insert_episode(
            ns="test",
            content="Rust is a systems language",
            source_type="memory",
        )

        output = engine.recall(
            ns="test",
            query="Rust",
            mode="hybrid",
            explain=True,
        )

        sc = output.explanation.scoring_config
        assert sc["lexical"] == pytest.approx(0.50)
        # Other defaults still present
        assert sc["vector"] == pytest.approx(0.30)
