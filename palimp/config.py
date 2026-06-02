"""Palimp configuration — env-based search weights and settings."""

from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        v = float(val)
        if v < 0:
            raise ValueError(f"{name} must be non-negative, got {v}")
        return v
    except ValueError as e:
        raise ValueError(f"Invalid {name}={val!r}: {e}")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        v = int(val)
        if v < 0:
            raise ValueError(f"{name} must be non-negative, got {v}")
        return v
    except ValueError as e:
        raise ValueError(f"Invalid {name}={val!r}: {e}")


class SearchWeights:
    """Configurable scoring weights loaded from environment variables."""

    def __init__(self) -> None:
        self.lexical = _env_float("PALIMP_WEIGHT_LEXICAL", 0.35)
        self.vector = _env_float("PALIMP_WEIGHT_VECTOR", 0.30)
        self.graph = _env_float("PALIMP_WEIGHT_GRAPH", 0.15)
        self.recency = _env_float("PALIMP_WEIGHT_RECENCY", 0.05)
        self.confidence = _env_float("PALIMP_WEIGHT_CONFIDENCE", 0.05)
        self.temporal = _env_float("PALIMP_WEIGHT_TEMPORAL", 0.05)
        self.category = _env_float("PALIMP_WEIGHT_CATEGORY", 0.05)

    def to_dict(self) -> dict[str, float]:
        return {
            "lexical": self.lexical,
            "vector": self.vector,
            "graph": self.graph,
            "recency": self.recency,
            "confidence": self.confidence,
            "temporal": self.temporal,
            "category": self.category,
        }

    def total(self) -> float:
        return sum(self.to_dict().values())


class GraphConfig:
    """Top-level Palimp configuration combining weights and graph settings."""

    def __init__(self) -> None:
        self.max_hops = _env_int("PALIMP_GRAPH_MAX_HOPS", 2)
        self.depth_decay = _env_float("PALIMP_GRAPH_DEPTH_DECAY", 0.55)
        self.max_expansions = _env_int("PALIMP_GRAPH_MAX_EXPANSIONS", 50)
        self.reranker_endpoint = os.environ.get("PALIMP_RERANKER_ENDPOINT")
        self.rerank_top_k = _env_int("PALIMP_RERANK_TOP_K", 20)
        self.weights = SearchWeights()


def get_config() -> GraphConfig:
    """Return a fresh GraphConfig populated from environment variables."""
    return GraphConfig()
