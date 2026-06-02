"""Optional reranker — local heuristic or HTTP endpoint.

Provides reranking for recall results. When PALIMP_RERANKER_ENDPOINT
is set, scores are boosted by an external cross-encoder.  Otherwise a
lightweight local heuristic reranks by term overlap, entity kind, and
temporal freshness.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx


def rerank_results(
    results: list[Any],
    query: str,
    endpoint: Optional[str] = None,
    top_k: int = 20,
) -> list[Any]:
    """Rerank results using endpoint or local heuristic.

    Parameters
    ----------
    results : list
        RecallResult objects to rerank.
    query : str
        The original query string.
    endpoint : str, optional
        HTTP endpoint for cross-encoder reranking (OpenAI-compatible).
        If ``None``, falls back to local heuristic.
    top_k : int
        Maximum number of results to rerank (cheaper to limit).

    Returns
    -------
    list
        Reranked results (may be shorter than input if top_k applies).
    """
    if not results:
        return results

    candidates = results[:top_k]

    if endpoint:
        return _http_rerank(candidates, query, endpoint)

    return _heuristic_rerank(candidates, query)


def _heuristic_rerank(results: list[Any], query: str) -> list[Any]:
    """Local heuristic reranking: matched terms + entity overlap + temporal match."""
    query_terms = set(query.lower().split())

    def rerank_score(r: Any) -> float:
        content = r.content if hasattr(r, "content") else r.get("content", "")
        content_lower = content.lower()

        # Term overlap
        content_terms = set(content_lower.split())
        term_overlap = len(query_terms & content_terms) / max(len(query_terms), 1)

        # Entity/knowledge bonus
        entity_bonus = 0.0
        kind = r.kind if hasattr(r, "kind") else r.get("kind", "")
        if kind == "knowledge":
            entity_bonus = 0.1

        # Temporal match bonus — prefer current facts
        temporal_bonus = 0.0
        temporal_status = r.temporal_status if hasattr(r, "temporal_status") else r.get("temporal_status", "")
        if temporal_status == "current":
            temporal_bonus = 0.1

        return term_overlap + entity_bonus + temporal_bonus

    results.sort(key=rerank_score, reverse=True)
    return results


def _http_rerank(results: list[Any], query: str, endpoint: str) -> list[Any]:
    """HTTP-based reranking via OpenAI-compatible endpoint."""
    try:
        documents = [
            r.content if hasattr(r, "content") else r.get("content", "")
            for r in results
        ]
        resp = httpx.post(
            endpoint,
            json={"query": query, "documents": documents},
            timeout=30,
        )
        resp.raise_for_status()
        scores = resp.json().get("scores", [])
        for r, score in zip(results, scores):
            if hasattr(r, "score"):
                r.score = r.score * 0.7 + score * 0.3
        results.sort(key=lambda r: r.score if hasattr(r, "score") else 0, reverse=True)
    except Exception:
        pass  # fallback to original order
    return results
