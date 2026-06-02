"""Query expansion for thinking-mode recall.

Splits complex queries into sub-queries, extracts entities, and merges
results from multiple expansions.
"""

from __future__ import annotations

import re
from typing import Any


def expand_query(query: str) -> list[str]:
    """Split complex query into sub-queries.

    Splits on: and, or, but, semicolons, commas with conjunctions.
    Also extracts named entities for targeted sub-queries.
    """
    # Split on conjunctions and punctuation
    splits = re.split(
        r"\s+(?:and|or|but)\s+|;\s*|,\s+(?:and|or)\s+",
        query,
        flags=re.IGNORECASE,
    )

    sub_queries = [s.strip() for s in splits if s.strip() and len(s.strip()) > 3]

    if not sub_queries:
        sub_queries = [query]

    # Add temporal expansions
    temporal_expansions: list[str] = []
    for sq in sub_queries[:]:  # copy to avoid modifying during iteration
        if any(cue in sq.lower() for cue in ["now", "currently", "today"]):
            cleaned = sq
            for word in ["now", "currently", "today"]:
                cleaned = re.sub(r"\b" + word + r"\b", "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) > 3:
                temporal_expansions.append(cleaned)
        if any(cue in sq.lower() for cue in ["before", "previously", "formerly"]):
            temporal_expansions.append(sq)

    # Extract entities for targeted queries (capitalized words > 2 chars)
    entity_queries: list[str] = []
    words = query.split()
    for word in words:
        if word[0:1].isupper() and len(word) > 2:
            # Strip trailing punctuation
            clean = re.sub(r"[^\w]+$", "", word)
            if clean and len(clean) > 2:
                entity_queries.append(clean)

    all_queries = sub_queries + temporal_expansions + entity_queries
    # Dedupe while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for q in all_queries:
        q_lower = q.lower().strip()
        if q_lower and q_lower not in seen and len(q_lower) > 2:
            seen.add(q_lower)
            result.append(q)

    return result if result else [query]


def merge_expansion_results(results_per_query: list[list[Any]]) -> list[Any]:
    """Merge results from multiple sub-queries, dedup by episode ID, keep best score."""
    by_id: dict[str, Any] = {}
    for results in results_per_query:
        for r in results:
            rid = r.id if hasattr(r, "id") else r.get("id", "")
            existing = by_id.get(rid)
            if not existing:
                by_id[rid] = r
            elif hasattr(r, "score") and hasattr(existing, "score"):
                if r.score > existing.score:
                    by_id[rid] = r
            elif hasattr(r, "score"):
                by_id[rid] = r
    return list(by_id.values())
