"""Query expansion for thinking-mode recall.

Splits complex queries into sub-queries, extracts entities, and merges
results from multiple expansions.
"""

from __future__ import annotations

import re
from typing import Any


def _flip_pronouns(q: str) -> str:
    """Flip first-person pronouns to third-person for matching transcripts."""
    replacements = [
        (r"\bI\b", "the user"),
        (r"\bmy\b", "the user's"),
        (r"\bme\b", "the user"),
        (r"\bwe\b", "the users"),
        (r"\bour\b", "the users'"),
        (r"\bus\b", "the users"),
        (r"\bMy\b", "the user's"),
        (r"\bMe\b", "the user"),
        (r"\bWe\b", "the users"),
        (r"\bOur\b", "the users'"),
        (r"\bUs\b", "the users"),
    ]
    res = q
    for pattern, rep in replacements:
        res = re.sub(pattern, rep, res)
    return res


def has_temporal_signal(query: str) -> bool:
    """Detect if query has cues referring to the past or general time/tense changes."""
    temporal_keywords = [
        "used to", "before", "previously", "in the past", "back when",
        "historically", "last year", "formerly", "old", "prior to",
        "when did", "when was", "what was", "what were", "old address",
        "was", "were", "had", "previously"
    ]
    q_lower = query.lower()
    return any(kw in q_lower for kw in temporal_keywords)


def _shift_tenses(q: str) -> str:
    """Shift present-tense verbs to past-tense for querying historical data."""
    def rep(match):
        val = match.group().lower()
        mapping = {"is": "was", "are": "were", "has": "had", "have": "had"}
        res = mapping.get(val, val)
        if match.group().istitle():
            return res.title()
        return res
    return re.sub(r"\b(is|are|has|have)\b", rep, q, flags=re.IGNORECASE)


def _extract_keywords(query: str) -> list[str]:
    """Extract stopword-filtered keywords to build cleaner search representations."""
    stop_question = {
        "what", "where", "who", "when", "which", "how", "did", "does",
        "do", "is", "are", "was", "were", "the", "a", "an", "of", "in",
        "to", "for", "on", "at", "by", "with", "from", "and", "or", "my",
        "i", "you", "me", "we", "our", "your", "their", "its", "tell",
        "about", "can", "could", "would", "should", "have", "had",
        "be", "been"
    }
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", query)
    return [t for t in tokens if t.lower() not in stop_question]


def expand_query(query: str) -> list[str]:
    """Split complex query into sub-queries.

    Splits on: and, or, but, semicolons, commas with conjunctions.
    Also extracts named entities, flips pronouns, shifts tenses, and creates keyword variants.
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

    # Flipping pronouns to match speaker dialogue/fact extraction perspectives
    flipped_queries: list[str] = []
    for sq in sub_queries:
        flipped = _flip_pronouns(sq)
        if flipped != sq:
            flipped_queries.append(flipped)

    # Tense shifting for queries with past temporal signals
    tense_queries: list[str] = []
    all_current = sub_queries + flipped_queries
    for sq in all_current:
        if has_temporal_signal(sq):
            shifted = _shift_tenses(sq)
            if shifted != sq:
                tense_queries.append(shifted)

    # Keyword-only query expansion to strip conversational noise
    keyword_queries: list[str] = []
    kw_list = _extract_keywords(query)
    if len(kw_list) >= 2:
        keyword_queries.append(" ".join(kw_list))

    # Extract entities for targeted queries (capitalized words > 2 chars)
    _STOPWORDS = {
        "what", "where", "when", "who", "why", "how", "which",
        "whose", "whom", "are", "was", "were", "been", "does",
        "did", "has", "have", "had", "can", "could", "should",
        "would", "will", "shall", "may", "might", "must", "the",
        "and", "but", "for", "nor", "yet", "you", "your", "they",
        "them", "their", "she", "her", "him", "his", "our", "its",
        "this", "that", "these", "those"
    }
    entity_queries: list[str] = []
    words = query.split()
    for word in words:
        if word[0:1].isupper() and len(word) > 2:
            # Strip trailing punctuation
            clean = re.sub(r"[^\w]+$", "", word)
            if clean and len(clean) > 2 and clean.lower() not in _STOPWORDS:
                entity_queries.append(clean)

    all_queries = (
        sub_queries
        + temporal_expansions
        + flipped_queries
        + tense_queries
        + keyword_queries
        + entity_queries
    )
    
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
