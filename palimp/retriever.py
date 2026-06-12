"""Palimp retrieval / recall engine.

Implements hybrid recall across memories and knowledge with:
- FTS5 lexical search
- Deterministic/HTTP vector embeddings + cosine similarity
- Multi-hop BFS graph traversal with depth decay
- Ebbinghaus forgetting curve decay (with pin support)
- Confidence scoring (avg entity/claim confidence from provenance)
- Three modes: fast, hybrid, thinking
- Provenance attachment and conflict/supersession warnings
- Explainable retrieval with per-component score breakdown
- Query term tracking and matched-term analysis
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from palimp.config import GraphConfig, get_config
from palimp.decay import DEFAULT_STABILITY, compute_decay_score
from palimp.embeddings import BaseEmbedder
from palimp.models import CATEGORY_PRIORITY, RecallExplanation, RecallResult, ScoreBreakdown
from palimp.query_expansion import expand_query
from palimp.reranker import rerank_results
from palimp.storage import SQLiteStore
from palimp.graph_traversal import bfs_graph_traversal, get_episodes_for_entities
from palimp.temporal import (
    classify_temporal_status,
    detect_temporal_cues,
    should_include_in_mode,
    temporal_score_boost,
)

# Scoring weights are now loaded from palimp.config (env-based).
# Kept as module-level fallbacks for backward compatibility.
WEIGHT_LEXICAL = 0.60
WEIGHT_VECTOR = 0.15
WEIGHT_GRAPH = 0.10
WEIGHT_RECENCY = 0.0
WEIGHT_CONFIDENCE = 0.05

# Model name used for deterministic embeddings
_DETERMINISTIC_MODEL = "deterministic-sha256"

# Minimum word length for query term extraction (filters stop words)
_MIN_QUERY_TERM_LENGTH = 2


def conn_execute_safe(store: SQLiteStore, sql: str, params: tuple) -> Any:
    """Execute a query safely and return the first row or None."""
    conn = store._conn()
    return conn.execute(sql, params).fetchone()


def _extract_query_terms(query: str) -> list[str]:
    """Extract meaningful terms from a query for lexical matching analysis.

    Returns lowercased, deduplicated tokens sorted alphabetically.
    Filters out very short tokens (< 2 chars) to reduce noise.
    """
    # Split on non-alphanumeric, lowercase, filter short tokens
    tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    terms = sorted(set(t for t in tokens if len(t) >= _MIN_QUERY_TERM_LENGTH))
    return terms


def _compute_matched_terms(query_terms: list[str], content: str) -> list[str]:
    """Determine which query terms appear in the content.

    Returns the subset of query_terms found in content (case-insensitive).
    """
    content_lower = content.lower()
    return [t for t in query_terms if t in content_lower]


def _parse_iso(ts: str) -> datetime:
    """Parse ISO timestamps, accepting the common trailing-Z form."""
    normalized = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _episode_recallable(episode: dict[str, Any], as_of: Optional[str] = None) -> bool:
    """Return whether an episode can participate in recall.

    Tombstones preserve audit history but must not surface. Governed forgetting
    is softer: the row and provenance remain inspectable, while recall excludes
    explicitly forgotten or expired episodes.
    """
    if episode.get("deleted_at") is not None or episode.get("tombstoned_at") is not None:
        return False
    if bool(episode.get("is_forgotten", 0)):
        return False
    forget_after = episode.get("forget_after")
    if forget_after:
        ref = _parse_iso(as_of) if as_of else datetime.now(timezone.utc)
        try:
            if ref >= _parse_iso(forget_after):
                return False
        except ValueError:
            return False
    return True


@dataclass
class RecallOutput:
    """Container for recall results plus optional explanation."""

    results: list[RecallResult]
    explanation: RecallExplanation = field(default_factory=RecallExplanation)


class RecallEngine:
    """Unified recall engine over memories and knowledge.

    Parameters
    ----------
    store : SQLiteStore
        The storage backend.
    embedder : BaseEmbedder
        The embedding provider (deterministic or HTTP).
    """

    def __init__(
        self,
        store: SQLiteStore,
        embedder: BaseEmbedder,
        config: Optional[GraphConfig] = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._config = config or get_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recall(
        self,
        ns: str,
        query: str,
        mode: Literal["fast", "hybrid", "thinking"] = "hybrid",
        search_mode: Literal["lexical", "vector", "graph", "hybrid"] = "hybrid",
        limit: int = 8,
        max_tokens: Optional[int] = None,
        include_provenance: bool = True,
        explain: bool = False,
        agent_tier: Optional[str] = None,
        as_of: Optional[str] = None,
        temporal_mode: str = "auto",
    ) -> RecallOutput:
        """Recall memories and knowledge for *ns* matching *query*.

        Parameters
        ----------
        ns : str
            Namespace to search within.
        query : str
            The natural-language query.
        mode : {"fast", "hybrid", "thinking"}
            Recall mode.
        limit : int
            Maximum number of results.
        include_provenance : bool
            Whether to attach provenance records.
        explain : bool
            If True, populate score_breakdown and why_retrieved on each result
            and build a full RecallExplanation.
        agent_tier : str, optional
            Agent tier for context management.
        as_of : str, optional
            ISO timestamp for temporal reference time.
        temporal_mode : str
            Temporal filtering mode: auto, current, historical, all.

        Returns
        -------
        RecallOutput
            Contains ``results`` (sorted by combined score descending) and
            ``explanation`` (populated when *explain* is True).
        """
        empty_output = RecallOutput(results=[], explanation=RecallExplanation())
        if not query or not query.strip():
            return empty_output

        total_start = time.monotonic()

        # Extract query terms for explanation tracking
        query_terms = _extract_query_terms(query)

        # Temporal cue detection: if auto, infer from query
        effective_temporal_mode = temporal_mode
        if temporal_mode == "auto":
            cue = detect_temporal_cues(query)
            if cue:
                effective_temporal_mode = cue

        # 1. FTS5 lexical results (guarded by search_mode)
        fts_start = time.monotonic()
        lexical_scores: dict[str, float] = {}
        if search_mode in ("lexical", "hybrid"):
            lexical_scores = self._search_lexical(ns, query, limit)
        fts_ms = (time.monotonic() - fts_start) * 1000

        # 2. Vector similarity results (guarded by search_mode)
        embed_start = time.monotonic()
        vector_scores: dict[str, float] = {}
        if search_mode in ("vector", "hybrid"):
            vector_scores = self._search_vector(ns, query)
        embedding_ms = (time.monotonic() - embed_start) * 1000
        vector_ms = 0.0

        # Track query expansions for explanation
        query_expansions: list[str] = []

        # Thinking mode: expand query into sub-queries, search each, merge
        if mode == "thinking":
            sub_queries = expand_query(query)
            if len(sub_queries) > 1:
                query_expansions = sub_queries
                for sq in sub_queries:
                    sq_lex = self._search_lexical(ns, sq, limit)
                    sq_vec = self._search_vector(ns, sq)
                    # Merge: keep best score per episode
                    for eid, score in sq_lex.items():
                        if score > lexical_scores.get(eid, 0.0):
                            lexical_scores[eid] = score
                    for eid, score in sq_vec.items():
                        if score > vector_scores.get(eid, 0.0):
                            vector_scores[eid] = score

        # Merge candidate episode IDs
        all_episode_ids: set[str] = set()
        all_episode_ids.update(lexical_scores.keys())
        all_episode_ids.update(vector_scores.keys())

        if not all_episode_ids:
            return empty_output

        # Load episodes and filter out tombstoned/deleted/forgotten/expired.
        episodes = self._store.get_episodes_by_ids(list(all_episode_ids))
        live_episodes: dict[str, dict[str, Any]] = {}
        for ep in episodes:
            if _episode_recallable(ep, as_of):
                live_episodes[ep["id"]] = ep

        # Profile memories are purpose-layer context: they are eligible even
        # when the task query does not lexically match them. They still pass
        # normal safety/lifecycle filters and scoring.
        if mode in ("hybrid", "thinking"):
            for ep in self._store.get_profile_episodes(ns):
                if _episode_recallable(ep, as_of):
                    live_episodes.setdefault(ep["id"], ep)

        if not live_episodes:
            return empty_output

        # 3. Graph traversal + score + rank + explanation
        return self._score_and_rank(
            ns=ns,
            query=query,
            live_episodes=live_episodes,
            lexical_scores=lexical_scores,
            vector_scores=vector_scores,
            mode=mode,
            search_mode=search_mode,
            limit=limit,
            max_tokens=max_tokens,
            include_provenance=include_provenance,
            explain=explain,
            as_of=as_of,
            temporal_mode=effective_temporal_mode,
            query_expansions=query_expansions,
            query_terms=query_terms,
            fts_ms=fts_ms,
            embedding_ms=embedding_ms,
            total_start=total_start,
            agent_tier=agent_tier,
        )

    def recall_refine(
        self,
        ns: str,
        query: str,
        previous_result_ids: list[str],
        mode: Literal["fast", "hybrid", "thinking"] = "hybrid",
        search_mode: Literal["lexical", "vector", "graph", "hybrid"] = "hybrid",
        limit: int = 8,
        max_tokens: Optional[int] = None,
        include_provenance: bool = True,
        explain: bool = False,
        as_of: Optional[str] = None,
        temporal_mode: str = "auto",
    ) -> RecallOutput:
        """Refine previous recall results with a new query.

        Re-scores only the episodes identified by *previous_result_ids*,
        running fresh lexical and vector searches but restricting the
        candidate set to the given IDs.  Graph traversal is skipped
        (narrowing, not discovery).

        Parameters
        ----------
        ns : str
            Namespace to search within.
        query : str
            The refinement query.
        previous_result_ids : list[str]
            Episode IDs from a prior recall call to narrow against.
        mode, search_mode, limit, max_tokens, include_provenance, explain, as_of, temporal_mode
            Same semantics as :meth:`recall`.
        """
        if not previous_result_ids:
            return self.recall(
                ns, query, mode=mode, search_mode=search_mode, limit=limit,
                max_tokens=max_tokens, include_provenance=include_provenance,
                explain=explain, as_of=as_of, temporal_mode=temporal_mode,
            )

        if not query or not query.strip():
            return RecallOutput(results=[], explanation=RecallExplanation())

        total_start = time.monotonic()
        query_terms = _extract_query_terms(query)

        # Temporal cue detection
        effective_temporal_mode = temporal_mode
        if temporal_mode == "auto":
            cue = detect_temporal_cues(query)
            if cue:
                effective_temporal_mode = cue

        # Build allowed-ID set
        allowed_ids = set(previous_result_ids)

        # Run FTS5 on full namespace, keep only matching IDs
        fts_start = time.monotonic()
        lexical_scores: dict[str, float] = {}
        if search_mode in ("lexical", "hybrid"):
            raw_lexical = self._search_lexical(ns, query, limit=limit * 3)
            lexical_scores = {eid: s for eid, s in raw_lexical.items() if eid in allowed_ids}
        fts_ms = (time.monotonic() - fts_start) * 1000

        # Run vector search, keep only matching IDs
        embed_start = time.monotonic()
        vector_scores: dict[str, float] = {}
        if search_mode in ("vector", "hybrid"):
            raw_vector = self._search_vector(ns, query)
            vector_scores = {eid: s for eid, s in raw_vector.items() if eid in allowed_ids}
        embedding_ms = (time.monotonic() - embed_start) * 1000

        # Load episodes — only those in allowed_ids
        candidate_ids = set(lexical_scores.keys()) | set(vector_scores.keys())
        # Also load any allowed_ids that had zero search scores (they still
        # participate via decay / confidence / profile boosts)
        candidate_ids = candidate_ids | allowed_ids
        episodes = self._store.get_episodes_by_ids(list(candidate_ids))
        live_episodes: dict[str, dict[str, Any]] = {}
        for ep in episodes:
            if ep["id"] in allowed_ids and _episode_recallable(ep, as_of):
                live_episodes[ep["id"]] = ep

        if not live_episodes:
            return RecallOutput(
                results=[],
                explanation=RecallExplanation(refined_from_count=len(previous_result_ids)),
            )

        # Score and rank (reuse shared pipeline)
        return self._score_and_rank(
            ns=ns,
            query=query,
            live_episodes=live_episodes,
            lexical_scores=lexical_scores,
            vector_scores=vector_scores,
            mode=mode,
            search_mode=search_mode,
            limit=limit,
            max_tokens=max_tokens,
            include_provenance=include_provenance,
            explain=explain,
            as_of=as_of,
            temporal_mode=effective_temporal_mode,
            query_expansions=[],
            query_terms=query_terms,
            fts_ms=fts_ms,
            embedding_ms=embedding_ms,
            total_start=total_start,
            refined_from_count=len(previous_result_ids),
        )

    # ------------------------------------------------------------------
    # Shared scoring + ranking pipeline
    # ------------------------------------------------------------------

    def _score_and_rank(
        self,
        ns: str,
        query: str,
        live_episodes: dict[str, dict[str, Any]],
        lexical_scores: dict[str, float],
        vector_scores: dict[str, float],
        mode: Literal["fast", "hybrid", "thinking"],
        search_mode: str,
        limit: int,
        max_tokens: Optional[int],
        include_provenance: bool,
        explain: bool,
        as_of: Optional[str],
        temporal_mode: str,
        query_expansions: list[str],
        query_terms: list[str],
        fts_ms: float,
        embedding_ms: float,
        total_start: float,
        refined_from_count: int = 0,
        agent_tier: Optional[str] = None,
    ) -> RecallOutput:
        """Score, rank, filter, and build explanation for live episodes.

        This is the shared tail of :meth:`recall` and :meth:`recall_refine`.
        """
        # Normalise lexical and vector scores to 0-1 across live candidates
        live_lexical = {eid: lexical_scores.get(eid, 0.0) for eid in live_episodes}
        live_vector = {eid: vector_scores.get(eid, 0.0) for eid in live_episodes}
        norm_lexical = self._normalize_scores(live_lexical)
        norm_vector = self._normalize_scores(live_vector)

        # Multi-hop graph traversal (BFS from seed entities)
        graph_start = time.monotonic()
        graph_path_scores: dict[str, float] = {}
        graph_paths_data: dict[str, list[dict[str, Any]]] = {}
        max_hop_count = 0
        best_path_score = 0.0
        vector_ms = 0.0

        if search_mode in ("graph", "hybrid") and mode in ("hybrid", "thinking"):
            seed_entity_ids: list[str] = []
            initial_combined: dict[str, float] = {}
            for eid in live_episodes:
                initial_combined[eid] = norm_lexical.get(eid, 0.0) + norm_vector.get(eid, 0.0)
            top_eps = sorted(initial_combined, key=lambda k: initial_combined[k], reverse=True)[:10]

            top_entities = self._store.get_entities_for_episodes(top_eps)
            for eid in top_eps:
                for ent in top_entities.get(eid, []):
                    if ent.get("deleted_at") is None:
                        seed_entity_ids.append(ent["id"])

            if seed_entity_ids:
                bfs_result = bfs_graph_traversal(
                    store=self._store,
                    ns=ns,
                    seed_entity_ids=seed_entity_ids,
                    max_hops=self._config.max_hops,
                    depth_decay=self._config.depth_decay,
                    max_expansions=self._config.max_expansions,
                )

                bfs_entity_ids = list(bfs_result.keys())
                entity_episodes = get_episodes_for_entities(self._store, bfs_entity_ids)

                for entity_id, bfs_info in bfs_result.items():
                    ep_ids = entity_episodes.get(entity_id, [])
                    for ep_id in ep_ids:
                        score = bfs_info["score"]
                        hop = bfs_info["hop"]

                        if ep_id not in graph_path_scores or score > graph_path_scores[ep_id]:
                            graph_path_scores[ep_id] = score

                        if ep_id not in graph_paths_data:
                            graph_paths_data[ep_id] = []
                        graph_paths_data[ep_id].append({
                            "entity_id": entity_id,
                            "hop": hop,
                            "score": round(score, 4),
                            "edge_relation": bfs_info.get("edge_relation", ""),
                        })

                        if hop > max_hop_count:
                            max_hop_count = hop
                        if score > best_path_score:
                            best_path_score = score

                new_ep_ids = set(graph_path_scores.keys()) - set(live_episodes.keys())
                if new_ep_ids:
                    new_eps = self._store.get_episodes_by_ids(list(new_ep_ids))
                    for ep in new_eps:
                        if _episode_recallable(ep, as_of):
                            live_episodes[ep["id"]] = ep
                            norm_lexical[ep["id"]] = 0.0
                            norm_vector[ep["id"]] = 0.0

        graph_ms = (time.monotonic() - graph_start) * 1000

        # Decay score
        decay_scores: dict[str, float] = {}
        if mode in ("hybrid", "thinking"):
            decay_scores = self._compute_decay_scores(live_episodes)

        # Confidence score
        confidence_scores: dict[str, float] = {}
        if mode in ("hybrid", "thinking"):
            confidence_scores = self._compute_confidence_scores(ns, live_episodes)

        # Category scores and trigger boosts
        category_scores: dict[str, float] = {}
        episode_categories: dict[str, str] = {}
        trigger_boosts: dict[str, float] = {}
        if mode in ("hybrid", "thinking"):
            episode_categories = self._store.get_memory_categories_batch(list(live_episodes.keys()))
            for eid, cat in episode_categories.items():
                category_scores[eid] = CATEGORY_PRIORITY.get(cat, 0.2)

            query_lower = query.lower()
            all_triggers = self._store.list_triggers(ns)
            if all_triggers:
                mem_to_ep: dict[str, str] = {}
                for eid in live_episodes:
                    mem_row = conn_execute_safe(
                        self._store, "SELECT id FROM memory WHERE episode_id = ?", (eid,)
                    )
                    if mem_row:
                        mem_to_ep[mem_row["id"]] = eid

                for trig in all_triggers:
                    term = trig["term"]
                    if term in query_lower:
                        mem_id = trig["memory_id"]
                        eid = mem_to_ep.get(mem_id)
                        if eid:
                            trigger_boosts[eid] = trigger_boosts.get(eid, 0.0) + 0.2

        # Combine scores per mode
        w = self._config.weights
        results: list[RecallResult] = []
        for eid, ep in live_episodes.items():
            lex = norm_lexical.get(eid, 0.0)
            vec = norm_vector.get(eid, 0.0)
            gb = graph_path_scores.get(eid, 0.0)
            rec = decay_scores.get(eid, 0.0)
            conf = confidence_scores.get(eid, 0.0)
            cat_score = category_scores.get(eid, 0.0)
            trig_boost = trigger_boosts.get(eid, 0.0)

            if mode == "fast":
                if vector_scores:
                    fast_total = w.lexical + w.vector
                    if fast_total <= 0:
                        combined = lex
                    else:
                        combined = ((w.lexical / fast_total) * lex) + (
                            (w.vector / fast_total) * vec
                        )
                else:
                    combined = lex
            else:
                if vector_scores:
                    combined = (
                        w.lexical * lex
                        + w.vector * vec
                        + w.graph * gb
                        + w.recency * rec
                        + w.confidence * conf
                        + w.category * cat_score
                    )
                else:
                    combined = (
                        (w.lexical + w.vector) * lex
                        + w.graph * gb
                        + w.recency * rec
                        + w.confidence * conf
                        + w.category * cat_score
                    )
                combined += trig_boost

            # Temporal classification
            entity_rows = self._store.get_entities_for_episode(eid)
            ep_valid_from: Optional[str] = None
            ep_valid_until: Optional[str] = None
            for ent_row in entity_rows:
                if ent_row.get("deleted_at") is not None:
                    continue
                vf = ent_row.get("valid_from")
                vu = ent_row.get("valid_until")
                if vf and not ep_valid_from:
                    ep_valid_from = vf
                if vu and not ep_valid_until:
                    ep_valid_until = vu
                if ep_valid_from and ep_valid_until:
                    break

            temporal_status, temporal_reason = classify_temporal_status(
                ep_valid_from, ep_valid_until, as_of
            )
            temporal_component = temporal_score_boost(temporal_status, temporal_mode)
            purpose_boost = 0.05 if ep.get("purpose") == "profile" and mode != "fast" else 0.0

            if mode != "fast":
                combined += w.temporal * temporal_component
                combined += purpose_boost
            combined *= temporal_component

            kind = self._determine_kind(eid)
            content = ep.get("content", "")

            prov: list[dict[str, Any]] = []
            if include_provenance:
                prov = self._get_episode_provenance(ns, eid, ep)

            safety: dict[str, bool] = {"treat_as_instruction": False}
            matched = _compute_matched_terms(query_terms, content)

            breakdown: Optional[ScoreBreakdown] = None
            why = ""
            if explain:
                breakdown = ScoreBreakdown(
                    lexical=round(lex, 4),
                    vector=round(vec, 4),
                    graph_boost=round(gb, 4),
                    recency=round(rec, 4),
                    confidence=round(conf, 4),
                    final=round(combined, 4),
                )
                why = _build_why_retrieved(lex, vec, gb, rec, conf, mode, cat_score, trig_boost)

            result = RecallResult(
                id=eid,
                kind=kind,
                content=content,
                score=round(combined, 4),
                score_breakdown=breakdown,
                why_retrieved=why,
                provenance=prov,
                warnings=[],
                safety=safety,
                temporal_status=temporal_status,
                valid_from=ep_valid_from,
                valid_until=ep_valid_until,
                temporal_reason=temporal_reason,
                category=episode_categories.get(eid, "other"),
            )
            result._matched_terms = matched  # type: ignore[attr-defined]
            results.append(result)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Temporal filtering
        results = [
            r for r in results
            if should_include_in_mode(r.temporal_status or "unknown", temporal_mode)
        ]

        # Reranking
        rerank_start = time.monotonic()
        results = rerank_results(
            results,
            query,
            endpoint=self._config.reranker_endpoint,
            top_k=self._config.rerank_top_k,
        )
        rerank_ms = (time.monotonic() - rerank_start) * 1000

        # Conflict/supersession warnings
        results = self._apply_conflict_warnings(ns, results)

        # Touch recalled episodes
        for r in results:
            self._store.touch_episode(r.id)

        # Token budget
        if max_tokens is not None:
            budget_results: list[RecallResult] = []
            used_tokens = 0
            for r in results:
                est = len(r.content) // 4
                if budget_results and used_tokens + est > max_tokens:
                    break
                budget_results.append(r)
                used_tokens += est
            results = budget_results

        # Context management
        if agent_tier is not None:
            from palimp.context_manager import ContextManager

            cm = ContextManager(store=self._store, agent_tier=agent_tier)
            managed = cm.manage_context(ns=ns, query=query, memories=results[:limit], mode=mode)

            ctx_explanation = managed.explanation
            for mem in managed.memories:
                if mem.provenance:
                    mem.provenance[0]["context_management"] = ctx_explanation
                else:
                    mem.provenance = [{"context_management": ctx_explanation}]

            results = managed.memories
        else:
            results = results[:limit]

        # Build explanation
        total_ms = (time.monotonic() - total_start) * 1000
        explanation = RecallExplanation()
        if explain:
            retrieval_breakdown: list[dict[str, Any]] = []
            for r in results:
                entry: dict[str, Any] = {"episode_id": r.id, "final_score": r.score}
                if r.score_breakdown:
                    entry["scores"] = r.score_breakdown.model_dump()
                    entry["purpose"] = live_episodes.get(r.id, {}).get("purpose", "search")
                    entry["version"] = live_episodes.get(r.id, {}).get("version", 1)
                    entry["is_latest"] = bool(live_episodes.get(r.id, {}).get("is_latest", 1))
                if r.why_retrieved:
                    entry["why_retrieved"] = r.why_retrieved
                matched = getattr(r, "_matched_terms", [])
                if matched:
                    entry["matched_terms"] = matched
                if r.id in graph_paths_data:
                    entry["graph_paths"] = graph_paths_data[r.id]
                retrieval_breakdown.append(entry)

            all_graph_paths: list[dict[str, Any]] = []
            for paths in graph_paths_data.values():
                all_graph_paths.extend(paths)

            explanation = RecallExplanation(
                query_expansions=query_expansions,
                query_terms=query_terms,
                retrieval_breakdown=retrieval_breakdown,
                latency_ms={
                    "fts": round(fts_ms, 1),
                    "embedding": round(embedding_ms, 1),
                    "vector": round(vector_ms, 1),
                    "graph": round(graph_ms, 1),
                    "rerank": round(rerank_ms, 1),
                    "total": round(total_ms, 1),
                },
                scoring_config=self._config.weights.to_dict(),
                graph_paths=all_graph_paths,
                hop_count=max_hop_count,
                path_score=round(best_path_score, 4),
                refined_from_count=refined_from_count,
            )

        return RecallOutput(results=results, explanation=explanation)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        """Min-max normalise scores to 0-1 range.

        If all scores are zero, returns all zeros.
        If all scores are equal and positive, returns all 1.0.
        Missing keys (score=0) are normalised relative to the range.
        """
        if not scores:
            return {}
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        spread = max_val - min_val
        if spread == 0:
            # All scores are equal
            if max_val > 0:
                # All have the same positive score -> all get 1.0
                return {k: 1.0 for k in scores}
            # All are zero -> all stay zero
            return {k: 0.0 for k in scores}
        return {k: (v - min_val) / spread for k, v in scores.items()}

    # ------------------------------------------------------------------
    # Per-query search helpers (used by thinking mode expansion)
    # ------------------------------------------------------------------

    def _search_lexical(
        self, ns: str, query: str, limit: int
    ) -> dict[str, float]:
        """Run FTS5 search and return {episode_id: score}."""
        fts_rows = self._store.search_fts(ns, query, limit=limit * 3)
        scores: dict[str, float] = {}
        for row in fts_rows:
            ep_id = row["episode_id"]
            raw_rank = abs(float(row.get("rank", 0.0)))
            scores[ep_id] = 1.0 / (1.0 + raw_rank)
        return scores

    def _search_vector(self, ns: str, query: str) -> dict[str, float]:
        """Run vector similarity search and return {episode_id: score}.

        For large datasets (>10K embeddings), skip vector search entirely
        and rely on FTS5 lexical search. Vector search with brute-force
        cosine similarity in Python is O(n) per query — too slow for 20K+ records.
        """
        model_name = "deterministic-sha256" if self._embedder.__class__.__name__ == "DeterministicEmbedder" else "http-embedder"

        # Fast count check — don't load all embeddings if we'll skip them
        MAX_EMBEDDINGS_FOR_VECTOR_SEARCH = 10000
        count = self._store.count_embeddings(ns, owner_type="episode", model=model_name)
        if count == 0 or count > MAX_EMBEDDINGS_FOR_VECTOR_SEARCH:
            return {}

        episode_embeddings = self._store.get_all_embeddings(
            ns, owner_type="episode", model=model_name
        )
        if not episode_embeddings:
            return {}

        query_vec = self._embedder.embed(query)
        scores: dict[str, float] = {}
        for emb_row in episode_embeddings:
            ep_id = emb_row["owner_id"]
            vec = _blob_to_vector(emb_row["vector_blob"])
            scores[ep_id] = self._cosine_similarity(query_vec, vec)
        return scores

    # ------------------------------------------------------------------
    # Component calculators
    # ------------------------------------------------------------------

    def _compute_graph_boosts(
        self, ns: str, episodes: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Compute 1-hop graph boost for each episode.

        Episodes that share entities with other episodes get a bonus.
        The boost is the fraction of candidate episodes that share at least
        one entity with the current episode (1-hop neighbor density).
        """
        # Batch fetch entities for all episodes
        episode_ids = list(episodes.keys())
        ep_entities_raw = self._store.get_entities_for_episodes(episode_ids)
        ep_entities: dict[str, set[str]] = {}
        for eid in episode_ids:
            ents = ep_entities_raw.get(eid, [])
            ep_entities[eid] = {e["id"] for e in ents if e.get("deleted_at") is None}

        # Compute boost: fraction of other episodes sharing at least one entity
        boosts: dict[str, float] = {}
        ep_ids = list(episodes.keys())
        for eid in ep_ids:
            my_ents = ep_entities.get(eid, set())
            if not my_ents:
                boosts[eid] = 0.0
                continue
            neighbor_count = 0
            total_others = len(ep_ids) - 1
            if total_others == 0:
                boosts[eid] = 0.0
                continue
            for other_id in ep_ids:
                if other_id == eid:
                    continue
                other_ents = ep_entities.get(other_id, set())
                if my_ents & other_ents:
                    neighbor_count += 1
            boosts[eid] = neighbor_count / total_others
        return boosts

    @staticmethod
    def _compute_decay_scores(
        episodes: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Compute decay scores using Ebbinghaus forgetting curve.

        Uses last_accessed_at if available, otherwise created_at.
        Pinned items always score 1.0.
        """
        scores: dict[str, float] = {}
        for eid, ep in episodes.items():
            created_at = ep.get("created_at", "")
            last_accessed = ep.get("last_accessed_at")
            stability = float(ep.get("stability", DEFAULT_STABILITY))
            pinned = bool(ep.get("pinned", 0))
            scores[eid] = compute_decay_score(
                created_at=created_at,
                last_accessed_at=last_accessed,
                stability=stability,
                pinned=pinned,
            )
        return scores

    def _compute_confidence_scores(
        self, ns: str, episodes: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Compute confidence score as avg entity/claim confidence linked to episode.

        Uses batch queries to avoid N+1 problem.
        """
        episode_ids = list(episodes.keys())
        if not episode_ids:
            return {}

        # Batch fetch entities for all episodes
        ep_entities = self._store.get_entities_for_episodes(episode_ids)

        # Collect all entity IDs
        all_entity_ids: list[str] = []
        for ents in ep_entities.values():
            for ent in ents:
                if ent.get("deleted_at") is None:
                    all_entity_ids.append(ent["id"])

        # Batch fetch claims for all entities
        all_claims = self._store.get_claims_for_entities(all_entity_ids) if all_entity_ids else {}

        # Compute scores
        scores: dict[str, float] = {}
        for eid in episode_ids:
            entities = ep_entities.get(eid, [])
            all_confidences: list[float] = []
            for ent in entities:
                if ent.get("deleted_at") is not None:
                    continue
                all_confidences.append(float(ent.get("confidence", 0.0)))
                for claim in all_claims.get(ent["id"], []):
                    if claim.get("deleted_at") is None:
                        all_confidences.append(float(claim.get("confidence", 0.0)))
            scores[eid] = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        return scores

    # ------------------------------------------------------------------
    # Kind detection
    # ------------------------------------------------------------------

    def _determine_kind(self, episode_id: str) -> Literal["memory", "knowledge"]:
        """Determine whether an episode is a memory or knowledge item."""
        conn = self._store._conn()
        mem = conn.execute(
            "SELECT id FROM memory WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if mem is not None:
            return "memory"
        return "knowledge"

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _get_episode_provenance(
        self, ns: str, episode_id: str, episode: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Build provenance list for an episode result."""
        source_ref = episode.get("source_ref")
        prov_entry: dict[str, Any] = {"episode_id": episode_id}
        if source_ref:
            prov_entry["source_ref"] = source_ref
        return [prov_entry]

    # ------------------------------------------------------------------
    # Conflict / supersession warnings (all modes)
    # ------------------------------------------------------------------

    def _apply_conflict_warnings(
        self, ns: str, results: list[RecallResult]
    ) -> list[RecallResult]:
        """Detect conflicts and supersession for all results.

        For each result, check if any of its entities are connected via
        CONTRADICTS or SUPERSEDES edges.  If so, attach warnings to both
        the result's ``warnings`` field and provenance metadata.
        """
        for result in results:
            # Find entities linked to this episode via provenance
            entity_rows = self._store.get_entities_for_episode(result.id)
            entity_ids = [e["id"] for e in entity_rows if e.get("deleted_at") is None]

            if not entity_ids:
                continue

            warnings: list[str] = []
            seen_edges: set[str] = set()

            for eid in entity_ids:
                # Check edges where this entity is source or target
                edges = self._store.get_edges_for_entity(eid)
                for edge in edges:
                    if edge.get("deleted_at") is not None:
                        continue
                    relation = edge.get("relation", "")
                    edge_id = edge["id"]
                    if edge_id in seen_edges:
                        continue
                    seen_edges.add(edge_id)

                    if relation == "CONTRADICTS":
                        warnings.append(
                            f"CONTRADICTS: entity {edge['source_entity_id']} "
                            f"contradicts {edge['target_entity_id']}"
                        )
                    elif relation == "SUPERSEDES":
                        warnings.append(
                            f"SUPERSEDES: entity {edge['source_entity_id']} "
                            f"supersedes {edge['target_entity_id']}"
                        )

            if warnings:
                # Attach warnings to RecallResult.warnings field
                result.warnings = warnings
                # Also attach to provenance for backward compatibility
                if result.provenance:
                    result.provenance[0]["warnings"] = warnings
                else:
                    result.provenance = [{"warnings": warnings}]

        return results


# ------------------------------------------------------------------
# Blob <-> vector helpers
# ------------------------------------------------------------------


def _vector_to_blob(vec: list[float]) -> bytes:
    """Convert a float vector to bytes (big-endian float32)."""
    import struct

    return b"".join(struct.pack("!f", v) for v in vec)


def _blob_to_vector(blob: bytes) -> list[float]:
    """Convert bytes (big-endian float32) back to a float list."""
    import struct

    count = len(blob) // 4
    return [struct.unpack_from("!f", blob, i * 4)[0] for i in range(count)]


# Threshold above which a score component is considered a strong signal.
_STRONG_THRESHOLD = 0.3


def _build_why_retrieved(
    lex: float,
    vec: float,
    gb: float,
    rec: float,
    conf: float,
    mode: str,
    cat_score: float = 0.0,
    trig_boost: float = 0.0,
) -> str:
    """Generate a human-readable explanation of why a result was retrieved.

    Returns a semicolon-separated list of contributing factors, e.g.
    ``"strong lexical match; graph edge boost; recent content"``.
    """
    reasons: list[str] = []
    if lex >= _STRONG_THRESHOLD:
        reasons.append("strong lexical match")
    if vec >= _STRONG_THRESHOLD:
        reasons.append("high vector similarity")
    if mode in ("hybrid", "thinking"):
        if gb >= _STRONG_THRESHOLD:
            reasons.append("multi-hop graph path")
        if rec >= _STRONG_THRESHOLD:
            reasons.append("recent content")
        if conf >= _STRONG_THRESHOLD:
            reasons.append("high confidence")
        if cat_score >= 0.7:
            reasons.append("high-priority category")
        if trig_boost > 0:
            reasons.append("trigger term match")
    if not reasons:
        reasons.append("partial match across signals")
    return "; ".join(reasons)
