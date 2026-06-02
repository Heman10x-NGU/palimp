"""AdaCoM-inspired rule-based context manager for Palimp.

Implements structured context state tracking, memory deduplication,
relevance scoring, and token-budget-aware compression.  Designed to
slot between RecallEngine.recall() and the API layer.

Reference: AdaCoM (arXiv:2605.30785)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from palimp.models import CATEGORY_PRIORITY, RecallResult
from palimp.storage import SQLiteStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Token estimation: ~4 characters per token (standard English average)
_CHARS_PER_TOKEN = 4

# Content overlap threshold for deduplication (Jaccard similarity on word sets)
_DEDUP_OVERLAP_THRESHOLD = 0.70

# Agent tier budget ratios — fraction of input memories to preserve
_AGENT_TIER_BUDGET: dict[str, float] = {
    "strong": 0.85,
    "medium": 0.65,
    "weak": 0.45,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ContextState(BaseModel):
    """Structured state maintained for each namespace + task."""

    namespace: str
    task_id: str = ""
    requirements: list[str] = Field(default_factory=list)
    resolved_constraints: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    current_leads: list[str] = Field(default_factory=list)
    rejected_candidates: list[dict[str, Any]] = Field(default_factory=list)
    failed_queries: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=_now_iso)


class ManagedContext(BaseModel):
    """Result of context management — compressed memories + explanation."""

    memories: list[RecallResult] = Field(default_factory=list)
    original_count: int = 0
    compressed_count: int = 0
    compression_ratio: float = 0.0
    explanation: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------


class ContextManager:
    """Rule-based context manager inspired by AdaCoM.

    Parameters
    ----------
    store : SQLiteStore
        The storage backend (used for FTS relevance scoring and state
        persistence).
    agent_tier : {"strong", "medium", "weak"}
        Determines the token budget — strong preserves more context,
        weak aggressively compresses.
    """

    def __init__(self, store: SQLiteStore, agent_tier: str = "medium") -> None:
        self._store = store
        self.agent_tier = agent_tier if agent_tier in _AGENT_TIER_BUDGET else "medium"
        self._ensure_context_tables()

    # ------------------------------------------------------------------
    # Schema extension — context tables
    # ------------------------------------------------------------------

    def _ensure_context_tables(self) -> None:
        """Create context_state and context_operations tables if missing."""
        conn = self._store._conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS context_state (
                id              TEXT PRIMARY KEY,
                namespace       TEXT NOT NULL,
                task_id         TEXT NOT NULL,
                requirements    TEXT,
                resolved        TEXT,
                evidence        TEXT,
                leads           TEXT,
                rejected        TEXT,
                failed_queries  TEXT,
                updated_at      TEXT NOT NULL,
                FOREIGN KEY (namespace) REFERENCES namespace(name)
            );

            CREATE TABLE IF NOT EXISTS context_operations (
                id              TEXT PRIMARY KEY,
                namespace       TEXT NOT NULL,
                task_id         TEXT NOT NULL,
                op_type         TEXT NOT NULL,
                target_ids      TEXT,
                original_content TEXT,
                new_content      TEXT,
                justification    TEXT,
                token_savings    INTEGER NOT NULL DEFAULT 0,
                created_at       TEXT NOT NULL,
                FOREIGN KEY (namespace) REFERENCES namespace(name)
            );
            """
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def manage_context(
        self,
        ns: str,
        query: str,
        memories: list[RecallResult],
        mode: Literal["fast", "hybrid", "thinking"] = "hybrid",
    ) -> ManagedContext:
        """Apply context management to a set of recall results.

        Steps:
        1. Extract requirements from the query.
        2. Deduplicate memories with >70% content overlap.
        3. Rank remaining memories by relevance to requirements.
        4. Apply token budget based on agent_tier.
        5. Return ManagedContext with compressed results + explanation.
        """
        original_count = len(memories)
        if original_count == 0:
            return ManagedContext(
                memories=[],
                original_count=0,
                compressed_count=0,
                compression_ratio=1.0,
                explanation={
                    "requirements_extracted": [],
                    "duplicates_merged": 0,
                    "memories_dropped": 0,
                    "token_savings": 0,
                },
            )

        # Step 1: Extract requirements
        requirements = self._extract_requirements(query)

        # Step 2: Deduplicate
        deduplicated, duplicates_merged = self._deduplicate(memories)

        # Step 3: Score by relevance
        scored: list[tuple[RecallResult, float]] = []
        for mem in deduplicated:
            relevance = self._compute_relevance(mem, requirements)
            # Blend recall score with relevance: 60% recall, 40% relevance
            blended = 0.6 * mem.score + 0.4 * relevance
            scored.append((mem, blended))

        # Sort by blended score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        sorted_memories = [m for m, _ in scored]

        # Step 4: Apply token budget
        budget_ratio = _AGENT_TIER_BUDGET.get(self.agent_tier, 0.65)
        budgeted = self._apply_token_budget(sorted_memories, budget_ratio)

        # Compute stats
        compressed_count = len(budgeted)
        original_tokens = sum(len(m.content) // _CHARS_PER_TOKEN for m in memories)
        compressed_tokens = sum(len(m.content) // _CHARS_PER_TOKEN for m in budgeted)
        token_savings = max(0, original_tokens - compressed_tokens)

        compression_ratio = (
            compressed_count / original_count if original_count > 0 else 1.0
        )

        # Log context operation
        self._log_operation(
            ns=ns,
            op_type="manage_context",
            target_ids=[m.id for m in budgeted],
            justification=f"agent_tier={self.agent_tier}, mode={mode}",
            token_savings=token_savings,
        )

        return ManagedContext(
            memories=budgeted,
            original_count=original_count,
            compressed_count=compressed_count,
            compression_ratio=round(compression_ratio, 4),
            explanation={
                "requirements_extracted": requirements,
                "duplicates_merged": duplicates_merged,
                "memories_dropped": original_count - compressed_count,
                "token_savings": token_savings,
            },
        )

    # ------------------------------------------------------------------
    # Requirement extraction
    # ------------------------------------------------------------------

    def _extract_requirements(self, query: str) -> list[str]:
        """Parse a query into discrete constraints/requirements.

        Strategy:
        - Split on sentence boundaries and clause connectors.
        - Strip and filter empty strings.
        - Each non-trivial fragment becomes a requirement.
        """
        if not query or not query.strip():
            return []

        # Split on sentence-ending punctuation and clause connectors
        # Also split on semicolons and commas that separate clauses
        raw_parts = re.split(r"[.;]|\band\b|\bor\b|\bbut\b", query, flags=re.IGNORECASE)

        requirements: list[str] = []
        for part in raw_parts:
            cleaned = part.strip()
            # Skip very short fragments (likely noise)
            if len(cleaned) >= 5:
                # Normalize: remove leading question words for cleaner requirement
                # but keep the full phrase as the requirement
                requirements.append(cleaned)

        # If splitting produced nothing useful, use the whole query
        if not requirements and query.strip():
            requirements = [query.strip()]

        return requirements

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, memories: list[RecallResult]) -> tuple[list[RecallResult], int]:
        """Merge memories with >70% content overlap.

        For each pair, compute Jaccard similarity on word sets.
        When overlap exceeds threshold, keep the one with the higher
        recall score and merge provenance from the dropped one.

        Returns (deduplicated_memories, number_of_merges).
        """
        if len(memories) <= 1:
            return list(memories), 0

        # Build word sets for each memory
        word_sets: list[set[str]] = []
        for mem in memories:
            words = set(re.findall(r"[a-z0-9]+", mem.content.lower()))
            word_sets.append(words)

        merges = 0
        merged_indices: set[int] = set()
        result: list[RecallResult] = []

        for i in range(len(memories)):
            if i in merged_indices:
                continue

            current = memories[i]
            current_words = word_sets[i]
            current_prov = list(current.provenance)

            for j in range(i + 1, len(memories)):
                if j in merged_indices:
                    continue

                other_words = word_sets[j]
                # Jaccard similarity
                intersection = current_words & other_words
                union = current_words | other_words
                if not union:
                    continue
                overlap = len(intersection) / len(union)

                if overlap > _DEDUP_OVERLAP_THRESHOLD:
                    # Keep the one with higher score; merge provenance
                    other = memories[j]
                    if other.score > current.score:
                        # Swap: keep 'other' as the survivor
                        current_prov.extend(current_prov)
                        current = RecallResult(
                            id=other.id,
                            kind=other.kind,
                            content=other.content,
                            score=other.score,
                            provenance=current_prov + list(other.provenance),
                            safety=other.safety,
                        )
                        current_words = other_words
                    else:
                        # Keep current, absorb other's provenance
                        current_prov.extend(other.provenance)
                        current = RecallResult(
                            id=current.id,
                            kind=current.kind,
                            content=current.content,
                            score=current.score,
                            provenance=current_prov,
                            safety=current.safety,
                        )
                    merged_indices.add(j)
                    merges += 1

            result.append(current)

        return result, merges

    # ------------------------------------------------------------------
    # Relevance scoring
    # ------------------------------------------------------------------

    def _compute_relevance(self, memory: RecallResult, requirements: list[str]) -> float:
        """Score how well a memory satisfies the extracted requirements.

        Uses FTS5 matching against requirements.  Falls back to simple
        word-overlap when FTS5 returns nothing.
        """
        if not requirements:
            return 0.0

        content_lower = memory.content.lower()
        content_words = set(re.findall(r"[a-z0-9]+", content_lower))

        # Try FTS5-based scoring against each requirement
        total_score = 0.0
        scored_reqs = 0

        for req in requirements:
            # FTS5 match against the memory content
            fts_score = self._fts_match_score(req, content_lower)
            if fts_score > 0:
                total_score += fts_score
                scored_reqs += 1
            else:
                # Fallback: word overlap
                req_words = set(re.findall(r"[a-z0-9]+", req.lower()))
                if req_words and content_words:
                    overlap = len(req_words & content_words) / len(req_words)
                    total_score += overlap
                    scored_reqs += 1

        if scored_reqs == 0:
            return 0.0

        return min(total_score / scored_reqs, 1.0)

    def _fts_match_score(self, query: str, content: str) -> float:
        """Compute an FTS5-inspired match score between query and content.

        Since we can't run FTS5 on individual strings directly, we use
        token matching as a proxy for what FTS5 would return.
        """
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        content_tokens = set(re.findall(r"[a-z0-9]+", content.lower()))

        if not query_tokens or not content_tokens:
            return 0.0

        matches = query_tokens & content_tokens
        if not matches:
            return 0.0

        # Score: fraction of query tokens found in content
        return len(matches) / len(query_tokens)

    # ------------------------------------------------------------------
    # Token budget
    # ------------------------------------------------------------------

    def _apply_token_budget(
        self, memories: list[RecallResult], budget_ratio: float
    ) -> list[RecallResult]:
        """Keep top memories until the token budget is exhausted.

        Token estimation: len(content) / 4.
        Budget = total_tokens * budget_ratio.
        Memories are assumed pre-sorted by relevance (best first).
        High-priority categories (identity, constraint, gotcha) get a
        category boost so they survive tighter budgets.
        """
        if not memories:
            return []

        # Apply category priority boost to scores for budget-aware sorting
        boosted: list[tuple[RecallResult, float]] = []
        for mem in memories:
            cat_priority = CATEGORY_PRIORITY.get(mem.category, 0.2)
            # Blend: 70% original score, 30% category priority
            boosted_score = 0.7 * mem.score + 0.3 * cat_priority
            boosted.append((mem, boosted_score))

        # Re-sort by boosted score
        boosted.sort(key=lambda x: x[1], reverse=True)
        sorted_memories = [m for m, _ in boosted]

        total_tokens = sum(len(m.content) // _CHARS_PER_TOKEN for m in sorted_memories)
        budget = max(1, int(total_tokens * budget_ratio))

        kept: list[RecallResult] = []
        used_tokens = 0

        for mem in sorted_memories:
            mem_tokens = len(mem.content) // _CHARS_PER_TOKEN
            if used_tokens + mem_tokens <= budget:
                kept.append(mem)
                used_tokens += mem_tokens
            else:
                # Always keep at least one memory
                if not kept:
                    kept.append(mem)
                break

        return kept

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self, state: ContextState) -> None:
        """Persist a ContextState to the database."""
        self._store.ensure_namespace(state.namespace)
        conn = self._store._conn()
        now = _now_iso()
        state_id = _new_id("ctx")

        # Check for existing state with same namespace + task_id
        existing = conn.execute(
            "SELECT id FROM context_state WHERE namespace = ? AND task_id = ?",
            (state.namespace, state.task_id),
        ).fetchone()

        with self._store._lock:
            if existing:
                conn.execute(
                    """UPDATE context_state
                       SET requirements = ?, resolved = ?, evidence = ?,
                           leads = ?, rejected = ?, failed_queries = ?,
                           updated_at = ?
                       WHERE id = ?""",
                    (
                        json.dumps(state.requirements),
                        json.dumps(state.resolved_constraints),
                        json.dumps(state.evidence),
                        json.dumps(state.current_leads),
                        json.dumps(state.rejected_candidates),
                        json.dumps(state.failed_queries),
                        now,
                        existing["id"],
                    ),
                )
            else:
                conn.execute(
                    """INSERT INTO context_state
                       (id, namespace, task_id, requirements, resolved, evidence,
                        leads, rejected, failed_queries, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        state_id,
                        state.namespace,
                        state.task_id,
                        json.dumps(state.requirements),
                        json.dumps(state.resolved_constraints),
                        json.dumps(state.evidence),
                        json.dumps(state.current_leads),
                        json.dumps(state.rejected_candidates),
                        json.dumps(state.failed_queries),
                        now,
                    ),
                )
            conn.commit()

    def load_state(self, ns: str, task_id: str) -> ContextState | None:
        """Load a persisted ContextState from the database."""
        conn = self._store._conn()
        row = conn.execute(
            "SELECT * FROM context_state WHERE namespace = ? AND task_id = ?",
            (ns, task_id),
        ).fetchone()

        if row is None:
            return None

        return ContextState(
            namespace=row["namespace"],
            task_id=row["task_id"],
            requirements=json.loads(row["requirements"]) if row["requirements"] else [],
            resolved_constraints=json.loads(row["resolved"]) if row["resolved"] else [],
            evidence=json.loads(row["evidence"]) if row["evidence"] else [],
            current_leads=json.loads(row["leads"]) if row["leads"] else [],
            rejected_candidates=json.loads(row["rejected"]) if row["rejected"] else [],
            failed_queries=json.loads(row["failed_queries"]) if row["failed_queries"] else [],
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------
    # Operation logging
    # ------------------------------------------------------------------

    def _log_operation(
        self,
        ns: str,
        op_type: str,
        target_ids: list[str],
        justification: str,
        token_savings: int,
    ) -> None:
        """Log a context management operation to context_operations."""
        self._store.ensure_namespace(ns)
        conn = self._store._conn()
        op_id = _new_id("cop")

        with self._store._lock:
            conn.execute(
                """INSERT INTO context_operations
                   (id, namespace, task_id, op_type, target_ids,
                    original_content, new_content, justification,
                    token_savings, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    op_id,
                    ns,
                    "",
                    op_type,
                    json.dumps(target_ids),
                    None,
                    None,
                    justification,
                    token_savings,
                    _now_iso(),
                ),
            )
            conn.commit()
