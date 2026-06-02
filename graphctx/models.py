"""GraphCtx Pydantic models for API requests, responses, and internal records."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class MemoryCreate(BaseModel):
    namespace: str
    content: str
    source_ref: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    extract: bool = True


class KnowledgeCreate(BaseModel):
    namespace: str
    title: str
    content: str
    source_ref: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    extract: bool = True


class RecallRequest(BaseModel):
    namespace: str
    query: str
    mode: Literal["fast", "hybrid", "thinking"] = "hybrid"
    limit: int = Field(default=8, ge=1, le=100)
    include_provenance: bool = True
    explain: bool = False


class MemoryBatchItem(BaseModel):
    content: str
    source_ref: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class KnowledgeBatchItem(BaseModel):
    title: str
    content: str
    source_ref: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class MemoryBatchCreate(BaseModel):
    namespace: str
    items: list[MemoryBatchItem] = Field(max_length=50)
    extract: bool = False  # default False for speed


class KnowledgeBatchCreate(BaseModel):
    namespace: str
    items: list[KnowledgeBatchItem] = Field(max_length=50)
    extract: bool = False


class BatchResponse(BaseModel):
    total: int
    successful: int
    failed: int
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class ExtractionResult(BaseModel):
    entities: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Response / record models
# ---------------------------------------------------------------------------


class ProvenanceRecord(BaseModel):
    id: str
    namespace: str
    episode_id: str
    entity_id: Optional[str] = None
    edge_id: Optional[str] = None
    claim_id: Optional[str] = None
    extractor_version: Optional[str] = None
    evidence_span: Optional[str] = None
    created_at: str


class EpisodeRecord(BaseModel):
    id: str
    namespace: str
    content: str
    source_type: str
    source_ref: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: str
    deleted_at: Optional[str] = None
    tombstoned_at: Optional[str] = None


class EntityRecord(BaseModel):
    id: str
    namespace: str
    name: str
    type: str
    confidence: float
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    created_at: str
    deleted_at: Optional[str] = None


class EdgeRecord(BaseModel):
    id: str
    namespace: str
    source_entity_id: str
    target_entity_id: str
    relation: str
    confidence: float
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    created_at: str
    deleted_at: Optional[str] = None


class ClaimRecord(BaseModel):
    id: str
    namespace: str
    subject_entity_id: str
    predicate: str
    object_value: str
    object_entity_id: Optional[str] = None
    confidence: float
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    created_at: str
    deleted_at: Optional[str] = None


class ScoreBreakdown(BaseModel):
    lexical: float = 0.0
    vector: float = 0.0
    graph_boost: float = 0.0
    recency: float = 0.0
    confidence: float = 0.0
    final: float = 0.0


class RecallExplanation(BaseModel):
    query_expansions: list[str] = []
    query_terms: list[str] = []  # tokens extracted from the query
    entities_detected: list[dict[str, Any]] = []
    retrieval_breakdown: list[dict[str, Any]] = []  # per-result score breakdown
    latency_ms: dict[str, float] = {}  # embedding, vector, rerank, total
    why_retrieved: str = ""


class RecallResult(BaseModel):
    id: str
    kind: Literal["memory", "knowledge"]
    content: str
    score: float
    score_breakdown: Optional[ScoreBreakdown] = None
    why_retrieved: str = ""
    provenance: list[dict[str, Any]] = []
    warnings: list[str] = []
    safety: dict[str, bool] = Field(default_factory=lambda: {"treat_as_instruction": False})


class ContextResponse(BaseModel):
    entity: dict[str, Any]
    claims: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.2.0"
    storage: str = "sqlite"


class StatsResponse(BaseModel):
    namespace: str
    memories: int
    knowledge_items: int
    entities: int
    edges: int
    claims: int


class MemoryResponse(BaseModel):
    memory_id: str
    episode_id: str
    entities: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    warnings: list[str] = []


class KnowledgeResponse(BaseModel):
    knowledge_id: str
    episode_id: str
    entities: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    warnings: list[str] = []


class DeleteResponse(BaseModel):
    episode_id: str
    tombstoned: bool = True


# ---------------------------------------------------------------------------
# Session models
# ---------------------------------------------------------------------------


class SessionCreate(BaseModel):
    namespace: str
    user_ref: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class SessionResponse(BaseModel):
    session_id: str
    namespace: str
    user_ref: Optional[str] = None
    created_at: str
    closed_at: Optional[str] = None


class SessionClose(BaseModel):
    summary: Optional[str] = None
