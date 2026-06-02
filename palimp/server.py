"""Palimp REST API server.

FastAPI application exposing HydraDB-style primitives:
memories, knowledge, recall, context, and source deletion.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from palimp.embeddings import BaseEmbedder, DeterministicEmbedder, HttpEmbedder
from palimp.errors import (
    EpisodeNotFoundError,
    PalimpError,
    NamespaceRequiredError,
)
from palimp.extractor import BaseExtractor, HttpExtractor, RuleBasedExtractor
from palimp.models import (
    BatchResponse,
    ContextPackRequest,
    ContextPackResult,
    ContextResponse,
    HealthResponse,
    KnowledgeBatchCreate,
    KnowledgeCreate,
    KnowledgeResponse,
    MemoryBatchCreate,
    MemoryCreate,
    MemoryResponse,
    RecallRequest,
    SessionClose,
    SessionCreate,
    SessionResponse,
    StatsResponse,
)
from palimp.ingest import ingest_knowledge, ingest_memory
from palimp.retriever import RecallEngine
from palimp.storage import SQLiteStore
from palimp.validate import (
    ValidationError,
    validate_batch_size,
    validate_namespace,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = "0.3.0"
MAX_REQUEST_BYTES = 1_048_576  # 1 MB


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create store, embedder, extractor, and recall engine on startup."""
    # DB path
    db_path = os.environ.get("PALIMP_DB", os.path.expanduser("~/.palimp/palimp.db"))

    # Embedder
    embedder_endpoint = os.environ.get("PALIMP_EMBEDDER_ENDPOINT")
    if embedder_endpoint:
        embedder: BaseEmbedder = HttpEmbedder(
            endpoint=embedder_endpoint,
            api_key=os.environ.get("PALIMP_EMBEDDER_API_KEY", ""),
            model=os.environ.get("PALIMP_EMBEDDER_MODEL", "text-embedding-3-small"),
            dim=int(os.environ.get("PALIMP_EMBEDDER_DIM", "384")),
        )
    else:
        embedder = DeterministicEmbedder()

    # Extractor
    extractor_endpoint = os.environ.get("PALIMP_EXTRACTOR_ENDPOINT")
    if extractor_endpoint:
        extractor: BaseExtractor = HttpExtractor(
            endpoint=extractor_endpoint,
            api_key=os.environ.get("PALIMP_EXTRACTOR_API_KEY", ""),
            model=os.environ.get("PALIMP_EXTRACTOR_MODEL", "gpt-4o-mini"),
        )
    else:
        extractor = RuleBasedExtractor()

    # Store
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    store = SQLiteStore(db_path)

    # Recall engine
    recall_engine = RecallEngine(store=store, embedder=embedder)

    # Attach to app.state
    app.state.store = store
    app.state.embedder = embedder
    app.state.extractor = extractor
    app.state.recall_engine = recall_engine

    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Palimp", version=VERSION, lifespan=lifespan)

# CORS — allow all origins for v0.1
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request size limit middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Reject requests with body > 1 MB."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": "Request too large", "detail": "Maximum request size is 1 MB."},
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.exception_handler(EpisodeNotFoundError)
async def episode_not_found_handler(_request: Request, exc: EpisodeNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": "Episode not found", "detail": str(exc)},
    )


@app.exception_handler(NamespaceRequiredError)
async def namespace_required_handler(_request: Request, exc: NamespaceRequiredError):
    return JSONResponse(
        status_code=422,
        content={"error": "Namespace required", "detail": str(exc)},
    )


@app.exception_handler(PalimpError)
async def graph_ctx_error_handler(_request: Request, exc: PalimpError):
    return JSONResponse(
        status_code=400,
        content={"error": "Palimp error", "detail": str(exc)},
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(_request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/health")
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/v1/stats")
def stats(namespace: str) -> StatsResponse:
    ns = validate_namespace(namespace)
    store: SQLiteStore = app.state.store
    data = store.get_stats(ns)
    return StatsResponse(namespace=ns, **data)


@app.post("/v1/memories")
def add_memory(body: MemoryCreate) -> MemoryResponse:
    ns = validate_namespace(body.namespace)
    store: SQLiteStore = app.state.store
    extractor: BaseExtractor = app.state.extractor
    embedder: BaseEmbedder = app.state.embedder

    result = ingest_memory(
        store=store,
        embedder=embedder,
        extractor=extractor,
        ns=ns,
        content=body.content,
        source_ref=body.source_ref,
        metadata=body.metadata,
        extract=body.extract,
    )

    return MemoryResponse(
        memory_id=result["memory_id"],
        episode_id=result["episode_id"],
        entities=result["entities"],
        claims=result["claims"],
        warnings=result["warnings"],
    )


@app.post("/v1/knowledge")
def add_knowledge(body: KnowledgeCreate) -> KnowledgeResponse:
    ns = validate_namespace(body.namespace)
    store: SQLiteStore = app.state.store
    extractor: BaseExtractor = app.state.extractor
    embedder: BaseEmbedder = app.state.embedder

    result = ingest_knowledge(
        store=store,
        embedder=embedder,
        extractor=extractor,
        ns=ns,
        title=body.title,
        content=body.content,
        source_ref=body.source_ref,
        metadata=body.metadata,
        extract=body.extract,
    )

    return KnowledgeResponse(
        knowledge_id=result["knowledge_id"],
        episode_id=result["episode_id"],
        entities=result["entities"],
        claims=result["claims"],
        warnings=result["warnings"],
    )


# ---------------------------------------------------------------------------
# Batch endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/memories/batch")
def add_memories_batch(body: MemoryBatchCreate) -> BatchResponse:
    """Insert up to 50 memories in one call."""
    ns = validate_namespace(body.namespace)
    validate_batch_size(body.items)
    store: SQLiteStore = app.state.store
    extractor: BaseExtractor = app.state.extractor
    embedder: BaseEmbedder = app.state.embedder

    # Convert to dicts for storage layer
    items = []
    for item in body.items:
        items.append({
            "content": item.content,
            "source_ref": item.source_ref,
            "metadata": item.metadata,
        })

    response = store.insert_memories_batch(ns, items)

    # Optionally run extraction + embeddings on successful items
    if body.extract:
        for result in response.results:
            try:
                ep = store.get_episode(result["episode_id"])
                if ep:
                    _store_embedding(store, embedder, ns, result["episode_id"], ep["content"])
                    _run_extraction(store, extractor, ns, ep["content"], result["episode_id"])
            except Exception:
                pass  # extraction failure should not break batch response
    else:
        # Still store embeddings even when extract=False
        for result in response.results:
            try:
                ep = store.get_episode(result["episode_id"])
                if ep:
                    _store_embedding(store, embedder, ns, result["episode_id"], ep["content"])
            except Exception:
                pass

    return response


@app.post("/v1/knowledge/batch")
def add_knowledge_batch(body: KnowledgeBatchCreate) -> BatchResponse:
    """Insert up to 50 knowledge items in one call."""
    ns = validate_namespace(body.namespace)
    validate_batch_size(body.items)
    store: SQLiteStore = app.state.store
    extractor: BaseExtractor = app.state.extractor
    embedder: BaseEmbedder = app.state.embedder

    items = []
    for item in body.items:
        items.append({
            "title": item.title,
            "content": item.content,
            "source_ref": item.source_ref,
            "metadata": item.metadata,
        })

    response = store.insert_knowledge_batch(ns, items)

    if body.extract:
        for result in response.results:
            try:
                ep = store.get_episode(result["episode_id"])
                if ep:
                    _store_embedding(store, embedder, ns, result["episode_id"], ep["content"])
                    _run_extraction(store, extractor, ns, ep["content"], result["episode_id"])
            except Exception:
                pass
    else:
        for result in response.results:
            try:
                ep = store.get_episode(result["episode_id"])
                if ep:
                    _store_embedding(store, embedder, ns, result["episode_id"], ep["content"])
            except Exception:
                pass

    return response


@app.post("/v1/recall")
def recall(body: RecallRequest) -> dict[str, Any]:
    ns = validate_namespace(body.namespace)
    engine: RecallEngine = app.state.recall_engine

    output = engine.recall(
        ns=ns,
        query=body.query,
        mode=body.mode,
        limit=body.limit,
        include_provenance=body.include_provenance,
        explain=body.explain,
        as_of=body.as_of,
        temporal_mode=body.temporal_mode,
    )

    # Collect top-level warnings from all results
    all_warnings: list[str] = []
    for r in output.results:
        all_warnings.extend(r.warnings)

    response: dict[str, Any] = {
        "query": body.query,
        "results": [r.model_dump() for r in output.results],
        "warnings": all_warnings,
    }

    if body.explain:
        response["explanation"] = output.explanation.model_dump()

    return response


@app.post("/v1/context/pack")
def context_pack(body: ContextPackRequest) -> ContextPackResult:
    """Build a compact context pack for a coding task.

    Returns runbook items + relevant memories/knowledge within token budget.
    """
    from palimp.cli import _build_context_pack

    ns = validate_namespace(body.namespace)
    store: SQLiteStore = app.state.store

    pack = _build_context_pack(
        store=store, ns=ns, task=body.task, budget_tokens=body.budget_tokens,
    )

    return ContextPackResult(
        items=pack["items"],
        total_tokens=pack["total_tokens"],
        safety=pack["safety"],
    )


@app.get("/v1/context/{entity_id}")
def context_entity(entity_id: str, namespace: str) -> ContextResponse:
    validate_namespace(namespace)
    store: SQLiteStore = app.state.store

    # Fetch entity
    entities = store.get_entities_by_ids([entity_id])
    if not entities:
        raise HTTPException(status_code=404, detail=f"Entity not found: {entity_id}")
    entity = entities[0]
    if entity.get("deleted_at"):
        raise HTTPException(status_code=404, detail=f"Entity not found: {entity_id}")

    # Claims
    claims = store.get_claims_for_entity(entity_id)

    # Edges
    edges = store.get_edges_for_entity(entity_id)

    # Provenance
    provenance = store.get_provenance_for(entity_id=entity_id)

    return ContextResponse(
        entity={"id": entity["id"], "name": entity["name"], "type": entity["type"]},
        claims=claims,
        edges=edges,
        provenance=provenance,
    )


@app.delete("/v1/sources/{episode_id}")
def delete_source(episode_id: str, namespace: str) -> dict[str, Any]:
    ns = validate_namespace(namespace)
    store: SQLiteStore = app.state.store

    # Verify episode exists and belongs to namespace
    episode = store.get_episode(episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")
    if episode["namespace"] != ns:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")

    store.tombstone_episode(episode_id)

    return {"status": "deleted", "episode_id": episode_id}


# ---------------------------------------------------------------------------
# Decay / pin endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/episodes/{episode_id}/pin")
def pin_episode(episode_id: str, namespace: str) -> dict[str, Any]:
    """Pin an episode so it never decays."""
    ns = validate_namespace(namespace)
    store: SQLiteStore = app.state.store

    episode = store.get_episode(episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")
    if episode["namespace"] != ns:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")

    store.pin_episode(episode_id)
    return {"status": "pinned", "episode_id": episode_id}


@app.post("/v1/episodes/{episode_id}/unpin")
def unpin_episode(episode_id: str, namespace: str) -> dict[str, Any]:
    """Unpin an episode so it resumes normal decay."""
    ns = validate_namespace(namespace)
    store: SQLiteStore = app.state.store

    episode = store.get_episode(episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")
    if episode["namespace"] != ns:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")

    store.unpin_episode(episode_id)
    return {"status": "unpinned", "episode_id": episode_id}


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/sessions")
def create_session(body: SessionCreate) -> SessionResponse:
    ns = validate_namespace(body.namespace)
    store: SQLiteStore = app.state.store
    session_id = store.create_session(ns=ns, user_ref=body.user_ref, metadata=body.metadata)
    session = store.get_session(session_id)
    return SessionResponse(
        session_id=session["id"],
        namespace=session["namespace"],
        user_ref=session.get("user_ref"),
        created_at=session["created_at"],
        closed_at=session.get("closed_at"),
    )


@app.post("/v1/sessions/{session_id}/close")
def close_session(session_id: str, body: SessionClose) -> dict[str, Any]:
    store: SQLiteStore = app.state.store
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    if session.get("closed_at"):
        raise HTTPException(status_code=409, detail="Session is already closed.")
    store.close_session(session_id, summary=body.summary)
    return {"status": "closed", "session_id": session_id}


@app.get("/v1/sessions")
def list_sessions(namespace: str) -> list[dict[str, Any]]:
    ns = validate_namespace(namespace)
    store: SQLiteStore = app.state.store
    sessions = store.get_sessions(ns)
    return [
        {
            "session_id": s["id"],
            "namespace": s["namespace"],
            "user_ref": s.get("user_ref"),
            "created_at": s["created_at"],
            "closed_at": s.get("closed_at"),
        }
        for s in sessions
    ]


@app.post("/v1/admin/decay/run")
def run_decay(namespace: str) -> dict[str, Any]:
    """Manually trigger a decay pass: touch all non-pinned episodes with expired retention."""
    from palimp.decay import compute_decay_score

    ns = validate_namespace(namespace)
    store: SQLiteStore = app.state.store
    conn = store._conn()

    rows = conn.execute(
        "SELECT id, created_at, last_accessed_at, stability, pinned FROM episode WHERE namespace = ? AND deleted_at IS NULL AND tombstoned_at IS NULL",
        (ns,),
    ).fetchall()

    archived = 0
    for row in rows:
        if row["pinned"]:
            continue
        score = compute_decay_score(
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            stability=float(row["stability"]),
        )
        if score < 0.1:
            store.tombstone_episode(row["id"])
            archived += 1

    return {"namespace": ns, "archived": archived}


# ---------------------------------------------------------------------------
# Extraction helper
# ---------------------------------------------------------------------------


def _store_embedding(
    store: SQLiteStore,
    embedder: BaseEmbedder,
    ns: str,
    episode_id: str,
    content: str,
) -> None:
    """Compute and store an embedding for an episode."""
    import struct

    vec = embedder.embed(content)
    blob = b"".join(struct.pack("!f", v) for v in vec)
    model_name = "deterministic-sha256" if isinstance(embedder, DeterministicEmbedder) else "http-embedder"
    try:
        store.insert_embedding(
            ns=ns,
            owner_type="episode",
            owner_id=episode_id,
            model=model_name,
            dimension=embedder.dimension,
            vector_blob=blob,
        )
    except Exception:
        # Embedding storage failure should not block the main operation
        pass


def _run_extraction(
    store: SQLiteStore,
    extractor: BaseExtractor,
    ns: str,
    content: str,
    episode_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Run extraction, store entities/edges/claims with provenance.

    Returns (entities, claims, warnings).
    """
    extraction = extractor.extract(content)
    warnings = list(extraction.warnings)

    entities_out: list[dict[str, Any]] = []
    claims_out: list[dict[str, Any]] = []
    name_to_id: dict[str, str] = {}

    # Insert entities
    for ent in extraction.entities:
        try:
            eid = store.insert_entity(
                ns=ns,
                name=ent["name"],
                entity_type=ent.get("type", "Entity"),
                confidence=ent.get("confidence", 0.85),
            )
            name_to_id[ent["name"]] = eid
            store.insert_provenance(
                ns=ns,
                episode_id=episode_id,
                entity_id=eid,
                extractor_version="rule-based-0.1",
            )
            entities_out.append({"id": eid, "name": ent["name"], "type": ent.get("type", "Entity")})
        except Exception as exc:
            warnings.append(f"Entity insert failed for '{ent['name']}': {exc}")

    # Insert edges
    for edge in extraction.edges:
        src_id = name_to_id.get(edge["source"])
        tgt_id = name_to_id.get(edge["target"])
        if src_id and tgt_id:
            try:
                edge_id = store.insert_edge(
                    ns=ns,
                    source_id=src_id,
                    target_id=tgt_id,
                    relation=edge["relation"],
                )
                store.insert_provenance(
                    ns=ns,
                    episode_id=episode_id,
                    edge_id=edge_id,
                    extractor_version="rule-based-0.1",
                )
            except Exception as exc:
                warnings.append(f"Edge insert failed: {exc}")

    # Insert claims
    for claim in extraction.claims:
        subj_id = name_to_id.get(claim["subject"])
        if subj_id:
            try:
                claim_id = store.insert_claim(
                    ns=ns,
                    subject_id=subj_id,
                    predicate=claim["predicate"],
                    object_value=claim["object"],
                )
                store.insert_provenance(
                    ns=ns,
                    episode_id=episode_id,
                    claim_id=claim_id,
                    extractor_version="rule-based-0.1",
                )
                claims_out.append({
                    "id": claim_id,
                    "subject": claim["subject"],
                    "predicate": claim["predicate"],
                    "object": claim["object"],
                })
            except Exception as exc:
                warnings.append(f"Claim insert failed: {exc}")

    return entities_out, claims_out, warnings
