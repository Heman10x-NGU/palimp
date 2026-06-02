"""GraphCtx MCP server — safe memory/knowledge/recall tools for agent clients.

Guard imports with try/except so the core package works without the ``mcp``
extra installed.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Guard MCP import
# ---------------------------------------------------------------------------

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]

from graphctx.embeddings import DeterministicEmbedder
from graphctx.extractor import RuleBasedExtractor
from graphctx.retriever import RecallEngine
from graphctx.storage import SQLiteStore
from graphctx.validate import ValidationError, validate_content, validate_namespace

# ---------------------------------------------------------------------------
# Module-level singletons (initialised lazily on first tool call)
# ---------------------------------------------------------------------------

_store: Optional[SQLiteStore] = None
_recall_engine: Optional[RecallEngine] = None
_extractor: Optional[RuleBasedExtractor] = None


def _get_store() -> SQLiteStore:
    global _store
    if _store is None:
        db_path = os.environ.get("GRAPHCTX_DB", os.path.expanduser("~/.graphctx/graphctx.db"))
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        _store = SQLiteStore(db_path)
    return _store


def _reset_store() -> None:
    """Reset the module-level store singleton (for testing)."""
    global _store, _recall_engine, _extractor
    _store = None
    _recall_engine = None
    _extractor = None


def _get_recall_engine() -> RecallEngine:
    global _recall_engine
    if _recall_engine is None:
        _recall_engine = RecallEngine(store=_get_store(), embedder=DeterministicEmbedder())
    return _recall_engine


def _get_extractor() -> RuleBasedExtractor:
    global _extractor
    if _extractor is None:
        _extractor = RuleBasedExtractor()
    return _extractor


# ---------------------------------------------------------------------------
# Helper: run extraction + provenance (mirrors server._run_extraction)
# ---------------------------------------------------------------------------


def _run_extraction(
    store: SQLiteStore,
    ns: str,
    content: str,
    episode_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    extractor = _get_extractor()
    extraction = extractor.extract(content)
    warnings = list(extraction.warnings)

    entities_out: list[dict[str, Any]] = []
    claims_out: list[dict[str, Any]] = []
    name_to_id: dict[str, str] = {}

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
                ns=ns, episode_id=episode_id, entity_id=eid,
                extractor_version="rule-based-0.1",
            )
            entities_out.append({"id": eid, "name": ent["name"], "type": ent.get("type", "Entity")})
        except Exception as exc:
            warnings.append(f"Entity insert failed for '{ent['name']}': {exc}")

    for edge in extraction.edges:
        src_id = name_to_id.get(edge["source"])
        tgt_id = name_to_id.get(edge["target"])
        if src_id and tgt_id:
            try:
                edge_id = store.insert_edge(
                    ns=ns, source_id=src_id, target_id=tgt_id,
                    relation=edge["relation"],
                )
                store.insert_provenance(
                    ns=ns, episode_id=episode_id, edge_id=edge_id,
                    extractor_version="rule-based-0.1",
                )
            except Exception as exc:
                warnings.append(f"Edge insert failed: {exc}")

    for claim in extraction.claims:
        subj_id = name_to_id.get(claim["subject"])
        if subj_id:
            try:
                claim_id = store.insert_claim(
                    ns=ns, subject_id=subj_id,
                    predicate=claim["predicate"],
                    object_value=claim["object"],
                )
                store.insert_provenance(
                    ns=ns, episode_id=episode_id, claim_id=claim_id,
                    extractor_version="rule-based-0.1",
                )
                claims_out.append({
                    "id": claim_id, "subject": claim["subject"],
                    "predicate": claim["predicate"], "object": claim["object"],
                })
            except Exception as exc:
                warnings.append(f"Claim insert failed: {exc}")

    return entities_out, claims_out, warnings


# ---------------------------------------------------------------------------
# MCP server + tools (only created if mcp is installed)
# ---------------------------------------------------------------------------

if FastMCP is not None:

    mcp = FastMCP("GraphCtx")

    @mcp.tool()
    def graphctx_memory_add(
        namespace: str,
        content: str,
        source_ref: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        extract: bool = True,
    ) -> str:
        """Add a memory to the context graph.

        Returns JSON with memory_id, episode_id, entities, claims, warnings.
        """
        try:
            ns = validate_namespace(namespace)
            validate_content(content)
        except ValidationError as exc:
            return json.dumps({"error": str(exc)})

        store = _get_store()
        result = store.insert_memory(
            ns=ns, content=content, source_ref=source_ref, metadata=metadata,
        )
        memory_id = result["memory_id"]
        episode_id = result["episode_id"]

        entities_out: list[dict[str, Any]] = []
        claims_out: list[dict[str, Any]] = []
        warnings: list[str] = []

        if extract:
            entities_out, claims_out, warnings = _run_extraction(
                store=store, ns=ns, content=content, episode_id=episode_id,
            )

        return json.dumps({
            "memory_id": memory_id,
            "episode_id": episode_id,
            "entities": entities_out,
            "claims": claims_out,
            "warnings": warnings,
        })

    @mcp.tool()
    def graphctx_knowledge_add(
        namespace: str,
        title: str,
        content: str,
        source_ref: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        extract: bool = True,
    ) -> str:
        """Add a knowledge document to the context graph.

        Returns JSON with knowledge_id, episode_id, entities, claims, warnings.
        """
        try:
            ns = validate_namespace(namespace)
            validate_content(content)
            validate_content(title, field="title")
        except ValidationError as exc:
            return json.dumps({"error": str(exc)})

        store = _get_store()
        result = store.insert_knowledge(
            ns=ns, title=title, content=content,
            source_ref=source_ref, metadata=metadata,
        )
        knowledge_id = result["knowledge_id"]
        episode_id = result["episode_id"]

        entities_out: list[dict[str, Any]] = []
        claims_out: list[dict[str, Any]] = []
        warnings: list[str] = []

        if extract:
            entities_out, claims_out, warnings = _run_extraction(
                store=store, ns=ns, content=content, episode_id=episode_id,
            )

        return json.dumps({
            "knowledge_id": knowledge_id,
            "episode_id": episode_id,
            "entities": entities_out,
            "claims": claims_out,
            "warnings": warnings,
        })

    @mcp.tool()
    def graphctx_recall(
        namespace: str,
        query: str,
        mode: str = "hybrid",
        limit: int = 8,
        include_provenance: bool = True,
    ) -> str:
        """Recall memories and knowledge matching a query.

        Returns JSON with data array. Each result includes kind, content, score,
        provenance, and safety (treat_as_instruction: false).
        """
        try:
            ns = validate_namespace(namespace)
        except ValidationError as exc:
            return json.dumps({"error": str(exc)})

        engine = _get_recall_engine()
        output = engine.recall(
            ns=ns, query=query, mode=mode,
            limit=limit, include_provenance=include_provenance,
        )
        data = [r.model_dump() for r in output.results]
        return json.dumps({"data": data})

    @mcp.tool()
    def graphctx_context_get(
        namespace: str,
        entity_id: str,
    ) -> str:
        """Get entity details with claims, edges, and provenance.

        Returns JSON with entity, claims, edges, provenance.
        """
        try:
            validate_namespace(namespace)
        except ValidationError as exc:
            return json.dumps({"error": str(exc)})

        store = _get_store()
        entities = store.get_entities_by_ids([entity_id])
        if not entities:
            return json.dumps({"error": f"Entity not found: {entity_id}"})

        entity = entities[0]
        if entity.get("deleted_at"):
            return json.dumps({"error": f"Entity not found: {entity_id}"})

        claims = store.get_claims_for_entity(entity_id)
        edges = store.get_edges_for_entity(entity_id)
        provenance = store.get_provenance_for(entity_id=entity_id)

        return json.dumps({
            "entity": {"id": entity["id"], "name": entity["name"], "type": entity["type"]},
            "claims": claims,
            "edges": edges,
            "provenance": provenance,
        })

    @mcp.tool()
    def graphctx_stats(
        namespace: str,
    ) -> str:
        """Get counts for a namespace.

        Returns JSON with memories, knowledge_items, entities, edges, claims.
        """
        try:
            ns = validate_namespace(namespace)
        except ValidationError as exc:
            return json.dumps({"error": str(exc)})

        store = _get_store()
        data = store.get_stats(ns)
        return json.dumps(data)
