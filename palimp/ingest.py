"""Shared ingestion service for REST and CLI paths."""

from __future__ import annotations

import struct
from typing import Any

from palimp.embeddings import BaseEmbedder, DeterministicEmbedder
from palimp.extractor import BaseExtractor
from palimp.storage import SQLiteStore
from palimp.validate import validate_content, validate_namespace

_DETERMINISTIC_MODEL = "deterministic-sha256"


def _store_embedding(
    store: SQLiteStore,
    embedder: BaseEmbedder,
    ns: str,
    episode_id: str,
    content: str,
) -> None:
    """Compute and store an embedding for an episode."""
    vec = embedder.embed(content)
    blob = b"".join(struct.pack("!f", v) for v in vec)
    model_name = _DETERMINISTIC_MODEL if isinstance(embedder, DeterministicEmbedder) else "http-embedder"
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

    for ent in extraction.entities:
        try:
            eid = store.insert_entity_with_alias(
                ns=ns,
                name=ent["name"],
                entity_type=ent.get("type", "Entity"),
                confidence=ent.get("confidence", 0.85),
                source_episode_id=episode_id,
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


def ingest_memory(
    store: SQLiteStore,
    embedder: BaseEmbedder,
    extractor: BaseExtractor | None,
    ns: str,
    content: str,
    source_ref: str | None = None,
    metadata: dict[str, Any] | None = None,
    extract: bool = True,
    category: str = "other",
) -> dict[str, Any]:
    """Ingest a memory: validate, store episode+memory, embed, extract."""
    ns = validate_namespace(ns)
    content = validate_content(content)

    result = store.insert_memory(ns, content, source_ref, metadata, category=category)
    episode_id = result["episode_id"]

    _store_embedding(store, embedder, ns, episode_id, content)

    entities: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    warnings: list[str] = []
    if extract and extractor:
        entities, claims, warnings = _run_extraction(
            store=store,
            extractor=extractor,
            ns=ns,
            content=content,
            episode_id=episode_id,
        )

    return {
        "memory_id": result["memory_id"],
        "episode_id": episode_id,
        "entities": entities,
        "claims": claims,
        "warnings": warnings,
    }


def ingest_knowledge(
    store: SQLiteStore,
    embedder: BaseEmbedder,
    extractor: BaseExtractor | None,
    ns: str,
    title: str,
    content: str,
    source_ref: str | None = None,
    metadata: dict[str, Any] | None = None,
    extract: bool = True,
    category: str = "other",
) -> dict[str, Any]:
    """Ingest a knowledge item: validate, store, embed, extract."""
    ns = validate_namespace(ns)
    content = validate_content(content)
    title = validate_content(title, field="title")

    result = store.insert_knowledge(ns, title, content, source_ref, metadata, category=category)
    episode_id = result["episode_id"]

    _store_embedding(store, embedder, ns, episode_id, content)

    entities: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    warnings: list[str] = []
    if extract and extractor:
        entities, claims, warnings = _run_extraction(
            store=store,
            extractor=extractor,
            ns=ns,
            content=content,
            episode_id=episode_id,
        )

    return {
        "knowledge_id": result["knowledge_id"],
        "episode_id": episode_id,
        "entities": entities,
        "claims": claims,
        "warnings": warnings,
    }
