"""Batch ingestion with parallel extraction.

Modeled after Graphiti's add_episode_bulk() and Mem0's phased batch pipeline.

Phase 1: Store all episodes + embeddings (fast, no LLM)
Phase 2: Parallel extraction (bounded by semaphore)
Phase 3: Store entities/edges/claims sequentially (SQLite constraint)
"""

from __future__ import annotations

import asyncio
import struct
import time
from dataclasses import dataclass
from typing import Any, Optional

from palimp.embeddings import BaseEmbedder
from palimp.extractor import ExtractionResult
from palimp.parallel_extractor import AsyncHttpExtractor
from palimp.storage import SQLiteStore
from palimp.validate import validate_content, validate_namespace

_DETERMINISTIC_MODEL = "deterministic-sha256"


@dataclass
class BatchItem:
    """Item for batch ingestion."""
    content: str
    source_ref: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    category: str = "other"


@dataclass
class BatchResult:
    """Result of batch ingestion."""
    episode_ids: list[str]
    extraction_results: list[ExtractionResult]
    store_ms: float
    extract_ms: float
    total_ms: float
    items_stored: int
    items_extracted: int


def ingest_batch_sync(
    store: SQLiteStore,
    embedder: BaseEmbedder,
    extractor: Optional[Any],
    ns: str,
    items: list[BatchItem],
    extract: bool = True,
) -> BatchResult:
    """Ingest multiple items with parallel extraction.

    Phase 1: Store all episodes + embeddings (fast, sync)
    Phase 2: Extract entities/edges/claims (parallel if async extractor)
    Phase 3: Store extraction results (sync, SQLite constraint)
    """
    ns = validate_namespace(ns)
    total_start = time.monotonic()

    # Phase 1: Store episodes + embeddings
    store_start = time.monotonic()
    episode_ids: list[str] = []
    for item in items:
        content = validate_content(item.content)
        result = store.insert_memory(ns, content, item.source_ref, item.metadata)
        episode_id = result["episode_id"]
        episode_ids.append(episode_id)

        # Store embedding
        try:
            vec = embedder.embed(content)
            blob = struct.pack(f"!{len(vec)}f", *vec)
            store.insert_embedding(
                ns, "episode", episode_id, _DETERMINISTIC_MODEL, len(vec), blob
            )
        except Exception:
            pass  # embedding failure is non-fatal
    store_ms = (time.monotonic() - store_start) * 1000

    # Phase 2: Extraction
    extract_start = time.monotonic()
    extraction_results: list[ExtractionResult] = []

    if extract and extractor:
        contents = [item.content for item in items]

        if isinstance(extractor, AsyncHttpExtractor):
            # Parallel extraction with semaphore
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    extraction_results = _extract_parallel_sync(extractor, contents)
                else:
                    extraction_results = loop.run_until_complete(
                        extractor.extract_batch(contents)
                    )
            except RuntimeError:
                extraction_results = asyncio.run(extractor.extract_batch(contents))
        else:
            # Sequential extraction
            for content in contents:
                try:
                    extraction_results.append(extractor.extract(content))
                except Exception as exc:
                    extraction_results.append(
                        ExtractionResult(warnings=[f"Extraction failed: {exc}"])
                    )
    else:
        extraction_results = [ExtractionResult() for _ in items]

    extract_ms = (time.monotonic() - extract_start) * 1000

    # Phase 3: Store extraction results
    for episode_id, extraction in zip(episode_ids, extraction_results):
        _store_extraction(store, ns, episode_id, extraction)

    total_ms = (time.monotonic() - total_start) * 1000

    return BatchResult(
        episode_ids=episode_ids,
        extraction_results=extraction_results,
        store_ms=store_ms,
        extract_ms=extract_ms,
        total_ms=total_ms,
        items_stored=len(episode_ids),
        items_extracted=len(extraction_results),
    )


def _extract_parallel_sync(
    extractor: AsyncHttpExtractor, contents: list[str]
) -> list[ExtractionResult]:
    """Run parallel extraction in a new event loop (sync wrapper)."""
    return asyncio.run(extractor.extract_batch(contents))


def _store_extraction(
    store: SQLiteStore, ns: str, episode_id: str, extraction: ExtractionResult
) -> None:
    """Store extraction results (entities, edges, claims) with provenance."""
    entity_ids: dict[str, str] = {}  # name -> entity_id

    # Store entities
    for ent in extraction.entities:
        name = ent.get("name", "").strip()
        if not name:
            continue
        ent_type = ent.get("type", "Unknown")
        confidence = float(ent.get("confidence", 0.85))
        try:
            ent_id = store.insert_entity_with_alias(
                ns, name, ent_type, confidence
            )
            entity_ids[name.lower()] = ent_id
            store.insert_provenance(
                ns, episode_id, entity_id=ent_id, extractor_version="parallel-v1"
            )
        except Exception:
            pass

    # Store edges
    for edge in extraction.edges:
        src_name = edge.get("source", "").strip().lower()
        tgt_name = edge.get("target", "").strip().lower()
        relation = edge.get("relation", "RELATES_TO")
        if src_name in entity_ids and tgt_name in entity_ids:
            try:
                edge_id = store.insert_edge(
                    ns, entity_ids[src_name], entity_ids[tgt_name], relation, 0.85
                )
                store.insert_provenance(
                    ns, episode_id, edge_id=edge_id, extractor_version="parallel-v1"
                )
            except Exception:
                pass

    # Store claims
    for claim in extraction.claims:
        subject = claim.get("subject", "").strip().lower()
        predicate = claim.get("predicate", "")
        obj = claim.get("object", "")
        if subject in entity_ids:
            try:
                claim_id = store.insert_claim(
                    ns, entity_ids[subject], predicate, obj, None, 0.85
                )
                store.insert_provenance(
                    ns, episode_id, claim_id=claim_id, extractor_version="parallel-v1"
                )
            except Exception:
                pass
