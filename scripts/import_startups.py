#!/usr/bin/env python3
"""Import startups from an existing SQLite database into Palimp.

Reads from a startup-funding-db SQLite database (tables: startups, enrichments)
and ingests each startup as a memory item via Palimp's ingest_memory().

After importing each startup, runs spaCy NER (via HybridExtractor) to extract
entities and create structured graph edges from the startup metadata.

Usage:
    python scripts/import_startups.py
    python scripts/import_startups.py --dry-run --limit 10
    python scripts/import_startups.py --source-db /path/to/startups.db --namespace my-startups
    python scripts/import_startups.py --no-extract-entities  # skip NER extraction
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so palimp imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from palimp.embeddings import DeterministicEmbedder
from palimp.extractor import RuleBasedExtractor
from palimp.ingest import ingest_memory
from palimp.parallel_extractor import HybridExtractor
from palimp.storage import SQLiteStore

logger = logging.getLogger(__name__)

# Maximum characters for large text fields before truncation
MAX_LARGE_FIELD_CHARS = 4096

# Fields that contain JSON arrays in the enrichments table
JSON_ARRAY_FIELDS = frozenset({
    "pros",
    "cons",
    "missing_elements",
    "differentiators",
    "categories",
    "tech_stack",
    "keywords",
})

# Fields to skip entirely (not useful for text content)
SKIP_FIELDS = frozenset({
    "id",           # enrichments auto-increment id
    "startup_id",   # FK handled via metadata
    "enriched_at",  # timestamp, not content
    "enrichment_version",
    "confidence_score",
    "llm_model",
})

# Enrichment fields that map directly to the text output
ENRICHMENT_TEXT_FIELDS = {
    "what_they_do": "What they do",
    "problem_solved": "Problem solved",
    "value_proposition": "Value proposition",
    "target_customer": "Target customer",
    "business_model": "Business model",
    "competitive_moat": "Competitive moat",
}

ENRICHMENT_SCORE_FIELDS = {
    "opportunity_score": "Opportunity score",
    "risk_level": "Risk level",
}

# Startup fields for the main text block
STARTUP_TEXT_FIELDS = [
    ("name", None),
    ("industry", "is a {value} startup"),
    ("source", "from {value}"),
    ("batch", "{value}"),
    ("one_liner", None),
    ("description", None),
    ("tags", "Tags: {value}"),
    ("status", "Status: {value}"),
    ("stage", "Stage: {value}"),
    ("location", "Location: {value}"),
    ("funding_amount", None),  # handled specially with funding_round
    ("funding_round", None),
    ("team_size", "Team size: {value}"),
]


def _safe_str(value: object | None) -> str:
    """Convert a value to a stripped string, returning '' for None/empty."""
    if value is None:
        return ""
    s = str(value).strip()
    return s


def _truncate(text: str, max_len: int = MAX_LARGE_FIELD_CHARS) -> str:
    """Truncate text to max_len characters, adding an indicator if truncated."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "... [truncated]"


def _parse_json_array(raw: str | None) -> list[str]:
    """Parse a JSON array string, returning a list of strings."""
    if not raw or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _format_startup_text(row: dict, enrichment: dict | None) -> str:
    """Format a startup row + optional enrichment into structured text content."""
    parts: list[str] = []

    # Header line: "{name} is a {industry} startup from {source} {batch}."
    name = _safe_str(row.get("name"))
    industry = _safe_str(row.get("industry"))
    source = _safe_str(row.get("source"))
    batch = _safe_str(row.get("batch"))

    header_parts = [name]
    if industry:
        header_parts.append(f"is a {industry} startup")
    if source or batch:
        from_parts = [p for p in [source, batch] if p]
        header_parts.append("from " + " ".join(from_parts))
    parts.append(" ".join(header_parts) + ".")

    # One-liner and description
    one_liner = _safe_str(row.get("one_liner"))
    if one_liner:
        parts.append(one_liner)

    description = _safe_str(row.get("description"))
    if description:
        parts.append(description)

    # Metadata lines
    tags = _safe_str(row.get("tags"))
    if tags:
        parts.append(f"Tags: {tags}.")

    status = _safe_str(row.get("status"))
    if status:
        parts.append(f"Status: {status}.")

    stage = _safe_str(row.get("stage"))
    if stage:
        parts.append(f"Stage: {stage}.")

    location = _safe_str(row.get("location"))
    if location:
        parts.append(f"Location: {location}.")

    # Funding (combine amount + round)
    funding_amount = _safe_str(row.get("funding_amount"))
    funding_round = _safe_str(row.get("funding_round"))
    if funding_amount or funding_round:
        funding_str = funding_amount
        if funding_round:
            funding_str = f"{funding_amount} ({funding_round})" if funding_amount else funding_round
        parts.append(f"Funding: {funding_str}.")

    team_size = row.get("team_size")
    if team_size is not None:
        parts.append(f"Team size: {team_size}.")

    # Enrichment section
    if enrichment:
        for field, label in ENRICHMENT_TEXT_FIELDS.items():
            value = _safe_str(enrichment.get(field))
            if field in ("full_analysis", "scraped_content"):
                value = _truncate(value)
            if value:
                parts.append(f"{label}: {value}")

        # Score fields
        opp_score = enrichment.get("opportunity_score")
        if opp_score is not None and opp_score != 0:
            parts.append(f"Opportunity score: {opp_score}/10")

        risk = _safe_str(enrichment.get("risk_level"))
        if risk:
            parts.append(f"Risk level: {risk}")

        # Flatten JSON array fields into the text
        for json_field in ("categories", "tech_stack", "keywords"):
            items = _parse_json_array(enrichment.get(json_field))
            if items:
                label = json_field.replace("_", " ").title()
                parts.append(f"{label}: {', '.join(items)}")

        # Pros and cons as bullet points
        pros = _parse_json_array(enrichment.get("pros"))
        if pros:
            parts.append("Pros:")
            for p in pros:
                parts.append(f"  - {p}")

        cons = _parse_json_array(enrichment.get("cons"))
        if cons:
            parts.append("Cons:")
            for c in cons:
                parts.append(f"  - {c}")

        differentiators = _parse_json_array(enrichment.get("differentiators"))
        if differentiators:
            parts.append("Differentiators:")
            for d in differentiators:
                parts.append(f"  - {d}")

    return "\n".join(parts)


def _build_metadata(row: dict, enrichment: dict | None, namespace: str) -> dict:
    """Build the JSON metadata dict for a startup."""
    tags_raw = _safe_str(row.get("tags"))
    tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

    meta: dict = {
        "source": _safe_str(row.get("source")),
        "batch": _safe_str(row.get("batch")),
        "industry": _safe_str(row.get("industry")),
        "tags": tags_list,
        "status": _safe_str(row.get("status")),
        "location": _safe_str(row.get("location")),
        "startup_id": _safe_str(row.get("id")),
    }

    # Add enrichment metadata if present
    if enrichment:
        opp_score = enrichment.get("opportunity_score")
        if opp_score is not None and opp_score != 0:
            meta["opportunity_score"] = opp_score

        risk = _safe_str(enrichment.get("risk_level"))
        if risk:
            meta["risk_level"] = risk

        categories = _parse_json_array(enrichment.get("categories"))
        if categories:
            meta["categories"] = categories

        business_model = _safe_str(enrichment.get("business_model"))
        if business_model:
            meta["business_model"] = business_model

    # Remove empty string values
    return {k: v for k, v in meta.items() if v != ""}


def _fetch_startups(conn: sqlite3.Connection, limit: int = 0) -> list[dict]:
    """Fetch all startup rows from the source database."""
    query = "SELECT * FROM startups ORDER BY id"
    if limit > 0:
        query += f" LIMIT {limit}"
    rows = conn.execute(query).fetchall()
    return [dict(r) for r in rows]


def _fetch_enrichments(conn: sqlite3.Connection) -> dict[str, dict]:
    """Fetch all enrichments keyed by startup_id."""
    rows = conn.execute("SELECT * FROM enrichments").fetchall()
    return {row["startup_id"]: dict(row) for row in rows}


def _count_startups(conn: sqlite3.Connection) -> int:
    """Count total startups in the source database."""
    row = conn.execute("SELECT COUNT(*) as cnt FROM startups").fetchone()
    return row["cnt"]


def _count_enrichments(conn: sqlite3.Connection) -> int:
    """Count total enrichments in the source database."""
    row = conn.execute("SELECT COUNT(*) as cnt FROM enrichments").fetchone()
    return row["cnt"]


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


# ---------------------------------------------------------------------------
# Entity extraction constants and functions
# ---------------------------------------------------------------------------

# Map spaCy NER labels to our entity types.
# spaCy uses: ORG, PERSON, GPE, LOC, PRODUCT, MONEY, etc.
_SPACY_TYPE_MAP: dict[str, str] = {
    "ORG": "Company",
    "PERSON": "Person",
    "GPE": "Location",
    "LOC": "Location",
    "PRODUCT": "Product",
}

# Fields from the startups table that should be extracted as structured entities.
# (field_name, entity_type, edge_relation)
_STRUCTURED_FIELDS: list[tuple[str, str, str]] = [
    ("industry", "Industry", "IN_INDUSTRY"),
    ("location", "Location", "LOCATED_IN"),
    ("batch", "Batch", "PART_OF_BATCH"),
    ("source", "Source", "FROM_SOURCE"),
]


def _extract_startup_entities(
    store: SQLiteStore,
    extractor: HybridExtractor,
    startup_row: dict,
    enrichment: dict | None,
    content: str,
    episode_id: str,
    ns: str,
) -> dict[str, int]:
    """Extract entities and edges from startup data using spaCy NER + structured metadata.

    Returns stats dict with keys: entities_created, edges_created,
    spacy_hits, llm_fallbacks.
    """
    stats = {
        "entities_created": 0,
        "edges_created": 0,
        "spacy_hits": 0,
        "llm_fallbacks": 0,
    }

    name_to_id: dict[str, str] = {}

    # --- 1. Create the company entity from the startup name ---
    company_name = _safe_str(startup_row.get("name"))
    if not company_name:
        return stats

    company_id = store.insert_entity_with_alias(
        ns=ns,
        name=company_name,
        entity_type="Company",
        confidence=0.95,
        source_episode_id=episode_id,
    )
    store.insert_provenance(
        ns=ns,
        episode_id=episode_id,
        entity_id=company_id,
        extractor_version="startup-importer-1.0",
        evidence_span=f"startup name: {company_name}",
    )
    name_to_id[company_name] = company_id
    stats["entities_created"] += 1

    # --- 2. Create structured entities + edges from metadata ---
    for field, entity_type, relation in _STRUCTURED_FIELDS:
        value = _safe_str(startup_row.get(field))
        if not value:
            continue

        ent_id = store.insert_entity_with_alias(
            ns=ns,
            name=value,
            entity_type=entity_type,
            confidence=0.95,
            source_episode_id=episode_id,
        )
        store.insert_provenance(
            ns=ns,
            episode_id=episode_id,
            entity_id=ent_id,
            extractor_version="startup-importer-1.0",
            evidence_span=f"startup {field}: {value}",
        )
        name_to_id[value] = ent_id
        stats["entities_created"] += 1

        edge_id = store.insert_edge(
            ns=ns,
            source_id=company_id,
            target_id=ent_id,
            relation=relation,
            confidence=0.95,
        )
        store.insert_provenance(
            ns=ns,
            episode_id=episode_id,
            edge_id=edge_id,
            extractor_version="startup-importer-1.0",
        )
        stats["edges_created"] += 1

    # --- 3. Tech stack from enrichment metadata ---
    if enrichment:
        tech_items = _parse_json_array(enrichment.get("tech_stack"))
        for tech in tech_items:
            tech = tech.strip()
            if not tech:
                continue
            ent_id = store.insert_entity_with_alias(
                ns=ns,
                name=tech,
                entity_type="Technology",
                confidence=0.90,
                source_episode_id=episode_id,
            )
            store.insert_provenance(
                ns=ns,
                episode_id=episode_id,
                entity_id=ent_id,
                extractor_version="startup-importer-1.0",
                evidence_span=f"tech_stack: {tech}",
            )
            name_to_id[tech] = ent_id
            stats["entities_created"] += 1

            edge_id = store.insert_edge(
                ns=ns,
                source_id=company_id,
                target_id=ent_id,
                relation="USES_TECHNOLOGY",
                confidence=0.90,
            )
            store.insert_provenance(
                ns=ns,
                episode_id=episode_id,
                edge_id=edge_id,
                extractor_version="startup-importer-1.0",
            )
            stats["edges_created"] += 1

    # --- 4. spaCy NER on the full content text ---
    prev_spacy = extractor._spacy_hits
    prev_llm = extractor._llm_fallbacks

    result = extractor.extract(content)

    stats["spacy_hits"] = extractor._spacy_hits - prev_spacy
    stats["llm_fallbacks"] = extractor._llm_fallbacks - prev_llm

    for ent in result.entities:
        ent_name = ent.get("name", "").strip()
        spacy_type = ent.get("type", "")

        if not ent_name or len(ent_name) <= 1:
            continue

        # Skip if already created from structured metadata
        if ent_name in name_to_id:
            continue

        # Map spaCy label to our type
        mapped_type = _SPACY_TYPE_MAP.get(spacy_type, "Entity")

        # Skip money/date entities -- not useful for graph
        if spacy_type in ("MONEY", "DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY"):
            continue

        ent_id = store.insert_entity_with_alias(
            ns=ns,
            name=ent_name,
            entity_type=mapped_type,
            confidence=ent.get("confidence", 0.90),
            source_episode_id=episode_id,
        )
        store.insert_provenance(
            ns=ns,
            episode_id=episode_id,
            entity_id=ent_id,
            extractor_version="spacy-ner-en_core_web_sm",
            evidence_span=f"NER entity: {ent_name} ({spacy_type})",
        )
        name_to_id[ent_name] = ent_id
        stats["entities_created"] += 1

        # Create edge from company to this entity based on type
        edge_relation = {
            "Company": "MENTIONED_WITH",
            "Person": "ASSOCIATED_WITH",
            "Location": "LOCATED_IN",
            "Product": "OFFERS_PRODUCT",
        }.get(mapped_type)

        if edge_relation:
            edge_id = store.insert_edge(
                ns=ns,
                source_id=company_id,
                target_id=ent_id,
                relation=edge_relation,
                confidence=ent.get("confidence", 0.90),
            )
            store.insert_provenance(
                ns=ns,
                episode_id=episode_id,
                edge_id=edge_id,
                extractor_version="spacy-ner-en_core_web_sm",
            )
            stats["edges_created"] += 1

    # --- 5. Regex-based edges from HybridExtractor ---
    for edge in result.edges:
        src_name = edge.get("source", "").strip()
        tgt_name = edge.get("target", "").strip()
        relation = edge.get("relation", "")

        if not src_name or not tgt_name or not relation:
            continue

        # Only create edge if both endpoints are known entities
        src_id = name_to_id.get(src_name)
        tgt_id = name_to_id.get(tgt_name)
        if src_id and tgt_id:
            edge_id = store.insert_edge(
                ns=ns,
                source_id=src_id,
                target_id=tgt_id,
                relation=relation,
                confidence=0.80,
            )
            store.insert_provenance(
                ns=ns,
                episode_id=episode_id,
                edge_id=edge_id,
                extractor_version="hybrid-regex-1.0",
            )
            stats["edges_created"] += 1

    return stats


def _extract_startup_entities_dry(
    extractor: HybridExtractor,
    startup_row: dict,
    enrichment: dict | None,
    content: str,
) -> dict[str, Any]:
    """Simulate entity extraction for dry-run mode (no DB writes).

    Returns stats + the list of entities and edges that would be created.
    """
    entities: list[dict[str, str]] = []
    edges: list[dict[str, str]] = []

    company_name = _safe_str(startup_row.get("name"))
    if company_name:
        entities.append({"name": company_name, "type": "Company"})

    # Structured fields
    for field, entity_type, relation in _STRUCTURED_FIELDS:
        value = _safe_str(startup_row.get(field))
        if value:
            entities.append({"name": value, "type": entity_type})
            if company_name:
                edges.append({"source": company_name, "relation": relation, "target": value})

    # Tech stack
    if enrichment:
        for tech in _parse_json_array(enrichment.get("tech_stack")):
            tech = tech.strip()
            if tech:
                entities.append({"name": tech, "type": "Technology"})
                if company_name:
                    edges.append({"source": company_name, "relation": "USES_TECHNOLOGY", "target": tech})

    # spaCy NER
    prev_spacy = extractor._spacy_hits
    prev_llm = extractor._llm_fallbacks
    result = extractor.extract(content)
    spacy_delta = extractor._spacy_hits - prev_spacy
    llm_delta = extractor._llm_fallbacks - prev_llm

    known = {e["name"] for e in entities}
    for ent in result.entities:
        ent_name = ent.get("name", "").strip()
        spacy_type = ent.get("type", "")
        if not ent_name or len(ent_name) <= 1 or ent_name in known:
            continue
        if spacy_type in ("MONEY", "DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY"):
            continue
        mapped_type = _SPACY_TYPE_MAP.get(spacy_type, "Entity")
        entities.append({"name": ent_name, "type": f"{mapped_type} ({spacy_type})"})
        known.add(ent_name)
        if company_name:
            edge_rel = {
                "Company": "MENTIONED_WITH",
                "Person": "ASSOCIATED_WITH",
                "Location": "LOCATED_IN",
                "Product": "OFFERS_PRODUCT",
            }.get(mapped_type)
            if edge_rel:
                edges.append({"source": company_name, "relation": edge_rel, "target": ent_name})

    return {
        "entities": entities,
        "edges": edges,
        "spacy_hits": spacy_delta,
        "llm_fallbacks": llm_delta,
    }


def print_summary(
    total_startups: int,
    total_enrichments: int,
    ingested: int,
    skipped: int,
    errors: int,
    elapsed: float,
    entity_stats: dict[str, int] | None = None,
) -> None:
    """Print a final summary of the import run."""
    rate = ingested / elapsed if elapsed > 0 else 0
    print("\n" + "=" * 60)
    print("IMPORT SUMMARY")
    print("=" * 60)
    print(f"  Total startups in source DB:  {total_startups}")
    print(f"  Total enrichments in source:  {total_enrichments}")
    print(f"  Successfully ingested:        {ingested}")
    print(f"  Skipped (empty content):      {skipped}")
    print(f"  Errors:                       {errors}")
    print(f"  Time elapsed:                 {_format_elapsed(elapsed)}")
    print(f"  Throughput:                   {rate:.1f} items/sec")

    if entity_stats:
        print()
        print("  ENTITY EXTRACTION STATS")
        print("  " + "-" * 40)
        print(f"  Total entities extracted:     {entity_stats.get('total_entities', 0)}")
        print(f"  Total edges created:          {entity_stats.get('total_edges', 0)}")
        print(f"  spaCy NER hits:               {entity_stats.get('spacy_hits', 0)}")
        print(f"  LLM fallbacks:                {entity_stats.get('llm_fallbacks', 0)}")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import startups from a SQLite database into Palimp.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-db",
        type=str,
        default="/Users/heman10x/Downloads/startup-funding-db/data/startups.db",
        help="Path to the source startups.db SQLite database.",
    )
    parser.add_argument(
        "--palimp-db",
        type=str,
        default=os.path.expanduser("~/.palimp/palimp.db"),
        help="Path to the Palimp database.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="startups",
        help="Namespace for the ingested data.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of startups to import (0 for all).",
    )
    parser.add_argument(
        "--skip-enrichments",
        action="store_true",
        help="Skip enrichment data, import startup data only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be imported without actually importing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for progress reporting (default: 50).",
    )
    parser.add_argument(
        "--no-extract-entities",
        action="store_true",
        help="Skip spaCy NER entity extraction and edge creation.",
    )

    args = parser.parse_args()

    # Validate source database exists
    source_db = Path(args.source_db)
    if not source_db.exists():
        print(f"ERROR: Source database not found: {source_db}", file=sys.stderr)
        sys.exit(1)

    # Open source database
    src_conn = sqlite3.connect(str(source_db))
    src_conn.row_factory = sqlite3.Row

    total_startups = _count_startups(src_conn)
    total_enrichments = _count_enrichments(src_conn)

    effective_limit = args.limit if args.limit > 0 else total_startups
    print(f"Source: {source_db}")
    print(f"  Startups: {total_startups}, Enrichments: {total_enrichments}")
    print(f"  Importing: {effective_limit} startups")
    if args.skip_enrichments:
        print("  Enrichments: SKIPPED")
    print(f"  Namespace: {args.namespace}")
    print()

    # Fetch data
    startups = _fetch_startups(src_conn, limit=args.limit)
    enrichments: dict[str, dict] = {}
    if not args.skip_enrichments:
        enrichments = _fetch_enrichments(src_conn)

    # Prepare store and dependencies for actual import
    store: SQLiteStore | None = None
    embedder: DeterministicEmbedder | None = None
    extractor: RuleBasedExtractor | None = None

    # Create HybridExtractor for spaCy NER (works in both dry-run and actual mode)
    hybrid_extractor: HybridExtractor | None = None
    if not args.no_extract_entities:
        fallback_extractor = RuleBasedExtractor()
        hybrid_extractor = HybridExtractor(http_extractor=fallback_extractor)
        if hybrid_extractor._nlp is None:
            print("WARNING: spaCy model not available. Entity extraction will use rule-based fallback.")
        else:
            print("spaCy NER model loaded. Entity extraction enabled.")

    if not args.dry_run:
        # Ensure parent directory exists for palimp db
        palimp_db = Path(args.palimp_db)
        palimp_db.parent.mkdir(parents=True, exist_ok=True)

        store = SQLiteStore(str(palimp_db))
        embedder = DeterministicEmbedder(dim=384)
        extractor = RuleBasedExtractor()

    # Track progress
    ingested = 0
    skipped = 0
    errors = 0
    enrichment_merged = 0
    total_entities = 0
    total_edges = 0
    total_spacy_hits = 0
    total_llm_fallbacks = 0
    t_start = time.monotonic()

    for idx, startup in enumerate(startups):
        startup_id = startup["id"]
        enrichment = enrichments.get(startup_id) if enrichments else None

        # Format content
        content = _format_startup_text(startup, enrichment)
        metadata = _build_metadata(startup, enrichment, args.namespace)

        # Skip if content is effectively empty
        if not content.strip():
            skipped += 1
            continue

        if args.dry_run:
            # Dry run: print what would be imported
            print(f"--- [{idx + 1}/{len(startups)}] {startup.get('name', 'UNKNOWN')} ---")
            print(content)
            print(f"  metadata: {json.dumps(metadata, indent=2)}")

            # Entity extraction (dry-run: show what would be extracted)
            if hybrid_extractor:
                dry_result = _extract_startup_entities_dry(
                    extractor=hybrid_extractor,
                    startup_row=startup,
                    enrichment=enrichment,
                    content=content,
                )
                ent_count = len(dry_result["entities"])
                edge_count = len(dry_result["edges"])
                total_entities += ent_count
                total_edges += edge_count
                total_spacy_hits += dry_result["spacy_hits"]
                total_llm_fallbacks += dry_result["llm_fallbacks"]

                print(f"  entities ({ent_count}):")
                for ent in dry_result["entities"]:
                    print(f"    - {ent['name']} [{ent['type']}]")
                print(f"  edges ({edge_count}):")
                for edge in dry_result["edges"]:
                    print(f"    - {edge['source']} --[{edge['relation']}]--> {edge['target']}")

            print()
            ingested += 1
            if enrichment:
                enrichment_merged += 1
        else:
            # Actual import
            assert store is not None
            assert embedder is not None
            source_ref = f"startup:{startup_id}"
            try:
                result = ingest_memory(
                    store=store,
                    embedder=embedder,
                    extractor=extractor,
                    ns=args.namespace,
                    content=content,
                    source_ref=source_ref,
                    metadata=metadata,
                    extract=False,  # Skip default extraction; we do startup-specific extraction below
                    category="knowledge",
                )
                ingested += 1
                if enrichment:
                    enrichment_merged += 1

                # Entity extraction with spaCy NER + structured metadata edges
                if hybrid_extractor:
                    episode_id = result["episode_id"]
                    ent_stats = _extract_startup_entities(
                        store=store,
                        extractor=hybrid_extractor,
                        startup_row=startup,
                        enrichment=enrichment,
                        content=content,
                        episode_id=episode_id,
                        ns=args.namespace,
                    )
                    total_entities += ent_stats["entities_created"]
                    total_edges += ent_stats["edges_created"]
                    total_spacy_hits += ent_stats["spacy_hits"]
                    total_llm_fallbacks += ent_stats["llm_fallbacks"]

            except Exception as exc:
                errors += 1
                print(f"  ERROR ingesting {startup_id}: {exc}", file=sys.stderr)

        # Progress reporting
        if (idx + 1) % args.batch_size == 0:
            elapsed = time.monotonic() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  Progress: {idx + 1}/{len(startups)} ({rate:.1f} items/sec)")

    elapsed = time.monotonic() - t_start

    src_conn.close()

    # Build entity stats
    entity_stats: dict[str, int] | None = None
    if not args.no_extract_entities:
        entity_stats = {
            "total_entities": total_entities,
            "total_edges": total_edges,
            "spacy_hits": total_spacy_hits,
            "llm_fallbacks": total_llm_fallbacks,
        }

    # Print summary
    print_summary(
        total_startups=total_startups,
        total_enrichments=total_enrichments,
        ingested=ingested,
        skipped=skipped,
        errors=errors,
        elapsed=elapsed,
        entity_stats=entity_stats,
    )

    if args.dry_run:
        print(f"\n  Enrichments merged: {enrichment_merged}")
        print("  (Dry run -- no data was written)")


if __name__ == "__main__":
    main()
