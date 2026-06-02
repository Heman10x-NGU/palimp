"""GraphCtx extraction layer.

Provides a base extractor interface plus:
- RuleBasedExtractor: deterministic, zero-dependency extractor for testing.
- HttpExtractor: optional OpenAI-compatible HTTP extractor via httpx.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx

from graphctx.models import ExtractionResult

logger = logging.getLogger(__name__)

# Allowed relation types for edges.
# Full set from AutoMem config; core subset is exposed in docs.
RELATIONS: list[str] = [
    "RELATES_TO",
    "LEADS_TO",
    "OCCURRED_BEFORE",
    "SIMILAR_TO",
    "PRECEDED_BY",
    "PREFERS_OVER",
    "EXEMPLIFIES",
    "CONTRADICTS",
    "REINFORCES",
    "INVALIDATED_BY",
    "EVOLVED_INTO",
    "DERIVED_FROM",
    "PART_OF",
    "USES",
    "WORKS_ON",
    "CREATED",
    "CONTAINS",
    "DEPENDS_ON",
    "SUPERSEDES",
]

# Extraction caps
MAX_ENTITIES = 20
MAX_EDGES = 30
MAX_CLAIMS = 30

# Confidence for rule-based extraction
RULE_CONFIDENCE = 0.85

# Articles to strip from entity names
_ARTICLES = {"a", "an", "the"}


def _normalize_entity_name(raw: str) -> str:
    """Strip leading articles and capitalize first letter."""
    name = raw.strip()
    parts = name.split()
    if parts and parts[0].lower() in _ARTICLES:
        parts = parts[1:]
    if not parts:
        return name
    normalized = " ".join(parts)
    return normalized[0].upper() + normalized[1:]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on common delimiters."""
    # Split on sentence-ending punctuation or semicolons.
    parts = re.split(r"[.;!?]+", text)
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Pattern definitions for rule-based extraction
# ---------------------------------------------------------------------------

# Each pattern: (regex, handler_fn)
# handler_fn(match) -> (entities, edges, claims)
#   entities: list of {"name": str, "type": str}
#   edges: list of {"source": str, "relation": str, "target": str}
#   claims: list of {"subject": str, "predicate": str, "object": str}


_PATTERNS: list[tuple[re.Pattern[str], Any]] = []


def _register(pattern: str):
    """Decorator to register a pattern handler."""

    def decorator(fn):
        _PATTERNS.append((re.compile(pattern, re.IGNORECASE), fn))
        return fn

    return decorator


@_register(r"^(.+?)\s+works?\s+on\s+(.+)$")
def _handle_works_on(m: re.Match) -> tuple:
    subj = _normalize_entity_name(m.group(1))
    obj = _normalize_entity_name(m.group(2))
    entities = [{"name": subj, "type": "Person"}, {"name": obj, "type": "Project"}]
    edges = [{"source": subj, "relation": "WORKS_ON", "target": obj}]
    return entities, edges, []


@_register(r"^(.+?)\s+uses?\s+(.+)$")
def _handle_uses(m: re.Match) -> tuple:
    subj = _normalize_entity_name(m.group(1))
    obj = _normalize_entity_name(m.group(2))
    entities = [{"name": subj, "type": "Entity"}, {"name": obj, "type": "Technology"}]
    edges = [{"source": subj, "relation": "USES", "target": obj}]
    return entities, edges, []


@_register(r"^(.+?)\s+prefers?\s+(.+)$")
def _handle_prefers(m: re.Match) -> tuple:
    subj = _normalize_entity_name(m.group(1))
    obj = m.group(2).strip()
    entities = [{"name": subj, "type": "Person"}]
    claims = [{"subject": subj, "predicate": "prefers", "object": obj}]
    return entities, [], claims


@_register(r"^(.+?)\s+is\s+an?\s+(.+)$")
def _handle_is_a(m: re.Match) -> tuple:
    subj = _normalize_entity_name(m.group(1))
    obj = m.group(2).strip()
    entities = [{"name": subj, "type": "Entity"}]
    claims = [{"subject": subj, "predicate": "is_a", "object": obj}]
    return entities, [], claims


@_register(r"^(.+?)\s+created?\s+(.+)$")
def _handle_created(m: re.Match) -> tuple:
    subj = _normalize_entity_name(m.group(1))
    obj = _normalize_entity_name(m.group(2))
    entities = [{"name": subj, "type": "Entity"}, {"name": obj, "type": "Entity"}]
    edges = [{"source": subj, "relation": "CREATED", "target": obj}]
    return entities, edges, []


@_register(r"^(.+?)\s+depends?\s+on\s+(.+)$")
def _handle_depends_on(m: re.Match) -> tuple:
    subj = _normalize_entity_name(m.group(1))
    obj = _normalize_entity_name(m.group(2))
    entities = [{"name": subj, "type": "Entity"}, {"name": obj, "type": "Entity"}]
    edges = [{"source": subj, "relation": "DEPENDS_ON", "target": obj}]
    return entities, edges, []


@_register(r"^(.+?)\s+contains?\s+(.+)$")
def _handle_contains(m: re.Match) -> tuple:
    subj = _normalize_entity_name(m.group(1))
    obj = _normalize_entity_name(m.group(2))
    entities = [{"name": subj, "type": "Entity"}, {"name": obj, "type": "Entity"}]
    edges = [{"source": subj, "relation": "CONTAINS", "target": obj}]
    return entities, edges, []


@_register(r"^(.+?)\s+contradicts?\s+(.+)$")
def _handle_contradicts(m: re.Match) -> tuple:
    subj = _normalize_entity_name(m.group(1))
    obj = _normalize_entity_name(m.group(2))
    entities = [{"name": subj, "type": "Entity"}, {"name": obj, "type": "Entity"}]
    edges = [{"source": subj, "relation": "CONTRADICTS", "target": obj}]
    return entities, edges, []


@_register(r"^(.+?)\s+supersedes?\s+(.+)$")
def _handle_supersedes(m: re.Match) -> tuple:
    subj = _normalize_entity_name(m.group(1))
    obj = _normalize_entity_name(m.group(2))
    entities = [{"name": subj, "type": "Entity"}, {"name": obj, "type": "Entity"}]
    edges = [{"source": subj, "relation": "SUPERSEDES", "target": obj}]
    return entities, edges, []


# ---------------------------------------------------------------------------
# Base extractor
# ---------------------------------------------------------------------------


class BaseExtractor(ABC):
    """Abstract base for all extractors."""

    @abstractmethod
    def extract(self, text: str) -> ExtractionResult:
        """Extract entities, edges, and claims from *text*.

        Returns an ``ExtractionResult``.  On failure the result should carry
        warnings and an empty entities/edges/claims list (never fabricated data).
        """


# ---------------------------------------------------------------------------
# Rule-based extractor
# ---------------------------------------------------------------------------


class RuleBasedExtractor(BaseExtractor):
    """Deterministic, zero-dependency extractor for testing.

    Applies simple regex patterns to sentences.  Entity names are normalised
    (leading articles stripped, first letter capitalised).  Extraction is
    capped at MAX_ENTITIES / MAX_EDGES / MAX_CLAIMS.
    """

    def extract(self, text: str) -> ExtractionResult:
        if not text or not text.strip():
            return ExtractionResult()

        sentences = _split_sentences(text)

        entity_map: dict[str, dict[str, str]] = {}  # name -> {"name": ..., "type": ...}
        edges: list[dict[str, Any]] = []
        claims: list[dict[str, Any]] = []
        warnings: list[str] = []

        for sentence in sentences:
            if len(entity_map) >= MAX_ENTITIES and len(edges) >= MAX_EDGES and len(claims) >= MAX_CLAIMS:
                break

            for pattern, handler in _PATTERNS:
                match = pattern.search(sentence)
                if match is None:
                    continue

                raw_entities, raw_edges, raw_claims = handler(match)

                # Add entities (deduplicated by name)
                for ent in raw_entities:
                    if ent["name"] not in entity_map and len(entity_map) < MAX_ENTITIES:
                        entity_map[ent["name"]] = ent

                # Add edges
                for edge in raw_edges:
                    if len(edges) < MAX_EDGES:
                        edges.append(edge)

                # Add claims
                for claim in raw_claims:
                    if len(claims) < MAX_CLAIMS:
                        claims.append(claim)

                break  # first matching pattern wins per sentence

        entities = [
            {"name": e["name"], "type": e["type"], "confidence": RULE_CONFIDENCE}
            for e in entity_map.values()
        ]

        return ExtractionResult(
            entities=entities[:MAX_ENTITIES],
            edges=edges[:MAX_EDGES],
            claims=claims[:MAX_CLAIMS],
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# HTTP (OpenAI-compatible) extractor
# ---------------------------------------------------------------------------


class HttpExtractor(BaseExtractor):
    """Optional OpenAI-compatible HTTP extractor via ``httpx``.

    Sends a structured JSON prompt to an OpenAI-compatible endpoint and
    parses the JSON response into an ``ExtractionResult``.

    On ANY failure the extractor logs a warning and returns an empty
    ``ExtractionResult`` with the warning recorded in ``warnings``.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def extract(self, text: str) -> ExtractionResult:
        if not text or not text.strip():
            return ExtractionResult()

        prompt = (
            "Extract structured knowledge from the following text. "
            "Return a JSON object with three keys: \"entities\" (list of "
            "{\"name\": str, \"type\": str, \"confidence\": float}), "
            "\"edges\" (list of {\"source\": str, \"relation\": str, "
            "\"target\": str}), and \"claims\" (list of {\"subject\": str, "
            "\"predicate\": str, \"object\": str}). "
            "Only include facts explicitly stated in the text. "
            "Respond with valid JSON only, no markdown fences.\n\n"
            f"Text:\n{text}"
        )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(self._endpoint, json=payload, headers=headers)
                resp.raise_for_status()
        except Exception as exc:
            msg = f"HttpExtractor request failed: {exc}"
            logger.warning(msg)
            return ExtractionResult(warnings=[msg])

        try:
            body = resp.json()
            content_str = body["choices"][0]["message"]["content"]
            data = json.loads(content_str)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            msg = f"HttpExtractor: failed to parse response as JSON: {exc}"
            logger.warning(msg)
            return ExtractionResult(warnings=[msg])

        # Validate structure loosely — missing keys default to empty lists.
        entities_raw = data.get("entities", [])
        edges_raw = data.get("edges", [])
        claims_raw = data.get("claims", [])

        # Validate that required keys exist in each item
        clean_entities: list[dict[str, Any]] = []
        for ent in entities_raw[:MAX_ENTITIES]:
            if isinstance(ent, dict) and "name" in ent:
                clean_entities.append({
                    "name": str(ent["name"]),
                    "type": str(ent.get("type", "Entity")),
                    "confidence": float(ent.get("confidence", 0.8)),
                })

        clean_edges: list[dict[str, Any]] = []
        for edge in edges_raw[:MAX_EDGES]:
            if isinstance(edge, dict) and all(k in edge for k in ("source", "relation", "target")):
                clean_edges.append({
                    "source": str(edge["source"]),
                    "relation": str(edge["relation"]),
                    "target": str(edge["target"]),
                })

        clean_claims: list[dict[str, Any]] = []
        for claim in claims_raw[:MAX_CLAIMS]:
            if isinstance(claim, dict) and all(k in claim for k in ("subject", "predicate", "object")):
                clean_claims.append({
                    "subject": str(claim["subject"]),
                    "predicate": str(claim["predicate"]),
                    "object": str(claim["object"]),
                })

        return ExtractionResult(
            entities=clean_entities,
            edges=clean_edges,
            claims=clean_claims,
            warnings=[],
        )
