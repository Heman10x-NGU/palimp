"""Parallel and hybrid extractors for fast benchmark ingestion.

Implements three optimization layers:
1. Parallel extraction with semaphore-bounded concurrency (Graphiti pattern)
2. Extraction cache (Graphiti LLMCache pattern)
3. Hybrid spaCy-first, LLM-fallback (MemoryOS/AutoMem pattern)
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional

import httpx

from palimp.extraction_cache import ExtractionCache
from palimp.extractor import BaseExtractor, ExtractionResult, HttpExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1: Parallel HttpExtractor
# ---------------------------------------------------------------------------


class AsyncHttpExtractor(BaseExtractor):
    """Async version of HttpExtractor with semaphore-bounded concurrency.

    Modeled after Graphiti's semaphore_gather() pattern.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        max_concurrent: int = 10,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def extract(self, text: str) -> ExtractionResult:
        """Synchronous fallback — runs async extract in event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in an async context — use sync httpx
                return self._extract_sync(text)
            return loop.run_until_complete(self.extract_async(text))
        except RuntimeError:
            return asyncio.run(self.extract_async(text))

    def _extract_sync(self, text: str) -> ExtractionResult:
        """Synchronous extraction using httpx.Client."""
        if not text or not text.strip():
            return ExtractionResult()
        prompt = self._build_prompt(text)
        is_anthropic = "anthropic" in self._endpoint or "/messages" in self._endpoint
        payload, headers = self._build_request(prompt, is_anthropic)
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(self._endpoint, json=payload, headers=headers)
                resp.raise_for_status()
            return self._parse_response(resp.json(), is_anthropic)
        except Exception as exc:
            return ExtractionResult(warnings=[f"HttpExtractor sync failed: {exc}"])

    async def extract_async(self, text: str) -> ExtractionResult:
        """Async extraction with semaphore-bounded concurrency."""
        if not text or not text.strip():
            return ExtractionResult()
        async with self._semaphore:
            return await self._extract_async_inner(text)

    async def _extract_async_inner(self, text: str) -> ExtractionResult:
        prompt = self._build_prompt(text)
        is_anthropic = "anthropic" in self._endpoint or "/messages" in self._endpoint
        payload, headers = self._build_request(prompt, is_anthropic)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._endpoint, json=payload, headers=headers)
                resp.raise_for_status()
            return self._parse_response(resp.json(), is_anthropic)
        except Exception as exc:
            return ExtractionResult(warnings=[f"HttpExtractor async failed: {exc}"])

    async def extract_batch(self, texts: list[str]) -> list[ExtractionResult]:
        """Extract from multiple texts concurrently with bounded parallelism."""
        tasks = [self.extract_async(t) for t in texts]
        return await asyncio.gather(*tasks)

    def _build_prompt(self, text: str) -> str:
        return (
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

    def _build_request(self, prompt: str, is_anthropic: bool) -> tuple[dict, dict]:
        if is_anthropic:
            payload = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.0,
            }
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            }
        else:
            payload = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            headers = {"Content-Type": "application/json"}
            if self._api_key and self._api_key.strip():
                headers["Authorization"] = f"Bearer {self._api_key}"
        return payload, headers

    def _parse_response(self, body: dict, is_anthropic: bool) -> ExtractionResult:
        import json as _json

        try:
            if is_anthropic:
                content_str = body["content"][0]["text"]
            else:
                content_str = body["choices"][0]["message"]["content"]
            content_str = content_str.strip()
            if "```json" in content_str:
                content_str = content_str.split("```json")[1].split("```")[0].strip()
            elif "```" in content_str:
                content_str = content_str.split("```")[1].split("```")[0].strip()
            data = _json.loads(content_str)
        except Exception as exc:
            return ExtractionResult(warnings=[f"Parse failed: {exc}"])

        return ExtractionResult(
            entities=data.get("entities", []),
            edges=data.get("edges", []),
            claims=data.get("claims", []),
            warnings=[],
        )


# ---------------------------------------------------------------------------
# Layer 2: Cached Extractor
# ---------------------------------------------------------------------------


class CachedExtractor(BaseExtractor):
    """Extractor with SQLite-backed cache. Wraps any BaseExtractor.

    Modeled after Graphiti's LLMCache pattern.
    """

    def __init__(self, inner: BaseExtractor, cache: Optional[ExtractionCache] = None) -> None:
        self._inner = inner
        self._cache = cache or ExtractionCache()
        self._model = getattr(inner, "_model", "unknown")
        self._hits = 0
        self._misses = 0

    def extract(self, text: str) -> ExtractionResult:
        cached = self._cache.get(self._model, text)
        if cached:
            self._hits += 1
            return ExtractionResult(**cached)
        self._misses += 1
        result = self._inner.extract(text)
        self._cache.set(self._model, text, result.model_dump())
        return result

    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, **self._cache.stats()}


# ---------------------------------------------------------------------------
# Layer 3: Hybrid Extractor (spaCy-first, LLM-fallback)
# ---------------------------------------------------------------------------

_SPACY_AVAILABLE = False
try:
    import spacy

    _SPACY_AVAILABLE = True
except ImportError:
    pass

# Regex patterns for edge extraction (from Mnemon's approach)
_EDGE_PATTERNS = [
    (re.compile(r"(\w+)\s+works?\s+(?:on|for|at)\s+(\w+)", re.I), "WORKS_ON"),
    (re.compile(r"(\w+)\s+uses?\s+(\w+)", re.I), "USES"),
    (re.compile(r"(\w+)\s+created?\s+(\w+)", re.I), "CREATED"),
    (re.compile(r"(\w+)\s+depends?\s+on\s+(\w+)", re.I), "DEPENDS_ON"),
    (re.compile(r"(\w+)\s+contains?\s+(\w+)", re.I), "CONTAINS"),
    (re.compile(r"(\w+)\s+leads?\s+to\s+(\w+)", re.I), "LEADS_TO"),
    (re.compile(r"(\w+)\s+relates?\s+to\s+(\w+)", re.I), "RELATES_TO"),
]


class HybridExtractor(BaseExtractor):
    """Hybrid extractor: spaCy NER first, LLM fallback for complex text.

    Modeled after MemoryOS and AutoMem patterns.
    spaCy handles ~70-80% of conversation turns without any LLM call.
    """

    def __init__(
        self,
        http_extractor: BaseExtractor,
        spacy_model: str = "en_core_web_sm",
        confidence_threshold: float = 0.7,
    ) -> None:
        self._http = http_extractor
        self._confidence_threshold = confidence_threshold
        self._nlp = None
        self._spacy_hits = 0
        self._llm_fallbacks = 0

        if _SPACY_AVAILABLE:
            try:
                self._nlp = spacy.load(spacy_model)
            except OSError:
                logger.warning(f"spaCy model '{spacy_model}' not found. Using LLM only.")

    def extract(self, text: str) -> ExtractionResult:
        if not text or not text.strip():
            return ExtractionResult()

        # Tier 1: spaCy NER (0ms, local)
        if self._nlp:
            spacy_result = self._extract_spacy(text)
            if spacy_result.entities:
                self._spacy_hits += 1
                return spacy_result

        # Tier 2: LLM fallback
        self._llm_fallbacks += 1
        return self._http.extract(text)

    def _extract_spacy(self, text: str) -> ExtractionResult:
        doc = self._nlp(text)
        entities = []
        seen = set()
        for ent in doc.ents:
            name = ent.text.strip()
            if name and name.lower() not in seen and len(name) > 1:
                seen.add(name.lower())
                entities.append({
                    "name": name,
                    "type": ent.label_,
                    "confidence": 0.9,
                })

        # Regex-based edge extraction (Mnemon pattern)
        edges = []
        for pattern, relation in _EDGE_PATTERNS:
            for match in pattern.finditer(text):
                src, tgt = match.group(1), match.group(2)
                if src.lower() != tgt.lower():
                    edges.append({
                        "source": src,
                        "relation": relation,
                        "target": tgt,
                    })

        return ExtractionResult(entities=entities, edges=edges, claims=[], warnings=[])

    def stats(self) -> dict[str, int]:
        return {
            "spacy_hits": self._spacy_hits,
            "llm_fallbacks": self._llm_fallbacks,
            "spacy_available": _SPACY_AVAILABLE,
        }
