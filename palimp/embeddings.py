"""Palimp embedding layer.

Provides a base embedder interface plus:
- DeterministicEmbedder: hash-based, zero LLM dependency, for testing.
- HttpEmbedder: optional HTTP embedding provider via httpx.
"""

from __future__ import annotations

import hashlib
import math
import struct
from abc import ABC, abstractmethod

import httpx


class BaseEmbedder(ABC):
    """Abstract base for all embedders."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into vectors."""


class DeterministicEmbedder(BaseEmbedder):
    """Hash-based embedding for testing. Zero LLM dependency.

    Produces deterministic, reproducible vectors from text using repeated
    SHA-256 hashing.  Vectors are L2-normalised so cosine similarity works
    correctly.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        """Embed text by expanding SHA-256 digest to *dim* floats via repeated hashing, then L2-normalise."""
        raw_bytes = text.encode("utf-8")
        floats: list[float] = []
        # We need ceil(dim / 8) hashes (SHA-256 produces 32 bytes = 8 floats of 4 bytes each).
        hashes_needed = (self._dim + 7) // 8
        for i in range(hashes_needed):
            h = hashlib.sha256(raw_bytes + i.to_bytes(4, "big")).digest()
            # Unpack 8 floats (32 bytes / 4 bytes per float)
            for j in range(8):
                if len(floats) >= self._dim:
                    break
                val = struct.unpack_from("!f", h, j * 4)[0]
                # Replace NaN/Inf with 0
                if math.isnan(val) or math.isinf(val):
                    val = 0.0
                floats.append(val)

        # L2-normalise
        norm = math.sqrt(sum(v * v for v in floats))
        if norm > 0:
            floats = [v / norm for v in floats]
        return floats[: self._dim]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        return [self.embed(t) for t in texts]


class HttpEmbedder(BaseEmbedder):
    """Optional HTTP embedding provider via ``httpx``.

    Sends text to an OpenAI-compatible ``/v1/embeddings`` endpoint and
    returns the resulting vector.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        dim: int = 384,
        timeout: float = 30.0,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._model = model
        self._dim = dim
        self._timeout = timeout

    @property
    def dimension(self) -> int:
        return self._dim

    def _request(self, texts: list[str]) -> list[list[float]]:
        """Send embedding request and return list of vectors."""
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key and self._api_key.strip():
            headers["Authorization"] = f"Bearer {self._api_key}"
        # Truncate each text to 6000 characters to fit context limits (Ollama/API)
        truncated_texts = [t[:6000] for t in texts]
        payload = {
            "model": self._model,
            "input": truncated_texts,
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(self._endpoint, json=payload, headers=headers)
            resp.raise_for_status()

        body = resp.json()
        data = body.get("data", [])
        # Sort by index to guarantee ordering
        data_sorted = sorted(data, key=lambda d: d.get("index", 0))
        return [item["embedding"] for item in data_sorted]

    def embed(self, text: str) -> list[float]:
        """Embed a single text via HTTP."""
        vectors = self._request([text])
        return vectors[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via HTTP."""
        return self._request(texts)
