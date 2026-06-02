"""Palimp Python async SDK.

Provides a thin async client for the Palimp REST API using httpx.

Usage::

    from palimp.sdk import PalimpClient

    async def main():
        client = PalimpClient()
        result = await client.add_memory("demo", "Alice prefers concise answers.")
        print(result["memory_id"])
        await client.close()
"""

from __future__ import annotations

from typing import Any, Optional

import httpx


class PalimpClient:
    """Async client for the Palimp REST API."""

    def __init__(self, base_url: str = "http://localhost:8420"):
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> PalimpClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Health / stats
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """Return server health status."""
        return (await self._client.get("/v1/health")).json()

    async def stats(self, namespace: str) -> dict[str, Any]:
        """Return namespace statistics."""
        return (await self._client.get("/v1/stats", params={"namespace": namespace})).json()

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    async def add_memory(self, namespace: str, content: str, **kwargs: Any) -> dict[str, Any]:
        """Add a single memory."""
        return (await self._client.post("/v1/memories", json={
            "namespace": namespace, "content": content, **kwargs
        })).json()

    async def add_memory_batch(
        self, namespace: str, items: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        """Add multiple memories in one call."""
        return (await self._client.post("/v1/memories/batch", json={
            "namespace": namespace, "items": items, **kwargs
        })).json()

    # ------------------------------------------------------------------
    # Knowledge
    # ------------------------------------------------------------------

    async def add_knowledge(
        self, namespace: str, title: str, content: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Add a knowledge item."""
        return (await self._client.post("/v1/knowledge", json={
            "namespace": namespace, "title": title, "content": content, **kwargs
        })).json()

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    async def recall(self, namespace: str, query: str, **kwargs: Any) -> dict[str, Any]:
        """Run a recall query."""
        return (await self._client.post("/v1/recall", json={
            "namespace": namespace, "query": query, **kwargs
        })).json()

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    async def context(self, namespace: str, entity_id: str) -> dict[str, Any]:
        """Get context for an entity."""
        return (await self._client.get(
            f"/v1/context/{entity_id}", params={"namespace": namespace}
        )).json()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def create_session(self, namespace: str, **kwargs: Any) -> dict[str, Any]:
        """Create a new session."""
        return (await self._client.post("/v1/sessions", json={
            "namespace": namespace, **kwargs
        })).json()

    async def close_session(
        self, session_id: str, summary: Optional[str] = None
    ) -> dict[str, Any]:
        """Close a session with an optional summary."""
        body: dict[str, Any] = {}
        if summary is not None:
            body["summary"] = summary
        return (await self._client.post(
            f"/v1/sessions/{session_id}/close", json=body
        )).json()

    # ------------------------------------------------------------------
    # Pin / unpin
    # ------------------------------------------------------------------

    async def pin(self, namespace: str, episode_id: str) -> dict[str, Any]:
        """Pin an episode so it never decays."""
        return (await self._client.post(
            f"/v1/episodes/{episode_id}/pin", params={"namespace": namespace}
        )).json()

    async def unpin(self, namespace: str, episode_id: str) -> dict[str, Any]:
        """Unpin an episode so it resumes normal decay."""
        return (await self._client.post(
            f"/v1/episodes/{episode_id}/unpin", params={"namespace": namespace}
        )).json()
