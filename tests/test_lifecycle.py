"""Tests for version chains, purpose-layer context, and governed forgetting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from palimp.embeddings import DeterministicEmbedder
from palimp.ingest import ingest_memory
from palimp.retriever import RecallEngine
from palimp.storage import SQLiteStore


def _engine(store: SQLiteStore) -> RecallEngine:
    return RecallEngine(store=store, embedder=DeterministicEmbedder(dim=384))


def test_version_chain_marks_parent_not_latest() -> None:
    store = SQLiteStore(":memory:")
    first = store.insert_memory("ns", "The API uses REST.")
    second = store.insert_memory(
        "ns",
        "The API uses GraphQL.",
        metadata={"parentMemoryId": first["memory_id"]},
    )

    parent = store.get_episode(first["episode_id"])
    child = store.get_episode(second["episode_id"])

    assert parent is not None
    assert child is not None
    assert parent["is_latest"] == 0
    assert child["is_latest"] == 1
    assert child["version"] == 2
    assert child["parent_episode_id"] == first["episode_id"]


def test_forgotten_episode_is_excluded_from_recall() -> None:
    store = SQLiteStore(":memory:")
    embedder = DeterministicEmbedder(dim=384)
    result = ingest_memory(
        store=store,
        embedder=embedder,
        extractor=None,
        ns="ns",
        content="Alice's secret project codename is Quartz.",
        extract=False,
    )
    store.forget_episode(result["episode_id"])

    output = RecallEngine(store=store, embedder=embedder).recall(
        "ns", "Quartz codename", mode="hybrid", limit=5
    )

    assert result["episode_id"] not in [r.id for r in output.results]


def test_forget_after_excludes_expired_episode() -> None:
    store = SQLiteStore(":memory:")
    embedder = DeterministicEmbedder(dim=384)
    expired = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    result = ingest_memory(
        store=store,
        embedder=embedder,
        extractor=None,
        ns="ns",
        content="Temporary build flag PALIMP_TMP_FLAG should expire.",
        metadata={"forgetAfter": expired},
        extract=False,
    )

    output = RecallEngine(store=store, embedder=embedder).recall(
        "ns", "PALIMP_TMP_FLAG", mode="hybrid", limit=5
    )

    assert result["episode_id"] not in [r.id for r in output.results]


def test_profile_purpose_is_available_without_lexical_match() -> None:
    store = SQLiteStore(":memory:")
    embedder = DeterministicEmbedder(dim=384)
    profile = ingest_memory(
        store=store,
        embedder=embedder,
        extractor=None,
        ns="ns",
        content="User prefers concise, evidence-first answers.",
        metadata={"purpose": "profile"},
        extract=False,
    )
    ingest_memory(
        store=store,
        embedder=embedder,
        extractor=None,
        ns="ns",
        content="The build command is uv build.",
        extract=False,
    )

    output = _engine(store).recall("ns", "build command", mode="hybrid", limit=5)

    assert profile["episode_id"] in [r.id for r in output.results]
