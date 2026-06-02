"""Tests for Ebbinghaus decay, pin/unpin, and touch_episode."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from graphctx.decay import DEFAULT_STABILITY, compute_decay_score, ebbinghaus_retention
from graphctx.embeddings import DeterministicEmbedder
from graphctx.retriever import RecallEngine
from graphctx.storage import SQLiteStore


class TestEbbinghausRetention:
    """Unit tests for the forgetting curve function."""

    def test_ebbinghaus_retention_fresh(self):
        """A just-accessed item should have retention ~1.0."""
        now = datetime.now(timezone.utc).isoformat()
        r = ebbinghaus_retention(now, DEFAULT_STABILITY)
        assert r > 0.99

    def test_ebbinghaus_retention_old(self):
        """30-day-old item with 30-day stability -> ~0.37 (e^-1)."""
        import math

        thirty_days_ago = (
            datetime.now(timezone.utc) - timedelta(days=30)
        ).isoformat()
        r = ebbinghaus_retention(thirty_days_ago, DEFAULT_STABILITY)
        expected = math.exp(-1.0)
        assert abs(r - expected) < 0.01

    def test_ebbinghaus_retention_pinned(self):
        """Pinned items always return 1.0 regardless of age."""
        long_ago = (
            datetime.now(timezone.utc) - timedelta(days=3650)
        ).isoformat()
        r = ebbinghaus_retention(long_ago, DEFAULT_STABILITY, pinned=True)
        assert r == 1.0

    def test_ebbinghaus_retention_bad_timestamp(self):
        """Invalid timestamp returns 0.0."""
        assert ebbinghaus_retention("not-a-date", DEFAULT_STABILITY) == 0.0

    def test_ebbinghaus_retention_none_timestamp(self):
        """None timestamp returns 0.0."""
        assert ebbinghaus_retention(None, DEFAULT_STABILITY) == 0.0


class TestPinUnpin:
    """Test pin/unpin cycle on storage."""

    def test_pin_unpin(self, memory_store: SQLiteStore):
        """Pin sets pinned=1, unpin sets pinned=0."""
        ns = "test-pin"
        ep_id = memory_store.insert_episode(ns, "hello", "memory")

        # Initially not pinned
        info = memory_store.get_decay_info(ep_id)
        assert info is not None
        assert info["pinned"] is False

        # Pin
        memory_store.pin_episode(ep_id)
        info = memory_store.get_decay_info(ep_id)
        assert info is not None
        assert info["pinned"] is True

        # Unpin
        memory_store.unpin_episode(ep_id)
        info = memory_store.get_decay_info(ep_id)
        assert info is not None
        assert info["pinned"] is False


class TestTouchEpisode:
    """Test that touch_episode updates last_accessed_at."""

    def test_touch_episode(self, memory_store: SQLiteStore):
        """Touching an episode sets last_accessed_at to now."""
        ns = "test-touch"
        ep_id = memory_store.insert_episode(ns, "world", "memory")

        # Initially no last_accessed_at
        info = memory_store.get_decay_info(ep_id)
        assert info is not None
        assert info["last_accessed_at"] is None

        # Touch
        memory_store.touch_episode(ep_id)
        info = memory_store.get_decay_info(ep_id)
        assert info is not None
        assert info["last_accessed_at"] is not None


class TestDecayInRecall:
    """Test that decay affects recall scoring."""

    def test_decay_in_recall(self, memory_store: SQLiteStore):
        """An old unaccessed episode scores lower than a recent one."""
        ns = "test-decay"
        embedder = DeterministicEmbedder()
        engine = RecallEngine(store=memory_store, embedder=embedder)

        # Insert two similar memories
        memory_store.insert_memory(ns, "python is a programming language")
        result2 = memory_store.insert_memory(ns, "python is a programming language")
        old_episode_id = result2["episode_id"]

        # Backdate the second episode's last_accessed_at to 60 days ago
        old_time = (
            datetime.now(timezone.utc) - timedelta(days=60)
        ).isoformat()
        conn = memory_store._conn()
        conn.execute(
            "UPDATE episode SET last_accessed_at = ? WHERE id = ?",
            (old_time, old_episode_id),
        )
        conn.commit()

        output = engine.recall(ns=ns, query="python", mode="hybrid", limit=10)
        assert len(output.results) == 2

        # The non-backdated episode should score higher
        scores = {r.id: r.score for r in output.results}
        fresh_id = [r.id for r in output.results if r.id != old_episode_id][0]
        assert scores[fresh_id] >= scores[old_episode_id]

    def test_pinned_episode_retains_score(self, memory_store: SQLiteStore):
        """A pinned old episode should still score well."""
        ns = "test-pinned"
        embedder = DeterministicEmbedder()
        engine = RecallEngine(store=memory_store, embedder=embedder)

        memory_store.insert_memory(ns, "rust is a systems language")
        result2 = memory_store.insert_memory(ns, "rust is a systems language")
        pinned_id = result2["episode_id"]

        # Backdate and pin
        old_time = (
            datetime.now(timezone.utc) - timedelta(days=60)
        ).isoformat()
        conn = memory_store._conn()
        conn.execute(
            "UPDATE episode SET last_accessed_at = ? WHERE id = ?",
            (old_time, pinned_id),
        )
        conn.commit()
        memory_store.pin_episode(pinned_id)

        output = engine.recall(ns=ns, query="rust", mode="hybrid", limit=10)
        assert len(output.results) == 2

        # Pinned episode should have a decent score (not crushed by decay)
        pinned_score = next(r.score for r in output.results if r.id == pinned_id)
        assert pinned_score > 0.0

    def test_compute_decay_score_uses_last_accessed(self):
        """compute_decay_score prefers last_accessed_at over created_at."""
        now = datetime.now(timezone.utc)
        recent = now.isoformat()
        old = (now - timedelta(days=90)).isoformat()

        # With last_accessed_at set to recent -> high score
        score_recent = compute_decay_score(
            created_at=old,
            last_accessed_at=recent,
        )
        # With only created_at (old) -> low score
        score_old = compute_decay_score(
            created_at=old,
            last_accessed_at=None,
        )
        assert score_recent > score_old
