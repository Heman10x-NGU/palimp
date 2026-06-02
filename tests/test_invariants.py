"""Deliberate break tests for Palimp invariants.

These tests verify critical safety and correctness properties:
- 11.2: Namespace leakage
- 11.5: Deletion cascade
- 11.6: Concurrency
- 11.7: Embedding dimension drift
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from palimp.errors import DimensionDriftError
from palimp.storage import SQLiteStore


class TestNamespaceLeakage:
    """11.2: Insert in 'private', query in 'public' returns nothing."""

    def test_namespace_leakage(self):
        store = SQLiteStore(":memory:")

        # Insert secret memory in 'private'
        store.insert_memory("private", "The secret password is swordfish.")
        store.insert_knowledge("private", "Secrets", "All the confidential data.")

        # Query in 'public' namespace
        results = store.search_fts("public", "secret password")
        assert len(results) == 0, "Namespace leakage: public query returned private data"

        # Stats should also be isolated
        public_stats = store.get_stats("public")
        assert public_stats["memories"] == 0
        assert public_stats["knowledge_items"] == 0

        # Verify private data exists
        private_stats = store.get_stats("private")
        assert private_stats["memories"] == 1
        assert private_stats["knowledge_items"] == 1


class TestDeletionCascade:
    """11.5: Tombstone episode, verify graph facts with sole provenance are tombstoned."""

    def test_deletion_cascade(self):
        store = SQLiteStore(":memory:")

        # Create an episode with entity + provenance
        ep_id = store.insert_episode("test-ns", "GraphCtx uses SQLite.", "knowledge")
        entity_id = store.insert_entity("test-ns", "GraphCtx", "Project")
        store.insert_provenance("test-ns", ep_id, entity_id=entity_id)

        # Entity should be visible
        entities = store.get_entities_for_episode(ep_id)
        assert len(entities) == 1

        # Tombstone the episode
        store.tombstone_episode(ep_id)

        # Entity should now be tombstoned (deleted_at set)
        conn = store._conn()
        entity_row = conn.execute("SELECT deleted_at FROM entity WHERE id = ?", (entity_id,)).fetchone()
        assert entity_row is not None
        assert entity_row["deleted_at"] is not None, "Entity not tombstoned after episode deletion"

        # Episode should be tombstoned
        episode = store.get_episode(ep_id)
        assert episode["deleted_at"] is not None
        assert episode["tombstoned_at"] is not None


class TestConcurrency:
    """11.6: 20 parallel inserts via ThreadPoolExecutor, no crash, audit count matches."""

    def test_concurrent_inserts(self, tmp_path):
        db_path = str(tmp_path / "concurrent_test.db")
        store = SQLiteStore(db_path)

        def insert_one(i: int) -> str:
            return store.insert_audit(
                "test-ns",
                actor=f"worker-{i}",
                action="insert",
                target_type="memory",
                target_id=f"test-{i}",
            )

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(insert_one, i) for i in range(20)]
            results = [f.result() for f in futures]

        # All 20 should have succeeded with unique IDs
        assert len(results) == 20
        assert len(set(results)) == 20, "Duplicate audit IDs detected"

        # Verify audit count
        conn = store._conn()
        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM audit_log WHERE namespace = 'test-ns'"
        ).fetchone()["cnt"]
        assert count == 20, f"Expected 20 audit entries, got {count}"


class TestEmbeddingDimensionDrift:
    """11.7: Insert 384-dim, attempt 768-dim, expect DimensionDriftError."""

    def test_embedding_dimension_drift(self):
        store = SQLiteStore(":memory:")
        import struct

        # Create a 384-dim embedding
        vec_384 = struct.pack(f"{384}f", *[0.1] * 384)
        store.insert_embedding("test-ns", "episode", "eps_001", "test-model", 384, vec_384)

        # Attempt to insert a 768-dim embedding with the same model
        vec_768 = struct.pack(f"{768}f", *[0.2] * 768)
        with pytest.raises(DimensionDriftError) as exc_info:
            store.insert_embedding("test-ns", "episode", "eps_002", "test-model", 768, vec_768)

        assert exc_info.value.expected == 384
        assert exc_info.value.got == 768

        # Verify only the first embedding was stored (no partial write)
        embeddings = store.get_all_embeddings("test-ns", "episode", "test-model")
        assert len(embeddings) == 1, "Partial write detected after dimension drift error"
