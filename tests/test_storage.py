"""Tests for the SQLite storage layer."""

from palimp.storage import SQLiteStore


class TestInsertMemoryAndRetrieve:
    def test_insert_memory_returns_ids(self):
        store = SQLiteStore(":memory:")
        result = store.insert_memory("test-ns", "Alice likes Python.", source_ref="chat:1")
        assert "memory_id" in result
        assert "episode_id" in result
        assert result["memory_id"].startswith("mem_")
        assert result["episode_id"].startswith("eps_")

    def test_insert_memory_episode_content(self):
        store = SQLiteStore(":memory:")
        result = store.insert_memory("test-ns", "Alice likes Python.")
        episode = store.get_episode(result["episode_id"])
        assert episode is not None
        assert episode["content"] == "Alice likes Python."
        assert episode["source_type"] == "memory"

    def test_memory_in_stats(self):
        store = SQLiteStore(":memory:")
        store.insert_memory("test-ns", "First memory.")
        store.insert_memory("test-ns", "Second memory.")
        stats = store.get_stats("test-ns")
        assert stats["memories"] == 2


class TestInsertKnowledgeAndRetrieve:
    def test_insert_knowledge_returns_ids(self):
        store = SQLiteStore(":memory:")
        result = store.insert_knowledge(
            "test-ns", "Architecture", "GraphCtx uses SQLite.", source_ref="docs/arch.md"
        )
        assert "knowledge_id" in result
        assert "episode_id" in result
        assert result["knowledge_id"].startswith("knw_")

    def test_insert_knowledge_episode_content(self):
        store = SQLiteStore(":memory:")
        result = store.insert_knowledge("test-ns", "Architecture", "GraphCtx uses SQLite.")
        episode = store.get_episode(result["episode_id"])
        assert episode is not None
        assert episode["content"] == "GraphCtx uses SQLite."
        assert episode["source_type"] == "knowledge"

    def test_knowledge_in_stats(self):
        store = SQLiteStore(":memory:")
        store.insert_knowledge("test-ns", "Doc 1", "Content 1.")
        store.insert_knowledge("test-ns", "Doc 2", "Content 2.")
        stats = store.get_stats("test-ns")
        assert stats["knowledge_items"] == 2


class TestInsertEpisodeAndTombstone:
    def test_tombstone_marks_deleted(self):
        store = SQLiteStore(":memory:")
        ep_id = store.insert_episode("test-ns", "Some content.", "memory")
        store.tombstone_episode(ep_id)
        episode = store.get_episode(ep_id)
        assert episode is not None
        assert episode["deleted_at"] is not None
        assert episode["tombstoned_at"] is not None

    def test_tombstone_idempotent(self):
        store = SQLiteStore(":memory:")
        ep_id = store.insert_episode("test-ns", "Some content.", "memory")
        store.tombstone_episode(ep_id)
        # Second tombstone should not crash
        store.tombstone_episode(ep_id)
        episode = store.get_episode(ep_id)
        assert episode["deleted_at"] is not None

    def test_get_nonexistent_episode(self):
        store = SQLiteStore(":memory:")
        assert store.get_episode("eps_nonexistent") is None


class TestNamespaceCreationAndUniqueness:
    def test_ensure_namespace_creates(self):
        store = SQLiteStore(":memory:")
        store.ensure_namespace("my-ns")
        stats = store.get_stats("my-ns")
        assert stats["memories"] == 0  # no crash, namespace exists

    def test_ensure_namespace_idempotent(self):
        store = SQLiteStore(":memory:")
        store.ensure_namespace("my-ns")
        store.ensure_namespace("my-ns")  # should not raise
        # Check only one namespace row
        conn = store._conn()
        count = conn.execute("SELECT COUNT(*) as cnt FROM namespace WHERE name = 'my-ns'").fetchone()["cnt"]
        assert count == 1

    def test_different_namespaces_are_separate(self):
        store = SQLiteStore(":memory:")
        store.insert_memory("ns-a", "Memory in A.")
        store.insert_memory("ns-b", "Memory in B.")
        stats_a = store.get_stats("ns-a")
        stats_b = store.get_stats("ns-b")
        assert stats_a["memories"] == 1
        assert stats_b["memories"] == 1


class TestProvenanceLinksToEpisode:
    def test_provenance_links_to_episode(self):
        store = SQLiteStore(":memory:")
        ep_id = store.insert_episode("test-ns", "Some content.", "memory")
        entity_id = store.insert_entity("test-ns", "Alice", "Person")
        prov_id = store.insert_provenance("test-ns", ep_id, entity_id=entity_id)
        assert prov_id.startswith("prv_")
        prov_list = store.get_provenance_for(entity_id=entity_id)
        assert len(prov_list) == 1
        assert prov_list[0]["episode_id"] == ep_id

    def test_provenance_links_to_edge(self):
        store = SQLiteStore(":memory:")
        ep_id = store.insert_episode("test-ns", "Edge content.", "memory")
        ent_a = store.insert_entity("test-ns", "Alice", "Person")
        ent_b = store.insert_entity("test-ns", "Bob", "Person")
        edge_id = store.insert_edge("test-ns", ent_a, ent_b, "RELATES_TO")
        prov_id = store.insert_provenance("test-ns", ep_id, edge_id=edge_id)
        prov_list = store.get_provenance_for(edge_id=edge_id)
        assert len(prov_list) == 1
        assert prov_list[0]["id"] == prov_id

    def test_provenance_links_to_claim(self):
        store = SQLiteStore(":memory:")
        ep_id = store.insert_episode("test-ns", "Claim content.", "memory")
        ent = store.insert_entity("test-ns", "GraphCtx", "Project")
        claim_id = store.insert_claim("test-ns", ent, "USES", "SQLite")
        store.insert_provenance("test-ns", ep_id, claim_id=claim_id)
        prov_list = store.get_provenance_for(claim_id=claim_id)
        assert len(prov_list) == 1


class TestFtsSearch:
    def test_fts_search_finds_content(self):
        store = SQLiteStore(":memory:")
        store.insert_memory("test-ns", "Alice prefers concise technical answers.")
        store.insert_memory("test-ns", "Bob likes verbose explanations.")
        results = store.search_fts("test-ns", "concise technical")
        assert len(results) >= 1
        assert any("concise" in r["content"] for r in results)

    def test_fts_search_respects_namespace(self):
        store = SQLiteStore(":memory:")
        store.insert_memory("ns-a", "Secret content about Alice.")
        results = store.search_fts("ns-b", "Alice")
        assert len(results) == 0

    def test_fts_search_no_match(self):
        store = SQLiteStore(":memory:")
        store.insert_memory("test-ns", "Alice prefers concise technical answers.")
        results = store.search_fts("test-ns", "quantum entanglement")
        assert len(results) == 0

    def test_fts_search_limit(self):
        store = SQLiteStore(":memory:")
        for i in range(10):
            store.insert_memory("test-ns", f"Memory number {i} about testing.")
        results = store.search_fts("test-ns", "testing", limit=3)
        assert len(results) <= 3
