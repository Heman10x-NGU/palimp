"""Tests for entity alias normalization and deduplication."""

import json

from graphctx.aliases import entities_compatible, normalize_entity_name
from graphctx.storage import SQLiteStore


# ---------------------------------------------------------------------------
# normalize_entity_name
# ---------------------------------------------------------------------------


class TestNormalizeEntityName:
    def test_strip_article_the(self):
        assert normalize_entity_name("The Python") == "python"

    def test_strip_article_a(self):
        assert normalize_entity_name("a widget") == "widget"

    def test_strip_article_an(self):
        assert normalize_entity_name("an apple") == "apple"

    def test_hyphen_collapsed(self):
        assert normalize_entity_name("graph-ctx") == "graph ctx"

    def test_underscore_collapsed(self):
        assert normalize_entity_name("my_project") == "my project"

    def test_whitespace_collapsed(self):
        assert normalize_entity_name("  hello   world  ") == "hello world"

    def test_lowercase(self):
        assert normalize_entity_name("PYTHON") == "python"

    def test_no_article(self):
        assert normalize_entity_name("Python") == "python"

    def test_mixed_case_with_article(self):
        assert normalize_entity_name("The GraphCtx") == "graphctx"

    def test_empty_after_strip(self):
        # "the" alone becomes empty after stripping article + collapse
        assert normalize_entity_name("the") == ""

    def test_only_article_with_space(self):
        assert normalize_entity_name("The ") == ""


# ---------------------------------------------------------------------------
# entities_compatible
# ---------------------------------------------------------------------------


class TestEntitiesCompatible:
    def test_same_type(self):
        assert entities_compatible("Person", "Person") is True

    def test_same_type_case_insensitive(self):
        assert entities_compatible("Person", "person") is True

    def test_different_type(self):
        assert entities_compatible("Company", "Fruit") is False

    def test_empty_type_a(self):
        assert entities_compatible("", "Person") is True

    def test_empty_type_b(self):
        assert entities_compatible("Person", "") is True

    def test_none_type_a(self):
        assert entities_compatible(None, "Person") is True

    def test_none_type_b(self):
        assert entities_compatible("Person", None) is True


# ---------------------------------------------------------------------------
# Alias dedup: same namespace
# ---------------------------------------------------------------------------


class TestAliasDedupSameNamespace:
    def test_same_normalized_name_reuses_entity(self):
        store = SQLiteStore(":memory:")
        ep_id = store.insert_episode("demo", "test content", "memory")
        eid1 = store.insert_entity_with_alias(
            "demo", "Python", "Technology", source_episode_id=ep_id
        )
        eid2 = store.insert_entity_with_alias(
            "demo", "python", "Technology", source_episode_id=ep_id
        )
        assert eid1 == eid2

    def test_article_variant_reuses_entity(self):
        store = SQLiteStore(":memory:")
        ep_id = store.insert_episode("demo", "test content", "memory")
        eid1 = store.insert_entity_with_alias(
            "demo", "Python", "Technology", source_episode_id=ep_id
        )
        eid2 = store.insert_entity_with_alias(
            "demo", "The Python", "Technology", source_episode_id=ep_id
        )
        assert eid1 == eid2

    def test_alias_row_inserted_for_new_surface_form(self):
        store = SQLiteStore(":memory:")
        ep_id = store.insert_episode("demo", "test content", "memory")
        store.insert_entity_with_alias(
            "demo", "Python", "Technology", source_episode_id=ep_id
        )
        store.insert_entity_with_alias(
            "demo", "The Python", "Technology", source_episode_id=ep_id
        )
        conn = store._conn()
        rows = conn.execute(
            "SELECT * FROM entity_alias WHERE alias = ?", ("python",)
        ).fetchall()
        # Should have at least one alias row (may be deduped if same normalized form)
        assert len(rows) >= 1


# ---------------------------------------------------------------------------
# Alias dedup: different types do NOT merge
# ---------------------------------------------------------------------------


class TestAliasNoMergeDifferentType:
    def test_different_types_create_separate_entities(self):
        store = SQLiteStore(":memory:")
        ep_id = store.insert_episode("demo", "test content", "memory")
        eid_company = store.insert_entity_with_alias(
            "demo", "Apple", "Company", source_episode_id=ep_id
        )
        eid_fruit = store.insert_entity_with_alias(
            "demo", "apple", "Fruit", source_episode_id=ep_id
        )
        assert eid_company != eid_fruit

    def test_find_entity_by_alias_returns_compatible_type(self):
        store = SQLiteStore(":memory:")
        ep_id = store.insert_episode("demo", "test content", "memory")
        eid_company = store.insert_entity_with_alias(
            "demo", "Apple", "Company", source_episode_id=ep_id
        )
        store.insert_entity_with_alias(
            "demo", "apple", "Fruit", source_episode_id=ep_id
        )
        # find_entity_by_alias returns the first match — should be the Company
        found = store.find_entity_by_alias("demo", "apple")
        assert found is not None
        assert found["id"] == eid_company


# ---------------------------------------------------------------------------
# Alias dedup: cross-namespace isolation
# ---------------------------------------------------------------------------


class TestAliasNoCrossNamespace:
    def test_same_name_different_namespace_different_entities(self):
        store = SQLiteStore(":memory:")
        ep_a = store.insert_episode("ns-a", "content a", "memory")
        ep_b = store.insert_episode("ns-b", "content b", "memory")
        eid_a = store.insert_entity_with_alias(
            "ns-a", "Python", "Technology", source_episode_id=ep_a
        )
        eid_b = store.insert_entity_with_alias(
            "ns-b", "Python", "Technology", source_episode_id=ep_b
        )
        assert eid_a != eid_b

    def test_find_entity_by_alias_respects_namespace(self):
        store = SQLiteStore(":memory:")
        ep_a = store.insert_episode("ns-a", "content a", "memory")
        store.insert_entity_with_alias(
            "ns-a", "Python", "Technology", source_episode_id=ep_a
        )
        found = store.find_entity_by_alias("ns-b", "python")
        assert found is None


# ---------------------------------------------------------------------------
# merge_entities
# ---------------------------------------------------------------------------


class TestMergeEntities:
    def _build_graph(self):
        """Create a small graph: A --WORKS_ON--> C, B --USES--> D, B has claim."""
        store = SQLiteStore(":memory:")
        ep = store.insert_episode("demo", "merge test", "memory")

        a = store.insert_entity_with_alias("demo", "Alice", "Person", source_episode_id=ep)
        b = store.insert_entity_with_alias("demo", "Bob", "Person", source_episode_id=ep)
        c = store.insert_entity_with_alias("demo", "GraphCtx", "Project", source_episode_id=ep)
        d = store.insert_entity_with_alias("demo", "Python", "Technology", source_episode_id=ep)

        edge_ac = store.insert_edge("demo", a, c, "WORKS_ON")
        edge_bd = store.insert_edge("demo", b, d, "USES")
        claim_b = store.insert_claim("demo", b, "prefers", "vim")

        store.insert_provenance("demo", ep, entity_id=a)
        store.insert_provenance("demo", ep, entity_id=b)
        store.insert_provenance("demo", ep, edge_id=edge_ac)
        store.insert_provenance("demo", ep, edge_id=edge_bd)
        store.insert_provenance("demo", ep, claim_id=claim_b)

        return store, a, b, c, d

    def test_merge_moves_edges(self):
        store, a, b, c, d = self._build_graph()
        store.merge_entities("demo", a, b, reason="same person")

        edges_a = store.get_edges_for_entity(a)
        relations = {e["relation"] for e in edges_a}
        assert "WORKS_ON" in relations
        assert "USES" in relations

    def test_merge_moves_claims(self):
        store, a, b, c, d = self._build_graph()
        store.merge_entities("demo", a, b, reason="same person")

        claims_a = store.get_claims_for_entity(a)
        predicates = {cl["predicate"] for cl in claims_a}
        assert "prefers" in predicates

    def test_merge_moves_provenance(self):
        store, a, b, c, d = self._build_graph()
        store.merge_entities("demo", a, b, reason="same person")

        prov_b = store.get_provenance_for(entity_id=b)
        # B's provenance should have been moved to A
        assert len(prov_b) == 0

    def test_merge_tombstones_b(self):
        store, a, b, c, d = self._build_graph()
        store.merge_entities("demo", a, b, reason="same person")

        entities_b = store.get_entities_by_ids([b])
        assert len(entities_b) == 1
        assert entities_b[0]["deleted_at"] is not None

    def test_merge_inserts_alias_for_b_name(self):
        store, a, b, c, d = self._build_graph()
        store.merge_entities("demo", a, b, reason="same person")

        # "bob" normalized alias should now point to entity A
        found = store.find_entity_by_alias("demo", "bob")
        assert found is not None
        assert found["id"] == a


# ---------------------------------------------------------------------------
# merge audit log
# ---------------------------------------------------------------------------


class TestMergeAuditLogged:
    def test_merge_creates_audit_log_entry(self):
        store = SQLiteStore(":memory:")
        ep = store.insert_episode("demo", "audit test", "memory")
        a = store.insert_entity_with_alias("demo", "Alice", "Person", source_episode_id=ep)
        b = store.insert_entity_with_alias("demo", "Alicia", "Person", source_episode_id=ep)

        store.merge_entities("demo", a, b, reason="same person alias")

        conn = store._conn()
        rows = conn.execute(
            """SELECT * FROM audit_log
               WHERE namespace = 'demo' AND action = 'entity_merge'""",
        ).fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["target_id"] == a
        meta = json.loads(row["metadata"])
        assert meta["merged_entity_id"] == b
        assert meta["merged_entity_name"] == "Alicia"
        assert meta["reason"] == "same person alias"
