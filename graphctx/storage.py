"""SQLite storage layer for GraphCtx.

Implements all 11 tables plus FTS5, WAL mode, and namespace isolation.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from graphctx.errors import DimensionDriftError, EpisodeNotFoundError
from graphctx.models import BatchResponse


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class SQLiteStore:
    """SQLite-backed store with WAL mode, namespace isolation, and provenance tracking."""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._local = threading.local()
        self._lock = threading.RLock()
        # Run schema on the main connection
        conn = self._conn()
        self._ensure_schema(conn)
        conn.commit()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Get a thread-local connection, creating one if needed."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._ensure_schema(conn)
            conn.commit()
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Create all tables if they do not exist."""
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS namespace (
                id          TEXT PRIMARY KEY,
                name        TEXT UNIQUE NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episode (
                id              TEXT PRIMARY KEY,
                namespace       TEXT NOT NULL,
                content         TEXT NOT NULL,
                source_type     TEXT NOT NULL,
                source_ref      TEXT,
                metadata        TEXT,
                created_at      TEXT NOT NULL,
                last_accessed_at TEXT,
                stability       REAL NOT NULL DEFAULT 30.0,
                pinned          INTEGER NOT NULL DEFAULT 0,
                deleted_at      TEXT,
                tombstoned_at   TEXT,
                FOREIGN KEY (namespace) REFERENCES namespace(name)
            );

            CREATE TABLE IF NOT EXISTS memory (
                id          TEXT PRIMARY KEY,
                episode_id  TEXT NOT NULL,
                namespace   TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episode(id),
                FOREIGN KEY (namespace) REFERENCES namespace(name)
            );

            CREATE TABLE IF NOT EXISTS knowledge (
                id          TEXT PRIMARY KEY,
                episode_id  TEXT NOT NULL,
                namespace   TEXT NOT NULL,
                title       TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episode(id),
                FOREIGN KEY (namespace) REFERENCES namespace(name)
            );

            CREATE TABLE IF NOT EXISTS entity (
                id          TEXT PRIMARY KEY,
                namespace   TEXT NOT NULL,
                name        TEXT NOT NULL,
                type        TEXT NOT NULL,
                confidence  REAL NOT NULL,
                valid_from  TEXT,
                valid_until TEXT,
                created_at  TEXT NOT NULL,
                deleted_at  TEXT,
                FOREIGN KEY (namespace) REFERENCES namespace(name)
            );

            CREATE TABLE IF NOT EXISTS entity_alias (
                id                  TEXT PRIMARY KEY,
                entity_id           TEXT NOT NULL,
                alias               TEXT NOT NULL,
                source_episode_id   TEXT NOT NULL,
                confidence          REAL NOT NULL,
                FOREIGN KEY (entity_id) REFERENCES entity(id),
                FOREIGN KEY (source_episode_id) REFERENCES episode(id)
            );

            CREATE TABLE IF NOT EXISTS edge (
                id                  TEXT PRIMARY KEY,
                namespace           TEXT NOT NULL,
                source_entity_id    TEXT NOT NULL,
                target_entity_id    TEXT NOT NULL,
                relation            TEXT NOT NULL,
                confidence          REAL NOT NULL,
                valid_from          TEXT,
                valid_until         TEXT,
                created_at          TEXT NOT NULL,
                deleted_at          TEXT,
                FOREIGN KEY (namespace) REFERENCES namespace(name),
                FOREIGN KEY (source_entity_id) REFERENCES entity(id),
                FOREIGN KEY (target_entity_id) REFERENCES entity(id)
            );

            CREATE TABLE IF NOT EXISTS claim (
                id                  TEXT PRIMARY KEY,
                namespace           TEXT NOT NULL,
                subject_entity_id   TEXT NOT NULL,
                predicate           TEXT NOT NULL,
                object_value        TEXT NOT NULL,
                object_entity_id    TEXT,
                confidence          REAL NOT NULL,
                valid_from          TEXT,
                valid_until         TEXT,
                created_at          TEXT NOT NULL,
                deleted_at          TEXT,
                FOREIGN KEY (namespace) REFERENCES namespace(name),
                FOREIGN KEY (subject_entity_id) REFERENCES entity(id),
                FOREIGN KEY (object_entity_id) REFERENCES entity(id)
            );

            CREATE TABLE IF NOT EXISTS provenance (
                id                  TEXT PRIMARY KEY,
                namespace           TEXT NOT NULL,
                episode_id          TEXT NOT NULL,
                entity_id           TEXT,
                edge_id             TEXT,
                claim_id            TEXT,
                extractor_version   TEXT,
                evidence_span       TEXT,
                created_at          TEXT NOT NULL,
                FOREIGN KEY (namespace) REFERENCES namespace(name),
                FOREIGN KEY (episode_id) REFERENCES episode(id),
                FOREIGN KEY (entity_id) REFERENCES entity(id),
                FOREIGN KEY (edge_id) REFERENCES edge(id),
                FOREIGN KEY (claim_id) REFERENCES claim(id)
            );

            CREATE TABLE IF NOT EXISTS embedding (
                id              TEXT PRIMARY KEY,
                namespace       TEXT NOT NULL,
                owner_type      TEXT NOT NULL,
                owner_id        TEXT NOT NULL,
                model           TEXT NOT NULL,
                dimension       INTEGER NOT NULL,
                vector_blob     BLOB NOT NULL,
                created_at      TEXT NOT NULL,
                FOREIGN KEY (namespace) REFERENCES namespace(name)
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id              TEXT PRIMARY KEY,
                namespace       TEXT NOT NULL,
                actor           TEXT NOT NULL,
                action          TEXT NOT NULL,
                target_type     TEXT NOT NULL,
                target_id       TEXT NOT NULL,
                metadata        TEXT,
                created_at      TEXT NOT NULL,
                FOREIGN KEY (namespace) REFERENCES namespace(name)
            );

            CREATE TABLE IF NOT EXISTS context_state (
                id              TEXT PRIMARY KEY,
                namespace       TEXT NOT NULL,
                task_id         TEXT NOT NULL,
                requirements    TEXT,
                resolved        TEXT,
                evidence        TEXT,
                leads           TEXT,
                rejected        TEXT,
                failed_queries  TEXT,
                updated_at      TEXT NOT NULL,
                FOREIGN KEY (namespace) REFERENCES namespace(name)
            );

            CREATE TABLE IF NOT EXISTS context_operations (
                id              TEXT PRIMARY KEY,
                namespace       TEXT NOT NULL,
                task_id         TEXT NOT NULL,
                op_type         TEXT NOT NULL,
                target_ids      TEXT,
                original_content TEXT,
                new_content      TEXT,
                justification    TEXT,
                token_savings    INTEGER NOT NULL DEFAULT 0,
                created_at       TEXT NOT NULL,
                FOREIGN KEY (namespace) REFERENCES namespace(name)
            );

            CREATE TABLE IF NOT EXISTS session (
                id              TEXT PRIMARY KEY,
                namespace       TEXT NOT NULL,
                user_ref        TEXT,
                metadata        TEXT,
                created_at      TEXT NOT NULL,
                closed_at       TEXT,
                summary         TEXT,
                FOREIGN KEY (namespace) REFERENCES namespace(name)
            );
            """
        )

        # Migration: add decay columns to episode table if missing
        for col, col_def in [
            ("last_accessed_at", "TEXT"),
            ("stability", "REAL NOT NULL DEFAULT 30.0"),
            ("pinned", "INTEGER NOT NULL DEFAULT 0"),
        ]:
            try:
                conn.execute(f"ALTER TABLE episode ADD COLUMN {col} {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists

        # FTS5 virtual table for episode content
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS episode_fts USING fts5(
                    episode_id UNINDEXED,
                    namespace UNINDEXED,
                    content
                )
                """
            )
        except sqlite3.OperationalError:
            # FTS5 already exists or not available; ignore
            pass

    # ------------------------------------------------------------------
    # Namespace
    # ------------------------------------------------------------------

    def ensure_namespace(self, ns: str) -> None:
        conn = self._conn()
        with self._lock:
            conn.execute(
                "INSERT OR IGNORE INTO namespace (id, name, created_at) VALUES (?, ?, ?)",
                (_new_id("ns"), ns, _now_iso()),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

    def insert_episode(
        self,
        ns: str,
        content: str,
        source_type: str,
        source_ref: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        self.ensure_namespace(ns)
        episode_id = _new_id("eps")
        now = _now_iso()
        conn = self._conn()
        with self._lock:
            conn.execute(
                """INSERT INTO episode
                   (id, namespace, content, source_type, source_ref, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (episode_id, ns, content, source_type, source_ref, json.dumps(metadata) if metadata else None, now),
            )
            # FTS index
            try:
                conn.execute(
                    "INSERT INTO episode_fts (episode_id, namespace, content) VALUES (?, ?, ?)",
                    (episode_id, ns, content),
                )
            except sqlite3.OperationalError:
                pass
            conn.commit()
        return episode_id

    def get_episode(self, episode_id: str) -> Optional[dict[str, Any]]:
        conn = self._conn()
        row = conn.execute("SELECT * FROM episode WHERE id = ?", (episode_id,)).fetchone()
        if row is None:
            return None
        return dict(row)

    def tombstone_episode(self, episode_id: str) -> None:
        now = _now_iso()
        conn = self._conn()
        with self._lock:
            cur = conn.execute(
                "UPDATE episode SET deleted_at = ?, tombstoned_at = ? WHERE id = ? AND deleted_at IS NULL",
                (now, now, episode_id),
            )
            if cur.rowcount == 0:
                return
            # Tombstone provenance-only graph facts (entities, edges, claims)
            # that are only supported by this episode
            # First get all provenance entries for this episode
            prov_rows = conn.execute(
                "SELECT entity_id, edge_id, claim_id FROM provenance WHERE episode_id = ?",
                (episode_id,),
            ).fetchall()

            for row in prov_rows:
                eid = row["entity_id"]
                edge_id = row["edge_id"]
                claim_id = row["claim_id"]

                if eid:
                    # Check if entity has other non-tombstoned provenance
                    other = conn.execute(
                        """SELECT COUNT(*) as cnt FROM provenance p
                           JOIN episode e ON p.episode_id = e.id
                           WHERE p.entity_id = ? AND e.deleted_at IS NULL""",
                        (eid,),
                    ).fetchone()
                    if other and other["cnt"] == 0:
                        conn.execute(
                            "UPDATE entity SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
                            (now, eid),
                        )

                if edge_id:
                    other = conn.execute(
                        """SELECT COUNT(*) as cnt FROM provenance p
                           JOIN episode e ON p.episode_id = e.id
                           WHERE p.edge_id = ? AND e.deleted_at IS NULL""",
                        (edge_id,),
                    ).fetchone()
                    if other and other["cnt"] == 0:
                        conn.execute(
                            "UPDATE edge SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
                            (now, edge_id),
                        )

                if claim_id:
                    other = conn.execute(
                        """SELECT COUNT(*) as cnt FROM provenance p
                           JOIN episode e ON p.episode_id = e.id
                           WHERE p.claim_id = ? AND e.deleted_at IS NULL""",
                        (claim_id,),
                    ).fetchone()
                    if other and other["cnt"] == 0:
                        conn.execute(
                            "UPDATE claim SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
                            (now, claim_id),
                        )

            conn.commit()

    def get_episodes_by_ids(self, episode_ids: list[str]) -> list[dict[str, Any]]:
        if not episode_ids:
            return []
        conn = self._conn()
        placeholders = ",".join("?" for _ in episode_ids)
        rows = conn.execute(
            f"SELECT * FROM episode WHERE id IN ({placeholders})", episode_ids
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Decay / pin support
    # ------------------------------------------------------------------

    def touch_episode(self, episode_id: str) -> None:
        """Update last_accessed_at to now."""
        now = _now_iso()
        conn = self._conn()
        with self._lock:
            conn.execute(
                "UPDATE episode SET last_accessed_at = ? WHERE id = ?",
                (now, episode_id),
            )
            conn.commit()

    def pin_episode(self, episode_id: str) -> None:
        """Set pinned=True on an episode."""
        conn = self._conn()
        with self._lock:
            conn.execute(
                "UPDATE episode SET pinned = 1 WHERE id = ?",
                (episode_id,),
            )
            conn.commit()

    def unpin_episode(self, episode_id: str) -> None:
        """Set pinned=False and reset last_accessed_at to now."""
        now = _now_iso()
        conn = self._conn()
        with self._lock:
            conn.execute(
                "UPDATE episode SET pinned = 0, last_accessed_at = ? WHERE id = ?",
                (now, episode_id),
            )
            conn.commit()

    def get_decay_info(self, episode_id: str) -> dict[str, Any] | None:
        """Return decay-related fields for an episode."""
        conn = self._conn()
        row = conn.execute(
            "SELECT last_accessed_at, stability, pinned FROM episode WHERE id = ?",
            (episode_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "last_accessed_at": row["last_accessed_at"],
            "stability": row["stability"],
            "pinned": bool(row["pinned"]),
        }

    def update_stability(self, episode_id: str, new_stability: float) -> None:
        """Update the stability value for an episode."""
        conn = self._conn()
        with self._lock:
            conn.execute(
                "UPDATE episode SET stability = ? WHERE id = ?",
                (new_stability, episode_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def insert_memory(
        self,
        ns: str,
        content: str,
        source_ref: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, str]:
        episode_id = self.insert_episode(ns, content, "memory", source_ref, metadata)
        memory_id = _new_id("mem")
        now = _now_iso()
        conn = self._conn()
        with self._lock:
            conn.execute(
                "INSERT INTO memory (id, episode_id, namespace, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (memory_id, episode_id, ns, content, now),
            )
            conn.commit()
        return {"memory_id": memory_id, "episode_id": episode_id}

    # ------------------------------------------------------------------
    # Knowledge
    # ------------------------------------------------------------------

    def insert_knowledge(
        self,
        ns: str,
        title: str,
        content: str,
        source_ref: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, str]:
        episode_id = self.insert_episode(ns, content, "knowledge", source_ref, metadata)
        knowledge_id = _new_id("knw")
        now = _now_iso()
        conn = self._conn()
        with self._lock:
            conn.execute(
                "INSERT INTO knowledge (id, episode_id, namespace, title, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (knowledge_id, episode_id, ns, title, content, now),
            )
            conn.commit()
        return {"knowledge_id": knowledge_id, "episode_id": episode_id}

    # ------------------------------------------------------------------
    # Batch insert
    # ------------------------------------------------------------------

    def insert_memories_batch(
        self,
        ns: str,
        items: list[dict[str, Any]],
    ) -> BatchResponse:
        """Insert multiple memories in a single transaction.

        Each item dict should have: content, source_ref (opt), metadata (opt).
        Returns BatchResponse with per-item results. Individual failures do
        not abort the entire batch.
        """
        self.ensure_namespace(ns)
        conn = self._conn()
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        t0 = time.monotonic()

        with self._lock:
            for idx, item in enumerate(items):
                try:
                    episode_id = _new_id("eps")
                    memory_id = _new_id("mem")
                    now = _now_iso()
                    content = item["content"]
                    source_ref = item.get("source_ref")
                    metadata = item.get("metadata")

                    conn.execute(
                        """INSERT INTO episode
                           (id, namespace, content, source_type, source_ref, metadata, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (episode_id, ns, content, "memory", source_ref,
                         json.dumps(metadata) if metadata else None, now),
                    )
                    try:
                        conn.execute(
                            "INSERT INTO episode_fts (episode_id, namespace, content) VALUES (?, ?, ?)",
                            (episode_id, ns, content),
                        )
                    except sqlite3.OperationalError:
                        pass
                    conn.execute(
                        "INSERT INTO memory (id, episode_id, namespace, content, created_at) VALUES (?, ?, ?, ?, ?)",
                        (memory_id, episode_id, ns, content, now),
                    )
                    results.append({"index": idx, "memory_id": memory_id, "episode_id": episode_id})
                except Exception as exc:
                    errors.append({"index": idx, "error": str(exc)})

            conn.commit()

        elapsed_ms = (time.monotonic() - t0) * 1000
        return BatchResponse(
            total=len(items),
            successful=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
            elapsed_ms=round(elapsed_ms, 2),
        )

    def insert_knowledge_batch(
        self,
        ns: str,
        items: list[dict[str, Any]],
    ) -> BatchResponse:
        """Insert multiple knowledge items in a single transaction."""
        self.ensure_namespace(ns)
        conn = self._conn()
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        t0 = time.monotonic()

        with self._lock:
            for idx, item in enumerate(items):
                try:
                    episode_id = _new_id("eps")
                    knowledge_id = _new_id("knw")
                    now = _now_iso()
                    title = item["title"]
                    content = item["content"]
                    source_ref = item.get("source_ref")
                    metadata = item.get("metadata")

                    conn.execute(
                        """INSERT INTO episode
                           (id, namespace, content, source_type, source_ref, metadata, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (episode_id, ns, content, "knowledge", source_ref,
                         json.dumps(metadata) if metadata else None, now),
                    )
                    try:
                        conn.execute(
                            "INSERT INTO episode_fts (episode_id, namespace, content) VALUES (?, ?, ?)",
                            (episode_id, ns, content),
                        )
                    except sqlite3.OperationalError:
                        pass
                    conn.execute(
                        "INSERT INTO knowledge (id, episode_id, namespace, title, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (knowledge_id, episode_id, ns, title, content, now),
                    )
                    results.append({"index": idx, "knowledge_id": knowledge_id, "episode_id": episode_id})
                except Exception as exc:
                    errors.append({"index": idx, "error": str(exc)})

            conn.commit()

        elapsed_ms = (time.monotonic() - t0) * 1000
        return BatchResponse(
            total=len(items),
            successful=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
            elapsed_ms=round(elapsed_ms, 2),
        )

    # ------------------------------------------------------------------
    # Entity
    # ------------------------------------------------------------------

    def insert_entity(
        self,
        ns: str,
        name: str,
        entity_type: str,
        confidence: float = 1.0,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
    ) -> str:
        self.ensure_namespace(ns)
        entity_id = _new_id("ent")
        now = _now_iso()
        conn = self._conn()
        with self._lock:
            conn.execute(
                """INSERT INTO entity
                   (id, namespace, name, type, confidence, valid_from, valid_until, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (entity_id, ns, name, entity_type, confidence, valid_from, valid_until, now),
            )
            conn.commit()
        return entity_id

    def get_entities_by_ids(self, entity_ids: list[str]) -> list[dict[str, Any]]:
        if not entity_ids:
            return []
        conn = self._conn()
        placeholders = ",".join("?" for _ in entity_ids)
        rows = conn.execute(
            f"SELECT * FROM entity WHERE id IN ({placeholders})", entity_ids
        ).fetchall()
        return [dict(r) for r in rows]

    def get_entities_for_episode(self, episode_id: str) -> list[dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            """SELECT e.* FROM entity e
               JOIN provenance p ON p.entity_id = e.id
               WHERE p.episode_id = ? AND e.deleted_at IS NULL""",
            (episode_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Edge
    # ------------------------------------------------------------------

    def insert_edge(
        self,
        ns: str,
        source_id: str,
        target_id: str,
        relation: str,
        confidence: float = 1.0,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
    ) -> str:
        self.ensure_namespace(ns)
        edge_id = _new_id("edg")
        now = _now_iso()
        conn = self._conn()
        with self._lock:
            conn.execute(
                """INSERT INTO edge
                   (id, namespace, source_entity_id, target_entity_id, relation,
                    confidence, valid_from, valid_until, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (edge_id, ns, source_id, target_id, relation, confidence, valid_from, valid_until, now),
            )
            conn.commit()
        return edge_id

    def get_edges_for_entity(self, entity_id: str) -> list[dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            """SELECT * FROM edge
               WHERE (source_entity_id = ? OR target_entity_id = ?)
               AND deleted_at IS NULL""",
            (entity_id, entity_id),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Claim
    # ------------------------------------------------------------------

    def insert_claim(
        self,
        ns: str,
        subject_id: str,
        predicate: str,
        object_value: str,
        object_entity_id: Optional[str] = None,
        confidence: float = 1.0,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
    ) -> str:
        self.ensure_namespace(ns)
        claim_id = _new_id("clm")
        now = _now_iso()
        conn = self._conn()
        with self._lock:
            conn.execute(
                """INSERT INTO claim
                   (id, namespace, subject_entity_id, predicate, object_value,
                    object_entity_id, confidence, valid_from, valid_until, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (claim_id, ns, subject_id, predicate, object_value, object_entity_id, confidence, valid_from, valid_until, now),
            )
            conn.commit()
        return claim_id

    def get_claims_for_entity(self, entity_id: str) -> list[dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            """SELECT * FROM claim
               WHERE (subject_entity_id = ? OR object_entity_id = ?)
               AND deleted_at IS NULL""",
            (entity_id, entity_id),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def insert_provenance(
        self,
        ns: str,
        episode_id: str,
        entity_id: Optional[str] = None,
        edge_id: Optional[str] = None,
        claim_id: Optional[str] = None,
        extractor_version: Optional[str] = None,
        evidence_span: Optional[str] = None,
    ) -> str:
        self.ensure_namespace(ns)
        prov_id = _new_id("prv")
        now = _now_iso()
        conn = self._conn()
        with self._lock:
            conn.execute(
                """INSERT INTO provenance
                   (id, namespace, episode_id, entity_id, edge_id, claim_id,
                    extractor_version, evidence_span, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (prov_id, ns, episode_id, entity_id, edge_id, claim_id, extractor_version, evidence_span, now),
            )
            conn.commit()
        return prov_id

    def get_provenance_for(
        self,
        entity_id: Optional[str] = None,
        edge_id: Optional[str] = None,
        claim_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        conn = self._conn()
        conditions = []
        params: list[Any] = []
        if entity_id:
            conditions.append("entity_id = ?")
            params.append(entity_id)
        if edge_id:
            conditions.append("edge_id = ?")
            params.append(edge_id)
        if claim_id:
            conditions.append("claim_id = ?")
            params.append(claim_id)
        if not conditions:
            return []
        where = " OR ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM provenance WHERE {where}", params
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def insert_embedding(
        self,
        ns: str,
        owner_type: str,
        owner_id: str,
        model: str,
        dimension: int,
        vector_blob: bytes,
    ) -> str:
        self.ensure_namespace(ns)
        conn = self._conn()

        # Check for dimension drift
        existing = conn.execute(
            """SELECT DISTINCT dimension FROM embedding
               WHERE namespace = ? AND model = ? LIMIT 1""",
            (ns, model),
        ).fetchone()
        if existing is not None and existing["dimension"] != dimension:
            raise DimensionDriftError(expected=existing["dimension"], got=dimension)

        emb_id = _new_id("emb")
        now = _now_iso()
        with self._lock:
            conn.execute(
                """INSERT INTO embedding
                   (id, namespace, owner_type, owner_id, model, dimension, vector_blob, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (emb_id, ns, owner_type, owner_id, model, dimension, vector_blob, now),
            )
            conn.commit()
        return emb_id

    def get_all_embeddings(
        self, ns: str, owner_type: str, model: str
    ) -> list[dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            """SELECT owner_id, vector_blob FROM embedding
               WHERE namespace = ? AND owner_type = ? AND model = ?""",
            (ns, owner_type, model),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # FTS search
    # ------------------------------------------------------------------

    def search_fts(
        self, ns: str, query: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        conn = self._conn()
        # Tokenize query into words and join with OR for broad matching
        import re
        tokens = re.findall(r'[a-zA-Z0-9]+', query.lower())
        if not tokens:
            return []
        # Use OR matching: any token present = match
        fts_expr = ' OR '.join(tokens)
        rows = conn.execute(
            """SELECT episode_id, content, rank FROM episode_fts
               WHERE namespace = ? AND content MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (ns, fts_expr, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, ns: str) -> dict[str, int]:
        conn = self._conn()
        memories = conn.execute(
            "SELECT COUNT(*) as cnt FROM memory WHERE namespace = ?", (ns,)
        ).fetchone()["cnt"]
        knowledge_items = conn.execute(
            "SELECT COUNT(*) as cnt FROM knowledge WHERE namespace = ?", (ns,)
        ).fetchone()["cnt"]
        entities = conn.execute(
            "SELECT COUNT(*) as cnt FROM entity WHERE namespace = ? AND deleted_at IS NULL", (ns,)
        ).fetchone()["cnt"]
        edges = conn.execute(
            "SELECT COUNT(*) as cnt FROM edge WHERE namespace = ? AND deleted_at IS NULL", (ns,)
        ).fetchone()["cnt"]
        claims = conn.execute(
            "SELECT COUNT(*) as cnt FROM claim WHERE namespace = ? AND deleted_at IS NULL", (ns,)
        ).fetchone()["cnt"]
        return {
            "memories": memories,
            "knowledge_items": knowledge_items,
            "entities": entities,
            "edges": edges,
            "claims": claims,
        }

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def insert_audit(
        self,
        ns: str,
        actor: str,
        action: str,
        target_type: str,
        target_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        self.ensure_namespace(ns)
        audit_id = _new_id("aud")
        now = _now_iso()
        conn = self._conn()
        with self._lock:
            conn.execute(
                """INSERT INTO audit_log
                   (id, namespace, actor, action, target_type, target_id, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (audit_id, ns, actor, action, target_type, target_id, json.dumps(metadata) if metadata else None, now),
            )
            conn.commit()
        return audit_id

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def create_session(
        self,
        ns: str,
        user_ref: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        self.ensure_namespace(ns)
        session_id = _new_id("ses")
        now = _now_iso()
        conn = self._conn()
        with self._lock:
            conn.execute(
                """INSERT INTO session
                   (id, namespace, user_ref, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, ns, user_ref, json.dumps(metadata) if metadata else None, now),
            )
            conn.commit()
        return session_id

    def close_session(self, session_id: str, summary: Optional[str] = None) -> None:
        now = _now_iso()
        conn = self._conn()
        with self._lock:
            conn.execute(
                "UPDATE session SET closed_at = ?, summary = ? WHERE id = ? AND closed_at IS NULL",
                (now, summary, session_id),
            )
            conn.commit()

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        conn = self._conn()
        row = conn.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_sessions(self, ns: str) -> list[dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM session WHERE namespace = ? ORDER BY created_at DESC",
            (ns,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Integrity / diagnostics
    # ------------------------------------------------------------------

    def integrity_check(self) -> dict[str, Any]:
        conn = self._conn()
        result = conn.execute("PRAGMA integrity_check").fetchone()
        table_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table'"
        ).fetchone()["cnt"]
        return {
            "integrity": result[0] if result else "unknown",
            "table_count": table_count,
        }

    def orphan_provenance_count(self) -> int:
        """Count provenance rows whose episode has been tombstoned."""
        conn = self._conn()
        row = conn.execute(
            """SELECT COUNT(*) as cnt FROM provenance p
               JOIN episode e ON p.episode_id = e.id
               WHERE e.deleted_at IS NOT NULL"""
        ).fetchone()
        return row["cnt"] if row else 0
