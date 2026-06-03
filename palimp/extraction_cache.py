"""Extraction cache — SQLite-backed LRU cache for extraction results.

Modeled after Graphiti's LLMCache. Avoids re-extracting identical content
across benchmark runs and production use.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional


class ExtractionCache:
    """SQLite-backed cache for extraction results.

    Key = MD5 of model:text. Value = JSON-serialized ExtractionResult.
    """

    def __init__(self, cache_path: str = ".palimp_cache/extraction.db") -> None:
        self._path = Path(cache_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS extraction_cache (
                key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        self._conn.commit()

    def _make_key(self, model: str, text: str) -> str:
        return hashlib.md5(f"{model}:{text}".encode()).hexdigest()

    def get(self, model: str, text: str) -> Optional[dict[str, Any]]:
        key = self._make_key(model, text)
        row = self._conn.execute(
            "SELECT result_json FROM extraction_cache WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def set(self, model: str, text: str, result: dict[str, Any]) -> None:
        key = self._make_key(model, text)
        self._conn.execute(
            "INSERT OR REPLACE INTO extraction_cache (key, result_json) VALUES (?, ?)",
            (key, json.dumps(result)),
        )
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        row = self._conn.execute("SELECT COUNT(*) FROM extraction_cache").fetchone()
        return {"entries": row[0] if row else 0}

    def close(self) -> None:
        self._conn.close()
