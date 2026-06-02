"""Palimp benchmark command implementation.

Generates synthetic fixtures, ingests them, measures recall latency,
and reports performance metrics.
"""

from __future__ import annotations

import os
import random
import statistics
import time
from typing import Any

from palimp.config import get_config
from palimp.embeddings import DeterministicEmbedder
from palimp.ingest import ingest_memory, ingest_knowledge
from palimp.retriever import RecallEngine
from palimp.storage import SQLiteStore

_DETERMINISTIC_MODEL = "deterministic-sha256"

# ---------------------------------------------------------------------------
# Synthetic fixture generators
# ---------------------------------------------------------------------------

_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank",
    "Ivy", "Jack", "Karen", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rosa", "Sam", "Tina", "Uma", "Vince", "Wendy", "Xander",
    "Yara", "Zane",
]

_PREFERENCES = [
    "prefers dark mode",
    "likes Python over Java",
    "prefers concise answers",
    "likes vim keybindings",
    "prefers TypeScript for frontend",
    "likes functional programming",
    "prefers REST over GraphQL",
    "likes test-driven development",
    "prefers monorepos",
    "likes containerized deployments",
    "prefers PostgreSQL over MySQL",
    "likes code reviews before merge",
    "prefers small pull requests",
    "likes detailed commit messages",
    "prefers pair programming",
    "likes automated CI/CD",
    "prefers documentation as code",
    "likes clean architecture",
    "prefers microservices",
    "likes domain-driven design",
]

_TECHNOLOGIES = [
    "SQLite", "PostgreSQL", "Redis", "Elasticsearch", "Kafka",
    "Docker", "Kubernetes", "Terraform", "Prometheus", "Grafana",
    "Rust", "Go", "Python", "TypeScript", "Java",
    "React", "Vue", "Angular", "Svelte", "Next.js",
    "gRPC", "GraphQL", "REST", "WebSocket", "MQTT",
    "AWS", "GCP", "Azure", "Cloudflare", "Vercel",
]

_PURPOSES = [
    "persistent storage", "caching", "search", "message queuing",
    "monitoring", "logging", "authentication", "authorization",
    "rate limiting", "load balancing", "service discovery",
    "configuration management", "secret management", "CI/CD",
    "container orchestration", "edge computing", "data pipeline",
    "real-time communication", "file storage", "API gateway",
]

_ARCH_TEMPLATES = [
    "The system uses {tech} for {purpose}",
    "{tech} is used for {purpose} in the backend",
    "The architecture relies on {tech} for {purpose}",
    "We use {tech} as our {purpose} solution",
    "{purpose} is handled by {tech}",
]

_TEMPORAL_CITIES = [
    "San Francisco", "New York", "London", "Berlin", "Tokyo",
    "Sydney", "Toronto", "Amsterdam", "Singapore", "Stockholm",
]

_CODING_DECISIONS = [
    ("Rust", "C++", "memory safety guarantees"),
    ("Go", "Java", "simpler concurrency model"),
    ("PostgreSQL", "MySQL", "better JSON support"),
    ("React", "Angular", "lighter bundle size"),
    ("Docker", "bare metal", "reproducible deployments"),
    ("SQLite", "PostgreSQL", "zero-config local storage"),
    ("TypeScript", "JavaScript", "type safety at scale"),
    ("pytest", "unittest", "more expressive assertions"),
    ("uv", "pip", "faster dependency resolution"),
    ("FastAPI", "Flask", "automatic OpenAPI generation"),
    ("gRPC", "REST", "efficient binary serialization"),
    ("Kafka", "RabbitMQ", "better throughput at scale"),
    ("Terraform", "Ansible", "declarative infrastructure"),
    ("WASM", "native", "cross-platform portability"),
    ("SQLite FTS5", "Elasticsearch", "zero-dependency full-text search"),
]

_QUERY_TEMPLATES = [
    "What does {name} prefer?",
    "What does {name} like?",
    "What technology does the system use for {purpose}?",
    "How is {purpose} handled?",
    "Where does {name} live now?",
    "Where did {name} live in 2022?",
    "What storage does the project use?",
    "Why did we choose {tech_a} over {tech_b}?",
    "What is the {purpose} solution?",
    "What are {name}'s preferences?",
    "Tell me about the {purpose} architecture",
    "What changed for {name} recently?",
    "What database does the system use?",
    "What deployment approach do we use?",
    "What is the coding style preference for {name}?",
]


def _generate_preferences(count: int) -> list[dict[str, str]]:
    """Generate preference memory fixtures."""
    items = []
    for i in range(count):
        name = _NAMES[i % len(_NAMES)]
        pref = _PREFERENCES[i % len(_PREFERENCES)]
        items.append({"content": f"{name} {pref}", "category": "preference"})
    return items


def _generate_architecture(count: int) -> list[dict[str, str]]:
    """Generate architecture knowledge fixtures."""
    items = []
    for i in range(count):
        tech = _TECHNOLOGIES[i % len(_TECHNOLOGIES)]
        purpose = _PURPOSES[i % len(_PURPOSES)]
        template = _ARCH_TEMPLATES[i % len(_ARCH_TEMPLATES)]
        content = template.format(tech=tech, purpose=purpose)
        items.append({"content": content, "category": "architecture"})
    return items


def _generate_temporal(count: int) -> list[dict[str, str]]:
    """Generate temporal fact fixtures."""
    items = []
    half = count // 2
    for i in range(half):
        name = _NAMES[i % len(_NAMES)]
        city = _TEMPORAL_CITIES[i % len(_TEMPORAL_CITIES)]
        items.append({
            "content": f"{name} lived in {_TEMPORAL_CITIES[(i + 3) % len(_TEMPORAL_CITIES)]} in 2022",
            "category": "temporal",
        })
        items.append({
            "content": f"{name} lives in {city} now",
            "category": "temporal",
        })
    return items[:count]


def _generate_contradictions(count: int) -> list[dict[str, str]]:
    """Generate contradiction fixtures."""
    items = []
    half = count // 2
    for i in range(half):
        tech = _TECHNOLOGIES[i % len(_TECHNOLOGIES)]
        purpose = _PURPOSES[i % len(_PURPOSES)]
        items.append({
            "content": f"The project uses {tech} for {purpose}",
            "category": "contradiction",
        })
        items.append({
            "content": f"The project no longer uses {tech} for {purpose}",
            "category": "contradiction",
        })
    return items[:count]


def _generate_decisions(count: int) -> list[dict[str, str]]:
    """Generate coding decision fixtures."""
    items = []
    for i in range(count):
        chosen, rejected, reason = _CODING_DECISIONS[i % len(_CODING_DECISIONS)]
        items.append({
            "content": f"We chose {chosen} over {rejected} because of {reason}",
            "category": "decision",
        })
    return items


def _generate_queries(count: int) -> list[str]:
    """Generate synthetic queries mixing all categories."""
    queries = []
    for i in range(count):
        template = _QUERY_TEMPLATES[i % len(_QUERY_TEMPLATES)]
        name = _NAMES[i % len(_NAMES)]
        purpose = _PURPOSES[i % len(_PURPOSES)]
        tech_a = _TECHNOLOGIES[i % len(_TECHNOLOGIES)]
        tech_b = _TECHNOLOGIES[(i + 5) % len(_TECHNOLOGIES)]
        queries.append(
            template.format(
                name=name,
                purpose=purpose,
                tech_a=tech_a,
                tech_b=tech_b,
            )
        )
    return queries


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def _ingest_fixtures(
    store: SQLiteStore,
    embedder: DeterministicEmbedder,
    ns: str,
    items: list[dict[str, str]],
) -> tuple[float, int]:
    """Ingest fixtures and return (total_ms, items_per_sec)."""
    t0 = time.monotonic()
    ingested = 0
    for item in items:
        content = item["content"]
        category = item["category"]
        if category == "architecture":
            ingest_knowledge(
                store=store,
                embedder=embedder,
                extractor=None,
                ns=ns,
                title=content[:80],
                content=content,
                extract=False,
            )
        else:
            ingest_memory(
                store=store,
                embedder=embedder,
                extractor=None,
                ns=ns,
                content=content,
                extract=False,
            )
        ingested += 1
    elapsed_ms = (time.monotonic() - t0) * 1000
    items_sec = (ingested / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0.0
    return round(elapsed_ms, 2), round(items_sec, 1)


# ---------------------------------------------------------------------------
# Recall measurement
# ---------------------------------------------------------------------------


def _measure_recall(
    engine: RecallEngine,
    ns: str,
    queries: list[str],
    mode: str,
    limit: int = 8,
) -> dict[str, Any]:
    """Measure recall latency and result counts for a set of queries."""
    latencies: list[float] = []
    result_counts: list[int] = []

    for query in queries:
        t0 = time.monotonic()
        output = engine.recall(ns=ns, query=query, mode=mode, limit=limit)
        elapsed_ms = (time.monotonic() - t0) * 1000
        latencies.append(elapsed_ms)
        result_counts.append(len(output.results))

    if not latencies:
        return {"p50_ms": 0, "p95_ms": 0, "max_ms": 0, "avg_result_count": 0}

    sorted_lat = sorted(latencies)
    p50_idx = int(len(sorted_lat) * 0.50)
    p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)

    return {
        "p50_ms": round(sorted_lat[p50_idx], 2),
        "p95_ms": round(sorted_lat[p95_idx], 2),
        "max_ms": round(sorted_lat[-1], 2),
        "avg_result_count": round(statistics.mean(result_counts), 1),
    }


# ---------------------------------------------------------------------------
# Tombstoned exclusion check
# ---------------------------------------------------------------------------


def _check_tombstoned_exclusion(
    store: SQLiteStore,
    embedder: DeterministicEmbedder,
    engine: RecallEngine,
    ns: str,
) -> dict[str, Any]:
    """Insert a memory, tombstone it, verify it is excluded from recall."""
    result = ingest_memory(
        store=store,
        embedder=embedder,
        extractor=None,
        ns=ns,
        content="Tombstoned test: Alice likes purple unicorns",
        extract=False,
    )
    episode_id = result["episode_id"]

    # Recall before tombstone
    before = engine.recall(ns=ns, query="purple unicorns Alice", mode="fast", limit=10)
    found_before = any(r.id == episode_id for r in before.results)

    # Tombstone
    store.tombstone_episode(episode_id)

    # Recall after tombstone
    after = engine.recall(ns=ns, query="purple unicorns Alice", mode="fast", limit=10)
    found_after = any(r.id == episode_id for r in after.results)

    return {
        "found_before_tombstone": found_before,
        "found_after_tombstone": found_after,
        "excluded": not found_after,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_benchmark(
    ns: str = "bench",
    items: int = 1000,
    queries: int = 100,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Run a full benchmark and return results as a dict.

    Parameters
    ----------
    ns : str
        Namespace for benchmark data.
    items : int
        Total number of synthetic items to ingest.
    queries : int
        Number of synthetic recall queries per mode.
    db_path : str | None
        Path to SQLite database. Defaults to /tmp/palimp_bench.db.

    Returns
    -------
    dict
        Benchmark results including ingest metrics, recall latency per mode,
        DB stats, and tombstoned exclusion check.
    """
    if db_path is None:
        db_path = "/tmp/palimp_bench.db"

    # Clean up any existing file
    if os.path.exists(db_path):
        os.remove(db_path)

    store = SQLiteStore(db_path)
    embedder = DeterministicEmbedder()
    engine = RecallEngine(store=store, embedder=embedder)

    # Generate fixtures
    per_category = items // 5
    all_fixtures: list[dict[str, str]] = []
    all_fixtures.extend(_generate_preferences(per_category))
    all_fixtures.extend(_generate_architecture(per_category))
    all_fixtures.extend(_generate_temporal(per_category))
    all_fixtures.extend(_generate_contradictions(per_category))
    all_fixtures.extend(_generate_decisions(per_category))

    # Shuffle to avoid category ordering bias
    random.seed(42)
    random.shuffle(all_fixtures)

    # Measure ingest
    ingest_total_ms, ingest_items_per_sec = _ingest_fixtures(
        store, embedder, ns, all_fixtures
    )

    # Generate queries
    query_set = _generate_queries(queries)

    # Measure recall for each mode
    recall_fast = _measure_recall(engine, ns, query_set, "fast")
    recall_hybrid = _measure_recall(engine, ns, query_set, "hybrid")
    recall_thinking = _measure_recall(engine, ns, query_set, "thinking")

    # DB stats
    db_size_bytes = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    conn = store._conn()
    embedding_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM embedding WHERE namespace = ?", (ns,)
    ).fetchone()["cnt"]

    # Category distribution
    category_counts: dict[str, int] = {}
    for fixture in all_fixtures:
        cat = fixture.get("category", "other")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Tombstoned exclusion
    tombstoned = _check_tombstoned_exclusion(store, embedder, engine, ns)

    # v3 config info
    config = get_config()

    # Cleanup
    del store
    try:
        os.remove(db_path)
    except OSError:
        pass

    return {
        "namespace": ns,
        "items_ingested": len(all_fixtures),
        "queries_per_mode": queries,
        "ingest": {
            "total_ms": ingest_total_ms,
            "items_per_sec": ingest_items_per_sec,
        },
        "recall": {
            "fast": recall_fast,
            "hybrid": recall_hybrid,
            "thinking": recall_thinking,
        },
        "db": {
            "size_bytes": db_size_bytes,
            "embedding_count": embedding_count,
        },
        "tombstoned_exclusion": tombstoned,
        "v3_config": {
            "graph_max_hops": config.max_hops,
            "temporal_filter_enabled": True,
            "alias_dedup_enabled": True,
            "reranker_enabled": config.reranker_endpoint is not None,
            "category_distribution": category_counts,
        },
    }
