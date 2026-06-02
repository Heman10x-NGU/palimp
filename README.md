# Palimp v0.3.0

> **Layers of truth for AI agents.** Local-first context graph with memories, knowledge, recall, provenance, and MCP access in a single SQLite file.

<!-- Badges — uncomment after PyPI publish
[![PyPI](https://img.shields.io/pypi/v/palimp)](https://pypi.org/project/palimp/)
[![Python](https://img.shields.io/pypi/pyversions/palimp)](https://pypi.org/project/palimp/)
[![License](https://img.shields.io/github/license/Heman10x-NGU/palimp)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/Heman10x-NGU/palimp/ci.yml)](https://github.com/Heman10x-NGU/palimp/actions)
-->

Most memory layers help agents remember. **Palimp helps agents remember safely.** Every fact is namespace-scoped, source-linked, provenance-aware, and returned through MCP as data — never as hidden instruction.

Named after the [palimpsest](https://en.wikipedia.org/wiki/Palimpsest) — a manuscript where old text is scraped away and rewritten, but traces of the original always remain. That's exactly what temporal truth with provenance is.

## Features at a Glance

| Feature | Status |
|---|---|
| ✅ SQLite-only — zero external dependencies | Built-in |
| ✅ Memory vs. Knowledge separation | First-class |
| ✅ Provenance on every fact | Episode-linked |
| ✅ MCP-safe output (`treat_as_instruction: false`) | Guaranteed |
| ✅ Multi-hop graph traversal (2-3 hops) | BFS with decay |
| ✅ Temporal validity + `as_of` queries | `valid_from`/`valid_until` |
| ✅ Entity alias deduplication | Namespace-scoped |
| ✅ Contradiction/supersession tracking | CONTRADICTS/SUPERSEDES |
| ✅ Coding-agent runbooks + preprompt hook | Gotcha/workflow/command_fix |
| ✅ Trigger keywords | Term-to-memory boost |
| ✅ AdaCoM-inspired context packing | Token-budget-aware |
| ✅ Explainable retrieval | 7-signal score breakdown |
| ✅ Ebbinghaus decay + pinning | Forgetting curve |
| ✅ Batch ingestion (50 items/request) | REST + CLI |

## Install

```bash
pip install palimp
```

## Quickstart — REST API

Start the server:

```bash
palimp serve --port 8420
```

Add a memory:

```bash
curl -X POST http://localhost:8420/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"namespace":"demo","content":"Alice prefers concise technical answers."}'
```

Add a knowledge item:

```bash
curl -X POST http://localhost:8420/v1/knowledge \
  -H "Content-Type: application/json" \
  -d '{"namespace":"demo","title":"Architecture","content":"Palimp uses SQLite, FTS, embeddings, and provenance."}'
```

Recall across memories and knowledge:

```bash
curl -X POST http://localhost:8420/v1/recall \
  -H "Content-Type: application/json" \
  -d '{"namespace":"demo","query":"What does Alice prefer?"}'
```

Check health:

```bash
curl http://localhost:8420/v1/health
```

## Quickstart — CLI

```bash
# Add a memory
palimp memory add --namespace demo "Alice prefers concise answers."

# Add a knowledge item
palimp knowledge add --namespace demo --title "Architecture" --content "Palimp uses SQLite."

# Recall
palimp recall --namespace demo "what does Alice prefer?"

# View stats
palimp stats --namespace demo

# Run diagnostics
palimp doctor --db ~/.palimp/palimp.db
```

## Quickstart — Coding-Agent Workflow

```bash
# 1. Add project gotchas to the runbook
palimp runbook add --namespace myproject --kind gotcha \
  --content "Storage tests need PALIMP_DB=:memory: to avoid file locks"
palimp runbook add --namespace myproject --kind workflow \
  --content "Run 'uv sync --all-extras' before pytest to catch dep conflicts"

# 2. Add a trigger keyword
palimp trigger add --namespace myproject --term "storage" --memory-id mem_...

# 3. Pack context before a coding task
palimp runbook pack --namespace myproject \
  --task "fix failing storage tests" --budget 2000

# 4. Recall with temporal filtering
palimp recall --namespace myproject "what changed in retriever last month?" \
  --temporal-mode historical

# 5. Preprompt hook for agent integration
palimp hook preprompt --namespace myproject \
  --task "refactor retriever module" --budget 2000 --json
```

## MCP Configuration

Add this to your Claude Desktop (or any MCP-compatible client) config:

```json
{
  "mcpServers": {
    "palimp": {
      "command": "palimp",
      "args": ["serve", "--port", "8420"]
    }
  }
}
```

Palimp exposes these MCP tools:

- `palimp_memory_add` — add a memory to a namespace
- `palimp_knowledge_add` — add a knowledge item to a namespace
- `palimp_recall` — unified retrieval across memories and knowledge
- `palimp_context_get` — get entity context from the graph
- `palimp_stats` — get namespace statistics
- `palimp_context_pack` — pack runbook + recall context for a coding-agent task

All MCP output is structured data. Memories and knowledge are returned as facts, never as instructions.

## How It Works

```
Clients (REST / CLI / MCP)
        |
        v
Boundary Layer (namespace required, payload limits, safe MCP output)
        |
        v
Context Engine (ingestion, extraction, normalization, conflict checking, recall scoring)
        |
        v
SQLite Store (memories, knowledge, episodes, entities, edges, claims, provenance, embeddings, FTS, audit log)
```

Public API uses HydraDB-style primitives (memories, knowledge, recall, context). Internal storage keeps episode/provenance-first architecture for correctness.

## Comparison

| Product | Stars | Best for | Palimp difference |
|---|---|---|---|
| **Mem0** | ~57K ★ | Agent memory platform | Palimp is local, SQLite-only, provenance-first |
| **Graphiti/Zep** | ~20K ★ | Temporal graph memory (Neo4j) | Palimp is smaller, zero external deps, MCP-safe |
| **AutoMem** | ~1.5K ★ | Multi-signal scoring | Palimp adds runbooks, trigger keywords, preprompt hook |
| **Raw vector DB** | — | Semantic search | Palimp adds memories, knowledge, graph, provenance |

See [docs/COMPARISON.md](docs/COMPARISON.md) for detailed differences.

## Safety and Provenance

- Every memory is namespace-scoped. No cross-namespace access.
- Every fact has provenance — linked to the source episode where it was extracted.
- Contradictions and supersessions are tracked. Recalling a fact that was later contradicted includes a warning.
- MCP output returns data, not instructions. The `safety.treat_as_instruction` field is always `false`.
- Deleted sources are tombstoned, not physically removed. Tombstoned content never appears in recall. Audit logs are preserved.

## What Palimp Is NOT

- NOT a full HydraDB replacement. HydraDB is a managed production platform with team workspaces, dashboards, and cloud sync. Palimp is a local context graph.
- NOT a hosted managed platform. You run it yourself on your own machine.
- NOT a production-grade multi-tenant system. It is designed for single-user or small-team local use.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/v1/health` | Health check |
| GET | `/v1/stats?namespace=demo` | Namespace statistics |
| POST | `/v1/memories` | Add a memory |
| POST | `/v1/knowledge` | Add a knowledge item |
| POST | `/v1/recall` | Unified retrieval |
| GET | `/v1/context/{entity_id}` | Entity context from graph |
| DELETE | `/v1/sources/{episode_id}` | Tombstone a source |

## Recall Modes

| Mode | Description |
|---|---|
| `fast` | FTS + deterministic/local embeddings only |
| `hybrid` | fast + multi-hop graph context boost |
| `thinking` | hybrid + temporal/conflict/provenance reranking + query expansion |

## Multi-Hop Graph Traversal

Recall traverses the entity graph up to 2 hops by default (configurable to 3). This means if A is connected to B and B is connected to C, a query about A can surface C even without a direct lexical match.

```bash
# Configure hop depth
export PALIMP_GRAPH_MAX_HOPS=2         # default, max 3
export PALIMP_GRAPH_DEPTH_DECAY=0.55   # hop-2 score multiplier
```

Explanation includes `graph_paths`, `hop_count`, and `path_score` for every multi-hop result.

## Temporal Validity

Facts carry `valid_from` / `valid_until` timestamps. Recall filters by temporal mode:

```bash
# Current facts only (default)
palimp recall --namespace demo "where does Alice live?"

# Historical facts
palimp recall --namespace demo "where did Alice live in 2022?" --temporal-mode historical

# Point-in-time query
palimp recall --namespace demo "project config" --as-of 2026-01-15T00:00:00Z
```

Temporal states: `current`, `historical`, `future`, `stale`, `unknown`.

## Entity Alias Deduplication

Entities are deduplicated across surface forms. "Python", "python", and "the Python" resolve to one entity within the same namespace.

```bash
# Merge two entities manually
palimp entity merge --namespace demo ent_a ent_b --reason "same project tool"
```

Alias normalization: lowercase, trim, strip leading articles, collapse punctuation variants. Cross-namespace and cross-type merges are never performed.

## Runbook Mode

Build a project runbook of gotchas, workflows, and command fixes. Pack a context bundle before each coding-agent task:

```bash
# Add runbook entries
palimp runbook add --namespace repo --kind gotcha --content "pytest requires PALIMP_DB=:memory: for isolated tests"
palimp runbook add --namespace repo --kind command_fix --content "uv build fails without --no-cache on CI"

# Pack context for a task
palimp runbook pack --namespace repo --task "fix failing storage tests" --budget 2000
```

Runbook kinds: `gotcha`, `workflow`, `command_fix`, `project_invariant`, `dependency_note`, `debug_trace`, `architecture_decision`.

## Trigger Keywords

Link trigger terms to memories so they surface even with weak lexical/vector scores:

```bash
palimp trigger add --namespace repo --term "pytest" --memory-id mem_...
palimp trigger list --namespace repo
```

When a query contains a trigger term, the linked memory gets a category/trigger boost in ranking.

## Explainable Retrieval

Every recall can return a full explanation of *why* each result was retrieved. Set `explain=True` on the recall endpoint or SDK call:

```bash
curl -X POST http://localhost:8420/v1/recall \
  -H "Content-Type: application/json" \
  -d '{"namespace":"demo","query":"What does Alice prefer?","explain":true}'
```

The response includes:

- `score_breakdown` per result: lexical, vector, graph_boost, recency, confidence scores
- `why_retrieved`: human-readable reason (e.g. "strong lexical match; recent content")
- Top-level `explanation`: query expansions, detected entities, latency per stage

In Python:

```python
from palimp.retriever import RecallEngine

output = engine.recall("demo", "What does Alice prefer?", explain=True)
for r in output.results:
    print(r.score_breakdown.lexical, r.score_breakdown.vector, r.why_retrieved)
```

## AdaCoM Context Management

Palimp uses an AdaCoM-inspired (arXiv:2605.30785) rule-based context manager that sits between recall and the API layer. It:

1. **Extracts requirements** from the query.
2. **Deduplicates** memories with >70% content overlap.
3. **Relevance-scores** remaining memories (60% recall score, 40% relevance).
4. **Applies a token budget** based on agent tier (`low`=0.35, `medium`=0.65, `high`=0.90).

```python
from palimp.context_manager import ContextManager

ctx_mgr = ContextManager(store, agent_tier="medium")
managed = ctx_mgr.manage_context("demo", "What does Alice prefer?", results, "hybrid")
print(managed.compression_ratio, managed.explanation)
```

The `managed.explanation` dict reports requirements extracted, duplicates merged, memories dropped, and token savings.

## Batch Ingestion

Add up to 50 memories or knowledge items in a single request:

```bash
# Batch memories
curl -X POST http://localhost:8420/v1/memories/batch \
  -H "Content-Type: application/json" \
  -d '{"namespace":"demo","items":[
    {"content":"Alice prefers concise answers."},
    {"content":"Bob prefers verbose explanations."}
  ]}'

# Batch knowledge
curl -X POST http://localhost:8420/v1/knowledge/batch \
  -H "Content-Type: application/json" \
  -d '{"namespace":"demo","items":[
    {"title":"API Style","content":"REST-first with JSON payloads."},
    {"title":"DB Choice","content":"SQLite with WAL mode."}
  ]}'
```

Both endpoints return a `BatchResponse` with created IDs and per-item status.

## Ebbinghaus Decay

Episodes decay over time using the Ebbinghaus forgetting curve:

```
R = e^(-t / S)
```

Where `t` = days since last access, `S` = stability (higher = slower decay). Pinned items have infinite stability and never decay.

```python
from palimp.decay import compute_decay_score

# Retention after 1 day with default stability
retention = compute_decay_score("2026-06-01T00:00:00Z")

# Pin an episode so it never decays
store.pin_episode(episode_id)
```

Decay scores feed into recall ranking: recent or pinned items rank higher.

## Session Lifecycle

Track conversation sessions with create/close semantics:

```bash
# Create a session
curl -X POST http://localhost:8420/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"namespace":"demo","user_ref":"alice"}'

# Close with summary
curl -X POST http://localhost:8420/v1/sessions/{session_id}/close \
  -H "Content-Type: application/json" \
  -d '{"summary":"Discussed architecture decisions."}'

# List sessions
curl "http://localhost:8420/v1/sessions?namespace=demo"
```

Sessions group related episodes and enable session-scoped recall.

## Python SDK

Palimp ships an async Python SDK (`PalimpClient`) for programmatic access to all REST endpoints:

```python
from palimp.sdk import PalimpClient

async def main():
    async with PalimpClient("http://localhost:8420") as client:
        # Memories
        await client.add_memory("demo", "Alice prefers concise answers.")
        await client.add_memory_batch("demo", [
            {"content": "Fact A."},
            {"content": "Fact B."},
        ])

        # Knowledge
        await client.add_knowledge("demo", "Architecture", "SQLite-first design.")

        # Recall
        results = await client.recall("demo", "What does Alice prefer?")

        # Sessions
        session = await client.create_session("demo", user_ref="alice")
        await client.close_session(session["session_id"], summary="Done.")

        # Pin/unpin
        await client.pin("demo", episode_id)
        await client.unpin("demo", episode_id)
```

## Benchmarks

Palimp has been evaluated against two long-term memory benchmarks using a **zero-dependency deterministic embedder** (mock hashing, no network calls or LLM API costs) to establish a baseline.

### LoCoMo-10

[LoCoMo](https://github.com/snap-research/locomo) evaluates long-term conversational memory across multi-session dialogue histories (~400 dialogue turns per session). Below is the turn-level retrieval recall at Top-10.

| Mode | Overall Recall | Multi-hop | Single-hop | Temporal | Open-domain | Adversarial |
| --- | --- | --- | --- | --- | --- | --- |
| **FAST** | 7.21% | 6.32% | 6.05% | 14.50% | 7.56% | 6.28% |
| **HYBRID** | 7.27% | 6.32% | 6.04% | 14.50% | 7.56% | 6.28% |
| **THINKING** | 1.03% | 0.70% | 0.95% | 7.45% | 0.46% | 1.05% |

> [!NOTE]
> In turn-level retrieval with mock deterministic (SHA-256 hash) embeddings, query expansion in **THINKING** mode introduces noise that dilutes exact-match rankings.

### LongMemEval-S

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) evaluates session-level recall across multi-session user history. Below is the session-level retrieval recall at Top-5 and Top-10.

| Mode | Recall_any@5 | Recall_any@10 | Recall_all@5 | Recall_all@10 |
| --- | --- | --- | --- | --- |
| **FAST** | 11.80% | 14.40% | 1.20% | 2.00% |
| **HYBRID** | 12.40% | 15.00% | 1.20% | 2.00% |
| **THINKING** | **40.80%** | **43.60%** | **13.20%** | **14.20%** |

> [!TIP]
> In session-level retrieval with larger context blocks, query expansion in **THINKING** mode is highly effective, yielding a **~3x boost** in recall compared to fast/hybrid modes.

## Development

```bash
git clone https://github.com/Heman10x-NGU/palimp.git
cd palimp
pip install -e ".[dev,mcp]"
pytest -q
```

## License

MIT
