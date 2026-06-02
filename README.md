# GraphCtx v0.3.0

> Open-source local context graph for AI agents -- memories, knowledge, recall, provenance, and MCP access in a SQLite-first package.

GraphCtx is an open-source, local-first alternative to HydraDB's core memory/context-graph primitive for developers who want inspectable agent context without a hosted black box.

Most memory layers help agents remember. GraphCtx helps agents remember safely. Every memory and knowledge item is namespace-scoped, source-linked, provenance-aware, and returned through MCP as data, not hidden instruction.

## Why GraphCtx?

GraphCtx is not trying to be the largest memory system. It is the smallest useful local context graph for coding agents that need source-linked project memory, temporal truth, aliases, multi-hop graph recall, and MCP-safe context packs.

Use GraphCtx for coding agents when you want local SQLite memory that is inspectable, MCP-safe, source-linked, and optimized for project experience: commands that failed, fixes that worked, architectural decisions, constraints, aliases, temporal truth, and compact evidence packs before each agent task.

## Install

```bash
pip install graphctx
```

## Quickstart -- REST API

Start the server:

```bash
graphctx serve --port 8420
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
  -d '{"namespace":"demo","title":"Architecture","content":"GraphCtx uses SQLite, FTS, embeddings, and provenance."}'
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

## Quickstart -- CLI

```bash
# Add a memory
graphctx memory add --namespace demo "Alice prefers concise answers."

# Add a knowledge item
graphctx knowledge add --namespace demo --title "Architecture" --content "GraphCtx uses SQLite."

# Recall
graphctx recall --namespace demo "what does Alice prefer?"

# View stats
graphctx stats --namespace demo

# Run diagnostics
graphctx doctor --db ~/.graphctx/graphctx.db
```

## Quickstart -- v0.3 Coding-Agent Workflow

```bash
# 1. Add project gotchas to the runbook
graphctx runbook add --namespace myproject --kind gotcha \
  --content "Storage tests need GRAPHCTX_DB=:memory: to avoid file locks"
graphctx runbook add --namespace myproject --kind workflow \
  --content "Run 'uv sync --all-extras' before pytest to catch dep conflicts"

# 2. Add a trigger keyword
graphctx trigger add --namespace myproject --term "storage" --memory-id mem_...

# 3. Pack context before a coding task
graphctx runbook pack --namespace myproject \
  --task "fix failing storage tests" --budget 2000

# 4. Recall with temporal filtering
graphctx recall --namespace myproject "what changed in retriever last month?" \
  --temporal-mode historical

# 5. Preprompt hook for agent integration
graphctx hook preprompt --namespace myproject \
  --task "refactor retriever module" --budget 2000 --json
```

## MCP Configuration

Add this to your Claude Desktop (or any MCP-compatible client) config:

```json
{
  "mcpServers": {
    "graphctx": {
      "command": "graphctx",
      "args": ["serve", "--port", "8420"]
    }
  }
}
```

GraphCtx exposes these MCP tools:

- `graphctx_memory_add` -- add a memory to a namespace
- `graphctx_knowledge_add` -- add a knowledge item to a namespace
- `graphctx_recall` -- unified retrieval across memories and knowledge
- `graphctx_context_get` -- get entity context from the graph
- `graphctx_stats` -- get namespace statistics
- `graphctx_context_pack` -- pack runbook + recall context for a coding-agent task (v0.3)

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

| Product | Best for | GraphCtx difference |
|---|---|---|
| HydraDB | Managed context infra | GraphCtx is local/open-source and inspectable |
| Mem0 | Agent memory platform | GraphCtx focuses on local context graph + provenance |
| Zep/Graphiti | Temporal graph memory | GraphCtx is smaller, SQLite-first, and MCP-local |
| Raw vector DB | Semantic search | GraphCtx adds memories, knowledge, graph, provenance |

See [docs/COMPARISON.md](docs/COMPARISON.md) for detailed differences.

## Safety and Provenance

- Every memory is namespace-scoped. No cross-namespace access.
- Every fact has provenance -- linked to the source episode where it was extracted.
- Contradictions and supersessions are tracked. Recalling a fact that was later contradicted includes a warning.
- MCP output returns data, not instructions. The `safety.treat_as_instruction` field is always `false`.
- Deleted sources are tombstoned, not physically removed. Tombstoned content never appears in recall. Audit logs are preserved.

## What GraphCtx Is NOT

- NOT a full HydraDB replacement. HydraDB is a managed production platform with team workspaces, dashboards, and cloud sync. GraphCtx is a local context graph.
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

## Multi-Hop Graph Traversal (v0.3)

Recall traverses the entity graph up to 2 hops by default (configurable to 3). This means if A is connected to B and B is connected to C, a query about A can surface C even without a direct lexical match.

```bash
# Configure hop depth
export GRAPHCTX_GRAPH_MAX_HOPS=2      # default, max 3
export GRAPHCTX_GRAPH_DEPTH_DECAY=0.55  # hop-2 score multiplier
```

Explanation includes `graph_paths`, `hop_count`, and `path_score` for every multi-hop result.

## Temporal Validity (v0.3)

Facts carry `valid_from` / `valid_until` timestamps. Recall filters by temporal mode:

```bash
# Current facts only (default)
graphctx recall --namespace demo "where does Alice live?"

# Historical facts
graphctx recall --namespace demo "where did Alice live in 2022?" --temporal-mode historical

# Point-in-time query
graphctx recall --namespace demo "project config" --as-of 2026-01-15T00:00:00Z
```

Temporal states: `current`, `historical`, `future`, `stale`, `unknown`.

## Entity Alias Deduplication (v0.3)

Entities are deduplicated across surface forms. "Python", "python", and "the Python" resolve to one entity within the same namespace.

```bash
# Merge two entities manually
graphctx entity merge --namespace demo ent_a ent_b --reason "same project tool"
```

Alias normalization: lowercase, trim, strip leading articles, collapse punctuation variants. Cross-namespace and cross-type merges are never performed.

## Runbook Mode (v0.3)

Build a project runbook of gotchas, workflows, and command fixes. Pack a context bundle before each coding-agent task:

```bash
# Add runbook entries
graphctx runbook add --namespace repo --kind gotcha --content "pytest requires GRAPHCTX_DB=:memory: for isolated tests"
graphctx runbook add --namespace repo --kind command_fix --content "uv build fails without --no-cache on CI"

# Pack context for a task
graphctx runbook pack --namespace repo --task "fix failing storage tests" --budget 2000
```

Runbook kinds: `gotcha`, `workflow`, `command_fix`, `project_invariant`, `dependency_note`, `debug_trace`, `architecture_decision`.

## Trigger Keywords (v0.3)

Link trigger terms to memories so they surface even with weak lexical/vector scores:

```bash
graphctx trigger add --namespace repo --term "pytest" --memory-id mem_...
graphctx trigger list --namespace repo
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
from graphctx.retriever import RecallEngine

output = engine.recall("demo", "What does Alice prefer?", explain=True)
for r in output.results:
    print(r.score_breakdown.lexical, r.score_breakdown.vector, r.why_retrieved)
```

## AdaCoM Context Management

GraphCtx uses an AdaCoM-inspired (arXiv:2605.30785) rule-based context manager that sits between recall and the API layer. It:

1. **Extracts requirements** from the query.
2. **Deduplicates** memories with >70% content overlap.
3. **Relevance-scores** remaining memories (60% recall score, 40% relevance).
4. **Applies a token budget** based on agent tier (`low`=0.35, `medium`=0.65, `high`=0.90).

```python
from graphctx.context_manager import ContextManager

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
from graphctx.decay import compute_decay_score

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

GraphCtx ships an async Python SDK (`GraphCtxClient`) for programmatic access to all REST endpoints:

```python
from graphctx.sdk import GraphCtxClient

async def main():
    async with GraphCtxClient("http://localhost:8420") as client:
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

## Development

```bash
git clone https://github.com/anthropics/graphctx.git
cd graphctx
pip install -e ".[dev,mcp]"
pytest -q
```

## License

MIT
