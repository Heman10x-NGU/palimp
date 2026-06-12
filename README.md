# Palimp

> **Local-first context graph for AI agents.** Memories, knowledge, recall, provenance, and MCP access — all in a single SQLite file.

Most memory layers help agents remember. **Palimp helps agents remember safely.** Every fact is namespace-scoped, source-linked, provenance-aware, and returned through MCP as data — never as hidden instruction.

Named after the [palimpsest](https://en.wikipedia.org/wiki/Palimpsest) — a manuscript where old text is scraped away and rewritten, but traces of the original always remain.

## Why Palimp?

| You want... | Use Palimp |
|---|---|
| Agent memory that works locally, no cloud | ✅ SQLite-only, zero external deps |
| To inspect what your agent "remembers" | ✅ Open the .db file, see everything |
| Provenance on every fact | ✅ Every result linked to source |
| MCP integration (Claude, Cursor, Codex) | ✅ 8 MCP tools, safety guaranteed |
| Multi-hop graph recall | ✅ 2-3 hop BFS with depth decay |
| Temporal truth (current vs historical) | ✅ `valid_from`/`valid_until` + `as_of` queries |
| Iterative search (progressive narrowing) | ✅ `recall_refine` for agentic search |
| Token-budgeted context | ✅ `max_tokens` parameter |

## Quick Demo

```bash
pip install palimp
palimp serve --port 8420 &

# Add a memory
curl -X POST http://localhost:8420/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"namespace":"demo","content":"Alice prefers concise technical answers."}'

# Add knowledge
curl -X POST http://localhost:8420/v1/knowledge \
  -H "Content-Type: application/json" \
  -d '{"namespace":"demo","title":"Architecture","content":"Palimp uses SQLite, FTS, embeddings, and provenance."}'

# Recall — searches across both memories and knowledge
curl -X POST http://localhost:8420/v1/recall \
  -H "Content-Type: application/json" \
  -d '{"namespace":"demo","query":"What does Alice prefer?"}'
```

Every result includes provenance (where it came from) and safety metadata (`treat_as_instruction: false`).

## Features

| Feature | Description |
|---|---|
| **Memories + Knowledge** | Dynamic context (memories) vs static docs (knowledge) — first-class separation |
| **Provenance** | Every fact traceable to source episode with extractor version and evidence span |
| **Multi-hop graph** | 2-3 hop BFS traversal with configurable depth decay |
| **Temporal validity** | `valid_from`/`valid_until` on entities, edges, claims. Query as-of any point in time |
| **Entity alias dedup** | "Python", "python", "the Python" → one entity. Namespace-scoped |
| **Contradiction tracking** | CONTRADICTS/SUPERSEDES edges with warnings on recall |
| **Search modes** | `lexical` (FTS5), `vector` (embeddings), `graph` (traversal), `hybrid` (all) |
| **Token budget** | `max_tokens` parameter limits total context returned |
| **Iterative recall** | `recall_refine` for progressive narrowing — inspired by Turbopuffer's agentic search |
| **AdaCoM context management** | Requirement extraction, dedup, relevance scoring, tier-based compression |
| **Runbook mode** | Project gotchas, workflows, command fixes — pack before coding tasks |
| **Trigger keywords** | Bind terms to memories for automatic surfacing |
| **Ebbinghaus decay** | Forgetting curve with pin support. Recent/pinned items rank higher |
| **Batch ingestion** | Up to 50 items per request |
| **Explainable retrieval** | Score breakdown per result: lexical, vector, graph, recency, confidence |
| **Session lifecycle** | Create/close sessions with summaries |
| **MCP-safe output** | `treat_as_instruction: false` on every result. No cross-namespace access |

## Install

```bash
pip install palimp
```

## CLI Quickstart

```bash
# Add a memory
palimp memory add --namespace demo "Alice prefers concise answers."

# Add knowledge
palimp knowledge add --namespace demo --title "Architecture" --content "SQLite-first design."

# Recall
palimp recall --namespace demo "what does Alice prefer?"

# Recall with search mode
palimp recall --namespace demo "authentication" --search-mode lexical

# Recall with token budget
palimp recall --namespace demo "architecture" --max-tokens 200

# Iterative recall (progressive narrowing)
palimp recall refine --namespace demo --query "infrastructure" --from-ids "eps_abc,eps_def"

# Runbook for coding agents
palimp runbook add --namespace repo --kind gotcha --content "pytest needs PALIMP_DB=:memory:"
palimp runbook pack --namespace repo --task "fix storage tests" --budget 2000

# Stats and diagnostics
palimp stats --namespace demo
palimp doctor --db ~/.palimp/palimp.db
```

## MCP Configuration

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

MCP tools: `palimp_memory_add`, `palimp_knowledge_add`, `palimp_recall`, `palimp_search`, `palimp_search_refine`, `palimp_context_get`, `palimp_stats`, `palimp_context_pack`

## Architecture

```
Clients (REST / CLI / MCP)
        │
        ▼
Boundary Layer (namespace validation, payload limits, safe MCP output)
        │
        ▼
Context Engine (ingestion, extraction, normalization, conflict checking, recall scoring)
        │
        ▼
SQLite Store (memories, knowledge, episodes, entities, edges, claims, provenance, embeddings, FTS, audit log)
```

## How It Compares

| Tool | Best for | Palimp's angle |
|---|---|---|
| **Mem0** | Hosted agent memory platform | Palimp is local, inspectable, provenance-first |
| **Graphiti/Zep** | Temporal graph memory (Neo4j) | Palimp is SQLite-only, zero external deps |
| **AutoMem** | Multi-signal scoring (FalkorDB + Qdrant) | Palimp adds runbooks, triggers, MCP safety |
| **HydraDB** | Managed cloud context infrastructure | Palimp is the local OSS alternative |

Palimp is not trying to be the biggest memory system. It is the smallest useful local context graph for agent builders who care about provenance, MCP safety, and installability.

## Safety

- Every memory is **namespace-scoped**. No cross-namespace access.
- Every fact has **provenance** — linked to the source episode.
- **Contradictions are tracked**, not silently overwritten.
- MCP output returns **data, not instructions**. `safety.treat_as_instruction` is always `false`.
- Deleted sources are **tombstoned**, not physically removed. Audit logs preserved.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/v1/health` | Health check |
| GET | `/v1/stats?namespace=demo` | Namespace statistics |
| POST | `/v1/memories` | Add a memory |
| POST | `/v1/memories/batch` | Batch add up to 50 memories |
| POST | `/v1/knowledge` | Add a knowledge item |
| POST | `/v1/knowledge/batch` | Batch add up to 50 knowledge items |
| POST | `/v1/recall` | Unified retrieval (search_mode, max_tokens) |
| POST | `/v1/recall/refine` | Iterative recall narrowing |
| GET | `/v1/context/{entity_id}` | Entity context from graph |
| DELETE | `/v1/sources/{episode_id}` | Tombstone a source |

## Development

```bash
git clone https://github.com/Heman10x-NGU/palimp.git
cd palimp
pip install -e ".[dev,mcp]"
pytest -q
```

## License

MIT

---

Built by [@heman10x](https://github.com/Heman10x-NGU) — inspired by [Turbopuffer's "RAG is dead" talk](https://www.youtube.com/watch?v=Kuba Rogut), [AdaCoM](https://arxiv.org/abs/2605.30785), and the local-first agent memory movement.
