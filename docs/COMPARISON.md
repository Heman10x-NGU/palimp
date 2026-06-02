# Detailed Comparison: GraphCtx vs Adjacent Tools

## GraphCtx vs HydraDB

HydraDB is managed cloud context infrastructure for production AI applications. It provides hosted context graphs, team workspaces, dashboards, and a managed API.

**When to use HydraDB:** You need a managed, multi-tenant, production-grade context platform with team workspaces, dashboards, cloud sync, and an SLA.

**When to use GraphCtx:** You want the same conceptual building blocks (memories, knowledge, recall, context) but locally, open-source, and inspectable -- without committing to hosted infrastructure.

**What GraphCtx borrows:** The public primitive vocabulary (memories, knowledge, recall, context) and the mental model of a context graph for agents. API shape inspiration only.

| Dimension | HydraDB | GraphCtx |
|---|---|---|
| Deployment | Hosted cloud | Local, self-hosted |
| Storage | Managed backend | SQLite (single file) |
| Inspectability | API-level access only | Direct database inspection |
| Multi-tenant | Yes | No (single-user/small-team) |
| Team workspaces | Yes | No |
| Dashboard | Yes | No |
| Cloud sync | Yes | No |
| Open source | No | Yes (MIT) |
| Provenance | Internal | Exposed in every recall result |
| MCP support | Yes | Yes (local) |
| Cost | Paid | Free |

GraphCtx is NOT a HydraDB replacement. It is the local OSS context graph that gives agent builders the core HydraDB-style workflow without committing to hosted infrastructure.

---

## GraphCtx vs Mem0 / OpenMemory

Mem0 is an agent memory platform that provides both hosted and self-hosted options for memory management with graph and vector recall. It publishes serious benchmark numbers (91.6 LoCoMo, 94.8 LongMemEval) and supports user/session/agent memory categories.

**When to use Mem0:** You need a production agent memory platform with hosted infrastructure, team features, multi-backend support, and published benchmark performance.

**When to use GraphCtx:** You want a local-first, SQLite-inspectable context graph with full provenance on every fact, separation of memories from knowledge, and MCP safety guarantees -- without a hosted platform.

**What GraphCtx borrows:** UX/API ergonomics for memory categorization. MCP memory tool surface design. Benchmark methodology discipline.

| Dimension | Mem0 | GraphCtx |
|---|---|---|
| Focus | Agent memory platform | Local context graph |
| Storage | Multiple backends | SQLite-first |
| Graph | Yes | Yes (episode/provenance-first) |
| Provenance | Partial | Full (every fact source-linked) |
| MCP | Yes | Yes |
| Hosted option | Yes | No |
| Knowledge docs | Limited | First-class `knowledge` primitive |
| Contradiction tracking | No | Yes (CONTRADICTS/SUPERSEDES) |
| Benchmark scores | Published (91.6 LoCoMo) | Local mini-eval only |

GraphCtx separates dynamic memories from static knowledge and tracks provenance on every extracted fact. Mem0 focuses on broader agent memory workflows with stronger benchmark performance.

---

## GraphCtx vs Graphiti / Zep

Graphiti (by Zep) provides temporal knowledge graph memory for AI agents. It focuses on episode-first storage with temporal validity, conflict-aware retrieval, and provenance. It uses a graph database backend (typically FalkorDB or Neo4j).

**When to use Graphiti:** You need a full temporal context graph with a dedicated graph database backend, episode-first architecture, and the production infrastructure to run it.

**When to use GraphCtx:** You want the same episode-first and temporal validity ideas but with zero external dependencies -- SQLite only, no graph database required, MCP-native.

**What GraphCtx borrows:** Episode-first storage architecture, temporal validity windows (valid_from, valid_until), supersession tracking, provenance on every extracted fact, and the distinction between historical and current recall.

| Dimension | Graphiti/Zep | GraphCtx |
|---|---|---|
| Architecture | Graph database backend | SQLite + FTS + embeddings |
| Temporal | Yes (valid_from, valid_until) | Yes (valid_from, valid_until) |
| Supersession | Yes | Yes |
| Size | Larger dependency tree | Minimal (SQLite only) |
| MCP | Yes | Yes |
| Deployment | Self-hosted or cloud | Local-only |
| Graph DB dependency | Yes (typically Neo4j/FalkorDB) | No (SQLite) |

GraphCtx borrows the episode-first and temporal validity ideas from Graphiti but keeps the dependency footprint minimal. No external graph database is required.

---

## GraphCtx vs AutoMem

AutoMem is a graph+vector memory stack that stores typed relationships (11 relationship types) alongside embeddings. It uses FalkorDB for the graph and Qdrant for vectors, with a 9-component scoring system. It benchmarks at 87% on LongMemEval and 84.74% on LoCoMo.

**When to use AutoMem:** You want a more advanced graph+vector memory stack with typed relationships, bridge discovery, and published benchmark performance. You are comfortable self-hosting Docker or deploying on Railway.

**When to use GraphCtx:** You want a simpler, dependency-light context graph that runs in one command with SQLite only. You prioritize inspectability, MCP safety, and installability over scoring sophistication.

**What GraphCtx borrows:** The 11 relationship taxonomy informed GraphCtx's relation enum. The 9-component scoring approach and LongMemEval/LoCoMo benchmark discipline shaped retrieval scoring and evaluation thinking.

| Dimension | AutoMem | GraphCtx |
|---|---|---|
| Graph backend | FalkorDB | SQLite (episode/edge tables) |
| Vector backend | Qdrant | Deterministic embeddings in SQLite |
| Relationship types | 11 typed relationships | Relation enum (simpler) |
| Scoring | 9-component hybrid | Multi-signal (lexical, vector, graph, recency, confidence) |
| Bridge discovery | Yes | Yes (multi-hop BFS, v0.3) |
| Benchmark | 87% LongMemEval | Local mini-eval only |
| Dependencies | FalkorDB + Qdrant + Docker | SQLite only |
| Install complexity | Docker Compose | `pip install graphctx` |

AutoMem is a more advanced graph+vector memory stack. GraphCtx is simpler, local-first, and installable in one command.

---

## GraphCtx vs MemoryOS

MemoryOS provides persistent, queryable memory for AI agents with temporal awareness, a knowledge graph, and sub-100ms retrieval. It uses Postgres (pgvector) and Redis, with Ebbinghaus-based decay and fast/thinking retrieval modes.

**When to use MemoryOS:** You need a full memory OS with Postgres+Redis infrastructure, Ebbinghaus decay, fast/thinking retrieval modes, and published performance numbers (78ms fast p50, 470ms thinking p50).

**When to use GraphCtx:** You want the same temporal KG and decay concepts but with SQLite-only storage, no external database dependencies, and MCP-native access.

**What GraphCtx borrows:** Temporal knowledge graph with append-only supersession. Ebbinghaus forgetting curve for memory decay. Fast vs. thinking retrieval mode naming. Query explanation traces.

| Dimension | MemoryOS | GraphCtx |
|---|---|---|
| Storage | Postgres (pgvector) + Redis | SQLite |
| Graph | Temporal KG | Episode/provenance graph |
| Decay | Ebbinghaus | Ebbinghaus |
| Retrieval modes | fast, thinking | fast, hybrid, thinking |
| Fast query p50 | ~78ms | Local benchmark available |
| Dependencies | Postgres + Redis + Docker | SQLite only |
| MCP | No (REST API) | Yes (native) |

MemoryOS is a more full-featured memory platform with external dependencies. GraphCtx borrows the core concepts (temporal KG, decay, mode naming) but keeps everything SQLite-local.

---

## GraphCtx vs Mnemon

Mnemon is an LLM-supervised persistent memory for AI agents. It is a single Go binary with zero API keys, using your host LLM as the supervisor. It provides `remember`, `link`, and `recall` primitives with intent-aware recall and importance decay.

**When to use Mnemon:** You want a single-binary memory tool with LLM-supervised memory management, Go-based performance, and a protocol-native approach (remember/link/recall as cognitive vocabulary).

**When to use GraphCtx:** You want a Python-based context graph with SQLite inspectability, separate memory/knowledge primitives, provenance tracking, and MCP-native access.

**What GraphCtx borrows:** The `init-agent` onboarding command concept. Agent-client setup workflow expectations. Simple install philosophy ("one command"). Memory protocol primitive naming.

| Dimension | Mnemon | GraphCtx |
|---|---|---|
| Language | Go | Python |
| Install | Single binary | `pip install graphctx` |
| LLM role | Supervisor (external) | Not required for storage/retrieval |
| Memory model | remember/link/recall | memory/knowledge/recall/context |
| Graph | Four-graph knowledge store | Episode/provenance graph |
| Decay | Importance decay | Ebbinghaus decay |
| MCP | Yes | Yes |
| API keys | None required | None required |

Mnemon takes a different architectural approach (LLM-supervised, Go binary). GraphCtx borrows the onboarding simplicity and protocol vocabulary ideas.

---

## GraphCtx vs Nocturne Memory

Nocturne Memory is a long-term memory server for MCP agents, using SQLite or PostgreSQL. It focuses on giving AI persistent personality and memory across sessions, with rollbackable MCP operations.

**When to use Nocturne Memory:** You want a memory server focused on AI personality persistence, with rollbackable MCP operations and Chinese-language documentation/community.

**When to use GraphCtx:** You want a context graph focused on coding agent workflows, with separate memory/knowledge primitives, provenance tracking, and structured recall modes.

**What GraphCtx borrows:** Rollback/delete test patterns for MCP memory operations. Local MCP ergonomics for persistent agent context. The idea that MCP memory should be rollbackable and inspectable.

| Dimension | Nocturne Memory | GraphCtx |
|---|---|---|
| Focus | AI personality persistence | Coding agent context graph |
| Storage | SQLite or PostgreSQL | SQLite |
| MCP | Yes (rollbackable) | Yes |
| Memory model | Unified memory | memory/knowledge separation |
| Provenance | Limited | Full (episode-linked) |
| Contradiction tracking | No | Yes |
| Primary language | Chinese/English | English |

Nocturne Memory focuses on AI personality persistence. GraphCtx borrows MCP ergonomics and rollback test patterns.

---

## GraphCtx vs Raw Vector Databases

Raw vector databases (Chroma, LanceDB, Qdrant, Pinecone) provide semantic search over embeddings.

**When to use a raw vector DB:** You only need embedding similarity search. You do not need provenance, knowledge separation, contradiction tracking, or MCP access.

**When to use GraphCtx:** You need a context graph layer on top of semantic search: provenance, knowledge separation, contradiction tracking, temporal validity, and MCP access.

**What GraphCtx borrows:** Nothing directly. GraphCtx uses deterministic embeddings stored in SQLite rather than a dedicated vector DB.

| Dimension | Raw Vector DB | GraphCtx |
|---|---|---|
| Semantic search | Yes | Yes |
| FTS | Sometimes | Yes (SQLite FTS5) |
| Knowledge graph | No | Yes (entities, edges, claims) |
| Provenance | No | Yes |
| Memory vs knowledge | No distinction | Separate primitives |
| Contradiction tracking | No | Yes |
| MCP | No (unless custom) | Yes (built-in) |
| Namespace isolation | Manual | Enforced |

GraphCtx adds the context graph layer on top of semantic search. If you only need embedding similarity, a raw vector DB is simpler. If you need provenance, knowledge separation, contradiction tracking, and MCP access, GraphCtx provides that structure.

---

## Summary

GraphCtx occupies a specific niche: it is a local, inspectable, SQLite-first context graph that provides HydraDB-style primitives (memories, knowledge, recall, context) for developers who want to self-host their agent's context layer.

**When to use each tool:**

- Use **Mem0** if you want managed memory/product maturity.
- Use **Graphiti** if you need a full temporal graph framework with graph DB backends.
- Use **AutoMem** if you want a more mature graph+vector memory stack with published benchmark discipline.
- Use **Mnemon/Nocturne** if their CLI/MCP workflow matches your host-agent style.
- Use **GraphCtx** if you want local SQLite, coding-agent runbooks, source-linked evidence, deterministic default embeddings, MCP-safe output, and no external graph/vector dependency.

**v0.3 differentiators:**

| Feature | GraphCtx v0.3 | Others |
|---|---|---|
| Multi-hop graph traversal | 2-3 hop BFS with depth decay over SQLite edges | Graphiti (graph DB), AutoMem (FalkorDB) |
| Temporal validity filtering | `as_of`, `temporal_mode` (current/historical/all) on SQLite | Graphiti (graph DB), MemoryOS (Postgres) |
| Entity alias deduplication | Normalized alias table, namespace-scoped merge | Not available in most local tools |
| Runbook mode | `gotcha`/`workflow`/`command_fix` evidence packs under token budget | No equivalent |
| Trigger keywords | Term-to-memory boost for weak-match surfacing | Nocturne (partial) |
| Preprompt hook | `graphctx hook preprompt` returns safe JSON for agent integration | Mnemon (LLM-supervised) |
| Configurable search weights | 7-component scoring via env vars | AutoMem (9-component, hardcoded) |

It is not the right choice if you need:

- A managed cloud platform (use HydraDB)
- A production memory platform with published benchmarks (use Mem0 or AutoMem)
- A full temporal graph with a dedicated graph database (use Graphiti)
- A memory OS with Postgres+Redis (use MemoryOS)
- An LLM-supervised single-binary tool (use Mnemon)
- Only vector search with no graph (use a raw vector DB)

It is the right choice if you want:

- Local-first, inspectable storage (SQLite, one file)
- HydraDB-style mental model (memories, knowledge, recall)
- Provenance on every fact
- Safe MCP output for agent consumption
- SQLite simplicity with no external dependencies
- Installable in one command (`pip install graphctx`)
- Coding-agent runbooks with source-linked evidence packs
- Temporal truth filtering and multi-hop graph recall
- Entity alias deduplication and trigger keywords
