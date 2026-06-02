# OSS Feature Map

This table tracks every open-source project that informed Palimp v0.3.0. Columns:

- **Source** -- project name
- **URL** -- canonical repository or site
- **Local `_refs`** -- path in the local `_refs/` directory, if present
- **License** -- license visible in the repository
- **Best Feature** -- the single most valuable idea from that project
- **Palimp Decision** -- what we did with it
- **Attribution Note** -- what was borrowed (concepts only, never code)
- **Code Reuse** -- always `none` unless a future line explicitly says otherwise

## Feature Map Table

| Source | URL | Local `_refs` | License | Best Feature | Palimp Decision | Attribution Note | Code Reuse |
|---|---|---|---|---|---|---|---|
| AutoMem | https://github.com/verygoodplugins/automem | `_refs/automem` | MIT | 11 relationship taxonomy, 9-signal scoring, bridge discovery, LoCoMo/LongMemEval benchmark discipline | implemented | Relation enum, benchmark evaluation approach, and multi-hop bridge discovery inspired by AutoMem's typed-relationship model, scoring discipline, and bridge traversal | none |
| MemoryOS | https://github.com/Per0x1de-1337/memoryos | `_refs/MemoryOS` | MIT | Temporal KG, Ebbinghaus decay, fast/thinking retrieval modes, query explanation traces | implemented | Temporal knowledge graph with append-only supersession and Ebbinghaus forgetting curve; retrieval mode naming (fast, thinking) and explanation traces | none |
| mem0 / OpenMemory | https://github.com/mem0ai/mem0 | `_refs/mem0` | Apache-2.0 | Graph+vector memory model, user/session/agent categories, MCP memory UX, benchmark seriousness | implement-v0.2 | UX/API ergonomics for memory categorization (user/session/agent); MCP tool surface design; benchmark methodology discipline | none |
| Graphiti / Zep | https://github.com/getzep/graphiti | `_refs/graphiti` | Apache-2.0 | Episode-first temporal context graph, validity windows, supersession, provenance, historical vs. current recall | implemented | Episode-first storage architecture, temporal validity (valid_from/valid_until) with as_of/temporal_mode filtering, supersession tracking, provenance on every extracted fact, historical vs. current recall distinction | none |
| Mnemon | https://github.com/mnemon-dev/mnemon | `_refs/mnemon` | Apache-2.0 | Simple install (single binary, zero API keys), LLM-supervised memory protocol, `remember`/`link`/`recall` primitives | implement-v0.2 | `init-agent` onboarding command concept; agent-client setup workflow expectations; memory protocol primitive naming | none |
| Nocturne Memory | https://github.com/nicholasgriffintn/nocturne-memory | `_refs/nocturne_memory` | MIT | Rollbackable MCP memory, local persistent agent context, SQLite/Postgres storage | implemented | Rollback/delete test patterns for MCP memory operations; local MCP ergonomics for persistent agent context; trigger keyword boost concept | none |
| HydraDB | N/A (proprietary) | N/A | N/A | Memories/knowledge/recall/context public primitive language | implemented | API shape and mental model (memories, knowledge, recall, context) as conceptual vocabulary only | none |
| sqlite-vec | https://github.com/asg017/sqlite-vec | N/A | MIT | Local vector search as a SQLite extension | v0.4 | Potential optional vector acceleration backend; not required in v0.3 | none |
| Kuzu | https://github.com/kuzudb/kuzu | N/A | MIT | Embedded graph database with Cypher-like query language | v0.4 | Potential optional graph backend; not required in v0.3 | none |
| ByteRover / Cipher | N/A | N/A | Elastic License 2.0 | Portable coding-agent memory CLI | reject | Rejected due to Elastic 2.0 license incompatibility with MIT; onboarding workflow expectations noted | none |
| MemOS | N/A | N/A | Apache-2.0 | Memory lifecycle management (create, update, archive, forget) | schema-hook | Lifecycle state ideas for memory items (active, archived, forgotten); not implementing proactive autonomy | none |
| memU | N/A | N/A | Apache-2.0 | Proactive/always-on memory with autonomous retrieval | reject | Rejected for v0.2 scope; proactive autonomy and always-on retrieval are out of scope for a deterministic local context graph | none |

## Decision Legend

- **implemented** -- concept present in Palimp v0.3.0 (or earlier, carried forward)
- **implement-v0.2** -- planned for v0.2.0 (UX ideas, onboarding, or API ergonomics)
- **schema-hook** -- reserved for future schema extension points; no implementation yet
- **v0.4** -- deferred to v0.4 (optional backends, advanced features)
- **reject** -- out of scope or license-incompatible

## Attribution Rule

Palimp reuses concepts, API ergonomics, and test ideas only -- not code -- unless a future line in this file explicitly says otherwise. See [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for the full attribution statement.
