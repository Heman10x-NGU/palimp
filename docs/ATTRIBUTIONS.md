# Attributions

Palimp draws inspiration from the following projects and ideas. **Palimp reuses concepts, API ergonomics, and test ideas only, not code, unless a future line in this file explicitly says otherwise.**

## Source-by-Source Attribution

### AutoMem
- **Repository:** https://github.com/verygoodplugins/automem
- **License:** MIT
- **What was borrowed:** The 11 typed relationship taxonomy informed Palimp's relation enum design. AutoMem's 9-component scoring approach and LongMemEval/LoCoMo benchmark discipline shaped how Palimp thinks about retrieval scoring and evaluation rigor. Bridge discovery concepts are noted for future v0.3 work.
- **Code reuse:** none

### MemoryOS
- **Repository:** https://github.com/Per0x1de-1337/memoryos
- **License:** MIT
- **What was borrowed:** The temporal knowledge graph model with append-only supersession (old facts superseded, never deleted) directly influenced Palimp's episode/provenance architecture. The Ebbinghaus forgetting curve for memory decay, fast vs. thinking retrieval mode naming, and query explanation traces are all concepts borrowed from MemoryOS.
- **Code reuse:** none

### mem0 / OpenMemory
- **Repository:** https://github.com/mem0ai/mem0
- **License:** Apache-2.0
- **What was borrowed:** UX/API ergonomics for memory categorization (user, session, agent scopes). MCP memory tool surface design patterns. Benchmark methodology seriousness (publishing methodology alongside numbers).
- **Code reuse:** none

### Graphiti / Zep
- **Repository:** https://github.com/getzep/graphiti
- **License:** Apache-2.0
- **What was borrowed:** The episode-first temporal context graph architecture is the strongest influence on Palimp. Temporal validity windows (valid_from, valid_until), supersession tracking, provenance on every extracted fact, and the distinction between historical and current recall all trace back to Graphiti's design.
- **Code reuse:** none

### Mnemon
- **Repository:** https://github.com/mnemon-dev/mnemon
- **License:** Apache-2.0
- **What was borrowed:** The `init-agent` onboarding command concept and agent-client setup workflow expectations. Mnemon's emphasis on simple install (single binary, zero API keys) informed Palimp's "one command" install philosophy. The memory protocol primitive naming (remember, recall) influenced Palimp's vocabulary.
- **Code reuse:** none

### Nocturne Memory
- **Repository:** https://github.com/nicholasgriffintn/nocturne-memory
- **License:** MIT
- **What was borrowed:** Rollback/delete test patterns for MCP memory operations. Local MCP ergonomics for persistent agent context. The idea that MCP memory should be rollbackable and inspectable.
- **Code reuse:** none

### HydraDB
- **Source:** Proprietary managed platform (no public repository)
- **License:** N/A
- **What was borrowed:** The public primitive vocabulary (memories, knowledge, recall, context) and the mental model of a context graph for agents. API shape inspiration only. Palimp is not a HydraDB replacement.
- **Code reuse:** none

### SQLite / FTS5
- **Source:** https://sqlite.org
- **License:** Public domain
- **What was borrowed:** SQLite as the storage engine and FTS5 as the full-text search layer. These are foundational infrastructure, not application-level borrowing.
- **Code reuse:** none (used as a dependency)

### FastAPI / Pydantic / MCP
- **Source:** FastAPI (https://fastapi.tiangolo.com), Pydantic (https://docs.pydantic.dev), MCP (https://modelcontextprotocol.io)
- **License:** MIT (FastAPI, Pydantic), MIT (MCP spec)
- **What was borrowed:** Standard framework and protocol usage. No application-level concepts borrowed.
- **Code reuse:** none (used as dependencies)

## v0.3 Design and Benchmark Sources

The following external sources informed Palimp v0.3 design decisions, benchmark methodology, and feature prioritization. Concepts and framing were borrowed; no code was reused.

### Mem0 Memory Evaluation
- **Source:** https://docs.mem0.ai/core-concepts/memory-evaluation
- **What was borrowed:** Memory evaluation methodology and benchmark framing for agent memory systems.

### Mem0 Token-Efficient Memory Research
- **Source:** https://mem0.ai/research
- **What was borrowed:** Token-efficient memory design patterns and benchmark discipline.

### Mem0 Temporal Reasoning
- **Source:** https://mem0.ai/blog/introducing-temporal-reasoning-in-mem0
- **What was borrowed:** Temporal reasoning benchmark discussion and the framing of temporal truth in agent memory.

### Graphiti / Zep Temporal Context Graph
- **Source:** https://www.getzep.com/platform/graphiti/
- **What was borrowed:** Temporal context graph architecture, validity windows, and the episode-first design that directly influenced Palimp's temporal filtering and supersession tracking.

### LongMemEval-V2
- **Source:** https://arxiv.org/abs/2605.12493 and https://xiaowu0162.github.io/longmemeval-v2/
- **What was borrowed:** The "experienced colleague" benchmark framing for long-term agent memory evaluation. This framing shaped Palimp's runbook mode and coding-agent context pack design.

### MemX Local-First Memory
- **Source:** https://arxiv.org/abs/2603.16171
- **What was borrowed:** Local-first memory design patterns and the case for SQLite-based agent memory without external dependencies.

### MemReranker
- **Source:** https://arxiv.org/abs/2605.06132
- **What was borrowed:** Reasoning-aware reranking context that informed Palimp's optional reranker hook design and thinking-mode query expansion.

## Licensing

Palimp is released under the MIT License.

## What This File Does NOT Claim

- Palimp does not claim to be a replacement for any listed project.
- Palimp does not claim feature parity with any listed project.
- Palimp does not use source code from any listed project unless explicitly noted above.
- "Concepts borrowed" means design ideas, vocabulary, API ergonomics, and test patterns -- not copied implementations.
