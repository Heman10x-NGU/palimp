# Palimp Launch — Twitter Thread

> 7-tweet thread for Palimp launch.
> Positioning: local OSS alternative to HydraDB's core context-graph primitive.
> Not a HydraDB replacement. Not a Mem0 clone. A local-first, inspectable context graph.

---

## Tweet 1 — Hook

I built Palimp — an open-source local context graph for AI agents.

Memories. Knowledge. Recall. Provenance. MCP access.
All in SQLite. No cloud. No vendor lock-in.

Here's why and how 🧵

---

## Tweet 2 — Problem

Most agent memory tools are either:

- Hosted black boxes (HydraDB, Mem0 managed)
- Simple vector stores with no graph, no provenance
- Complex stacks (Neo4j + Qdrant + custom code)

None of them let you inspect what the agent actually "remembers."

---

## Tweet 3 — Solution

Palimp gives you HydraDB-style primitives locally:

- memories (dynamic context)
- knowledge (static docs/specs)
- recall (unified retrieval)
- context graph (entities, claims, relations)
- provenance (every fact linked to its source)

All in a single `pip install palimp`.

---

## Tweet 4 — Safety

The key difference: safety.

Every memory is namespace-scoped.
Every fact has provenance.
Contradictions are tracked, not silently overwritten.
MCP output returns data, not instructions.

Your agent remembers safely, not just "more."

---

## Tweet 5 — Tech

Stack:

- SQLite (WAL mode, FTS5, numpy vectors)
- FastAPI (7 endpoints)
- Typer CLI (7 commands)
- MCP server (5 tools)
- Pydantic models
- Zero paid LLM dependencies for tests

---

## Tweet 6 — How to Try

```
pip install palimp
palimp serve --port 8420

curl -X POST localhost:8420/v1/memories \
  -d '{"namespace":"demo","content":"Alice prefers concise answers"}'
```

Every result includes provenance and safety metadata.

---

## Tweet 7 — CTA

Palimp is not a HydraDB replacement. It's the local OSS alternative to the core context-graph idea.

If you build agents and want inspectable, safe, local memory — try it.

- Star on GitHub
- Open issues
- PRs especially welcome

---

## Tweet 8 — AdaCoM Context Management

Palimp now includes AdaCoM-inspired context management (arXiv:2605.30785).

- Deduplicates memories with >70% content overlap
- Relevance-scores and compresses to fit agent token budgets
- Every compression step is explainable

Your agent's context window stays clean without losing critical memories.

---

## Tweet 9 — MemoryOS-Inspired Features

Adopted ideas from MemoryOS:

- Ebbinghaus decay: memories fade naturally unless pinned or accessed
- Session lifecycle: create, close, and summarize conversations
- Batch ingestion: add up to 50 memories/knowledge items per request
- Explainable retrieval: every result includes score breakdowns and why-retrieved reasons

Local-first memory that behaves like real memory — it forgets, unless you tell it not to.

---

## Positioning Notes

From the plan (section 1):

> Palimp is an open-source, local-first alternative to HydraDB's core memory/context-graph primitive for developers who want inspectable agent context without a hosted black box.

**Do not claim:**
- "Full HydraDB replacement."
- "HydraDB clone."
- "Production-grade managed memory platform."
- "Better than Mem0/Zep/HydraDB."

**Do claim:**
- Local OSS alternative to the core context-graph idea.
- Same conceptual primitives: memories, knowledge, recall, context graph.
- Different deployment philosophy: local-first, SQLite-first, inspectable.
- Stronger safety wedge: provenance + namespace isolation + MCP-safe output.
