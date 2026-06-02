# Palimp v0.3 — Social Media Posts

## Twitter Thread (7 tweets)

### Tweet 1 — Hook

```
I built Palimp v0.3 — a local-first context graph for coding agents.

It makes your agent remember like an experienced teammate:
• project gotchas
• commands that failed
• fixes that worked
• temporal decisions
• multi-hop relationships

All in SQLite. No cloud. No vendor lock-in.

🧵 Here's what's new ↓
```

### Tweet 2 — The Problem

```
Most agent memory tools are either:
• Hosted black boxes (HydraDB, Mem0 managed)
• Simple vector stores with no graph
• Complex stacks (Neo4j + Qdrant + custom code)

None of them understand that "Python" and "python" and "the Python" are the same thing.

None of them know that a fact from 2022 is different from a fact from 2026.
```

### Tweet 3 — Multi-Hop + Temporal

```
Palimp v0.3 adds multi-hop graph traversal.

Your agent can now follow relationships:
Alice → works_on → Palimp → uses → SQLite

Ask "what does Alice's project use?" and it finds SQLite through 2 hops.

Plus temporal validity: "where does Alice live NOW?" returns current fact.
"Where did Alice live in 2022?" returns historical fact.

Same data. Different truth windows.
```

### Tweet 4 — Alias Dedup + Categories

```
Entity alias deduplication:
"Python", "python", "the Python" → one entity, not three fragmented memories.

12 memory categories with priority ranking:
identity > constraint > gotcha > architecture_decision > preference > ...

Critical memories survive context compression. Operational noise gets filtered.
```

### Tweet 5 — Runbook Mode

```
The killer feature: coding-agent runbook mode.

palimp runbook add --kind gotcha --content "pytest requires :memory: for isolated tests"
palimp runbook pack --task "fix failing storage tests" --budget 2000

Your agent gets a compact evidence pack with:
• source-linked facts
• confidence scores
• safety metadata
• under 2k tokens

Like asking an experienced teammate "what should I know before touching this?"
```

### Tweet 6 — Tech Stack

```
Stack:
• SQLite (WAL mode, FTS5, 14 tables)
• FastAPI (12 endpoints)
• Typer CLI (15+ commands)
• MCP server (6 tools)
• 378 tests passing
• Zero paid LLM dependencies for tests
• Deterministic embeddings (SHA-256 based)

pip install palimp
palimp init-agent --client claude
```

### Tweet 7 — CTA

```
Palimp is not the biggest memory system.

It is the smallest useful local context graph for coding agents that need:
• source-linked project memory
• temporal truth
• aliases
• multi-hop graph recall
• MCP-safe context packs

⭐ Star: github.com/Heman10x-NGU/palimp
🐛 Issues welcome
🤝 PRs especially welcome

If you build agents and want them to remember like a teammate — try it.
```

---

## LinkedIn Post

```
🚀 Palimp v0.3 — Local Context Graph for Coding Agents

I've been building Palimp, an open-source local-first context graph that makes coding agents remember like an experienced teammate.

The problem: Every time you start a new coding session, your agent forgets everything. Your preferences, architecture decisions, project conventions — gone.

The solution: Palimp gives agents persistent memory with:

✅ Multi-hop graph traversal (follow relationships across entities)
✅ Temporal truth (current vs historical facts with validity windows)
✅ Entity alias deduplication ("Python" = "python" = "the Python")
✅ 12 memory categories with priority ranking
✅ Runbook mode (compact evidence packs for coding tasks)
✅ Full provenance (every fact traceable to its source)
✅ MCP-safe output (context as data, not instruction)
✅ SQLite-inspectable (open the DB, see everything)

What makes it different:
• Zero infrastructure — no Docker, no Postgres, no Redis
• 378 tests passing, deterministic embeddings
• Works with Claude, Cursor, Codex, any MCP-compatible agent
• MIT licensed, local-first, no vendor lock-in

The key insight: Most memory tools help agents "remember more." Palimp helps agents "remember safely" — with provenance, namespace isolation, and temporal validity.

pip install palimp
palimp init-agent --client claude

Try it: github.com/Heman10x-NGU/palimp

#AI #OpenSource #CodingAgents #MCP #SQLite #GraphDatabase #DeveloperTools
```

---

## Reddit Post (r/MachineLearning or r/LocalLLaMA)

```
Title: Palimp v0.3 — Local SQLite context graph for coding agents (378 tests, MIT licensed)

I built Palimp, an open-source local-first context graph for coding agents.

Key features:
- Multi-hop graph traversal (2-3 hops with depth decay)
- Temporal validity filtering (current vs historical facts)
- Entity alias deduplication
- 12 memory categories with priority ranking
- Runbook mode for coding-agent evidence packs
- MCP server with 6 tools
- SQLite storage, zero external dependencies
- 378 tests passing

It's designed for developers who want their coding agent to remember project gotchas, commands that failed, fixes that worked, and architectural decisions — all with full provenance and temporal truth.

pip install palimp
palimp init-agent --client claude

GitHub: https://github.com/Heman10x-NGU/palimp

Not trying to be the biggest memory system. Just the smallest useful one for coding agents that care about source-linked memory and MCP safety.
```
