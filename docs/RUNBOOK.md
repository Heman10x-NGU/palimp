# Runbook Mode

Runbook mode lets coding agents build a project-specific knowledge base of gotchas, workflows, command fixes, and architectural decisions. Before each task, the agent packs a compact evidence bundle under a token budget.

## Runbook Kinds

| Kind | Use for |
|---|---|
| `gotcha` | Non-obvious behaviors that trip up agents (e.g., "pytest needs :memory: DB") |
| `workflow` | Step sequences that work (e.g., "sync deps before running tests") |
| `command_fix` | Commands that failed and the fix that worked |
| `project_invariant` | Constraints that must hold (e.g., "never modify migrations/ directly") |
| `dependency_note` | Version pins, conflicts, or quirks |
| `debug_trace` | Investigation paths that led to a resolution |
| `architecture_decision` | Why a design choice was made |

## CLI Commands

### Add a runbook entry

```bash
palimp runbook add --namespace repo --kind gotcha \
  --content "pytest requires PALIMP_DB=:memory: for isolated tests"
```

### List runbook entries

```bash
palimp runbook list --namespace repo
```

### Pack a context bundle

```bash
palimp runbook pack --namespace repo \
  --task "fix failing storage tests" --budget 2000
```

Output is a compact JSON bundle containing:
- Matched runbook entries
- Relevant recalled memories and knowledge
- Source references with episode IDs
- Confidence scores
- Category labels
- Why each item was included
- Safety metadata (`treat_as_instruction: false`)

### Delete a runbook entry

```bash
palimp runbook delete --namespace repo --entry-id <id>
```

## MCP Tool

The `palimp_context_pack` MCP tool provides the same functionality as `palimp runbook pack`:

```json
{
  "tool": "palimp_context_pack",
  "arguments": {
    "namespace": "repo",
    "task": "fix failing storage tests",
    "budget_tokens": 2000
  }
}
```

Returns a structured context pack suitable for injection into an agent's system prompt or context window.

## Preprompt Hook

For agent integration without running a REST server:

```bash
palimp hook preprompt --namespace repo \
  --task "refactor retriever module" --budget 2000 --json
```

Returns:
- Compact context pack
- Runbook items relevant to the task
- Recent relevant memories
- Project constraints
- Safety metadata (`treat_as_instruction: false`)

The JSON shape is stable and documented. The `safety.treat_as_instruction` field is always `false` -- the pack is data, not instructions.

## Coding-Agent Workflow Example

```bash
# Day 1: Build the runbook as you work
palimp runbook add --namespace myapp --kind gotcha \
  --content "SQLite WAL mode required for concurrent reads during tests"
palimp runbook add --namespace myapp --kind command_fix \
  --content "uv build failed with --locked; fix: run uv lock first"
palimp runbook add --namespace myapp --kind workflow \
  --content "Run palimp eval mini before every release"

# Day N: Pack context before a task
palimp runbook pack --namespace myapp \
  --task "prepare v0.3 release" --budget 2000

# Or use the preprompt hook for agent integration
palimp hook preprompt --namespace myapp \
  --task "prepare v0.3 release" --budget 2000 --json
```

## Token Budget

The `--budget` flag (default 2000) controls the maximum token count for the packed context. The packer:

1. Scores all candidate items by relevance to the task
2. Fills the budget highest-relevance-first
3. Drops items that would exceed the budget
4. Reports dropped items in the explanation

Category-priority memories (gotcha, project_invariant, constraint) are ranked higher under tight budgets.

## Safety

- The context pack is always data, never instructions.
- `safety.treat_as_instruction` is always `false`.
- Prompt-injection content in memories is flagged and never emitted as instruction.
- Source provenance is included for every item in the pack.
