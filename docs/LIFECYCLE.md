# Lifecycle, Purpose, and Forgetting

Palimp keeps exact episode text as the stable ground truth, then layers lifecycle metadata over it.

## Version Chains

Episodes support append-only version chains:

- `version`
- `parent_episode_id`
- `is_latest`

When a new memory is inserted with `metadata.parentMemoryId`, `metadata.parent_memory_id`, `metadata.parentEpisodeId`, or `metadata.parent_episode_id`, Palimp resolves the parent memory or episode, increments the version, and marks the parent episode as not latest.

Example metadata:

```json
{
  "parentMemoryId": "mem_abc123",
  "version": 2
}
```

This preserves historical evidence instead of overwriting old facts.

## Purpose Layer

Episodes have a `purpose`:

- `search`: query-specific recall, the default
- `profile`: always-relevant user/project context

Profile memories are eligible in hybrid and thinking recall even when the query does not lexically match them. They still pass namespace, tombstone, and governed-forgetting filters.

Example metadata:

```json
{
  "purpose": "profile"
}
```

## Governed Forgetting

Palimp supports soft forgetting without destroying auditability:

- `forget_after`: ISO timestamp after which the episode is excluded from recall
- `is_forgotten`: explicit recall exclusion flag

The row, source text, and provenance remain inspectable. Recall excludes forgotten or expired episodes.

Example metadata:

```json
{
  "forgetAfter": "2026-07-01T00:00:00Z"
}
```

Use `SQLiteStore.forget_episode(episode_id)` for explicit soft forgetting in code.

## Scoring Defaults

Palimp defaults to FTS5-primary retrieval:

- lexical: `0.60`
- vector: `0.15`
- graph: `0.10`
- temporal: `0.10`
- confidence: `0.05`
- recency: `0.00`
- category: `0.00`

This keeps exact episodic grounding primary while retaining semantic embeddings as a weak secondary boost.
