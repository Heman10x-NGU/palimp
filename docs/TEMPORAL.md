# Temporal Validity

GraphCtx tracks when facts are valid. Every memory and knowledge item carries `valid_from` and `valid_until` timestamps that control whether it appears in current, historical, or point-in-time queries.

## Temporal States

Each recalled result carries a `temporal_status`:

| Status | Meaning |
|---|---|
| `current` | `valid_until IS NULL` or `valid_until >= now` |
| `historical` | `valid_until < now` (fact was true in the past, superseded) |
| `future` | `valid_from > now` (fact will become true) |
| `stale` | Explicitly marked as no longer valid |
| `unknown` | No temporal metadata available |

## Temporal Modes

Control how recall filters by time:

| Mode | Behavior |
|---|---|
| `auto` | Default. Prefers current facts, includes historical if query has temporal cues |
| `current` | Only returns facts where `valid_until IS NULL OR valid_until >= now` |
| `historical` | Preserves old facts, useful when query contains "before", "previously", "formerly", "in 2022", "last year" |
| `all` | Returns all facts regardless of temporal status |

## CLI Usage

### Recall with temporal mode

```bash
# Current facts only (default)
graphctx recall --namespace demo "where does Alice live?"

# Historical facts
graphctx recall --namespace demo "where did Alice live in 2022?" \
  --temporal-mode historical

# All facts regardless of time
graphctx recall --namespace demo "Alice address" \
  --temporal-mode all
```

### Point-in-time query with `as_of`

```bash
# What was true on January 15, 2026?
graphctx recall --namespace demo "project config" \
  --as-of 2026-01-15T00:00:00Z
```

The `as_of` parameter changes which facts are considered current at that point in time. Facts with `valid_from <= as_of` and (`valid_until IS NULL` or `valid_until >= as_of`) are treated as current.

### Combine both

```bash
graphctx recall --namespace demo "architecture decisions" \
  --as-of 2025-12-01T00:00:00Z \
  --temporal-mode historical
```

## REST API

```bash
curl -X POST http://localhost:8420/v1/recall \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "demo",
    "query": "where does Alice live?",
    "temporal_mode": "current",
    "as_of": "2026-01-15T00:00:00Z",
    "explain": true
  }'
```

## Explanation Fields

When `explain=true`, each result includes temporal metadata:

```json
{
  "temporal_status": "current",
  "valid_from": "2026-01-01T00:00:00Z",
  "valid_until": null,
  "temporal_reason": "fact is current (no expiry set)"
}
```

## How Temporal Filtering Works

1. **Ingestion:** When a new fact supersedes an old one, the old fact's `valid_until` is set to the current timestamp. The new fact's `valid_from` is set to now.

2. **Current recall:** Filters to facts where `valid_until IS NULL OR valid_until >= now`. Stale facts are excluded or downranked.

3. **Historical recall:** Preserves superseded facts. Triggered when the query contains temporal cues (before, previous, formerly, in YYYY, last year, etc.).

4. **Point-in-time (`as_of`):** Evaluates `valid_from <= as_of AND (valid_until IS NULL OR valid_until >= as_of)`.

## Supersession

When a new fact contradicts an existing one:

```
Old fact:  "Alice lives in Boston"     valid_until = 2026-03-01
New fact:  "Alice lives in Portland"   valid_from  = 2026-03-01
```

- Current recall returns "Alice lives in Portland"
- Historical recall (or `as-of 2026-01-01`) returns "Alice lives in Boston"
- The contradiction is tracked with a `SUPERSEDES` edge

## Examples

### Project config evolution

```bash
# Record the original config
graphctx memory add --namespace myapp \
  --content "Build system uses Make" --valid-from 2025-01-01T00:00:00Z

# Record the change
graphctx memory add --namespace myapp \
  --content "Build system migrated to uv" --valid-from 2026-04-01T00:00:00Z

# Current: returns "Build system migrated to uv"
graphctx recall --namespace myapp "build system"

# Historical: returns "Build system uses Make"
graphctx recall --namespace myapp "build system" --temporal-mode historical

# Point-in-time: returns "Build system uses Make"
graphctx recall --namespace myapp "build system" --as-of 2025-06-01T00:00:00Z
```

### Command fix history

```bash
# Old fix
graphctx memory add --namespace myapp \
  --content "Fix import error: pip install -e ."

# New fix
graphctx memory add --namespace myapp \
  --content "Fix import error: uv sync --all-extras"

# Historical recall finds both
graphctx recall --namespace myapp "import error fix" --temporal-mode all
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GRAPHCTX_WEIGHT_TEMPORAL` | `0.05` | Weight of temporal match in scoring |

Temporal scoring boosts facts whose temporal status matches the query mode (current for current queries, historical for historical queries).
