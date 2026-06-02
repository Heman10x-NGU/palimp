# Quickstart: GraphCtx for Coding Agent Users

Get GraphCtx running with your coding agent in under 5 minutes.

## 1. Install

```bash
pip install graphctx
```

Verify:

```bash
graphctx --version
```

## 2. Initialize for Your Agent

Run the one-step setup command for your agent client:

```bash
graphctx init-agent --client claude
```

Supported `--client` values: `claude`, `cursor`, `codex`, `generic`.

This command:

- Creates the database at `~/.graphctx/graphctx.db` (or verifies it exists)
- Runs diagnostics (`doctor`)
- Inserts a starter memory and starter knowledge item
- Prints the exact MCP configuration JSON to paste into your client
- Prints a health check command and first recall command

## 3. Import Your Project Docs

Ingest your project's documentation as knowledge items:

```bash
# Single file
graphctx import --namespace demo --path README.md

# Recursive directory
graphctx import --namespace demo --path docs --ext .md --recursive

# Multiple extensions
graphctx import --namespace demo --path . --ext .md .txt .rst --recursive
```

Each file becomes a knowledge item with the file path as `source_ref`. Files over 1 MB are skipped by default (override with `--max-bytes`).

## 4. Add Memories

```bash
# Add a memory via CLI
graphctx memory add --namespace demo "Alice prefers concise technical answers."

# Add knowledge
graphctx knowledge add --namespace demo --title "Architecture" --content "GraphCtx uses SQLite with WAL mode."
```

## 5. Recall

```bash
# Basic recall
graphctx recall --namespace demo "what does Alice prefer?"

# With explanation
graphctx recall --namespace demo "what does Alice prefer?" --explain

# Specify mode and limit
graphctx recall --namespace demo "architecture decisions" --mode thinking --limit 5
```

## 6. Configure MCP

Paste the MCP config printed by `init-agent` into your agent client.

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "graphctx": {
      "command": "graphctx",
      "args": ["mcp"]
    }
  }
}
```

**Cursor** (Settings > MCP > Add new global MCP server):

```json
{
  "mcpServers": {
    "graphctx": {
      "command": "graphctx",
      "args": ["mcp"]
    }
  }
}
```

**Codex / Generic:**

```json
{
  "mcpServers": {
    "graphctx": {
      "command": "graphctx",
      "args": ["mcp"]
    }
  }
}
```

See [MCP_SETUP.md](MCP_SETUP.md) for detailed configuration and troubleshooting.

## 7. Verify Everything Works

```bash
# Check health
graphctx doctor --db ~/.graphctx/graphctx.db

# View stats
graphctx stats --namespace demo

# Run mini evaluation
graphctx eval mini --namespace demo --json
```

## What You Get

After setup, your coding agent can:

- **Remember** facts across sessions (memories)
- **Know** project documentation (knowledge)
- **Recall** relevant context with provenance (where each fact came from)
- **Track** contradictions and supersessions (old facts are superseded, not deleted)
- **Inspect** everything in SQLite (`~/.graphctx/graphctx.db`)

All context is returned as data, not instruction. The `safety.treat_as_instruction` field is always `false`.

## Next Steps

- [MCP Setup Guide](MCP_SETUP.md) -- detailed MCP configuration and troubleshooting
- [Benchmarks](BENCHMARKS.md) -- run local performance benchmarks
- [Comparison](COMPARISON.md) -- how GraphCtx compares to other tools
- [Attributions](ATTRIBUTIONS.md) -- what concepts were borrowed from other projects
