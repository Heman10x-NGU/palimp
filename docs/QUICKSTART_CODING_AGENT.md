# Quickstart: Palimp for Coding Agent Users

Get Palimp running with your coding agent in under 5 minutes.

## 1. Install

```bash
pip install palimp
```

Verify:

```bash
palimp --version
```

## 2. Initialize for Your Agent

Run the one-step setup command for your agent client:

```bash
palimp init-agent --client claude
```

Supported `--client` values: `claude`, `cursor`, `codex`, `generic`.

This command:

- Creates the database at `~/.palimp/palimp.db` (or verifies it exists)
- Runs diagnostics (`doctor`)
- Inserts a starter memory and starter knowledge item
- Prints the exact MCP configuration JSON to paste into your client
- Prints a health check command and first recall command

## 3. Import Your Project Docs

Ingest your project's documentation as knowledge items:

```bash
# Single file
palimp import --namespace demo --path README.md

# Recursive directory
palimp import --namespace demo --path docs --ext .md --recursive

# Multiple extensions
palimp import --namespace demo --path . --ext .md .txt .rst --recursive
```

Each file becomes a knowledge item with the file path as `source_ref`. Files over 1 MB are skipped by default (override with `--max-bytes`).

## 4. Add Memories

```bash
# Add a memory via CLI
palimp memory add --namespace demo "Alice prefers concise technical answers."

# Add knowledge
palimp knowledge add --namespace demo --title "Architecture" --content "Palimp uses SQLite with WAL mode."
```

## 5. Recall

```bash
# Basic recall
palimp recall --namespace demo "what does Alice prefer?"

# With explanation
palimp recall --namespace demo "what does Alice prefer?" --explain

# Specify mode and limit
palimp recall --namespace demo "architecture decisions" --mode thinking --limit 5
```

## 6. Configure MCP

Paste the MCP config printed by `init-agent` into your agent client.

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "palimp": {
      "command": "palimp",
      "args": ["mcp"]
    }
  }
}
```

**Cursor** (Settings > MCP > Add new global MCP server):

```json
{
  "mcpServers": {
    "palimp": {
      "command": "palimp",
      "args": ["mcp"]
    }
  }
}
```

**Codex / Generic:**

```json
{
  "mcpServers": {
    "palimp": {
      "command": "palimp",
      "args": ["mcp"]
    }
  }
}
```

See [MCP_SETUP.md](MCP_SETUP.md) for detailed configuration and troubleshooting.

## 7. Verify Everything Works

```bash
# Check health
palimp doctor --db ~/.palimp/palimp.db

# View stats
palimp stats --namespace demo

# Run mini evaluation
palimp eval mini --namespace demo --json
```

## What You Get

After setup, your coding agent can:

- **Remember** facts across sessions (memories)
- **Know** project documentation (knowledge)
- **Recall** relevant context with provenance (where each fact came from)
- **Track** contradictions and supersessions (old facts are superseded, not deleted)
- **Inspect** everything in SQLite (`~/.palimp/palimp.db`)

All context is returned as data, not instruction. The `safety.treat_as_instruction` field is always `false`.

## Next Steps

- [MCP Setup Guide](MCP_SETUP.md) -- detailed MCP configuration and troubleshooting
- [Benchmarks](BENCHMARKS.md) -- run local performance benchmarks
- [Comparison](COMPARISON.md) -- how Palimp compares to other tools
- [Attributions](ATTRIBUTIONS.md) -- what concepts were borrowed from other projects
