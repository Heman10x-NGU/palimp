# MCP Setup Guide

GraphCtx exposes MCP (Model Context Protocol) tools for use with any MCP-compatible client. This guide covers configuration for Claude Desktop, Cursor, Codex, and generic MCP clients.

## Claude Desktop

Add the following to your Claude Desktop configuration file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

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

If `graphctx` is not on your PATH, use the full path to the installed binary:

```json
{
  "mcpServers": {
    "graphctx": {
      "command": "/Users/yourname/.local/bin/graphctx",
      "args": ["mcp"]
    }
  }
}
```

Restart Claude Desktop after saving the config.

## Cursor

Add the following to your Cursor MCP settings (Settings > MCP > Add new global MCP server):

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

Alternatively, add it to `.cursor/mcp.json` in your project root for project-specific configuration.

## Codex

Add the following to your Codex MCP configuration:

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

## Generic MCP Client

Any MCP-compatible client can connect to GraphCtx. The MCP server is started via:

```bash
graphctx mcp
```

This launches a stdio-based MCP server. Configure your client to run this command and communicate over stdin/stdout.

For a streamable HTTP transport (if supported by your client):

```bash
graphctx mcp --transport streamable-http --port 8421
```

## Available MCP Tools

Once configured, these tools are available to your agent:

| Tool | Description |
|---|---|
| `graphctx_memory_add` | Add a memory to a namespace |
| `graphctx_knowledge_add` | Add a knowledge item to a namespace |
| `graphctx_recall` | Unified retrieval across memories and knowledge |
| `graphctx_context_get` | Get entity context from the graph |
| `graphctx_stats` | Get namespace statistics |

All MCP output is structured data. Memories and knowledge are returned as facts, never as instructions. The `safety.treat_as_instruction` field is always `false`.

## Verifying the Setup

After configuration, test that GraphCtx is working:

1. Open your MCP client (e.g., Claude Desktop).
2. Ask the agent: "What tools do you have from GraphCtx?" -- it should list the tools above.
3. Ask the agent: "Add a memory to namespace demo: I prefer concise answers." -- it should call `graphctx_memory_add`.
4. Ask the agent: "What do I prefer?" -- it should call `graphctx_recall` and return the memory.

## Troubleshooting

### "command not found: graphctx"

GraphCtx is not on your PATH. Either:

1. Install globally: `pip install graphctx`
2. Use the full path in your MCP config: `"/path/to/venv/bin/graphctx"`
3. Activate your virtual environment before starting your MCP client

### "namespace is required" errors

All GraphCtx MCP tools require a `namespace` parameter. If your agent is not passing it, instruct it to always include a namespace (e.g., `"demo"` or your project name).

### MCP server does not appear in client

1. Verify the config file path is correct for your client.
2. Verify the JSON is valid (no trailing commas, correct quoting).
3. Restart your MCP client after changing the config.
4. Check that `graphctx mcp` runs without errors from your terminal.

### "database is locked" errors

SQLite uses file-level locking. If you have multiple GraphCtx instances (REST server + MCP server) pointing at the same database, they may contend. Use WAL mode (enabled by default) or use separate database files for different clients.

### Slow first query

The first recall query after starting the MCP server may be slow due to SQLite page cache warming. Subsequent queries are faster.
