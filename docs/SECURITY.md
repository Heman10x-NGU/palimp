# Security Model

## Namespace Isolation

Every read and write operation in Palimp is scoped to a namespace. There is no mechanism to query across namespaces. Inserting data into namespace `A` and querying namespace `B` will return zero results.

## MCP Safety

When Palimp exposes memories and knowledge through MCP tools, the output is always structured data. It is never presented as an instruction to the consuming model. The `safety.treat_as_instruction` field is always `false`.

## Local-First

Palimp stores all data in a local SQLite database. No data is sent to external services unless an optional HTTP embedding provider is explicitly configured. The default configuration is fully offline.

## Tombstoning

Deleted sources are tombstoned, not physically removed. Tombstoned episodes and their dependent graph facts are excluded from all recall queries. Audit logs are preserved.

## Threat Model

Palimp is designed for single-user or small-team local use. It does not provide:

- Multi-tenant authentication
- Network-level encryption
- Role-based access control
- Rate limiting beyond what the OS provides

For production multi-user deployments, place Palimp behind a reverse proxy with authentication.
