"""GraphCtx CLI — Typer app for memory, knowledge, recall, and diagnostics."""

from __future__ import annotations

import json
import os
from typing import Optional

import typer

from graphctx.validate import ValidationError, validate_content, validate_namespace

app = typer.Typer(help="GraphCtx — local context graph for AI agents.")

# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
    port: int = typer.Option(8420, help="Port for the HTTP server."),
) -> None:
    """Start the GraphCtx REST + MCP server."""
    import uvicorn

    resolved = os.path.expanduser(db)
    os.environ["GRAPHCTX_DB"] = resolved
    typer.echo(f"Starting GraphCtx server on port {port} (db: {resolved})")
    uvicorn.run("graphctx.server:app", host="0.0.0.0", port=port, log_level="info")


# ---------------------------------------------------------------------------
# memory add
# ---------------------------------------------------------------------------

memory_app = typer.Typer(help="Memory commands.")
app.add_typer(memory_app, name="memory")


@memory_app.command("batch")
def memory_batch(
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON lines file (one memory JSON object per line)."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Batch insert memories from a JSON lines file.

    Each line should be a JSON object with at least a "content" key.
    Optional keys: "source_ref", "metadata".
    """
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)

    items: list[dict] = []
    with open(file) as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    result = store.insert_memories_batch(ns=ns, items=items)
    typer.echo(f"total:      {result.total}")
    typer.echo(f"successful: {result.successful}")
    typer.echo(f"failed:     {result.failed}")
    typer.echo(f"elapsed_ms: {result.elapsed_ms}")
    if result.errors:
        for err in result.errors:
            typer.echo(f"  error at index {err['index']}: {err['error']}")


@memory_app.command("add")
def memory_add(
    content: str = typer.Argument(help="Memory content text."),
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    category: str = typer.Option(
        "other", "--category", "-c", help="Memory category (identity, preference, project_config, constraint, architecture_decision, workflow, gotcha, bug_fix, command_result, tool_usage, knowledge, other)."
    ),
    source_ref: Optional[str] = typer.Option(
        None, "--source-ref", "-s", help="Source reference."
    ),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Add a memory to the context graph."""
    from graphctx.models import MEMORY_CATEGORIES

    if category not in MEMORY_CATEGORIES:
        typer.echo(f"Error: invalid category '{category}'. Choose from: {', '.join(MEMORY_CATEGORIES)}", err=True)
        raise typer.Exit(code=1)

    try:
        ns = validate_namespace(namespace)
        validate_content(content)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.embeddings import DeterministicEmbedder
    from graphctx.extractor import RuleBasedExtractor
    from graphctx.ingest import ingest_memory
    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    embedder = DeterministicEmbedder()
    extractor = RuleBasedExtractor()
    result = ingest_memory(
        store=store,
        embedder=embedder,
        extractor=extractor,
        ns=ns,
        content=content,
        source_ref=source_ref,
        category=category,
    )
    typer.echo(f"memory_id: {result['memory_id']}")
    typer.echo(f"episode_id: {result['episode_id']}")
    typer.echo(f"entities: {len(result['entities'])}")
    typer.echo(f"claims: {len(result['claims'])}")
    typer.echo(f"warnings: {len(result['warnings'])}")


# ---------------------------------------------------------------------------
# knowledge add
# ---------------------------------------------------------------------------

knowledge_app = typer.Typer(help="Knowledge commands.")
app.add_typer(knowledge_app, name="knowledge")


@knowledge_app.command("batch")
def knowledge_batch(
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON lines file (one knowledge JSON object per line)."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Batch insert knowledge items from a JSON lines file.

    Each line should be a JSON object with "title" and "content" keys.
    Optional keys: "source_ref", "metadata".
    """
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)

    items: list[dict] = []
    with open(file) as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    result = store.insert_knowledge_batch(ns=ns, items=items)
    typer.echo(f"total:      {result.total}")
    typer.echo(f"successful: {result.successful}")
    typer.echo(f"failed:     {result.failed}")
    typer.echo(f"elapsed_ms: {result.elapsed_ms}")
    if result.errors:
        for err in result.errors:
            typer.echo(f"  error at index {err['index']}: {err['error']}")


@knowledge_app.command("add")
def knowledge_add(
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    title: str = typer.Option(..., "--title", "-t", help="Knowledge title."),
    category: str = typer.Option(
        "other", "--category", "-k", help="Knowledge category (identity, preference, project_config, constraint, architecture_decision, workflow, gotcha, bug_fix, command_result, tool_usage, knowledge, other)."
    ),
    file: Optional[str] = typer.Option(
        None, "--file", "-f", help="Path to content file."
    ),
    content: Optional[str] = typer.Option(
        None, "--content", "-c", help="Inline content text."
    ),
    source_ref: Optional[str] = typer.Option(
        None, "--source-ref", "-s", help="Source reference."
    ),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Add a knowledge document to the context graph."""
    from graphctx.models import MEMORY_CATEGORIES

    if category not in MEMORY_CATEGORIES:
        typer.echo(f"Error: invalid category '{category}'. Choose from: {', '.join(MEMORY_CATEGORIES)}", err=True)
        raise typer.Exit(code=1)

    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    if file:
        with open(file) as fh:
            body = fh.read()
    elif content:
        body = content
    else:
        typer.echo("Error: provide --file or --content", err=True)
        raise typer.Exit(code=1)

    try:
        validate_content(body)
        validate_content(title, field="title")
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.embeddings import DeterministicEmbedder
    from graphctx.extractor import RuleBasedExtractor
    from graphctx.ingest import ingest_knowledge
    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    embedder = DeterministicEmbedder()
    extractor = RuleBasedExtractor()
    result = ingest_knowledge(
        store=store,
        embedder=embedder,
        extractor=extractor,
        ns=ns,
        title=title,
        content=body,
        source_ref=source_ref,
        category=category,
    )
    typer.echo(f"knowledge_id: {result['knowledge_id']}")
    typer.echo(f"episode_id: {result['episode_id']}")
    typer.echo(f"entities: {len(result['entities'])}")
    typer.echo(f"claims: {len(result['claims'])}")
    typer.echo(f"warnings: {len(result['warnings'])}")


# ---------------------------------------------------------------------------
# entity merge
# ---------------------------------------------------------------------------

entity_app = typer.Typer(help="Entity commands.")
app.add_typer(entity_app, name="entity")


@entity_app.command("merge")
def entity_merge(
    entity_a: str = typer.Argument(help="Entity ID to keep (merge target)."),
    entity_b: str = typer.Argument(help="Entity ID to merge and tombstone."),
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    reason: str = typer.Option("", "--reason", "-r", help="Reason for the merge."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Merge entity B into entity A: move edges, claims, provenance, then tombstone B."""
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)

    # Verify both entities exist
    entities_a = store.get_entities_by_ids([entity_a])
    entities_b = store.get_entities_by_ids([entity_b])
    if not entities_a:
        typer.echo(f"Error: entity not found: {entity_a}", err=True)
        raise typer.Exit(code=1)
    if not entities_b:
        typer.echo(f"Error: entity not found: {entity_b}", err=True)
        raise typer.Exit(code=1)

    try:
        result_id = store.merge_entities(ns, entity_a, entity_b, reason=reason)
    except Exception as exc:
        typer.echo(f"Error: merge failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Merged {entity_b} into {result_id}")
    typer.echo(f"Entity B ({entities_b[0]['name']}) tombstoned.")


# ---------------------------------------------------------------------------
# trigger
# ---------------------------------------------------------------------------

trigger_app = typer.Typer(help="Trigger glossary commands.")
app.add_typer(trigger_app, name="trigger")


@trigger_app.command("add")
def trigger_add(
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    term: str = typer.Option(..., "--term", "-t", help="Trigger term."),
    memory_id: str = typer.Option(..., "--memory-id", "-m", help="Memory ID to link."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Add a trigger term linked to a memory."""
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    trigger_id = store.insert_trigger(ns=ns, term=term, memory_id=memory_id)
    typer.echo(f"trigger_id: {trigger_id}")
    typer.echo(f"term:       {term.lower()}")
    typer.echo(f"memory_id:  {memory_id}")


@trigger_app.command("list")
def trigger_list(
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """List trigger terms for a namespace."""
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    triggers = store.list_triggers(ns=ns)
    if not triggers:
        typer.echo("No triggers found.")
        return
    for t in triggers:
        typer.echo(f"[{t['id']}] term={t['term']}  memory_id={t['memory_id']}")


@trigger_app.command("delete")
def trigger_delete(
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    term: str = typer.Option(..., "--term", "-t", help="Trigger term to delete."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Delete a trigger term by name."""
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    deleted = store.delete_trigger(ns=ns, term=term)
    if deleted:
        typer.echo(f"Deleted trigger '{term}'")
    else:
        typer.echo(f"Trigger not found: '{term}'", err=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# runbook
# ---------------------------------------------------------------------------

_VALID_RUNBOOK_KINDS = {
    "gotcha",
    "workflow",
    "command_fix",
    "project_invariant",
    "dependency_note",
    "debug_trace",
    "architecture_decision",
}

runbook_app = typer.Typer(help="Runbook commands for coding-agent context.")
app.add_typer(runbook_app, name="runbook")


@runbook_app.command("add")
def runbook_add(
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    kind: str = typer.Option(..., "--kind", "-k", help="Runbook kind."),
    content: str = typer.Option(..., "--content", "-c", help="Runbook content."),
    source_ref: Optional[str] = typer.Option(
        None, "--source-ref", "-s", help="Source reference."
    ),
    confidence: float = typer.Option(1.0, "--confidence", help="Confidence 0-1."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Add a runbook entry (gotcha, workflow, command_fix, etc.)."""
    try:
        ns = validate_namespace(namespace)
        validate_content(content)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    if kind not in _VALID_RUNBOOK_KINDS:
        typer.echo(
            f"Error: invalid kind '{kind}'. Choose from: {', '.join(sorted(_VALID_RUNBOOK_KINDS))}",
            err=True,
        )
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    rb_id = store.insert_runbook(
        ns=ns, kind=kind, content=content,
        source_ref=source_ref, confidence=confidence,
    )
    typer.echo(f"runbook_id: {rb_id}")


@runbook_app.command("list")
def runbook_list(
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    kind: Optional[str] = typer.Option(None, "--kind", "-k", help="Filter by kind."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """List runbook entries for a namespace."""
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    entries = store.list_runbook(ns=ns, kind=kind)
    if not entries:
        typer.echo("No runbook entries found.")
        return
    for entry in entries:
        typer.echo(f"[{entry['id']}] ({entry['kind']}) {entry['content']}")
        if entry.get("source_ref"):
            typer.echo(f"  source: {entry['source_ref']}")


@runbook_app.command("delete")
def runbook_delete(
    runbook_id: str = typer.Argument(help="Runbook entry ID to delete."),
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Delete a runbook entry by ID."""
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    deleted = store.delete_runbook(ns=ns, runbook_id=runbook_id)
    if deleted:
        typer.echo(f"Deleted runbook entry {runbook_id}")
    else:
        typer.echo(f"Runbook entry not found: {runbook_id}", err=True)
        raise typer.Exit(code=1)


def _build_context_pack(
    store: Any,
    ns: str,
    task: str,
    budget_tokens: int,
) -> dict[str, Any]:
    """Build a compact evidence pack for a task.

    Queries memories + knowledge + runbook entries relevant to the task,
    applies token budget, and returns a structured pack.
    """
    from graphctx.embeddings import DeterministicEmbedder
    from graphctx.retriever import RecallEngine

    embedder = DeterministicEmbedder()
    engine = RecallEngine(store=store, embedder=embedder)

    # Recall relevant memories/knowledge
    output = engine.recall(
        ns=ns, query=task, mode="hybrid",
        limit=20, include_provenance=True, explain=True,
    )

    # Get runbook entries
    runbook_entries = store.list_runbook(ns=ns)

    # Build items list with token counting
    chars_per_token = 4
    used_tokens = 0
    items: list[dict[str, Any]] = []

    # Add runbook items first (highest priority for coding agents)
    for entry in runbook_entries:
        entry_tokens = len(entry["content"]) // chars_per_token
        if used_tokens + entry_tokens > budget_tokens:
            break
        items.append({
            "category": "runbook",
            "kind": entry["kind"],
            "content": entry["content"],
            "source_ref": entry.get("source_ref"),
            "confidence": entry.get("confidence", 1.0),
            "why_included": f"runbook {entry['kind']}",
            "safety": {"treat_as_instruction": False},
        })
        used_tokens += entry_tokens

    # Add recalled memories/knowledge
    for result in output.results:
        item_tokens = len(result.content) // chars_per_token
        if used_tokens + item_tokens > budget_tokens:
            break
        source_ref = None
        if result.provenance:
            source_ref = result.provenance[0].get("source_ref")
        items.append({
            "category": result.kind,
            "kind": "memory" if result.kind == "memory" else "knowledge",
            "content": result.content,
            "source_ref": source_ref,
            "confidence": result.score,
            "why_included": result.why_retrieved or "recall match",
            "safety": result.safety,
        })
        used_tokens += item_tokens

    return {
        "namespace": ns,
        "task": task,
        "budget_tokens": budget_tokens,
        "items": items,
        "total_tokens": used_tokens,
        "safety": {"treat_as_instruction": False},
    }


@runbook_app.command("pack")
def runbook_pack(
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    task: str = typer.Option(..., "--task", "-t", help="Task description."),
    budget: int = typer.Option(2000, "--budget", "-b", help="Token budget."),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Build a compact evidence pack for a coding task."""
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    pack = _build_context_pack(store=store, ns=ns, task=task, budget_tokens=budget)

    if json_output:
        typer.echo(json.dumps(pack, indent=2))
        return

    typer.echo(f"Task: {pack['task']}")
    typer.echo(f"Budget: {pack['budget_tokens']} tokens (used: {pack['total_tokens']})")
    typer.echo(f"Items: {len(pack['items'])}")
    typer.echo(f"Safety: {pack['safety']}")
    typer.echo("")
    for i, item in enumerate(pack["items"], 1):
        typer.echo(f"--- Item {i} [{item['category']}/{item['kind']}] ---")
        typer.echo(f"  content:    {item['content']}")
        if item.get("source_ref"):
            typer.echo(f"  source:     {item['source_ref']}")
        typer.echo(f"  confidence: {item['confidence']}")
        typer.echo(f"  why:        {item['why_included']}")


# ---------------------------------------------------------------------------
# hook
# ---------------------------------------------------------------------------

hook_app = typer.Typer(help="Hook commands for agent integration.")
app.add_typer(hook_app, name="hook")


@hook_app.command("preprompt")
def hook_preprompt(
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    task: str = typer.Option(..., "--task", "-t", help="Task description."),
    budget: int = typer.Option(2000, "--budget", "-b", help="Token budget."),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Generate a preprompt context pack for an agent task.

    Returns context pack + runbook items + recent memories + safety metadata.
    Works without running REST server (direct SQLite access).
    """
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    pack = _build_context_pack(store=store, ns=ns, task=task, budget_tokens=budget)

    if json_output:
        typer.echo(json.dumps(pack, indent=2))
        return

    typer.echo(f"namespace: {pack['namespace']}")
    typer.echo(f"task:      {pack['task']}")
    typer.echo(f"budget:    {pack['budget_tokens']} tokens")
    typer.echo(f"used:      {pack['total_tokens']} tokens")
    typer.echo(f"safety:    {pack['safety']}")
    typer.echo(f"items:     {len(pack['items'])}")
    for item in pack["items"]:
        typer.echo(f"  [{item['category']}] {item['content'][:80]}")


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


@app.command()
def recall(
    query: str = typer.Argument(help="Recall query text."),
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    mode: str = typer.Option(
        "hybrid", "--mode", "-m", help="Recall mode: fast, hybrid, thinking."
    ),
    limit: int = typer.Option(8, "--limit", "-l", help="Max results."),
    explain: bool = typer.Option(
        False, "--explain", "-e", help="Show score breakdown, why_retrieved, and provenance."
    ),
    as_of: Optional[str] = typer.Option(
        None, "--as-of", help="ISO timestamp for temporal reference time."
    ),
    temporal_mode: str = typer.Option(
        "auto", "--temporal-mode", help="Temporal filtering: auto, current, historical, all."
    ),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Recall memories and knowledge matching a query."""
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.embeddings import DeterministicEmbedder
    from graphctx.retriever import RecallEngine
    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    embedder = DeterministicEmbedder()
    engine = RecallEngine(store=store, embedder=embedder)

    output = engine.recall(
        ns=ns,
        query=query,
        mode=mode,
        limit=limit,
        include_provenance=True,
        explain=explain,
        as_of=as_of,
        temporal_mode=temporal_mode,
    )

    if not output.results:
        typer.echo("No results found.")
        return

    for i, r in enumerate(output.results, 1):
        typer.echo(f"--- Result {i} ---")
        typer.echo(f"  kind:        {r.kind}")
        typer.echo(f"  score:       {r.score:.4f}")
        typer.echo(f"  content:     {r.content}")
        if r.provenance:
            for p in r.provenance:
                ep = p.get("episode_id", "?")
                ref = p.get("source_ref", "")
                prov_str = f"    episode: {ep}"
                if ref:
                    prov_str += f"  source: {ref}"
                typer.echo(prov_str)
        if r.warnings:
            for w in r.warnings:
                typer.echo(f"  WARNING:     {w}")
        typer.echo(f"  safety:      {r.safety}")

        # Explain mode: print detailed breakdown
        if explain and r.score_breakdown:
            typer.echo(f"  --- Score Breakdown ---")
            typer.echo(f"    lexical:     {r.score_breakdown.lexical:.4f}")
            typer.echo(f"    vector:      {r.score_breakdown.vector:.4f}")
            typer.echo(f"    graph_boost: {r.score_breakdown.graph_boost:.4f}")
            typer.echo(f"    recency:     {r.score_breakdown.recency:.4f}")
            typer.echo(f"    confidence:  {r.score_breakdown.confidence:.4f}")
            typer.echo(f"    final:       {r.score_breakdown.final:.4f}")
        if explain and r.why_retrieved:
            typer.echo(f"  why:         {r.why_retrieved}")

        typer.echo("")

    # Explain mode: print explanation summary
    if explain:
        explanation = output.explanation
        if explanation.query_terms:
            typer.echo(f"Query terms: {', '.join(explanation.query_terms)}")
        if explanation.latency_ms:
            typer.echo(f"Latency (ms): {explanation.latency_ms}")


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------


@app.command()
def context(
    entity_id: str = typer.Argument(help="Entity ID to inspect."),
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Inspect an entity and its claims, edges, and provenance."""
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)

    entities = store.get_entities_by_ids([entity_id])
    if not entities:
        typer.echo(f"Entity not found: {entity_id}")
        raise typer.Exit(code=1)

    entity = entities[0]
    typer.echo(f"entity:    {entity['id']}  name={entity['name']}  type={entity['type']}")

    claims = store.get_claims_for_entity(entity_id)
    typer.echo(f"claims:    {len(claims)}")
    for c in claims:
        typer.echo(f"  - {c['subject_entity_id']} {c['predicate']} {c['object_value']}")

    edges = store.get_edges_for_entity(entity_id)
    typer.echo(f"edges:     {len(edges)}")
    for e in edges:
        typer.echo(f"  - {e['source_entity_id']} --[{e['relation']}]--> {e['target_entity_id']}")

    provenance = store.get_provenance_for(entity_id=entity_id)
    typer.echo(f"provenance: {len(provenance)}")
    for p in provenance:
        typer.echo(f"  - episode: {p['episode_id']}")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@app.command()
def stats(
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Print counts for a namespace."""
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    data = store.get_stats(ns)
    typer.echo(f"namespace:       {ns}")
    typer.echo(f"memories:        {data['memories']}")
    typer.echo(f"knowledge_items: {data['knowledge_items']}")
    typer.echo(f"entities:        {data['entities']}")
    typer.echo(f"edges:           {data['edges']}")
    typer.echo(f"claims:          {data['claims']}")
    typer.echo(f"runbook_items:   {data.get('runbook_items', 0)}")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

_REQUIRED_TABLES = {
    "namespace",
    "episode",
    "memory",
    "knowledge",
    "entity",
    "entity_alias",
    "edge",
    "claim",
    "provenance",
    "embedding",
    "audit_log",
    "runbook",
}


@app.command()
def doctor(
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Run integrity and health checks on the database."""
    import sqlite3

    resolved = os.path.expanduser(db)
    if not os.path.exists(resolved):
        typer.echo(f"Database not found: {resolved}")
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    store = SQLiteStore(resolved)
    conn = store._conn()
    errors: list[str] = []

    # 1. Integrity check
    row = conn.execute("PRAGMA integrity_check").fetchone()
    integrity = row[0] if row else "unknown"
    if integrity != "ok":
        errors.append(f"integrity_check: {integrity}")
    typer.echo(f"integrity_check: {integrity}")

    # 2. Required tables
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    existing = {r["name"] for r in rows}
    missing = _REQUIRED_TABLES - existing
    if missing:
        errors.append(f"missing tables: {sorted(missing)}")
    typer.echo(f"tables: {len(existing & _REQUIRED_TABLES)}/{len(_REQUIRED_TABLES)} present")
    if missing:
        typer.echo(f"  missing: {sorted(missing)}")

    # 3. Embedding dimension metadata
    emb_row = conn.execute(
        "SELECT DISTINCT model, dimension FROM embedding LIMIT 5"
    ).fetchall()
    if emb_row:
        for r in emb_row:
            typer.echo(f"embedding: model={r['model']}  dim={r['dimension']}")
    else:
        typer.echo("embedding: no embeddings stored yet")

    # 4. Orphan provenance count
    orphan_count = store.orphan_provenance_count()
    typer.echo(f"orphan_provenance: {orphan_count}")
    if orphan_count > 0:
        errors.append(f"orphan provenance rows: {orphan_count}")

    # 5. Tombstoned exclusion check
    tombstoned = conn.execute(
        "SELECT COUNT(*) as cnt FROM episode WHERE tombstoned_at IS NOT NULL"
    ).fetchone()["cnt"]
    typer.echo(f"tombstoned_episodes: {tombstoned}")

    if errors:
        typer.echo(f"\nDOCTOR: {len(errors)} issue(s) found.")
        for e in errors:
            typer.echo(f"  - {e}")
        raise typer.Exit(code=1)
    else:
        typer.echo("\nDOCTOR: all checks passed.")


# ---------------------------------------------------------------------------
# decay commands
# ---------------------------------------------------------------------------


@app.command()
def pin(
    episode_id: str = typer.Argument(help="Episode ID to pin."),
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Pin an episode so it never decays."""
    try:
        validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    store.pin_episode(episode_id)
    typer.echo(f"Pinned episode {episode_id}")


@app.command()
def unpin(
    episode_id: str = typer.Argument(help="Episode ID to unpin."),
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Unpin an episode so it resumes normal decay."""
    try:
        validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    store.unpin_episode(episode_id)
    typer.echo(f"Unpinned episode {episode_id}")


@app.command()
def decay(
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Run a decay pass: archive episodes with very low retention."""
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    from graphctx.decay import compute_decay_score
    from graphctx.storage import SQLiteStore

    resolved = os.path.expanduser(db)
    store = SQLiteStore(resolved)
    conn = store._conn()

    rows = conn.execute(
        "SELECT id, created_at, last_accessed_at, stability, pinned FROM episode WHERE namespace = ? AND deleted_at IS NULL AND tombstoned_at IS NULL",
        (ns,),
    ).fetchall()

    archived = 0
    for row in rows:
        if row["pinned"]:
            continue
        score = compute_decay_score(
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            stability=float(row["stability"]),
        )
        if score < 0.1:
            store.tombstone_episode(row["id"])
            archived += 1

    typer.echo(f"Decay pass complete: {archived} episode(s) archived.")


# ---------------------------------------------------------------------------
# init-agent
# ---------------------------------------------------------------------------

_MCP_CONFIGS: dict[str, dict] = {
    "claude": {
        "mcpServers": {
            "graphctx": {
                "command": "graphctx",
                "args": ["serve", "--port", "8420"],
            }
        }
    },
    "cursor": {
        "mcpServers": {
            "graphctx": {
                "command": "graphctx",
                "args": ["serve", "--port", "8420"],
                "env": {
                    "GRAPHCTX_DB": "~/.graphctx/graphctx.db",
                },
            }
        }
    },
    "codex": {
        "mcpServers": {
            "graphctx": {
                "command": "graphctx",
                "args": ["serve", "--port", "8420"],
                "env": {
                    "GRAPHCTX_DB": "~/.graphctx/graphctx.db",
                },
            }
        }
    },
    "generic": {
        "mcpServers": {
            "graphctx": {
                "command": "graphctx",
                "args": ["serve"],
            }
        }
    },
}

_STARTER_MEMORY = (
    "This agent uses GraphCtx for persistent context. "
    "Memories are namespace-scoped and provenance-aware."
)

_STARTER_KNOWLEDGE_TITLE = "GraphCtx Quick Reference"
_STARTER_KNOWLEDGE_CONTENT = (
    "GraphCtx is a local context graph for AI agents. Key commands:\n"
    "- graphctx memory add: Add a memory\n"
    "- graphctx knowledge add: Add knowledge\n"
    "- graphctx recall: Query context\n"
    "- graphctx stats: Show namespace statistics\n"
    "- graphctx doctor: Run integrity checks\n"
    "- graphctx serve: Start REST+MCP server"
)


@app.command()
def init_agent(
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
    client: str = typer.Option(
        "generic",
        "--client",
        "-c",
        help="Agent client: claude, cursor, codex, generic.",
    ),
) -> None:
    """One-step agent onboarding: create DB, seed data, print MCP config."""
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    valid_clients = set(_MCP_CONFIGS)
    if client not in valid_clients:
        typer.echo(
            f"Error: unknown client '{client}'. Choose from: {', '.join(sorted(valid_clients))}",
            err=True,
        )
        raise typer.Exit(code=1)

    resolved = os.path.expanduser(db)
    os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)

    from graphctx.embeddings import DeterministicEmbedder
    from graphctx.extractor import RuleBasedExtractor
    from graphctx.ingest import ingest_knowledge, ingest_memory
    from graphctx.storage import SQLiteStore

    store = SQLiteStore(resolved)
    embedder = DeterministicEmbedder()
    extractor = RuleBasedExtractor()

    # Doctor check
    conn = store._conn()
    row = conn.execute("PRAGMA integrity_check").fetchone()
    integrity = row[0] if row else "unknown"
    if integrity != "ok":
        typer.echo(f"Error: integrity check failed: {integrity}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"DB ready: {resolved}")
    typer.echo(f"integrity_check: {integrity}")

    # Insert starter memory
    mem_result = ingest_memory(
        store=store,
        embedder=embedder,
        extractor=extractor,
        ns=ns,
        content=_STARTER_MEMORY,
        source_ref="init-agent",
    )
    typer.echo(f"Starter memory inserted: {mem_result['memory_id']}")

    # Insert starter knowledge
    kn_result = ingest_knowledge(
        store=store,
        embedder=embedder,
        extractor=extractor,
        ns=ns,
        title=_STARTER_KNOWLEDGE_TITLE,
        content=_STARTER_KNOWLEDGE_CONTENT,
        source_ref="init-agent",
    )
    typer.echo(f"Starter knowledge inserted: {kn_result['knowledge_id']}")

    # Print MCP config
    typer.echo("\n--- MCP Config ---")
    typer.echo(json.dumps(_MCP_CONFIGS[client], indent=2))

    # Print REST health command
    typer.echo("\n--- Verify ---")
    typer.echo("curl http://localhost:8420/v1/health")

    # Print first recall command
    typer.echo(f'\ngraphctx recall "GraphCtx" --namespace {ns}')


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

_DEFAULT_EXTENSIONS = {".md", ".txt", ".rst"}
_DEFAULT_MAX_BYTES = 1_048_576  # 1 MB


@app.command("import")
def import_files(
    namespace: str = typer.Option(..., "--namespace", "-n", help="Namespace."),
    path: str = typer.Option(..., "--path", "-p", help="File or directory to import."),
    ext: Optional[str] = typer.Option(
        None, "--ext", help="Comma-separated extensions to filter (e.g. .md,.txt)."
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Recurse into subdirectories."
    ),
    max_bytes: int = typer.Option(
        _DEFAULT_MAX_BYTES, "--max-bytes", help="Skip files larger than this (bytes)."
    ),
    db: str = typer.Option(
        "~/.graphctx/graphctx.db", help="Path to SQLite database."
    ),
) -> None:
    """Import files as knowledge items into the context graph."""
    try:
        ns = validate_namespace(namespace)
    except ValidationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    resolved_path = os.path.expanduser(path)
    if not os.path.exists(resolved_path):
        typer.echo(f"Error: path not found: {resolved_path}", err=True)
        raise typer.Exit(code=1)

    # Determine allowed extensions
    if ext:
        allowed_extensions = {e.strip() if e.strip().startswith(".") else f".{e.strip()}" for e in ext.split(",")}
    else:
        allowed_extensions = _DEFAULT_EXTENSIONS

    # Collect files
    files_to_import: list[str] = []
    if os.path.isfile(resolved_path):
        files_to_import.append(resolved_path)
    elif os.path.isdir(resolved_path):
        if recursive:
            for dirpath, _dirnames, filenames in os.walk(resolved_path):
                for fname in filenames:
                    files_to_import.append(os.path.join(dirpath, fname))
        else:
            for fname in os.listdir(resolved_path):
                full = os.path.join(resolved_path, fname)
                if os.path.isfile(full):
                    files_to_import.append(full)

    # Filter by extension
    files_to_import = [
        f for f in files_to_import
        if os.path.splitext(f)[1].lower() in allowed_extensions
    ]

    resolved = os.path.expanduser(db)
    os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)

    from graphctx.embeddings import DeterministicEmbedder
    from graphctx.extractor import RuleBasedExtractor
    from graphctx.ingest import ingest_knowledge
    from graphctx.storage import SQLiteStore

    store = SQLiteStore(resolved)
    embedder = DeterministicEmbedder()
    extractor = RuleBasedExtractor()

    # Compute base for relative paths
    if os.path.isfile(resolved_path):
        base_dir = os.path.dirname(resolved_path)
    else:
        base_dir = resolved_path

    total = len(files_to_import)
    success = 0
    failed = 0
    skipped = 0

    for fpath in files_to_import:
        # Skip files over max_bytes
        file_size = os.path.getsize(fpath)
        if file_size > max_bytes:
            skipped += 1
            continue

        rel_path = os.path.relpath(fpath, base_dir)
        try:
            with open(fpath, encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except Exception as exc:
            typer.echo(f"  failed to read {rel_path}: {exc}", err=True)
            failed += 1
            continue

        if not content.strip():
            skipped += 1
            continue

        title = os.path.splitext(os.path.basename(fpath))[0]
        try:
            ingest_knowledge(
                store=store,
                embedder=embedder,
                extractor=extractor,
                ns=ns,
                title=title,
                content=content,
                source_ref=rel_path,
            )
            success += 1
        except Exception as exc:
            typer.echo(f"  failed to import {rel_path}: {exc}", err=True)
            failed += 1

    typer.echo(f"total:   {total}")
    typer.echo(f"success: {success}")
    typer.echo(f"failed:  {failed}")
    typer.echo(f"skipped: {skipped}")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@app.command()
def benchmark(
    namespace: str = typer.Option("bench", "--namespace", "-n", help="Namespace."),
    items: int = typer.Option(1000, "--items", "-i", help="Number of synthetic items."),
    queries: int = typer.Option(100, "--queries", "-q", help="Number of recall queries per mode."),
    json_output: bool = typer.Option(False, "--json", help="Output JSON instead of table."),
    db: str = typer.Option(
        "/tmp/graphctx_bench.db", help="Path to temporary SQLite database."
    ),
) -> None:
    """Run ingest and recall benchmarks with synthetic data."""
    from graphctx.benchmark import run_benchmark

    resolved = os.path.expanduser(db)
    results = run_benchmark(ns=namespace, items=items, queries=queries, db_path=resolved)

    if json_output:
        typer.echo(json.dumps(results, indent=2))
        return

    # Human-readable table
    typer.echo(f"{'=' * 60}")
    typer.echo(f"  GraphCtx Benchmark Results")
    typer.echo(f"{'=' * 60}")
    typer.echo(f"  Namespace:        {results['namespace']}")
    typer.echo(f"  Items ingested:   {results['items_ingested']}")
    typer.echo(f"  Queries per mode: {results['queries_per_mode']}")
    typer.echo(f"{'=' * 60}")

    ing = results["ingest"]
    typer.echo(f"\n  Ingest")
    typer.echo(f"    Total time:     {ing['total_ms']:.1f} ms")
    typer.echo(f"    Items/sec:      {ing['items_per_sec']:.1f}")

    for mode in ("fast", "hybrid", "thinking"):
        m = results["recall"][mode]
        typer.echo(f"\n  Recall [{mode}]")
        typer.echo(f"    p50 latency:    {m['p50_ms']:.2f} ms")
        typer.echo(f"    p95 latency:    {m['p95_ms']:.2f} ms")
        typer.echo(f"    max latency:    {m['max_ms']:.2f} ms")
        typer.echo(f"    avg results:    {m['avg_result_count']:.1f}")

    db_info = results["db"]
    typer.echo(f"\n  Database")
    typer.echo(f"    Size:           {db_info['size_bytes']:,} bytes")
    typer.echo(f"    Embeddings:     {db_info['embedding_count']}")

    ts = results["tombstoned_exclusion"]
    typer.echo(f"\n  Tombstoned Exclusion")
    typer.echo(f"    Found before:   {ts['found_before_tombstone']}")
    typer.echo(f"    Found after:    {ts['found_after_tombstone']}")
    typer.echo(f"    Excluded:       {ts['excluded']}")

    # v3 config
    v3 = results.get("v3_config", {})
    if v3:
        typer.echo(f"\n  V3 Config")
        typer.echo(f"    Graph max hops:       {v3.get('graph_max_hops', '?')}")
        typer.echo(f"    Temporal filter:      {v3.get('temporal_filter_enabled', '?')}")
        typer.echo(f"    Alias dedup:          {v3.get('alias_dedup_enabled', '?')}")
        typer.echo(f"    Reranker enabled:     {v3.get('reranker_enabled', '?')}")
        cat_dist = v3.get("category_distribution", {})
        if cat_dist:
            typer.echo(f"    Category distribution:")
            for cat, count in sorted(cat_dist.items()):
                typer.echo(f"      {cat}: {count}")

    typer.echo(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# eval mini
# ---------------------------------------------------------------------------

eval_app = typer.Typer(help="Evaluation commands.")
app.add_typer(eval_app, name="eval")


@eval_app.command("mini")
def eval_mini(
    namespace: str = typer.Option("eval", "--namespace", "-n", help="Namespace."),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
    db: str = typer.Option(
        "/tmp/graphctx_eval.db", help="Path to temporary SQLite database."
    ),
) -> None:
    """Run the mini evaluation suite (15 categories)."""
    from graphctx.eval import run_eval_mini

    resolved = os.path.expanduser(db)
    results = run_eval_mini(ns=namespace, db_path=resolved)

    if json_output:
        typer.echo(json.dumps(results, indent=2))
        return

    # Human-readable output
    summary = results["summary"]
    typer.echo(f"{'=' * 60}")
    typer.echo(f"  GraphCtx Mini Eval Results")
    typer.echo(f"{'=' * 60}")
    typer.echo(f"  Passed: {summary['passed']}/{summary['total']}")
    typer.echo(f"{'=' * 60}")

    for cat in results["categories"]:
        status = "PASS" if cat["pass"] else "FAIL"
        typer.echo(f"\n  [{status}] {cat['category']}")
        typer.echo(f"    Expected:  {cat['expected']}")
        typer.echo(f"    Top score: {cat['top_score']:.4f}")
        if cat["notes"]:
            typer.echo(f"    Notes:     {cat['notes']}")

    typer.echo(f"\n{'=' * 60}")
    if summary["all_passed"]:
        typer.echo("  All categories passed.")
    else:
        typer.echo("  Some categories failed.")
    typer.echo(f"{'=' * 60}")


if __name__ == "__main__":
    app()
