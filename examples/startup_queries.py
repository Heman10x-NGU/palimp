#!/usr/bin/env python3
"""Startup query examples for Palimp.

Demonstrates hybrid recall over imported startup data with 5 example queries:
  1. Find AI agent startups
  2. Find fintech startups with high opportunity score
  3. Find competitors to Zep AI
  4. Find W26 (Winter 2026) batch startups
  5. Find agent infrastructure / developer tools

Usage:
    # Run with a small embedded test dataset (creates temp DB):
    python examples/startup_queries.py

    # Run against existing imported data:
    python examples/startup_queries.py --db ~/.palimp/palimp.db

CLI equivalents (after importing data):
    palimp recall --namespace startups "AI agent startups" --limit 10
    palimp recall --namespace startups "fintech opportunity score 9" --limit 5
    palimp recall --namespace startups "Zep AI competitors" --limit 5
    palimp recall --namespace startups "Winter 2026 batch" --limit 10
    palimp recall --namespace startups "agent infrastructure API developer tools" --limit 10
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from palimp.embeddings import DeterministicEmbedder
from palimp.ingest import ingest_memory
from palimp.retriever import RecallEngine, RecallOutput
from palimp.storage import SQLiteStore

# ---------------------------------------------------------------------------
# Small test dataset: 10 representative startups for local testing
# ---------------------------------------------------------------------------

_TEST_STARTUPS: list[dict[str, str]] = [
    {
        "content": (
            "Zep AI is an AI infrastructure startup from YC W24. "
            "Zep provides a long-term memory service for AI agents and assistants. "
            "It helps developers build agents that remember user preferences, "
            "past conversations, and important context over time. "
            "Category: AI agent infrastructure. "
            "Tags: memory, context, LLM, agents, developer tools."
        ),
        "source_ref": "startup:zep-ai",
    },
    {
        "content": (
            "Fixie.ai is an AI agent platform startup from YC W23. "
            "Fixie provides a platform for building and deploying AI agents "
            "that can connect to external APIs and services. "
            "It offers a hosted runtime for agent workflows. "
            "Category: AI agent platform. "
            "Tags: agents, platform, API, workflows, developer tools."
        ),
        "source_ref": "startup:fixie-ai",
    },
    {
        "content": (
            "Cognition Labs is an AI startup building Devin, "
            "an autonomous AI software engineer. "
            "Devin can plan, write code, debug, and ship software with minimal human input. "
            "Category: AI coding agent. "
            "Tags: coding, autonomous, software engineering, agents."
        ),
        "source_ref": "startup:cognition-labs",
    },
    {
        "content": (
            "Mercury is a fintech startup from YC W23. "
            "Mercury offers a banking platform for startups with API access, "
            "expense management, and treasury features. "
            "Opportunity score: 9/10. Risk level: low. "
            "Category: fintech, banking. "
            "Tags: banking, API, startups, fintech, treasury."
        ),
        "source_ref": "startup:mercury",
    },
    {
        "content": (
            "Ramp is a fintech startup focused on corporate card and expense management. "
            "Ramp provides intelligent spend management with AI-powered savings insights. "
            "Opportunity score: 9/10. Risk level: low. "
            "Category: fintech, spend management. "
            "Tags: fintech, corporate card, expenses, AI, savings."
        ),
        "source_ref": "startup:ramp",
    },
    {
        "content": (
            "Dust.tt is an AI agent infrastructure startup from YC W24. "
            "Dust provides a platform for building custom AI assistants "
            "that connect to company knowledge bases and tools. "
            "It focuses on enterprise agent deployment. "
            "Category: AI agent infrastructure. "
            "Tags: agents, enterprise, knowledge, assistants, developer tools."
        ),
        "source_ref": "startup:dust-tt",
    },
    {
        "content": (
            "Letta (formerly MemGPT) is an AI startup building stateful AI agents. "
            "Letta provides a framework for creating agents with persistent memory "
            "and self-editing capabilities. "
            "Category: AI agent framework. "
            "Tags: memory, agents, framework, stateful, LLM."
        ),
        "source_ref": "startup:letta",
    },
    {
        "content": (
            "LangChain is an AI developer tools startup from YC W23. "
            "LangChain provides a framework for building LLM-powered applications "
            "with chains, agents, and retrieval-augmented generation. "
            "Category: AI developer tools. "
            "Tags: LLM, framework, chains, RAG, developer tools, agents."
        ),
        "source_ref": "startup:langchain",
    },
    {
        "content": (
            "Clay is a go-to-market AI startup from YC W23. "
            "Clay provides a data enrichment and outreach platform for sales teams, "
            "using AI to personalize outbound at scale. "
            "Opportunity score: 8/10. "
            "Category: AI sales tools. "
            "Tags: sales, outreach, enrichment, GTM, AI."
        ),
        "source_ref": "startup:clay",
    },
    {
        "content": (
            "Replit Agent is an AI coding assistant built by Replit. "
            "It can create full-stack applications from natural language prompts, "
            "handle deployments, and iterate on code based on feedback. "
            "Category: AI coding tools. "
            "Tags: coding, IDE, agents, deployment, developer tools."
        ),
        "source_ref": "startup:replit-agent",
    },
]


def _seed_test_data(db_path: str, ns: str) -> int:
    """Ingest the test dataset into a fresh Palimp database.

    Returns the number of items ingested.
    """
    store = SQLiteStore(db_path)
    embedder = DeterministicEmbedder()
    count = 0
    for item in _TEST_STARTUPS:
        ingest_memory(
            store=store,
            embedder=embedder,
            extractor=None,
            ns=ns,
            content=item["content"],
            source_ref=item["source_ref"],
            category="knowledge",
            extract=False,
        )
        count += 1
    return count


def _print_results(label: str, output: RecallOutput) -> float:
    """Print recall results and return elapsed time in ms.

    Args:
        label: Human-readable query description.
        output: RecallOutput from the engine.

    Returns:
        The total latency in milliseconds from the explanation.
    """
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    if not output.results:
        print("  No results found.")
        total_ms = output.explanation.latency_ms.get("total", 0.0)
        print(f"  Latency: {total_ms:.1f} ms")
        return total_ms

    print(f"  Results: {len(output.results)}")
    print()
    for r in output.results:
        content_preview = r.content[:80].replace("\n", " ")
        print(f"  [{r.kind:>8}] {content_preview}... score={r.score:.2f}")

    total_ms = output.explanation.latency_ms.get("total", 0.0)
    print(f"\n  Latency: {total_ms:.1f} ms")
    return total_ms


def run_queries(engine: RecallEngine, ns: str) -> None:
    """Run all 5 example queries and print results."""
    latencies: list[float] = []

    # Example 1: Find AI agent startups
    output = engine.recall(ns, "AI agent startups", mode="hybrid", limit=10, explain=True)
    latencies.append(_print_results("Example 1: AI Agent Startups", output))

    # Example 2: Find fintech startups with high opportunity score
    output = engine.recall(ns, "fintech opportunity score 9", mode="hybrid", limit=5, explain=True)
    latencies.append(_print_results("Example 2: High-Score Fintech", output))

    # Example 3: Find competitors to Zep AI
    output = engine.recall(ns, "Zep AI competitors context memory", mode="hybrid", limit=5, explain=True)
    latencies.append(_print_results("Example 3: Zep AI Competitors", output))

    # Example 4: Find W26 startups
    output = engine.recall(ns, "Winter 2026 batch", mode="hybrid", limit=10, explain=True)
    latencies.append(_print_results("Example 4: W26 Startups", output))

    # Example 5: Find agent infrastructure
    output = engine.recall(ns, "agent infrastructure API developer tools", mode="hybrid", limit=10, explain=True)
    latencies.append(_print_results("Example 5: Agent Infrastructure", output))

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for i, lat in enumerate(latencies, 1):
        print(f"  Query {i} latency: {lat:.1f} ms")
    if latencies:
        print(f"  Average latency: {sum(latencies) / len(latencies):.1f} ms")
    print(f"{'=' * 60}")


def print_namespace_stats(store: SQLiteStore, ns: str) -> None:
    """Print namespace statistics."""
    stats = store.get_stats(ns)
    print(f"\n{'=' * 60}")
    print("  NAMESPACE STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Namespace:       {ns}")
    print(f"  Memories:        {stats.get('memories', 0)}")
    print(f"  Knowledge items: {stats.get('knowledge_items', 0)}")
    print(f"  Entities:        {stats.get('entities', 0)}")
    print(f"  Edges:           {stats.get('edges', 0)}")
    print(f"  Claims:          {stats.get('claims', 0)}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run startup query examples against Palimp.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to an existing Palimp database with imported startup data. "
             "If omitted, creates a temporary DB with a small test dataset.",
    )
    parser.add_argument(
        "--namespace",
        "-n",
        type=str,
        default="startups",
        help="Namespace to query (default: startups).",
    )
    args = parser.parse_args()

    ns = args.namespace
    temp_dir = None

    if args.db:
        db_path = os.path.expanduser(args.db)
        if not os.path.exists(db_path):
            print(f"ERROR: Database not found: {db_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Using existing database: {db_path}")
    else:
        # Create a temporary DB with test data
        temp_dir = tempfile.mkdtemp(prefix="palimp_startup_demo_")
        db_path = os.path.join(temp_dir, "palimp.db")
        print(f"Creating temporary test database: {db_path}")

        t0 = time.monotonic()
        count = _seed_test_data(db_path, ns)
        elapsed = (time.monotonic() - t0) * 1000
        print(f"Seeded {count} test startups in {elapsed:.0f} ms")

    # Build the recall engine
    store = SQLiteStore(db_path)
    embedder = DeterministicEmbedder()
    engine = RecallEngine(store=store, embedder=embedder)

    # Print namespace stats
    print_namespace_stats(store, ns)

    # Run all example queries
    run_queries(engine, ns)

    # Print CLI equivalents
    print(f"\n{'=' * 60}")
    print("  CLI EQUIVALENTS")
    print(f"{'=' * 60}")
    print(f"  palimp recall --namespace {ns} \"AI agent startups\" --limit 10")
    print(f"  palimp recall --namespace {ns} \"fintech opportunity score 9\" --limit 5")
    print(f"  palimp recall --namespace {ns} \"Zep AI competitors\" --limit 5")
    print(f"  palimp recall --namespace {ns} \"Winter 2026 batch\" --limit 10")
    print(f"  palimp recall --namespace {ns} \"agent infrastructure API developer tools\" --limit 10")
    print(f"{'=' * 60}")

    if temp_dir:
        print(f"\n  (Temp DB at: {db_path})")


if __name__ == "__main__":
    main()
