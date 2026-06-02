"""Basic usage example for GraphCtx SDK.

Demonstrates adding memories, adding knowledge, running recall,
and managing sessions through the async Python SDK.

Run with:
    python examples/basic_usage.py

Requires the GraphCtx server to be running on localhost:8420.
Start it with: graphctx serve
"""

import asyncio

from graphctx.sdk import GraphCtxClient


async def main():
    client = GraphCtxClient()

    # Add memory
    result = await client.add_memory("demo", "Alice prefers concise answers.")
    print(f"Memory: {result['memory_id']}")

    # Add knowledge
    result = await client.add_knowledge("demo", "Architecture", "GraphCtx uses SQLite.")
    print(f"Knowledge: {result['knowledge_id']}")

    # Recall with explanation
    results = await client.recall("demo", "What does Alice prefer?", explain=True)
    for r in results["results"]:
        print(f"  [{r['kind']}] {r['content'][:60]}... score={r['score']:.2f}")
        if r.get("score_breakdown"):
            print(f"    breakdown: {r['score_breakdown']}")

    # Create session
    session = await client.create_session("demo", user_ref="alice")
    print(f"Session: {session['session_id']}")

    await client.close()


asyncio.run(main())
