"""Multi-hop BFS graph traversal for recall enhancement."""

from __future__ import annotations

from collections import deque
from typing import Any


def bfs_graph_traversal(
    store: Any,
    ns: str,
    seed_entity_ids: list[str],
    max_hops: int = 2,
    depth_decay: float = 0.55,
    max_expansions: int = 50,
) -> dict[str, dict[str, Any]]:
    """BFS from seed entities, return scored entity paths.

    Returns dict of entity_id -> {score, hop, path: list of (entity_id, edge_id)}.
    """
    visited: dict[str, dict[str, Any]] = {}
    queue: deque[tuple[str, int, list[tuple[str, str]]]] = deque()

    for eid in seed_entity_ids:
        if eid not in visited:
            visited[eid] = {"score": 1.0, "hop": 0, "path": []}
            queue.append((eid, 0, []))

    expansions = 0

    while queue and expansions < max_expansions:
        current_id, hop, path = queue.popleft()

        if hop >= max_hops:
            continue

        # Get edges where current entity is source or target
        edges = store.get_edges_for_entity(current_id)

        for edge in edges:
            if expansions >= max_expansions:
                break
            if edge.get("deleted_at") is not None:
                continue

            # Get neighbor entity
            if edge["source_entity_id"] == current_id:
                neighbor_id = edge["target_entity_id"]
            else:
                neighbor_id = edge["source_entity_id"]

            if neighbor_id in visited:
                continue

            # Check neighbor is not tombstoned
            neighbor = store.get_entities_by_ids([neighbor_id])
            if not neighbor or neighbor[0].get("deleted_at") is not None:
                continue

            new_hop = hop + 1
            score = depth_decay ** new_hop
            new_path = path + [(current_id, edge["id"])]

            visited[neighbor_id] = {
                "score": score,
                "hop": new_hop,
                "path": new_path,
                "edge_relation": edge.get("relation", ""),
            }

            queue.append((neighbor_id, new_hop, new_path))
            expansions += 1

    return visited


def get_episodes_for_entities(store: Any, entity_ids: list[str]) -> dict[str, list[str]]:
    """Map entity_ids to their source episode_ids via provenance."""
    result: dict[str, list[str]] = {}
    for eid in entity_ids:
        prov = store.get_provenance_for(entity_id=eid)
        episodes = [p["episode_id"] for p in prov if p.get("episode_id")]
        if episodes:
            result[eid] = episodes
    return result
