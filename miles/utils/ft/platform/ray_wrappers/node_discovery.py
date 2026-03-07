"""Shared Ray node discovery utilities.

Centralises ray.nodes() traversal logic used by both RayTrainingJob
(node-id resolution) and standalone NCCL diagnostics (GPU node discovery).
"""

from __future__ import annotations

import logging
from typing import Any

import ray

logger = logging.getLogger(__name__)


def get_alive_gpu_nodes(
    node_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return alive Ray nodes that have GPUs.

    If *node_ids* is given, only return nodes whose NodeID is in the list.
    """
    nodes = [n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0]

    if node_ids is not None:
        allowed = set(node_ids)
        nodes = [n for n in nodes if n["NodeID"] in allowed]

    return nodes


def resolve_to_ray_node_ids(identifiers: list[str]) -> list[str]:
    """Map node identifiers (K8s names, IPs, or Ray hex IDs) to Ray node IDs.

    Looks up each identifier against NodeName, NodeManagerAddress, and NodeID
    of alive Ray nodes. Identifiers already matching a NodeID pass through.
    Unresolvable identifiers are logged and skipped.
    """
    lookup: dict[str, str] = {}
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        ray_id = node["NodeID"]
        lookup[ray_id] = ray_id
        if name := node.get("NodeName"):
            lookup[name] = ray_id
        if addr := node.get("NodeManagerAddress"):
            lookup[addr] = ray_id

    seen: set[str] = set()
    resolved: list[str] = []
    for ident in identifiers:
        ray_id = lookup.get(ident)
        if ray_id is not None and ray_id not in seen:
            seen.add(ray_id)
            resolved.append(ray_id)
        elif ray_id is None:
            logger.warning("resolve_to_ray_node_ids: %s not found in Ray cluster, skipping", ident)
    return resolved


def build_node_address_map(nodes: list[dict[str, Any]]) -> dict[str, str]:
    """Build a mapping from NodeID to NodeManagerAddress."""
    return {node["NodeID"]: addr for node in nodes if (addr := node.get("NodeManagerAddress", ""))}
