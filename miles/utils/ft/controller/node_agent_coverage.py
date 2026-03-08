"""Sliding-window checker that warns when training nodes lack a registered node agent."""

from __future__ import annotations

import logging
from collections import deque

logger = logging.getLogger(__name__)


class NodeAgentCoverageChecker:
    """Tracks node agent coverage over a sliding window and warns on sustained gaps.

    Each call to ``check()`` records whether each training node has a
    registered agent. Only after a node has been uncovered for
    *threshold* consecutive checks does a warning fire, filtering out
    transient registration delays. Once warned, a node is not warned
    again until coverage is restored and then lost again.
    """

    def __init__(
        self,
        window_size: int = 5,
        threshold: int = 5,
    ) -> None:
        self._windows: dict[str, deque[bool]] = {}
        self._window_size = window_size
        self._threshold = threshold
        self._alerted: set[str] = set()

    def check(
        self,
        training_node_ids: set[str],
        registered_agent_node_ids: set[str],
    ) -> None:
        uncovered = training_node_ids - registered_agent_node_ids
        covered = training_node_ids & registered_agent_node_ids

        for node_id in covered:
            if node_id in self._alerted:
                logger.info("Node agent coverage restored: %s", node_id)
                self._alerted.discard(node_id)
            self._windows.pop(node_id, None)

        for node_id in uncovered:
            window = self._windows.setdefault(
                node_id,
                deque(maxlen=self._window_size),
            )
            window.append(True)

            if len(window) >= self._threshold and node_id not in self._alerted:
                logger.warning(
                    "Node %s has been running training without node agent " "for %d consecutive checks",
                    node_id,
                    len(window),
                )
                self._alerted.add(node_id)
