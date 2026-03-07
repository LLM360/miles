from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable

from miles.utils.ft.agents.diagnostics.stack_trace import PySpyThread, StackTraceDiagnostic
from miles.utils.ft.controller.diagnostics.stack_trace.aggregator import StackTraceAggregator
from miles.utils.ft.protocols.agents import NodeAgentProtocol

logger = logging.getLogger(__name__)


async def collect_stack_trace_suspects(
    agents: dict[str, NodeAgentProtocol],
    rank_pids_provider: Callable[[str], dict[int, int]],
    default_timeout_seconds: int,
) -> list[str]:
    """Collect stack traces from all nodes and identify suspects via aggregation."""
    traces: dict[str, list[PySpyThread]] = {}
    suspect_from_failures: list[str] = []

    async def _collect_node(node_id: str) -> None:
        try:
            rank_pids = rank_pids_provider(node_id)
        except Exception:
            suspect_from_failures.append(node_id)
            logger.warning(
                "rank_pids_provider_failed node=%s",
                node_id,
                exc_info=True,
            )
            return

        if not rank_pids:
            return

        diag = StackTraceDiagnostic(pids=list(rank_pids.values()))

        try:
            result = await diag.run(
                node_id=node_id,
                timeout_seconds=default_timeout_seconds,
            )
            if result.passed:
                threads = [PySpyThread.model_validate(t) for t in json.loads(result.details)]
                traces[node_id] = threads
            else:
                suspect_from_failures.append(node_id)
                logger.info(
                    "stack_trace_collection_failed node=%s details=%s",
                    node_id,
                    result.details,
                )
        except Exception:
            suspect_from_failures.append(node_id)
            logger.warning(
                "stack_trace_collect_exception node=%s",
                node_id,
                exc_info=True,
            )

    await asyncio.gather(*(_collect_node(nid) for nid in agents))

    suspect_from_aggregation = StackTraceAggregator().aggregate(traces=traces)
    all_suspects = sorted(set(suspect_from_failures) | set(suspect_from_aggregation))

    logger.info(
        "collect_stack_trace_suspects_done traces_collected=%d suspect_from_failures=%s suspect_from_aggregation=%s",
        len(traces),
        suspect_from_failures,
        suspect_from_aggregation,
    )
    return all_suspects
