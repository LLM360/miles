"""Pure core of the agentic rollout abort: no torch/sglang imports.

The trainer wrapper in ``inference_rollout_train`` injects the real transports
(harbor signal, engine abort_all).
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def collect_finished_rollouts(
    pendings,
    *,
    partial_rollout: bool,
    rollout_id: int,
) -> list:
    """Await every pending rollout task and collect partial samples.

    A task that raises is logged and does not abort the run. When partial_rollout
    is set, each drained group with a response is stamped with ``start_rollout_id``
    (its origin step) if not already set, and kept.

    Returns the list of collected partial-sample groups (``list[list[Sample]]``).
    """
    aborted_samples: list = []
    for task in asyncio.as_completed(set(pendings)):
        try:
            group = await task
        except Exception as exc:  # a failed pending task must not kill the run
            logger.error(f"[abort] pending rollout task raised: {exc!r}", exc_info=exc)
            continue
        if not partial_rollout or group is None:
            continue
        for sample in group:
            if getattr(sample, "response", None) and "start_rollout_id" not in sample.metadata:
                sample.metadata["start_rollout_id"] = rollout_id
        aborted_samples.append(group)
    return aborted_samples
