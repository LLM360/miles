"""Pure core of the agentic rollout abort: no torch/sglang imports.

The trainer wrapper in ``inference_rollout_train`` injects the real transports
(harbor signal, engine abort_all, /get_load fetch).
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def engines_idle(fetch_loads, *, retries: int, interval: float) -> bool:
    """Return True once every worker shard reports num_reqs==0 and num_waiting_reqs==0.

    ``fetch_loads`` returns (or awaits to) a flat list of per-shard load dicts.
    Bounded-retry because /get_load lags an abort by ~1 scheduler tick: an aborted
    request stays counted until the next forward pass runs check_finished +
    filter_batch.
    """

    def _idle(w) -> bool:
        # Fail closed: an entry missing the expected keys is treated as busy, so a
        # future sglang /get_load schema change cannot make the gate falsely pass
        # (which would let a live decode poison the weight update).
        if not isinstance(w, dict) or "num_reqs" not in w or "num_waiting_reqs" not in w:
            return False
        return int(w["num_reqs"]) == 0 and int(w["num_waiting_reqs"]) == 0

    for _ in range(max(1, retries)):
        loads = fetch_loads()
        if asyncio.iscoroutine(loads):
            loads = await loads
        if loads and all(_idle(w) for w in loads):
            return True
        if interval:
            await asyncio.sleep(interval)
    return False


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
