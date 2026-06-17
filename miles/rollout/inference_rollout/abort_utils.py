"""Pure, dependency-light core of the agent-first rollout abort (Phase 3).

Kept free of torch/sglang imports so the three-layer protocol and the Layer-3
quiescence gate are unit-testable without the training container. The trainer
wrapper in ``inference_rollout_train`` injects the real transports
(harbor signal, engine abort_all, /get_load fetch).
"""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


async def engines_quiescent(fetch_loads, *, retries: int, interval: float) -> bool:
    """Return True once every worker shard reports num_reqs==0 and num_waiting_reqs==0.

    ``fetch_loads`` returns (or awaits to) a flat list of per-shard load dicts.
    Bounded-retry because /get_load lags an abort by ~1 scheduler tick: an aborted
    request stays counted until the next forward pass runs check_finished +
    filter_batch (abort design spec section 13.4).
    """
    def _idle(w) -> bool:
        # Fail closed: an entry missing the expected keys is treated as busy, so a
        # future sglang /get_load schema change cannot make the invariant gate
        # falsely pass (which would let a live decode poison the weight update).
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


async def run_abort_protocol(
    pendings,
    *,
    partial_rollout: bool,
    rollout_id: int,
    is_agentic: bool,
    t1: float,
    t2: float,
    signal_harbor,
    abort_all_engines,
) -> list:
    """Drive the three-layer abort and collect partial samples.

    Layer 1 (agentic): ``signal_harbor()`` then await cooperative teardown until T1.
    Layer 2: re-``signal_harbor()`` until T2, then cancel still-pending tasks and
    ``abort_all_engines()``. Vanilla (non-agentic): ``abort_all_engines()`` then an
    unbounded drain (the historical behavior).

    Partial-sample preservation is unchanged: every drained group with a response
    is stamped with ``start_rollout_id`` (origin step) if not already set, and kept.
    A pending task that raises is logged and does NOT abort the run.

    Returns the list of collected partial-sample groups (``list[list[Sample]]``).
    """
    aborted_samples: list = []
    pend = set(pendings)

    def _collect(group):
        if not partial_rollout or group is None:
            return
        for sample in group:
            if getattr(sample, "response", None) and "start_rollout_id" not in sample.metadata:
                sample.metadata["start_rollout_id"] = rollout_id
        aborted_samples.append(group)

    async def _drain_until(deadline):
        nonlocal pend
        while pend:
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            if timeout == 0.0:
                return
            done, pend = await asyncio.wait(pend, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if not done:  # timed out before any task completed
                return
            for t in done:
                try:
                    _collect(t.result())
                except Exception as exc:  # a failed pending task must not kill the run
                    logger.error(f"[abort] pending task raised: {exc!r}", exc_info=exc)

    t0 = time.monotonic()
    if is_agentic:
        await signal_harbor()  # Layer 1
        await _drain_until(t0 + t1)
        if pend:
            await signal_harbor()  # Layer 2: re-signal
            await _drain_until(t0 + t2)
        if pend:  # Layer 2 hard cutoff
            logger.warning(f"[abort] {len(pend)} tasks still pending at T2 -- cancel + abort_all")
            for t in pend:
                t.cancel()
            await abort_all_engines()
            for r in await asyncio.gather(*pend, return_exceptions=True):
                if not isinstance(r, BaseException):
                    _collect(r)
            pend = set()
    else:
        await abort_all_engines()  # vanilla: blunt engine abort
        await _drain_until(None)  # unbounded drain (historical behavior)

    return aborted_samples
