import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.inference_rollout.inference_rollout_common import _acquire_or_abort


@pytest.mark.asyncio
async def test_abort_wakes_queued_admission_without_leaking_a_permit():
    state = SimpleNamespace(
        generate_fn_semaphore=asyncio.Semaphore(0),
        abort_event=asyncio.Event(),
        aborted=False,
    )
    queued = asyncio.create_task(_acquire_or_abort(state))
    await asyncio.sleep(0)

    state.aborted = True
    state.abort_event.set()

    assert await asyncio.wait_for(queued, timeout=0.1) is False

    simultaneous = SimpleNamespace(
        generate_fn_semaphore=asyncio.Semaphore(1),
        abort_event=asyncio.Event(),
        aborted=True,
    )
    simultaneous.abort_event.set()

    assert await _acquire_or_abort(simultaneous) is False
    await asyncio.wait_for(simultaneous.generate_fn_semaphore.acquire(), timeout=0.1)
