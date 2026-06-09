"""Worker operations that keep their caller's session lock safe on cancellation."""

import asyncio
from collections.abc import Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


async def run_session_worker(function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    """Finish a started state operation before letting its caller release the lock.

    The caller holds the session lock. Canceling a thread await cannot stop the
    thread, so shield and drain it, including repeated cancellation. The entire
    checkpoint/record publication must be inside one worker operation.
    """
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancellation = None
    while True:
        try:
            result = await asyncio.shield(worker)
            break
        except asyncio.CancelledError as exc:
            if worker.cancelled():
                raise
            cancellation = exc
    if cancellation is not None:
        raise cancellation
    return result
