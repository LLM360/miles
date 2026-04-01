from __future__ import annotations

import abc
import asyncio
import heapq
import time


class Clock(abc.ABC):
    @abc.abstractmethod
    def time(self) -> float: ...

    @abc.abstractmethod
    async def sleep(self, seconds: float) -> None: ...


class RealClock(Clock):
    def time(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class FakeClock(Clock):
    """Deterministic clock for testing async time-dependent code.

    ``sleep()`` suspends the caller until ``elapse()`` advances the clock
    past the target time. This gives tests precise control over which
    sleeps resolve and when.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._waiters: list[tuple[float, int, asyncio.Future[None]]] = []
        self._counter: int = 0

    def time(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return

        target = self._now + seconds
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._counter += 1
        heapq.heappush(self._waiters, (target, self._counter, future))
        await future

    async def elapse(self, seconds: float) -> None:
        assert seconds >= 0, f"Cannot elapse negative time: {seconds}"
        self._now += seconds
        self._resolve_ready()
        await asyncio.sleep(0)

    def _resolve_ready(self) -> None:
        while self._waiters and self._waiters[0][0] <= self._now:
            _, _, future = heapq.heappop(self._waiters)
            if not future.done():
                future.set_result(None)

    @property
    def pending_count(self) -> int:
        return sum(1 for _, _, f in self._waiters if not f.done())
