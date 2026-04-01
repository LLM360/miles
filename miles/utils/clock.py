from __future__ import annotations

import asyncio
import time
from typing import Protocol


class Clock(Protocol):
    def time(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


class RealClock:
    def time(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    async def sleep(self, seconds: float) -> None:
        pass
