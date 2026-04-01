from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from miles.utils.clock import Clock, RealClock

logger = logging.getLogger(__name__)


class SimpleHealthChecker:
    """Periodic async health checker. Calls *check_fn*; on failure calls *on_failure*."""

    def __init__(
        self,
        *,
        name: str,
        check_fn: Callable[[], Coroutine[Any, Any, None]],
        on_failure: Callable[[], None],
        interval: float,
        first_wait: float = 0.0,
        clock: Clock | None = None,
    ) -> None:
        self._name = name
        self._check_fn = check_fn
        self._on_failure = on_failure
        self._interval = interval
        self._first_wait = first_wait
        self._clock = clock or RealClock()

        self._paused: bool = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    async def _loop(self) -> None:
        await self._clock.sleep(self._first_wait)

        while True:
            if not self._paused:
                try:
                    await self._check_fn()
                except Exception:
                    logger.error(f"Health check failed for {self._name}", exc_info=True)
                    self._on_failure()

            await self._clock.sleep(self._interval)


# TODO: should move when Rollout FT is implemented
def create_rollout_cell_health_checker(
    *,
    cell_id: str,
    get_engines: Callable[[], list[object]],
    interval: float,
    timeout: float,
    on_failure: Callable[[], None],
) -> SimpleHealthChecker:

    async def _check() -> None:
        engines = get_engines()
        if not engines:
            raise RuntimeError("No engines")

        lead_engine = engines[0]
        if lead_engine is None:
            raise RuntimeError("Lead engine is None")

        await asyncio.wait_for(lead_engine.health_generate.remote(), timeout=timeout)

    return SimpleHealthChecker(
        name=f"rollout-cell-{cell_id}",
        check_fn=_check,
        on_failure=on_failure,
        interval=interval,
    )
