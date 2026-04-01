from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miles.ray.train.cell import RayTrainCell

logger = logging.getLogger(__name__)


class SimpleHealthChecker:
    """Periodic async health checker. Calls *check_fn*; on failure calls *on_failure* and stops."""

    def __init__(
        self,
        *,
        name: str,
        check_fn: Callable[[], Coroutine[Any, Any, None]],
        on_failure: Callable[[], None],
        interval: float,
        first_wait: float = 0.0,
    ) -> None:
        self._name = name
        self._check_fn = check_fn
        self._on_failure = on_failure
        self._interval = interval
        self._first_wait = first_wait

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
        await asyncio.sleep(self._first_wait)

        while True:
            if not self._paused:
                try:
                    await self._check_fn()
                except Exception:
                    logger.error(f"Health check failed for {self._name}", exc_info=True)
                    self._on_failure()

            await asyncio.sleep(self._interval)


def create_trainer_cell_health_checker(
    *,
    cell: "RayTrainCell",
    interval: float,
    timeout: float,
    staleness: float,
    first_wait: float,
) -> SimpleHealthChecker:

    async def _check() -> None:
        if not cell.is_alive:
            return

        now = time.time()
        futures = [actor.heartbeat.remote() for actor in cell._get_actor_handles()]

        for future in futures:
            status = await asyncio.wait_for(future, timeout=timeout)
            delta = now - status.last_active_timestamp
            if delta > staleness:
                raise RuntimeError(
                    f"Heartbeat stale: last_active={status.last_active_timestamp:.1f}, "
                    f"now={now:.1f}, delta={delta:.1f}s, bump_count={status.bump_count}"
                )

    return SimpleHealthChecker(
        name=f"trainer-cell-{cell.cell_index}",
        check_fn=_check,
        on_failure=cell._mark_as_errored,
        interval=interval,
        first_wait=first_wait,
    )


# TODO: should move when Rollout Ft is implemented
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
