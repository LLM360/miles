import asyncio
import logging
import time

import ray

from miles.ray.train.cell import RayTrainCell

logger = logging.getLogger(__name__)


class TrainerHeartbeatMonitor:
    def __init__(
        self,
        *,
        cells: list[RayTrainCell],
        first_wait: float,
        interval: float,
        timeout: float,
        staleness: float,
    ) -> None:
        self._cells = cells
        self._first_wait = first_wait
        self._interval = interval
        self._timeout = timeout
        self._staleness = staleness

        self._paused: bool = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        await asyncio.sleep(self._first_wait)

        while True:
            if not self._paused:
                await self.check()

            await asyncio.sleep(self._interval)

    async def check(self) -> None:
        now = time.time()

        alive_cells = [cell for cell in self._cells if cell.is_alive]
        all_futures: list[tuple[RayTrainCell, ray.ObjectRef]] = []
        for cell in alive_cells:
            for actor in cell._get_actor_handles():
                all_futures.append((cell, actor.heartbeat.remote()))

        for cell, future in all_futures:
            if cell.is_errored:
                continue
            try:
                status = await asyncio.wait_for(future, timeout=self._timeout)
                if now - status.last_active_timestamp > self._staleness:
                    logger.error(
                        f"Cell {cell.cell_index} heartbeat stale: "
                        f"last_active={status.last_active_timestamp:.1f}, now={now:.1f}, "
                        f"delta={now - status.last_active_timestamp:.1f}s, bump_count={status.bump_count}"
                    )
                    cell._mark_as_errored()
            except Exception:
                logger.error(f"Cell {cell.cell_index} heartbeat failed", exc_info=True)
                cell._mark_as_errored()

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
