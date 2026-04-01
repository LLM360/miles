import logging
import threading
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

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        if self._stop_event.wait(timeout=self._first_wait):
            return

        while not self._stop_event.is_set():
            if not self._pause_event.is_set():
                self.check()

            if self._stop_event.wait(timeout=self._interval):
                break

    def check(self) -> None:
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
                status = ray.get(future, timeout=self._timeout)
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
        self._stop_event.set()
        self._thread.join(timeout=10.0)

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()
