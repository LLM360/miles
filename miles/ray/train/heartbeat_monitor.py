import logging

from miles.ray.train.cell import RayTrainCell
from miles.utils.health_checker import SimpleHealthChecker, create_trainer_cell_health_checker

logger = logging.getLogger(__name__)


class TrainerHeartbeatMonitor:
    """Per-cell heartbeat monitors for trainer actors.

    Creates one ``SimpleHealthChecker`` per cell. Each checker periodically
    calls ``heartbeat()`` on every actor in the cell and verifies the returned
    timestamp is not stale.
    """

    def __init__(
        self,
        *,
        cells: list[RayTrainCell],
        first_wait: float,
        interval: float,
        timeout: float,
        staleness: float,
    ) -> None:
        self._checkers: list[SimpleHealthChecker] = [
            create_trainer_cell_health_checker(
                cell=cell,
                interval=interval,
                timeout=timeout,
                staleness=staleness,
                first_wait=first_wait,
            )
            for cell in cells
        ]

    async def start(self) -> None:
        for checker in self._checkers:
            await checker.start()

    def stop(self) -> None:
        for checker in self._checkers:
            checker.stop()

    def pause(self) -> None:
        for checker in self._checkers:
            checker.pause()

    def resume(self) -> None:
        for checker in self._checkers:
            checker.resume()
