import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from miles.utils.prometheus_utils import set_prometheus_gauge

logger = logging.getLogger(__name__)

_METRIC_NAME = "miles_rollout_cell_alive"
_LABEL_KEYS = ["session_id", "run_name", "cell_id"]


@dataclass(frozen=True)
class CellEntry:
    cell_id: str
    get_engines: Callable[[], list[object]]


class RolloutCellHealth:
    """Async health checker that periodically probes rollout cells and reports a Prometheus gauge.

    Each cell's lead engine is probed via ``engine.health_generate.remote()``.
    The gauge ``miles_rollout_cell_alive`` is set to 1.0 (healthy) or 0.0 (unhealthy).
    """

    def __init__(
        self,
        *,
        cells: list[CellEntry],
        session_id: str,
        run_name: str,
        check_interval: float = 30.0,
        timeout: float = 30.0,
    ) -> None:
        self._cells = {entry.cell_id: entry for entry in cells}
        self._session_id = session_id
        self._run_name = run_name
        self._check_interval = check_interval
        self._timeout = timeout
        self._paused = False
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the health check loop. Must be called from an async context."""
        if self._task is not None:
            return
        self._task = asyncio.ensure_future(self._loop())
        logger.info("rollout cell health checker started: num_cells=%d", len(self._cells))

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    async def shutdown(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def _loop(self) -> None:
        try:
            while True:
                if not self._paused:
                    await asyncio.gather(
                        *(
                            _check_one_cell(
                                entry=e, session_id=self._session_id, run_name=self._run_name, timeout=self._timeout
                            )
                            for e in self._cells.values()
                        )
                    )
                await asyncio.sleep(self._check_interval)
        except asyncio.CancelledError:
            raise

    async def is_paused(self) -> bool:
        return self._paused


async def _check_one_cell(*, entry: CellEntry, session_id: str, run_name: str, timeout: float) -> None:
    is_healthy = False
    try:
        is_healthy = await _probe_cell(engines=entry.get_engines(), timeout=timeout)
    except Exception:
        logger.warning("Health probe failed for cell %s", entry.cell_id, exc_info=True)

    _report(session_id=session_id, run_name=run_name, cell_id=entry.cell_id, is_healthy=is_healthy)


async def _probe_cell(*, engines: list[object], timeout: float) -> bool:
    if not engines:
        return False

    lead_engine = engines[0]
    if lead_engine is None:
        return False

    await asyncio.wait_for(lead_engine.health_generate.remote(), timeout=timeout)  # type: ignore[union-attr]
    return True


def _report(*, session_id: str, run_name: str, cell_id: str, is_healthy: bool) -> None:
    set_prometheus_gauge(
        name=_METRIC_NAME,
        label_keys=_LABEL_KEYS,
        label_values=[session_id, run_name, cell_id],
        value=1.0 if is_healthy else 0.0,
    )
