import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from miles.utils.pydantic_utils import StrictBaseModel

logger = logging.getLogger(__name__)


class CellSnapshot(StrictBaseModel):
    name: str
    healthy_status: str
    healthy_reason: str | None


@dataclass
class _CellBackoff:
    consecutive_failures: int = 0
    next_attempt_at: float = 0.0
    given_up: bool = False


class MiniFTController:
    def __init__(
        self,
        *,
        get_cells: Callable[[], Awaitable[list[CellSnapshot]]],
        suspend_cell: Callable[[str], Awaitable[None]],
        resume_cell: Callable[[str], Awaitable[None]],
        poll_interval: float = 10.0,
        resume_delay: float = 5.0,
        max_consecutive_failures: int = 5,
    ) -> None:
        self._get_cells = get_cells
        self._suspend_cell = suspend_cell
        self._resume_cell = resume_cell
        self._poll_interval = poll_interval
        self._resume_delay = resume_delay
        self._max_consecutive_failures = max_consecutive_failures

        self._running: bool = False
        self._cell_backoffs: dict[str, _CellBackoff] = {}

    async def run(self) -> None:
        self._running = True
        while self._running:
            start = time.monotonic()
            await self._poll_and_heal()
            elapsed = time.monotonic() - start
            await asyncio.sleep(max(0, self._poll_interval - elapsed))

    def request_stop(self) -> None:
        self._running = False

    async def _poll_and_heal(self) -> None:
        try:
            cells = await self._get_cells()

            seen_cell_names: set[str] = set()
            for cell in cells:
                seen_cell_names.add(cell.name)

                if cell.healthy_status == "True":
                    continue

                if cell.healthy_reason == "Degraded":
                    logger.warning("Cell %s is Degraded, skipping heal", cell.name)
                    continue

                if cell.healthy_status == "False" and cell.healthy_reason == "Fatal":
                    backoff = self._cell_backoffs.setdefault(cell.name, _CellBackoff())

                    if backoff.given_up:
                        continue

                    now = time.monotonic()
                    if now < backoff.next_attempt_at:
                        continue

                    await self._heal(cell_name=cell.name, backoff=backoff)

            stale_keys = set(self._cell_backoffs) - seen_cell_names
            for key in stale_keys:
                del self._cell_backoffs[key]
        except Exception:
            logger.error("Error in _poll_and_heal", exc_info=True)

    async def _heal(self, *, cell_name: str, backoff: _CellBackoff) -> None:

        try:
            logger.info("Healing cell %s: suspending", cell_name)
            await self._suspend_cell(cell_name)

            await asyncio.sleep(self._resume_delay)

            logger.info("Healing cell %s: resuming", cell_name)
            await self._resume_cell(cell_name)

            backoff.consecutive_failures = 0
            backoff.next_attempt_at = 0.0
            logger.info("Successfully healed cell %s", cell_name)
        except Exception:
            backoff.consecutive_failures += 1
            delay = min(5 * (2 ** backoff.consecutive_failures), 300)
            backoff.next_attempt_at = time.monotonic() + delay

            if backoff.consecutive_failures >= self._max_consecutive_failures:
                backoff.given_up = True
                logger.error(
                    "Giving up on cell %s after %d consecutive failures",
                    cell_name,
                    backoff.consecutive_failures,
                    exc_info=True,
                )
            else:
                logger.warning(
                    "Failed to heal cell %s (attempt %d), next attempt in %.0fs",
                    cell_name,
                    backoff.consecutive_failures,
                    delay,
                    exc_info=True,
                )
