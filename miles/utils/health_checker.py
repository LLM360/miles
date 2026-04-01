from __future__ import annotations

import abc
import argparse
import asyncio
import logging
from collections.abc import Callable, Coroutine
from enum import StrEnum, auto
from typing import Any

from miles.utils.clock import Clock, RealClock
from miles.utils.pydantic_utils import StrictBaseModel

logger = logging.getLogger(__name__)


class SimpleHealthCheckerConfig(StrictBaseModel):
    interval: float = 30.0
    timeout: float = 10.0
    staleness: float = 90.0
    first_wait: float = 0.0

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser, *, prefix: str) -> None:
        parser.add_argument(
            f"--{prefix}-interval",
            type=float,
            default=30.0,
            help=f"Interval in seconds between {prefix} health checks.",
        )
        parser.add_argument(
            f"--{prefix}-timeout",
            type=float,
            default=10.0,
            help=f"Timeout in seconds for a single {prefix} health check RPC.",
        )
        parser.add_argument(
            f"--{prefix}-staleness",
            type=float,
            default=90.0,
            help=f"Maximum allowed staleness (seconds) before marking as errored.",
        )
        parser.add_argument(
            f"--{prefix}-first-wait",
            type=float,
            default=300.0,
            help=f"Initial grace period (seconds) before starting {prefix} health checks.",
        )

    @staticmethod
    def from_args(args: object, *, prefix: str) -> SimpleHealthCheckerConfig:
        attr_prefix = prefix.replace("-", "_")
        return SimpleHealthCheckerConfig(
            interval=getattr(args, f"{attr_prefix}_interval"),
            timeout=getattr(args, f"{attr_prefix}_timeout"),
            staleness=getattr(args, f"{attr_prefix}_staleness"),
            first_wait=getattr(args, f"{attr_prefix}_first_wait"),
        )


class HealthStatus(StrEnum):
    HEALTHY = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


class BaseHealthChecker(abc.ABC):
    @property
    @abc.abstractmethod
    def status(self) -> HealthStatus: ...

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def pause(self) -> None: ...

    @abc.abstractmethod
    def resume(self) -> None: ...


class SimpleHealthChecker(BaseHealthChecker):
    """Periodic async health checker. Calls *check_fn*; reports result via *on_result*.

    After each ``resume()``, waits ``first_wait`` seconds before the first check
    (matching ``RolloutHealthMonitor._need_first_wait`` semantics).
    """

    def __init__(
        self,
        *,
        name: str,
        check_fn: Callable[[], Coroutine[Any, Any, None]],
        on_result: Callable[[bool], None] | None = None,
        interval: float,
        first_wait: float = 0.0,
        clock: Clock | None = None,
    ) -> None:
        self._name = name
        self._check_fn = check_fn
        self._on_result = on_result
        self._interval = interval
        self._first_wait = first_wait
        self._clock = clock or RealClock()

        self._status = HealthStatus.UNKNOWN
        self._paused: bool = False
        self._need_first_wait: bool = True
        self._task: asyncio.Task[None] | None = None

    @property
    def status(self) -> HealthStatus:
        return self._status

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._status = HealthStatus.UNKNOWN

    def pause(self) -> None:
        self._paused = True
        self._status = HealthStatus.UNKNOWN

    def resume(self) -> None:
        self._paused = False
        self._need_first_wait = True
        self._status = HealthStatus.UNKNOWN

    async def _loop(self) -> None:
        while True:
            if self._need_first_wait:
                self._need_first_wait = False
                await self._clock.sleep(self._first_wait)

            if not self._paused:
                success = False
                try:
                    await self._check_fn()
                    success = True
                except Exception:
                    logger.error(f"Health check failed for {self._name}", exc_info=True)

                self._status = HealthStatus.HEALTHY if success else HealthStatus.UNHEALTHY
                if self._on_result is not None:
                    self._on_result(success)

            await self._clock.sleep(self._interval)


class NoopHealthChecker(BaseHealthChecker):
    @property
    def status(self) -> HealthStatus:
        return HealthStatus.UNKNOWN

    async def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass


# TODO: should move when Rollout FT is implemented
def create_rollout_cell_health_checker(
    *,
    cell_id: str,
    get_engines: Callable[[], list[object]],
    config: SimpleHealthCheckerConfig,
    on_result: Callable[[bool], None] | None = None,
) -> SimpleHealthChecker:

    async def _check() -> None:
        engines = get_engines()
        if not engines:
            raise RuntimeError("No engines")

        lead_engine = engines[0]
        if lead_engine is None:
            raise RuntimeError("Lead engine is None")

        await asyncio.wait_for(lead_engine.health_generate.remote(), timeout=config.timeout)

    return SimpleHealthChecker(
        name=f"rollout-cell-{cell_id}",
        check_fn=_check,
        on_result=on_result,
        interval=config.interval,
    )
