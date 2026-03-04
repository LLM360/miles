import asyncio
from abc import ABC, abstractmethod

from miles.utils.ft.models import CollectorOutput, MetricSample


class BaseCollector(ABC):
    collect_interval: float = 10.0

    async def collect(self) -> CollectorOutput:
        metrics = await asyncio.to_thread(self._collect_sync)
        return CollectorOutput(metrics=metrics)

    @abstractmethod
    def _collect_sync(self) -> list[MetricSample]:
        ...

    async def close(self) -> None:
        pass
