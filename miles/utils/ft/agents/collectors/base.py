from abc import ABC, abstractmethod

from miles.utils.ft.models import CollectorOutput


class BaseCollector(ABC):
    collect_interval: float = 10.0

    @abstractmethod
    async def collect(self) -> CollectorOutput:
        ...

    async def close(self) -> None:
        pass
