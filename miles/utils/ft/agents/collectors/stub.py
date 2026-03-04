from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models import CollectorOutput


class StubCollector(BaseCollector):
    async def collect(self) -> CollectorOutput:
        return CollectorOutput(metrics=[])
