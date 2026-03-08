from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.types import MetricSample


class StubCollector(BaseCollector):
    def _collect_sync(self) -> list[MetricSample]:
        return []
