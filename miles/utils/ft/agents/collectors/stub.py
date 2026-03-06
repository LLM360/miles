from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models.metrics import MetricSample


class StubCollector(BaseCollector):
    def _collect_sync(self) -> list[MetricSample]:
        return []
