from typing import Literal

from miles.utils.ft.models.base import FtBaseModel


class MetricSample(FtBaseModel):
    name: str
    labels: dict[str, str]
    value: float
    metric_type: Literal["gauge", "counter"] = "gauge"


class CollectorOutput(FtBaseModel):
    metrics: list[MetricSample]
