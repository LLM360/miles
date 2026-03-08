from datetime import datetime
from typing import Annotated, Literal, NamedTuple

from pydantic import Field

from miles.utils.ft.models.base import FtBaseModel


class StepValue(NamedTuple):
    step: int
    value: float


class TimedStepValue(NamedTuple):
    step: int
    timestamp: datetime
    value: float


class _MetricSampleBase(FtBaseModel):
    name: str
    labels: dict[str, str]


class GaugeSample(_MetricSampleBase):
    value: float
    metric_type: Literal["gauge"] = "gauge"


class CounterSample(_MetricSampleBase):
    delta: float
    metric_type: Literal["counter"] = "counter"


MetricSample = Annotated[
    GaugeSample | CounterSample,
    Field(discriminator="metric_type"),
]


class CollectorOutput(FtBaseModel):
    metrics: list[MetricSample]
