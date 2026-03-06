from __future__ import annotations

from prometheus_client.parser import text_string_to_metric_families

from miles.utils.ft.models.metrics import MetricSample


def parse_prometheus_text(text: str) -> list[MetricSample]:
    samples: list[MetricSample] = []
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            samples.append(
                MetricSample(
                    name=sample.name,
                    labels=dict(sample.labels),
                    value=sample.value,
                )
            )
    return samples
