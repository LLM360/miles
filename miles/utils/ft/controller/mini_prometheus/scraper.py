from __future__ import annotations

import re

from miles.utils.ft.models import MetricSample

_METRIC_LINE_RE = re.compile(r"^([\w:]+)(\{([^}]*)\})?\s+(.+?)(\s+\d+)?$")


def parse_prometheus_text(text: str) -> list[MetricSample]:
    samples: list[MetricSample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        line_match = _METRIC_LINE_RE.match(line)
        if not line_match:
            continue

        name = line_match.group(1)
        labels_str = line_match.group(3) or ""
        value_str = line_match.group(4)

        labels: dict[str, str] = {}
        if labels_str:
            for pair in labels_str.split(","):
                pair = pair.strip()
                if "=" in pair:
                    key, val = pair.split("=", 1)
                    labels[key.strip()] = val.strip().strip('"')

        try:
            value = float(value_str)
        except ValueError:
            continue

        samples.append(MetricSample(name=name, labels=labels, value=value))

    return samples
