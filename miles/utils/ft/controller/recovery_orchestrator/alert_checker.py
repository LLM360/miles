from __future__ import annotations

from miles.utils.ft.controller.detectors.hardware_checks import check_all_hardware_faults
from miles.utils.ft.controller.metrics.protocol import MetricStoreProtocol
from miles.utils.ft.models import unique_node_ids


class AlertChecker:
    def __init__(self, metric_store: MetricStoreProtocol) -> None:
        self._metric_store = metric_store

    def check_alerts(self) -> tuple[list[str], list[str]]:
        """Return (sorted bad_node_ids, reasons)."""
        faults = check_all_hardware_faults(self._metric_store)
        bad_node_ids = sorted(unique_node_ids(faults))
        reasons = [f.reason for f in faults]
        return bad_node_ids, reasons
