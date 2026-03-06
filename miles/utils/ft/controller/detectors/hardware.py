from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.hardware_checks import (
    DISK_AVAILABLE_THRESHOLD_BYTES,
    check_all_hardware_faults,
)
from miles.utils.ft.models.fault import Decision


class HighConfidenceHardwareDetector(BaseFaultDetector):
    is_critical = True

    def __init__(
        self,
        disk_available_threshold_bytes: float = DISK_AVAILABLE_THRESHOLD_BYTES,
    ) -> None:
        self._disk_available_threshold_bytes = disk_available_threshold_bytes

    def evaluate(self, ctx: DetectorContext) -> Decision:
        faults = check_all_hardware_faults(
            metric_store=ctx.metric_store,
            disk_available_threshold_bytes=self._disk_available_threshold_bytes,
        )

        return Decision.from_node_faults(
            faults,
            fallback_reason="no high-confidence hardware faults",
        )
