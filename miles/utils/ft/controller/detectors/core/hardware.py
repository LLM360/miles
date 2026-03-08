from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.checks.hardware import check_all_hardware_faults
from miles.utils.ft.controller.types import Decision, TriggerType


class HighConfidenceHardwareDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        faults = check_all_hardware_faults(metric_store=ctx.metric_store)

        return Decision.from_node_faults(
            faults,
            fallback_reason="no high-confidence hardware faults",
            trigger=TriggerType.HARDWARE,
        )
