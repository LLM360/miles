from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.hang import HangDetector
from miles.utils.ft.controller.detectors.hardware import HighConfidenceHardwareDetector
from miles.utils.ft.controller.detectors.mfu_decline import MfuDeclineDetector
from miles.utils.ft.controller.detectors.nan_loss import NanLossDetector
from miles.utils.ft.controller.detectors.network import NetworkAlertDetector
from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector

__all__ = [
    "BaseFaultDetector",
    "DetectorContext",
    "HangDetector",
    "HighConfidenceHardwareDetector",
    "MfuDeclineDetector",
    "NanLossDetector",
    "NetworkAlertDetector",
    "TrainingCrashDetector",
    "build_detector_chain",
]


def build_detector_chain() -> list[BaseFaultDetector]:
    """Build the default detector chain in priority order (highest first)."""
    return [
        HighConfidenceHardwareDetector(),
        NetworkAlertDetector(),
        TrainingCrashDetector(),
        HangDetector(),
        NanLossDetector(),
        MfuDeclineDetector(),
    ]
