from tests.fast.utils.ft.utils import inject_disk_fault, make_detector_context, make_fake_metric_store

from miles.utils.ft.controller.detectors.core.disk_space import DiskSpaceLowDetector
from miles.utils.ft.controller.types import ActionType


class TestDiskSpaceLowDetector:
    def test_disk_space_low_returns_notify_human(self) -> None:
        store = make_fake_metric_store()
        inject_disk_fault(store, node_id="node-0", available_bytes=500e6)
        detector = DiskSpaceLowDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NOTIFY_HUMAN
        assert "disk space low" in decision.reason
        assert decision.bad_node_ids == []

    def test_disk_space_sufficient_returns_none(self) -> None:
        store = make_fake_metric_store()
        inject_disk_fault(store, node_id="node-0", available_bytes=50e9)
        detector = DiskSpaceLowDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_empty_metric_store_returns_none(self) -> None:
        store = make_fake_metric_store()
        detector = DiskSpaceLowDetector()

        decision = detector.evaluate(make_detector_context(metric_store=store))

        assert decision.action == ActionType.NONE

    def test_has_no_is_critical_attribute(self) -> None:
        assert not hasattr(DiskSpaceLowDetector, "is_critical")
