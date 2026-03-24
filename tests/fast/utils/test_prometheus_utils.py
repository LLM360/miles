from unittest.mock import MagicMock, patch

import pytest

from miles.utils.prometheus_utils import _PrometheusCollector, set_prometheus_gauge


@pytest.fixture()
def collector() -> _PrometheusCollector:
    with patch("prometheus_client.start_http_server"):
        return _PrometheusCollector(prometheus_port=9090, run_name="test-run")


class TestSetGaugeWithLabels:
    def test_set_gauge_with_labels_creates_gauge(self, collector: _PrometheusCollector) -> None:
        """First call with a new name should create the gauge and set its value."""
        collector.set_gauge_with_labels(
            name="miles_rollout_cell_alive",
            label_keys=["session_id", "cell_id"],
            label_values=["sess-1", "cell-0"],
            value=1.0,
        )

        assert "miles_rollout_cell_alive" in collector._custom_gauges

    def test_set_gauge_with_labels_updates_existing(self, collector: _PrometheusCollector) -> None:
        """Calling twice with the same name reuses the gauge (no KeyError on second call)."""
        collector.set_gauge_with_labels(
            name="my_gauge",
            label_keys=["cell_id"],
            label_values=["cell-0"],
            value=1.0,
        )
        collector.set_gauge_with_labels(
            name="my_gauge",
            label_keys=["cell_id"],
            label_values=["cell-0"],
            value=0.0,
        )

        assert len(collector._custom_gauges) == 1

    def test_set_gauge_with_labels_different_label_values(self, collector: _PrometheusCollector) -> None:
        """Same gauge name but different label values should work (multi-cell scenario)."""
        collector.set_gauge_with_labels(
            name="miles_rollout_cell_alive",
            label_keys=["session_id", "cell_id"],
            label_values=["sess-1", "cell-0"],
            value=1.0,
        )
        collector.set_gauge_with_labels(
            name="miles_rollout_cell_alive",
            label_keys=["session_id", "cell_id"],
            label_values=["sess-1", "cell-1"],
            value=0.0,
        )

        assert len(collector._custom_gauges) == 1


class TestSetPrometheusGauge:
    @patch("miles.utils.prometheus_utils.get_prometheus", return_value=None)
    def test_set_prometheus_gauge_no_collector_is_noop(self, _mock_get: MagicMock) -> None:
        """When no collector is initialised, set_prometheus_gauge is a silent no-op."""
        set_prometheus_gauge(
            name="miles_rollout_cell_alive",
            label_keys=["session_id", "cell_id"],
            label_values=["sess-1", "cell-0"],
            value=1.0,
        )
