from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

import miles.utils.prometheus_utils as prometheus_mod
from miles.utils.prometheus_utils import _PrometheusCollector, set_prometheus_gauge


def _make_args(
    prometheus_port: int = 9090,
    prometheus_run_name: str | None = None,
    wandb_group: str | None = None,
) -> Namespace:
    return Namespace(
        prometheus_port=prometheus_port,
        prometheus_run_name=prometheus_run_name,
        wandb_group=wandb_group,
    )


@pytest.fixture()
def collector() -> _PrometheusCollector:
    with patch("prometheus_client.start_http_server"):
        return _PrometheusCollector(_make_args(prometheus_run_name="test-run"))


class TestPrometheusCollectorInit:
    def test_run_name_from_prometheus_run_name(self) -> None:
        with patch("prometheus_client.start_http_server"):
            c = _PrometheusCollector(_make_args(prometheus_run_name="my-run"))
        assert c._run_name == "my-run"

    def test_run_name_falls_back_to_wandb_group(self) -> None:
        with patch("prometheus_client.start_http_server"):
            c = _PrometheusCollector(_make_args(wandb_group="wandb-group"))
        assert c._run_name == "wandb-group"

    def test_run_name_defaults_to_miles_training(self) -> None:
        with patch("prometheus_client.start_http_server"):
            c = _PrometheusCollector(_make_args())
        assert c._run_name == "miles_training"

    def test_prometheus_run_name_takes_priority_over_wandb_group(self) -> None:
        with patch("prometheus_client.start_http_server"):
            c = _PrometheusCollector(
                _make_args(prometheus_run_name="prom", wandb_group="wandb")
            )
        assert c._run_name == "prom"

    def test_starts_http_server_on_given_port(self) -> None:
        with patch("prometheus_client.start_http_server") as mock_start:
            _PrometheusCollector(_make_args(prometheus_port=8888))
        mock_start.assert_called_once_with(8888)

    def test_ping_returns_true(self, collector: _PrometheusCollector) -> None:
        assert collector.ping() is True


class TestSetGauge:
    def test_creates_gauge_on_first_call(self, collector: _PrometheusCollector) -> None:
        collector.set_gauge("my_metric", 42.0)
        assert "my_metric" in collector._gauges

    def test_reuses_gauge_on_second_call(self, collector: _PrometheusCollector) -> None:
        collector.set_gauge("my_metric", 1.0)
        collector.set_gauge("my_metric", 2.0)
        assert len(collector._gauges) == 1

    def test_gauge_uses_run_name_label(self, collector: _PrometheusCollector) -> None:
        collector.set_gauge("my_metric", 1.0)
        gauge = collector._gauges["my_metric"]
        assert gauge._labelnames == ("run_name",)


class TestUpdate:
    def test_adds_prefix_and_sets_gauge(self, collector: _PrometheusCollector) -> None:
        collector.update({"loss": 0.5, "mfu": 0.3})
        assert "miles_metric_loss" in collector._gauges
        assert "miles_metric_mfu" in collector._gauges

    def test_skips_non_numeric_values(self, collector: _PrometheusCollector) -> None:
        collector.update({"loss": 0.5, "name": "hello"})
        assert "miles_metric_loss" in collector._gauges
        assert "miles_metric_name" not in collector._gauges

    def test_sanitizes_slash(self, collector: _PrometheusCollector) -> None:
        collector.update({"train/loss": 1.0})
        assert "miles_metric_train_loss" in collector._gauges

    def test_sanitizes_dash(self, collector: _PrometheusCollector) -> None:
        collector.update({"grad-norm": 2.0})
        assert "miles_metric_grad_norm" in collector._gauges

    def test_sanitizes_at_sign(self, collector: _PrometheusCollector) -> None:
        collector.update({"lr@step": 3.0})
        assert "miles_metric_lr_at_step" in collector._gauges

    def test_int_values_accepted(self, collector: _PrometheusCollector) -> None:
        collector.update({"step": 100})
        assert "miles_metric_step" in collector._gauges

    def test_empty_dict_is_noop(self, collector: _PrometheusCollector) -> None:
        collector.update({})
        assert len(collector._gauges) == 0


class TestSetGaugeWithLabels:
    def test_creates_custom_gauge_on_first_call(
        self, collector: _PrometheusCollector
    ) -> None:
        collector.set_gauge_with_labels(
            name="miles_rollout_cell_alive",
            label_keys=["session_id", "cell_id"],
            label_values=["sess-1", "cell-0"],
            value=1.0,
        )
        assert "miles_rollout_cell_alive" in collector._custom_gauges

    def test_reuses_gauge_on_second_call(
        self, collector: _PrometheusCollector
    ) -> None:
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

    def test_different_label_values_same_gauge(
        self, collector: _PrometheusCollector
    ) -> None:
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

    def test_does_not_interfere_with_regular_gauges(
        self, collector: _PrometheusCollector
    ) -> None:
        collector.set_gauge("regular_metric", 1.0)
        collector.set_gauge_with_labels(
            name="custom_metric",
            label_keys=["k"],
            label_values=["v"],
            value=2.0,
        )

        assert "regular_metric" in collector._gauges
        assert "regular_metric" not in collector._custom_gauges
        assert "custom_metric" in collector._custom_gauges
        assert "custom_metric" not in collector._gauges


class TestSetPrometheusGauge:
    @patch.object(prometheus_mod, "_collector_handle", None)
    def test_noop_when_collector_is_none(self) -> None:
        set_prometheus_gauge(
            name="miles_rollout_cell_alive",
            label_keys=["cell_id"],
            label_values=["cell-0"],
            value=1.0,
        )

    def test_happy_path_calls_remote(self) -> None:
        mock_handle = MagicMock()
        mock_remote_ref = MagicMock()
        mock_handle.set_gauge_with_labels.remote.return_value = mock_remote_ref

        with (
            patch.object(prometheus_mod, "_collector_handle", mock_handle),
            patch.object(prometheus_mod.ray, "get") as mock_ray_get,
        ):
            set_prometheus_gauge(
                name="miles_rollout_cell_alive",
                label_keys=["session_id", "cell_id"],
                label_values=["sess-1", "cell-0"],
                value=1.0,
            )

        mock_handle.set_gauge_with_labels.remote.assert_called_once_with(
            "miles_rollout_cell_alive",
            ["session_id", "cell_id"],
            ["sess-1", "cell-0"],
            1.0,
        )
        mock_ray_get.assert_called_once_with(mock_remote_ref)

    def test_swallows_exception_and_logs_warning(self) -> None:
        mock_handle = MagicMock()
        mock_handle.set_gauge_with_labels.remote.side_effect = RuntimeError(
            "ray down"
        )

        with (
            patch.object(prometheus_mod, "_collector_handle", mock_handle),
            patch.object(prometheus_mod, "logger") as mock_logger,
        ):
            set_prometheus_gauge(
                name="my_gauge",
                label_keys=["k"],
                label_values=["v"],
                value=1.0,
            )

        mock_logger.warning.assert_called_once()
        assert "my_gauge" in mock_logger.warning.call_args[0][1]
