"""Unit tests for TrainingRankMetricExporter.

TrainingRankMetricExporter owns the Prometheus exporter and metric gauges
(iteration + phase). These tests verify gauge creation, updates, and the
HTTP exposition endpoint.
"""

from collections.abc import Iterator

import httpx
import pytest

from miles.utils.ft.agents.utils.training_rank_metric_exporter import TrainingRankMetricExporter


def _parse_gauge(text: str, metric_name: str, labels: dict[str, str]) -> float:
    """Extract a gauge value from Prometheus text exposition format."""
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if metric_name not in line:
            continue
        label_match = all(f'{k}="{v}"' in line for k, v in labels.items())
        if label_match:
            value_str = line.rsplit(" ", 1)[-1]
            return float(value_str)
    raise ValueError(f"{metric_name} with labels {labels} not found in metrics output")


@pytest.fixture()
def metric_exporter() -> Iterator[TrainingRankMetricExporter]:
    exporter = TrainingRankMetricExporter(rank=0, node_id="test-node")
    yield exporter
    exporter.shutdown()


class TestTrainingRankMetricExporterExporter:
    @pytest.mark.anyio
    async def test_exporter_returns_prometheus_format(
        self, metric_exporter: TrainingRankMetricExporter
    ) -> None:
        address = metric_exporter.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

    @pytest.mark.anyio
    async def test_exporter_address_has_port(
        self, metric_exporter: TrainingRankMetricExporter
    ) -> None:
        address = metric_exporter.get_exporter_address()
        assert address.startswith("http://localhost:")
        port = int(address.split(":")[-1])
        assert port > 0

    @pytest.mark.anyio
    async def test_initial_gauge_values(
        self, metric_exporter: TrainingRankMetricExporter
    ) -> None:
        address = metric_exporter.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        text = response.text
        assert "miles_ft_training_iteration" in text
        assert "miles_ft_training_phase" in text
        assert 'rank="0"' in text


class TestTrainingRankMetricExporterStep:
    @pytest.mark.anyio
    async def test_step_updates_iteration_gauge(
        self, metric_exporter: TrainingRankMetricExporter
    ) -> None:
        metric_exporter.step(iteration=42)

        address = metric_exporter.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        text = response.text
        assert "miles_ft_training_iteration" in text
        assert "42.0" in text

    def test_step_warns_on_non_increasing_iteration(
        self, metric_exporter: TrainingRankMetricExporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        metric_exporter.step(iteration=5)
        metric_exporter.step(iteration=5)
        assert "non-increasing iteration" in caplog.text
        assert metric_exporter._last_iteration == 5

    def test_step_warns_on_decreasing_iteration(
        self, metric_exporter: TrainingRankMetricExporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        metric_exporter.step(iteration=5)
        metric_exporter.step(iteration=3)
        assert "non-increasing iteration" in caplog.text
        assert metric_exporter._last_iteration == 5

    @pytest.mark.anyio
    async def test_step_iteration_monotonic_across_phases(
        self, metric_exporter: TrainingRankMetricExporter
    ) -> None:
        """Simulate a full rollout cycle with split set_phase/step API."""
        address = metric_exporter.get_exporter_address()
        labels = {"rank": "0"}

        metric_exporter.set_phase("training")
        for step_id in range(4):
            metric_exporter.step(iteration=step_id)

        metric_exporter.set_phase("idle")
        metric_exporter.set_phase("checkpoint_saving")
        metric_exporter.set_phase("idle")

        metric_exporter.set_phase("training")
        for step_id in range(4, 8):
            metric_exporter.step(iteration=step_id)

        metric_exporter.set_phase("idle")

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{address}/metrics")
        iteration = _parse_gauge(resp.text, "miles_ft_training_iteration", labels)
        phase = _parse_gauge(resp.text, "miles_ft_training_phase", labels)
        assert iteration == 7.0
        assert phase == 0.0

    def test_step_exception_does_not_propagate(
        self, metric_exporter: TrainingRankMetricExporter
    ) -> None:
        from unittest.mock import patch

        with patch.object(
            metric_exporter, "_iteration_child", **{"set.side_effect": RuntimeError("boom")}
        ):
            metric_exporter.step(iteration=1)


class TestTrainingRankMetricExporterSetPhase:
    @pytest.mark.anyio
    async def test_set_phase_updates_phase_gauge(
        self, metric_exporter: TrainingRankMetricExporter
    ) -> None:
        metric_exporter.set_phase("checkpoint_saving")

        address = metric_exporter.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        labels = {"rank": "0"}
        phase = _parse_gauge(response.text, "miles_ft_training_phase", labels)
        assert phase == 2.0

    @pytest.mark.anyio
    async def test_set_phase_idle_preserves_iteration(
        self, metric_exporter: TrainingRankMetricExporter
    ) -> None:
        metric_exporter.step(iteration=10)
        metric_exporter.set_phase("idle")

        address = metric_exporter.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        labels = {"rank": "0"}
        iteration = _parse_gauge(response.text, "miles_ft_training_iteration", labels)
        phase = _parse_gauge(response.text, "miles_ft_training_phase", labels)
        assert iteration == 10.0
        assert phase == 0.0
