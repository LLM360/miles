"""Backend conformance: MiniPrometheus and real Prometheus must behave
identically when both scrape the same HTTP /metrics endpoint.

Every test is parametrized via the ``backend`` fixture and runs once
against MiniBackend (ScrapeLoop → HTTP GET → ingest) and once against
LiveBackend (real Prometheus binary → PrometheusClient HTTP API).
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests.fast.utils.ft.integration.prometheus_live.conftest import MetricBackend

pytestmark = pytest.mark.integration


class TestQueryLatest:
    async def test_returns_pushed_value(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_latest", 42.0)
        await backend.flush()

        df = backend.store.query_latest("conformance_latest")

        assert not df.is_empty()
        assert "__name__" in df.columns
        assert "value" in df.columns
        assert df["value"][0] == 42.0

    async def test_with_label_filters(self, backend: MetricBackend) -> None:
        backend.set_gauge(
            "conformance_labeled",
            1.0,
            labels={"node_id": "node-0", "device": "ib0"},
        )
        backend.set_gauge(
            "conformance_labeled",
            0.0,
            labels={"node_id": "node-0", "device": "ib1"},
        )
        await backend.flush()

        df = backend.store.query_latest(
            "conformance_labeled",
            label_filters={"device": "ib0"},
        )

        assert df.shape[0] == 1
        assert df["value"][0] == 1.0

    async def test_empty_for_nonexistent_metric(self, backend: MetricBackend) -> None:
        df = backend.store.query_latest("conformance_no_such_metric_xyz")

        assert df.is_empty()
        assert "__name__" in df.columns
        assert "value" in df.columns


class TestQueryRange:
    async def test_returns_data_in_window(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_range", 1.0)
        await backend.flush()

        backend.set_gauge("conformance_range", 2.0)
        await backend.flush()

        df = backend.store.query_range(
            "conformance_range",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert "__name__" in df.columns
        assert "timestamp" in df.columns
        assert "value" in df.columns


class TestRangeAggregations:
    async def test_changes_detects_value_change(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_changes", 1.0)
        await backend.flush()

        backend.set_gauge("conformance_changes", 2.0)
        await backend.flush()

        df = backend.store.changes(
            "conformance_changes",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert df["value"][0] >= 1.0

    async def test_count_over_time_counts_samples(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_count", 10.0)
        await backend.flush()

        backend.set_gauge("conformance_count", 20.0)
        await backend.flush()

        df = backend.store.count_over_time(
            "conformance_count",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        assert df["value"][0] >= 1.0

    async def test_avg_over_time_computes_average(self, backend: MetricBackend) -> None:
        backend.set_gauge("conformance_avg", 1.0)
        await backend.flush()

        backend.set_gauge("conformance_avg", 3.0)
        await backend.flush()

        df = backend.store.avg_over_time(
            "conformance_avg",
            window=timedelta(minutes=5),
        )

        assert not df.is_empty()
        avg = df["value"][0]
        assert 1.0 <= avg <= 3.0
