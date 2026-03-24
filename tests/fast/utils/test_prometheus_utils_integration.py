from __future__ import annotations

import logging
import socket
import time
import urllib.request
from argparse import Namespace
from typing import Any

import pytest
import ray

import miles.utils.prometheus_utils as prometheus_mod
from miles.utils.prometheus_utils import (
    _PrometheusCollector,
    get_prometheus,
    init_prometheus,
    set_prometheus_gauge,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _fetch_metrics(port: int) -> str:
    return urllib.request.urlopen(
        f"http://localhost:{port}/metrics", timeout=5
    ).read().decode()


def _make_args(port: int, run_name: str = "test-run") -> Namespace:
    return Namespace(
        prometheus_port=port,
        prometheus_run_name=run_name,
        wandb_group=None,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ray_context() -> Any:
    ray.init(num_cpus=2)
    yield
    ray.shutdown()


@pytest.fixture()
def prometheus_port() -> int:
    return _find_free_port()


@pytest.fixture()
def prometheus_server(ray_context: Any, prometheus_port: int) -> Any:
    args = _make_args(port=prometheus_port)
    init_prometheus(args, start_server=True)
    yield prometheus_port

    # Teardown: kill the named actor and reset the module-level handle
    try:
        actor = ray.get_actor("miles_prometheus_collector")
        ray.kill(actor)
    except ValueError:
        pass
    prometheus_mod._collector_handle = None

    # Wait briefly for the actor to be fully removed so the next test
    # can re-create it on a fresh port without name collisions.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            ray.get_actor("miles_prometheus_collector")
            time.sleep(0.1)
        except ValueError:
            break


def _wait_for_server(port: int, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _fetch_metrics(port)
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"Prometheus HTTP server on port {port} not ready after {timeout}s")


# ---------------------------------------------------------------------------
# Tests — init_prometheus
# ---------------------------------------------------------------------------


class TestInitPrometheus:
    def test_start_server_creates_named_ray_actor(self, prometheus_server: int) -> None:
        actor = ray.get_actor("miles_prometheus_collector")
        assert actor is not None

    def test_start_server_false_discovers_existing_actor(self, prometheus_server: int) -> None:
        prometheus_mod._collector_handle = None
        assert get_prometheus() is None

        init_prometheus(
            _make_args(port=0, run_name="ignored"),
            start_server=False,
        )

        assert get_prometheus() is not None

    def test_ping_via_ray_remote(self, prometheus_server: int) -> None:
        handle = get_prometheus()
        assert handle is not None
        assert ray.get(handle.ping.remote()) is True


# ---------------------------------------------------------------------------
# Tests — HTTP metrics endpoint
# ---------------------------------------------------------------------------


class TestMetricsViaHttp:
    def test_set_gauge_visible_on_http_endpoint(self, prometheus_server: int) -> None:
        _wait_for_server(prometheus_server)
        handle = get_prometheus()
        assert handle is not None

        ray.get(handle.set_gauge.remote("test_http_sg", 42.0))
        body = _fetch_metrics(prometheus_server)

        assert 'test_http_sg{run_name="test-run"} 42.0' in body

    def test_update_visible_on_http_endpoint(self, prometheus_server: int) -> None:
        _wait_for_server(prometheus_server)
        handle = get_prometheus()
        assert handle is not None

        ray.get(handle.update.remote({"loss": 0.5}))
        body = _fetch_metrics(prometheus_server)

        assert 'miles_metric_loss{run_name="test-run"} 0.5' in body

    def test_set_gauge_with_labels_visible_on_http_endpoint(self, prometheus_server: int) -> None:
        _wait_for_server(prometheus_server)
        handle = get_prometheus()
        assert handle is not None

        ray.get(handle.set_gauge_with_labels.remote(
            "test_cell_alive", ["cell_id"], ["c0"], 1.0,
        ))
        body = _fetch_metrics(prometheus_server)

        assert 'test_cell_alive{cell_id="c0"} 1.0' in body

    def test_multiple_label_values_all_visible(self, prometheus_server: int) -> None:
        _wait_for_server(prometheus_server)
        handle = get_prometheus()
        assert handle is not None

        ray.get(handle.set_gauge_with_labels.remote(
            "test_cell_alive", ["cell_id"], ["c0"], 1.0,
        ))
        ray.get(handle.set_gauge_with_labels.remote(
            "test_cell_alive", ["cell_id"], ["c1"], 0.0,
        ))
        body = _fetch_metrics(prometheus_server)

        assert 'test_cell_alive{cell_id="c0"} 1.0' in body
        assert 'test_cell_alive{cell_id="c1"} 0.0' in body


# ---------------------------------------------------------------------------
# Tests — set_prometheus_gauge end-to-end
# ---------------------------------------------------------------------------


class TestSetPrometheusGaugeEndToEnd:
    def test_end_to_end_via_http(self, prometheus_server: int) -> None:
        _wait_for_server(prometheus_server)

        set_prometheus_gauge(
            name="test_e2e_gauge",
            label_keys=["k"],
            label_values=["v"],
            value=99.0,
        )
        body = _fetch_metrics(prometheus_server)

        assert 'test_e2e_gauge{k="v"} 99.0' in body

    def test_noop_without_init(self) -> None:
        original = prometheus_mod._collector_handle
        try:
            prometheus_mod._collector_handle = None
            set_prometheus_gauge(
                name="should_not_exist",
                label_keys=["x"],
                label_values=["y"],
                value=1.0,
            )
        finally:
            prometheus_mod._collector_handle = original
