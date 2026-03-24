from __future__ import annotations

import socket
import time
from argparse import Namespace
from typing import Any

import httpx
import pytest
import ray

import miles.utils.prometheus_utils as prometheus_mod
from miles.utils.prometheus_utils import get_prometheus, init_prometheus


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _fetch_metrics(port: int) -> str:
    return httpx.get(f"http://localhost:{port}/metrics", timeout=5).text


def _make_args(port: int, run_name: str = "test-run") -> Namespace:
    return Namespace(
        prometheus_port=port,
        prometheus_run_name=run_name,
        wandb_group=None,
    )


def _wait_for_server(port: int, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _fetch_metrics(port)
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"Prometheus HTTP server on port {port} not ready after {timeout}s")


@pytest.fixture(scope="module")
def ray_context() -> Any:
    ray.init(num_cpus=2)
    yield
    ray.shutdown()


@pytest.fixture()
def prometheus_server(ray_context: Any) -> Any:
    port = _find_free_port()
    init_prometheus(_make_args(port=port), start_server=True)
    _wait_for_server(port)

    yield port

    try:
        actor = ray.get_actor("miles_prometheus_collector")
        ray.kill(actor)
    except ValueError:
        pass
    prometheus_mod._collector_handle = None

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            ray.get_actor("miles_prometheus_collector")
            time.sleep(0.1)
        except ValueError:
            break


class TestInitPrometheus:
    def test_creates_named_ray_actor(self, prometheus_server: int) -> None:
        actor = ray.get_actor("miles_prometheus_collector")
        assert actor is not None

    def test_start_server_false_discovers_existing_actor(self, prometheus_server: int) -> None:
        prometheus_mod._collector_handle = None
        assert get_prometheus() is None

        init_prometheus(_make_args(port=0, run_name="ignored"), start_server=False)

        assert get_prometheus() is not None

    def test_ping_via_ray_remote(self, prometheus_server: int) -> None:
        handle = get_prometheus()
        assert handle is not None
        assert ray.get(handle.ping.remote()) is True


class TestSetGaugeViaHttp:
    def test_set_gauge_visible(self, prometheus_server: int) -> None:
        handle = get_prometheus()
        ray.get(handle.set_gauge.remote("test_http_sg", 42.0))

        body = _fetch_metrics(prometheus_server)
        assert 'test_http_sg{run_name="test-run"} 42.0' in body

    def test_update_visible(self, prometheus_server: int) -> None:
        handle = get_prometheus()
        ray.get(handle.update.remote({"loss": 0.5}))

        body = _fetch_metrics(prometheus_server)
        assert 'miles_metric_loss{run_name="test-run"} 0.5' in body

    def test_extra_labels_merged_with_run_name(self, prometheus_server: int) -> None:
        handle = get_prometheus()
        ray.get(handle.set_gauge.remote("test_cell_alive", 1.0, extra_labels={"cell_id": "c0"}))

        body = _fetch_metrics(prometheus_server)
        assert 'test_cell_alive{cell_id="c0",run_name="test-run"} 1.0' in body

    def test_multiple_extra_label_values_independent(self, prometheus_server: int) -> None:
        handle = get_prometheus()
        ray.get(handle.set_gauge.remote("test_multi_cell", 1.0, extra_labels={"cell_id": "c0"}))
        ray.get(handle.set_gauge.remote("test_multi_cell", 0.0, extra_labels={"cell_id": "c1"}))

        body = _fetch_metrics(prometheus_server)
        assert 'test_multi_cell{cell_id="c0",run_name="test-run"} 1.0' in body
        assert 'test_multi_cell{cell_id="c1",run_name="test-run"} 0.0' in body

    def test_update_overwrites_previous_value(self, prometheus_server: int) -> None:
        handle = get_prometheus()
        ray.get(handle.set_gauge.remote("test_overwrite", 1.0))
        ray.get(handle.set_gauge.remote("test_overwrite", 99.0))

        body = _fetch_metrics(prometheus_server)
        assert 'test_overwrite{run_name="test-run"} 99.0' in body

    def test_update_sanitizes_special_chars(self, prometheus_server: int) -> None:
        handle = get_prometheus()
        ray.get(handle.update.remote({"train/loss": 1.0, "grad-norm": 2.0, "lr@step": 3.0}))

        body = _fetch_metrics(prometheus_server)
        assert 'miles_metric_train_loss{run_name="test-run"} 1.0' in body
        assert 'miles_metric_grad_norm{run_name="test-run"} 2.0' in body
        assert 'miles_metric_lr_at_step{run_name="test-run"} 3.0' in body

    def test_update_skips_non_numeric(self, prometheus_server: int) -> None:
        handle = get_prometheus()
        ray.get(handle.update.remote({"good": 1.0, "bad": "hello"}))

        body = _fetch_metrics(prometheus_server)
        assert "miles_metric_good" in body
        assert "miles_metric_bad" not in body
