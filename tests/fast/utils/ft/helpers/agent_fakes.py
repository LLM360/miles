from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.models.metrics import MetricSample


# ---------------------------------------------------------------------------
# Collector test helpers
# ---------------------------------------------------------------------------


class TestCollector(BaseCollector):
    def __init__(
        self,
        metrics: list[MetricSample] | None = None,
        collect_interval: float = 10.0,
    ) -> None:
        self._metrics = metrics or []
        self.collect_interval = collect_interval

    def set_metrics(self, metrics: list[MetricSample]) -> None:
        self._metrics = metrics

    def _collect_sync(self) -> list[MetricSample]:
        return list(self._metrics)


class FailingCollector(BaseCollector):
    """Collector that always raises on collect. Tracks call count."""

    def __init__(self, collect_interval: float = 10.0) -> None:
        self.collect_interval = collect_interval
        self.call_count = 0

    def _collect_sync(self) -> list[MetricSample]:
        self.call_count += 1
        raise RuntimeError("collect failed")


class FailingCloseCollector(BaseCollector):
    """Collector whose close() always raises. Tracks whether close was called."""

    def __init__(self, collect_interval: float = 10.0) -> None:
        self.collect_interval = collect_interval
        self.close_called = False

    def _collect_sync(self) -> list[MetricSample]:
        return []

    async def close(self) -> None:
        self.close_called = True
        raise RuntimeError("close failed")


# ---------------------------------------------------------------------------
# HW-collector test helpers
# ---------------------------------------------------------------------------


class FakeKmsgReader:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self._consumed = False

    def read_new_lines(self) -> list[str]:
        if self._consumed:
            return []
        self._consumed = True
        return list(self._lines)

    def close(self) -> None:
        pass


def make_mock_pynvml(
    device_count: int = 8,
    temperature: int = 65,
    remap_info: tuple[int, int, int, int] = (0, 0, 0, 0),
    pcie_throughput_kb: int = 1048576,
    utilization_gpu: int = 50,
    failing_handle_indices: set[int] | None = None,
    ecc_uncorrectable: int = 0,
    retired_pages: list[object] | None = None,
    power_state: int = 0,
) -> MagicMock:
    failing = failing_handle_indices or set()
    mock = MagicMock()
    mock.NVML_TEMPERATURE_GPU = 0
    mock.NVML_PCIE_UTIL_TX_BYTES = 1
    mock.NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
    mock.NVML_VOLATILE_ECC_COUNTER_TYPE = 0
    mock.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR = 0

    mock.nvmlInit.return_value = None
    mock.nvmlShutdown.return_value = None
    mock.nvmlDeviceGetCount.return_value = device_count

    def get_handle(index: int) -> object:
        if index in failing:
            raise RuntimeError(f"GPU {index} handle failed")
        return f"handle-{index}"

    mock.nvmlDeviceGetHandleByIndex.side_effect = get_handle
    mock.nvmlDeviceGetTemperature.return_value = temperature
    mock.nvmlDeviceGetRemappedRows.return_value = remap_info
    mock.nvmlDeviceGetPcieThroughput.return_value = pcie_throughput_kb
    mock.nvmlDeviceGetUtilizationRates.return_value = SimpleNamespace(gpu=utilization_gpu)
    mock.nvmlDeviceGetTotalEccErrors.return_value = ecc_uncorrectable
    mock.nvmlDeviceGetRetiredPages.return_value = retired_pages or []
    mock.nvmlDeviceGetPowerState.return_value = power_state

    return mock


def create_sysfs_interface(
    base: Path,
    name: str,
    operstate: str = "up",
    rx_errors: int = 0,
    tx_errors: int = 0,
    rx_dropped: int = 0,
    tx_dropped: int = 0,
) -> None:
    iface_dir = base / name
    iface_dir.mkdir(parents=True, exist_ok=True)

    (iface_dir / "operstate").write_text(operstate + "\n")

    stats_dir = iface_dir / "statistics"
    stats_dir.mkdir(exist_ok=True)
    (stats_dir / "rx_errors").write_text(str(rx_errors) + "\n")
    (stats_dir / "tx_errors").write_text(str(tx_errors) + "\n")
    (stats_dir / "rx_dropped").write_text(str(rx_dropped) + "\n")
    (stats_dir / "tx_dropped").write_text(str(tx_dropped) + "\n")


# ---------------------------------------------------------------------------
# Stack trace test helpers
# ---------------------------------------------------------------------------

SAMPLE_PYSPY_OUTPUT_NORMAL = """\
Thread 0x7F1234 (active): "MainThread"
    _wait_for_data (selectors.py:451)
    select (selectors.py:469)
    _run_once (asyncio/base_events.py:1922)
    run_forever (asyncio/base_events.py:604)
Thread 0x7F5678 (idle): "WorkerThread-1"
    wait (threading.py:320)
    get (queue.py:171)
    _worker (concurrent/futures/thread.py:83)
"""

SAMPLE_PYSPY_OUTPUT_STUCK = """\
Thread 0x7F1234 (active): "MainThread"
    nccl_allreduce (nccl_ops.py:42)
    all_reduce (torch/distributed/distributed_c10d.py:1234)
    forward (model.py:100)
    train_step (train.py:50)
Thread 0x7F5678 (idle): "WorkerThread-1"
    wait (threading.py:320)
    get (queue.py:171)
    _worker (concurrent/futures/thread.py:83)
"""

SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK = """\
Thread 0x7F1234 (active): "MainThread"
    recv (socket.py:123)
    _receive_data (network.py:456)
    fetch_batch (dataloader.py:78)
Thread 0x7F5678 (idle): "WorkerThread-1"
    wait (threading.py:320)
    get (queue.py:171)
    _worker (concurrent/futures/thread.py:83)
"""


def make_rank_pids_provider(
    mapping: dict[str, dict[int, int]],
) -> Callable[[str], dict[int, int]]:
    def provider(node_id: str) -> dict[int, int]:
        return mapping.get(node_id, {})

    return provider


def make_trace_result(
    node_id: str,
    passed: bool = True,
    details: str = "trace output",
) -> DiagnosticResult:
    return DiagnosticResult(
        diagnostic_type="stack_trace",
        node_id=node_id,
        passed=passed,
        details=details,
    )
