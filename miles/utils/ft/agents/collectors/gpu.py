from __future__ import annotations

import asyncio
import logging
from types import ModuleType

import miles.utils.ft.metric_names as mn
from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models import CollectorOutput, MetricSample

logger = logging.getLogger(__name__)


class GpuCollector(BaseCollector):
    def __init__(self) -> None:
        self._pynvml: ModuleType | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
        except Exception:
            logger.warning("pynvml unavailable — GpuCollector will report all GPUs as unavailable")

    async def collect(self) -> CollectorOutput:
        metrics = await asyncio.to_thread(self._collect_sync)
        return CollectorOutput(metrics=metrics)

    async def close(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                logger.warning("nvmlShutdown failed", exc_info=True)
            self._pynvml = None

    def _collect_sync(self) -> list[MetricSample]:
        pynvml = self._pynvml
        if pynvml is None:
            return []

        samples: list[MetricSample] = []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            logger.warning("nvmlDeviceGetCount failed", exc_info=True)
            return []

        for index in range(device_count):
            gpu_label = {"gpu": str(index)}

            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except Exception:
                logger.warning("Cannot get handle for GPU %d", index)
                samples.append(MetricSample(name=mn.GPU_AVAILABLE, labels=gpu_label, value=0.0))
                continue

            samples.append(MetricSample(name=mn.GPU_AVAILABLE, labels=gpu_label, value=1.0))
            self._collect_temperature(pynvml, handle, gpu_label, samples)
            self._collect_row_remap(pynvml, handle, gpu_label, samples)
            self._collect_pcie_bandwidth(pynvml, handle, gpu_label, samples)
            self._collect_utilization(pynvml, handle, gpu_label, samples)

        return samples

    @staticmethod
    def _collect_temperature(
        pynvml: ModuleType,
        handle: object,
        gpu_label: dict[str, str],
        samples: list[MetricSample],
    ) -> None:
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            samples.append(MetricSample(name=mn.DCGM_FI_DEV_GPU_TEMP, labels=gpu_label, value=float(temp)))
        except Exception:
            logger.warning("Failed to get temperature for GPU %s", gpu_label["gpu"])

    @staticmethod
    def _collect_row_remap(
        pynvml: ModuleType,
        handle: object,
        gpu_label: dict[str, str],
        samples: list[MetricSample],
    ) -> None:
        try:
            _correctable, _uncorrectable, pending, failure = pynvml.nvmlDeviceGetRemappedRows(handle)
            samples.append(MetricSample(name=mn.DCGM_FI_DEV_ROW_REMAP_PENDING, labels=gpu_label, value=float(pending)))
            samples.append(MetricSample(name=mn.DCGM_FI_DEV_ROW_REMAP_FAILURE, labels=gpu_label, value=float(failure)))
        except Exception:
            logger.warning("Failed to get row remap for GPU %s", gpu_label["gpu"])

    @staticmethod
    def _collect_pcie_bandwidth(
        pynvml: ModuleType,
        handle: object,
        gpu_label: dict[str, str],
        samples: list[MetricSample],
    ) -> None:
        try:
            throughput_kb_per_s = pynvml.nvmlDeviceGetPcieThroughput(
                handle, pynvml.NVML_PCIE_UTIL_TX_BYTES,
            )
            bytes_per_s = throughput_kb_per_s * 1024
            samples.append(MetricSample(name=mn.DCGM_FI_DEV_PCIE_TX_THROUGHPUT, labels=gpu_label, value=float(bytes_per_s)))
        except Exception:
            logger.warning("Failed to get PCIe bandwidth for GPU %s", gpu_label["gpu"])

    @staticmethod
    def _collect_utilization(
        pynvml: ModuleType,
        handle: object,
        gpu_label: dict[str, str],
        samples: list[MetricSample],
    ) -> None:
        try:
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            samples.append(MetricSample(name=mn.DCGM_FI_DEV_GPU_UTIL, labels=gpu_label, value=float(rates.gpu)))
        except Exception:
            logger.warning("Failed to get utilization for GPU %s", gpu_label["gpu"])
