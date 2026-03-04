from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.prometheus_exporter import PrometheusExporter
from miles.utils.ft.models import DiagnosticResult

logger = logging.getLogger(__name__)


class FtNodeAgent:
    def __init__(
        self,
        node_id: str,
        collectors: list[BaseCollector] | None = None,
        collect_interval_seconds: float | None = None,
    ) -> None:
        self._node_id = node_id
        self._collectors = collectors or []
        self._stopped = False

        if collect_interval_seconds is not None:
            for collector in self._collectors:
                collector.collect_interval = collect_interval_seconds

        self._exporter = PrometheusExporter()
        self._collector_tasks: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_exporter_address(self) -> str:
        return self._exporter.get_address()

    async def start(self) -> None:
        if self._stopped or self._collector_tasks:
            return

        loop = asyncio.get_running_loop()
        for collector in self._collectors:
            task = loop.create_task(self._run_single_collector(collector))
            self._collector_tasks.append(task)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        for task in self._collector_tasks:
            task.cancel()
        await asyncio.gather(*self._collector_tasks, return_exceptions=True)
        self._collector_tasks.clear()

        for collector in self._collectors:
            try:
                await collector.close()
            except Exception:
                logger.warning(
                    "Collector %s.close() failed on node %s",
                    type(collector).__name__,
                    self._node_id,
                    exc_info=True,
                )

        self._exporter.shutdown()

    # ------------------------------------------------------------------
    # Stub methods (future milestones)
    # ------------------------------------------------------------------

    async def collect_logs(self) -> dict[str, str]:
        raise NotImplementedError(
            "collect_logs will be implemented in diag-framework milestone"
        )

    async def run_diagnostic(self, diagnostic_type: str) -> DiagnosticResult:
        raise NotImplementedError(
            "run_diagnostic will be implemented in diag-framework milestone"
        )

    async def cleanup_training_processes(self, training_job_id: str) -> None:
        logger.info(
            "cleanup_training_processes node_id=%s job_id=%s (stub — no-op)",
            self._node_id, training_job_id,
        )

    # ------------------------------------------------------------------
    # Per-collector background task
    # ------------------------------------------------------------------

    async def _run_single_collector(self, collector: BaseCollector) -> None:
        collector_name = type(collector).__name__
        while True:
            try:
                result = await collector.collect()
                self._exporter.update_metrics(result.metrics)
            except Exception:
                logger.warning(
                    "Collector %s failed on node %s",
                    collector_name,
                    self._node_id,
                    exc_info=True,
                )

            await asyncio.sleep(collector.collect_interval)
