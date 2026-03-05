from __future__ import annotations

import logging
from http.server import HTTPServer

from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

import miles.utils.ft.metric_names as mn
from miles.utils.ft.models import RECOVERY_PHASE_TO_INT, RecoveryPhase
from miles.utils.ft.platform.protocols import JobStatus

logger = logging.getLogger(__name__)

_JOB_STATUS_TO_NUMERIC: dict[JobStatus, int] = {
    JobStatus.RUNNING: 1,
    JobStatus.STOPPED: 0,
    JobStatus.FAILED: -1,
    JobStatus.PENDING: 2,
}


class ControllerExporter:
    """Exposes Controller operational metrics via a dedicated Prometheus HTTP endpoint.

    Uses an isolated CollectorRegistry to avoid polluting the global REGISTRY
    shared by NodeAgent / MegatronAgent exporters.
    """

    def __init__(
        self,
        port: int = 9400,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self._port = port
        self._registry = registry or CollectorRegistry()
        self._httpd: HTTPServer | None = None

        self._mode = Gauge(
            mn.CONTROLLER_MODE,
            "Controller mode (0=monitoring, 1=recovery)",
            registry=self._registry,
        )
        self._tick_count = Counter(
            mn.CONTROLLER_TICK_COUNT,
            "Cumulative tick count",
            registry=self._registry,
        )
        self._recovery_phase = Gauge(
            mn.CONTROLLER_RECOVERY_PHASE,
            "Recovery phase encoding (0=none, 1=check_alerts, 2=reattempting, ...)",
            registry=self._registry,
        )

        self._training_job_status = Gauge(
            mn.TRAINING_JOB_STATUS,
            "Training job status (-1=FAILED, 0=STOPPED, 1=RUNNING, 2=PENDING)",
            registry=self._registry,
        )
        self._training_loss_latest = Gauge(
            mn.TRAINING_LOSS_LATEST,
            "Latest training loss from rank 0",
            registry=self._registry,
        )
        self._training_mfu_latest = Gauge(
            mn.TRAINING_MFU_LATEST,
            "Latest training MFU from rank 0",
            registry=self._registry,
        )

    @property
    def address(self) -> str:
        return f"http://localhost:{self._port}"

    def start(self) -> None:
        self._httpd, _thread = start_http_server(port=self._port, registry=self._registry)
        logger.info("controller_exporter_started port=%d", self._port)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
            logger.info("controller_exporter_stopped")

    def update_mode(self, *, is_recovery: bool) -> None:
        self._mode.set(1 if is_recovery else 0)

    def update_tick_count(self) -> None:
        self._tick_count.inc()

    def update_recovery_phase(self, phase: RecoveryPhase | None) -> None:
        self._recovery_phase.set(RECOVERY_PHASE_TO_INT.get(phase, 0) if phase else 0)

    def update_training_job_status(self, status: JobStatus) -> None:
        self._training_job_status.set(_JOB_STATUS_TO_NUMERIC.get(status, 0))

    def update_training_metrics(
        self,
        loss: float | None,
        mfu: float | None,
    ) -> None:
        if loss is not None:
            self._training_loss_latest.set(loss)
        if mfu is not None:
            self._training_mfu_latest.set(mfu)
