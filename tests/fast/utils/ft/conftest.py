from datetime import timedelta

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.controller.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import CollectorOutput, DiagnosticResult, MetricSample


def make_metric(
    name: str,
    value: float,
    labels: dict[str, str] | None = None,
) -> MetricSample:
    return MetricSample(name=name, labels=labels or {}, value=value)


def make_fake_metric_store(
    metrics: list[MetricSample] | None = None,
    target_id: str = "node-0",
) -> MiniPrometheus:
    store = MiniPrometheus(config=MiniPrometheusConfig(
        retention=timedelta(minutes=60),
    ))
    if metrics:
        store.ingest_samples(target_id=target_id, samples=metrics)
    return store


def make_fake_mini_wandb(
    steps: dict[int, dict[str, float]] | None = None,
    run_id: str = "test-run",
    rank: int = 0,
) -> MiniWandb:
    wandb = MiniWandb(active_run_id=run_id)
    if steps:
        for step_num, metrics in sorted(steps.items()):
            wandb.log_step(run_id=run_id, rank=rank, step=step_num, metrics=metrics)
    return wandb


# ---------------------------------------------------------------------------
# Agent test helpers
# ---------------------------------------------------------------------------


class TestCollector(BaseCollector):
    def __init__(self, metrics: list[MetricSample] | None = None) -> None:
        self._metrics = metrics or []

    def set_metrics(self, metrics: list[MetricSample]) -> None:
        self._metrics = metrics

    async def collect(self) -> CollectorOutput:
        return CollectorOutput(metrics=self._metrics)


class FakeNodeAgent:
    def __init__(
        self,
        diagnostic_results: dict[str, DiagnosticResult] | None = None,
    ) -> None:
        self._diagnostic_results = diagnostic_results or {}
        self.cleanup_called: bool = False
        self.cleanup_job_id: str | None = None

    async def run_diagnostic(self, diagnostic_type: str) -> DiagnosticResult:
        return self._diagnostic_results[diagnostic_type]

    async def cleanup_training_processes(self, training_job_id: str) -> None:
        self.cleanup_called = True
        self.cleanup_job_id = training_job_id
