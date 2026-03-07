import logging
from datetime import datetime, timedelta, timezone

from pydantic import ConfigDict, field_validator

from miles.utils.ft.models.metric_names import DCGM_FI_DEV_GPU_TEMP
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.protocols.metrics import MetricQueryProtocol, TrainingMetricStoreProtocol

logger = logging.getLogger(__name__)


class MfuDeclineDetectorConfig(FtBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mfu_baseline: float = 0.0
    mfu_threshold_ratio: float = 0.8
    consecutive_steps: int = 10
    temperature_delta_threshold: float = 20.0
    decline_timeout_minutes: float = 30.0
    baseline_steps: int = 50
    mfu_absolute_minimum: float = 0.0

    @field_validator("mfu_threshold_ratio")
    @classmethod
    def _ratio_in_range(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("mfu_threshold_ratio must be in (0, 1]")
        return value

    @field_validator("consecutive_steps", "baseline_steps")
    @classmethod
    def _must_be_at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("temperature_delta_threshold", "decline_timeout_minutes")
    @classmethod
    def _must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be > 0")
        return value

    @field_validator("mfu_absolute_minimum")
    @classmethod
    def _must_be_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("mfu_absolute_minimum must be >= 0")
        return value


class MfuDeclineDetector(BaseFaultDetector):
    def __init__(self, config: MfuDeclineDetectorConfig | None = None) -> None:
        self._config = config or MfuDeclineDetectorConfig()

        self._mfu_baseline = self._config.mfu_baseline
        self._mfu_threshold_ratio = self._config.mfu_threshold_ratio
        self._consecutive_steps = self._config.consecutive_steps
        self._temperature_delta_threshold = self._config.temperature_delta_threshold
        self._decline_timeout_minutes = self._config.decline_timeout_minutes
        self._baseline_steps = self._config.baseline_steps
        self._mfu_absolute_minimum = self._config.mfu_absolute_minimum

        self._baseline_locked: bool = False
        self._locked_baseline: float | None = None

    def evaluate(self, ctx: DetectorContext) -> Decision:
        recent_mfu = ctx.mini_wandb.query_last_n_steps("mfu", last_n=self._consecutive_steps)
        if len(recent_mfu) < self._consecutive_steps:
            return Decision.no_fault(reason="insufficient MFU data")

        mfu_values = [value for _, value in recent_mfu]
        avg_mfu = sum(mfu_values) / len(mfu_values)

        if self._mfu_absolute_minimum > 0 and avg_mfu < self._mfu_absolute_minimum:
            return Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason=f"MFU {avg_mfu:.4f} below absolute minimum {self._mfu_absolute_minimum:.4f}",
                trigger=TriggerType.MISC,
            )

        baseline = self._get_baseline(ctx.mini_wandb)
        if baseline <= 0:
            return Decision.no_fault(reason="no valid MFU baseline")

        threshold = baseline * self._mfu_threshold_ratio
        mfu_stats = f"{avg_mfu:.4f} < {threshold:.4f}"

        if avg_mfu >= threshold:
            return Decision.no_fault(reason="MFU within acceptable range")

        high_temp_node = self._find_high_temperature_node(ctx.metric_store, ctx.rank_placement)
        if high_temp_node is not None:
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=[high_temp_node],
                reason=f"MFU decline ({mfu_stats}) correlated with high temperature on {high_temp_node}",
                trigger=TriggerType.HARDWARE,
            )

        elapsed_minutes = self._compute_decline_duration_minutes(ctx, threshold)

        if elapsed_minutes >= self._decline_timeout_minutes:
            return Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason=f"MFU decline ({mfu_stats}) persisted for {elapsed_minutes:.1f}min without identifiable cause",
                trigger=TriggerType.MISC,
            )

        return Decision.no_fault(
            reason=f"MFU declining ({mfu_stats}), monitoring ({elapsed_minutes:.1f}min)",
        )

    def _compute_decline_duration_minutes(
        self, ctx: DetectorContext, threshold: float,
    ) -> float:
        """Derive how long MFU has been below *threshold* from time-series data.

        Queries a window wider than the timeout so the "last healthy" reading
        is visible even when the decline started exactly at the timeout boundary.
        """
        lookup_window = timedelta(minutes=self._decline_timeout_minutes * 2)
        timed_mfu = ctx.mini_wandb.query_time_window(
            "mfu", window=lookup_window,
        )
        if not timed_mfu:
            return 0.0

        now = datetime.now(timezone.utc)

        last_healthy_time: datetime | None = None
        for _, ts, value in timed_mfu:
            if value >= threshold:
                last_healthy_time = ts

        if last_healthy_time is not None:
            return (now - last_healthy_time).total_seconds() / 60

        return (now - timed_mfu[0].timestamp).total_seconds() / 60

    def _get_baseline(self, mini_wandb: TrainingMetricStoreProtocol) -> float:
        if self._mfu_baseline > 0:
            return self._mfu_baseline

        if self._baseline_locked and self._locked_baseline is not None:
            return self._locked_baseline

        total_needed = self._baseline_steps + self._consecutive_steps
        all_data = mini_wandb.query_last_n_steps("mfu", last_n=total_needed)

        baseline_data = all_data[:-self._consecutive_steps] if len(all_data) > self._consecutive_steps else []
        if not baseline_data:
            return 0.0

        baseline = sum(v for _, v in baseline_data) / len(baseline_data)

        self._locked_baseline = baseline
        self._baseline_locked = True
        logger.info("MFU baseline locked at %.4f from %d steps", baseline, len(baseline_data))

        return baseline

    def _find_high_temperature_node(
        self,
        metric_store: MetricQueryProtocol,
        rank_placement: dict[int, str],
    ) -> str | None:
        if not rank_placement:
            return None

        df = metric_store.query_latest(DCGM_FI_DEV_GPU_TEMP)
        if df is None or df.is_empty():
            return None

        node_ids = set(rank_placement.values())

        node_temps: dict[str, list[float]] = {}
        for row in df.iter_rows(named=True):
            node_id = row["node_id"]
            if node_id in node_ids:
                node_temps.setdefault(node_id, []).append(row["value"])

        if not node_temps:
            return None

        node_avg_temps: dict[str, float] = {
            node_id: sum(temps) / len(temps)
            for node_id, temps in node_temps.items()
        }

        overall_avg = sum(node_avg_temps.values()) / len(node_avg_temps)

        for node_id, avg_temp in node_avg_temps.items():
            if avg_temp > overall_avg + self._temperature_delta_threshold:
                return node_id

        return None
