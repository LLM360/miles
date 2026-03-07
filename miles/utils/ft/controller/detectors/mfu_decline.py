from datetime import datetime, timedelta, timezone

import polars as pl
from pydantic import ConfigDict, Field

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.models.metric_names import DCGM_FI_DEV_GPU_TEMP
from miles.utils.ft.protocols.metrics import MetricQueryProtocol, TrainingMetricStoreProtocol


class MfuDeclineDetectorConfig(FtBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mfu_baseline: float | None = None
    mfu_threshold_ratio: float = Field(default=0.8, gt=0, le=1)
    consecutive_steps: int = Field(default=10, ge=1)
    temperature_delta_threshold: float = Field(default=20.0, gt=0)
    decline_timeout_minutes: float = Field(default=30.0, gt=0)
    baseline_steps: int = Field(default=50, ge=1)
    mfu_absolute_minimum: float = Field(default=0.0, ge=0)


class MfuDeclineDetector(BaseFaultDetector):
    def __init__(self, config: MfuDeclineDetectorConfig | None = None) -> None:
        self._config = config or MfuDeclineDetectorConfig()

    def evaluate(self, ctx: DetectorContext) -> Decision:
        cfg = self._config

        recent_mfu = ctx.mini_wandb.query_last_n_steps("mfu", last_n=cfg.consecutive_steps)
        if len(recent_mfu) < cfg.consecutive_steps:
            return Decision.no_fault(reason="insufficient MFU data")

        avg_mfu = sum(sv.value for sv in recent_mfu) / len(recent_mfu)

        if cfg.mfu_absolute_minimum > 0 and avg_mfu < cfg.mfu_absolute_minimum:
            return Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason=f"MFU {avg_mfu:.4f} below absolute minimum {cfg.mfu_absolute_minimum:.4f}",
                trigger=TriggerType.MISC,
            )

        baseline = self._compute_baseline(ctx.mini_wandb)
        if baseline <= 0:
            return Decision.no_fault(reason="no valid MFU baseline")

        threshold = baseline * cfg.mfu_threshold_ratio
        if avg_mfu >= threshold:
            return Decision.no_fault(reason="MFU within acceptable range")

        decline_summary = f"{avg_mfu:.4f} < {threshold:.4f}"

        high_temp_node_id = self._find_high_temperature_node(ctx.metric_store, ctx.rank_placement)
        if high_temp_node_id is not None:
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                bad_node_ids=[high_temp_node_id],
                reason=f"MFU decline ({decline_summary}) correlated with high temperature on {high_temp_node_id}",
                trigger=TriggerType.HARDWARE,
            )

        elapsed_minutes = self._compute_decline_duration_minutes(ctx.mini_wandb, threshold)
        if elapsed_minutes >= cfg.decline_timeout_minutes:
            return Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason=f"MFU decline ({decline_summary}) persisted for {elapsed_minutes:.1f}min without identifiable cause",
                trigger=TriggerType.MISC,
            )

        return Decision.no_fault(
            reason=f"MFU declining ({decline_summary}), monitoring ({elapsed_minutes:.1f}min)",
        )

    def _compute_baseline(self, mini_wandb: TrainingMetricStoreProtocol) -> float:
        cfg = self._config
        if cfg.mfu_baseline is not None:
            return cfg.mfu_baseline

        total_needed = cfg.baseline_steps + cfg.consecutive_steps
        all_data = mini_wandb.query_last_n_steps("mfu", last_n=total_needed)

        baseline_data = all_data[:-cfg.consecutive_steps]
        if not baseline_data:
            return 0.0

        return sum(sv.value for sv in baseline_data) / len(baseline_data)

    def _compute_decline_duration_minutes(
        self, mini_wandb: TrainingMetricStoreProtocol, threshold: float,
    ) -> float:
        """Derive how long MFU has been below *threshold* from time-series data.

        Queries a window wider than the timeout so the "last healthy" reading
        is visible even when the decline started exactly at the timeout boundary.
        """
        lookup_window = timedelta(minutes=self._config.decline_timeout_minutes * 2)
        timed_mfu = mini_wandb.query_time_window("mfu", window=lookup_window)
        if not timed_mfu:
            return 0.0

        healthy_times = [ts for _, ts, value in timed_mfu if value >= threshold]
        decline_start = healthy_times[-1] if healthy_times else timed_mfu[0].timestamp
        return (datetime.now(timezone.utc) - decline_start).total_seconds() / 60

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
        df = df.filter(pl.col("node_id").is_in(node_ids))
        if df.is_empty():
            return None

        node_avgs = df.group_by("node_id").agg(pl.col("value").mean().alias("avg_temp"))
        overall_avg = node_avgs["avg_temp"].mean()

        outliers = node_avgs.filter(
            pl.col("avg_temp") > overall_avg + self._config.temperature_delta_threshold,
        )
        if outliers.is_empty():
            return None

        return outliers.sort("avg_temp", descending=True)["node_id"][0]
