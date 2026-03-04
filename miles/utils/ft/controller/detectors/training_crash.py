from miles.utils.ft.controller.detectors.base import (
    BaseFaultDetector,
    DetectorContext,
    _get_non_finite_loss,
)
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType, Decision
from miles.utils.ft.platform.protocols import JobStatus


class TrainingCrashDetector(BaseFaultDetector):
    def evaluate(self, ctx: DetectorContext) -> Decision:
        if ctx.job_status != JobStatus.FAILED:
            return Decision(action=ActionType.NONE, reason="training job not failed")

        trigger = self._determine_trigger(ctx.mini_wandb)

        return Decision(
            action=ActionType.ENTER_RECOVERY,
            reason=f"training job failed (trigger={trigger})",
            trigger=trigger,
        )

    def _determine_trigger(self, mini_wandb: MiniWandb) -> str:
        if _get_non_finite_loss(mini_wandb) is not None:
            return "nan_loss"

        return "crash"
