from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext, get_non_finite_loss
from miles.utils.ft.models import ActionType, Decision, TrainingMetricStoreProtocol, TriggerType
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

    def _determine_trigger(self, mini_wandb: TrainingMetricStoreProtocol) -> TriggerType:
        if get_non_finite_loss(mini_wandb) is not None:
            return TriggerType.NAN_LOSS

        return TriggerType.CRASH
