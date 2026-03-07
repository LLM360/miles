from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext, get_non_finite_loss
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType


class NanLossDetector(BaseFaultDetector):
    def evaluate(self, ctx: DetectorContext) -> Decision:
        bad_loss = get_non_finite_loss(ctx.mini_wandb)

        if bad_loss is not None:
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason=f"loss is {bad_loss}",
                trigger=TriggerType.NAN_LOSS,
            )

        return Decision.no_fault(reason="loss is normal")
