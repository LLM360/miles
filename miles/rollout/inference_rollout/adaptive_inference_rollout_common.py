import logging

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn

logger = logging.getLogger(__name__)


class AdaptiveInferenceRolloutFn(InferenceRolloutFn):
    """Opt-in adaptive oversampling rollout.

    Use with:
      MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1
      --rollout-function-path miles.rollout.inference_rollout.adaptive_inference_rollout_common.AdaptiveInferenceRolloutFn
    """

    KEEP_RATE_EMA_ALPHA = 0.8

    def __init__(self, input: RolloutFnConstructorInput):
        super().__init__(input)
        self.keep_rate_ema: float | None = None

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        from miles.rollout.inference_rollout.adaptive_inference_rollout_train import generate_rollout_async

        output, aborted_samples = await generate_rollout_async(
            self.state,
            input.rollout_id,
            self.data_source.get_samples,
            keep_rate_getter=self._get_keep_rate,
            keep_rate_updater=self._update_keep_rate,
        )
        self.data_source.add_samples(aborted_samples)
        return output

    def _get_keep_rate(self) -> float | None:
        return self.keep_rate_ema

    def _update_keep_rate(self, observed_keep_rate: float) -> float:
        observed_keep_rate = max(0.0, min(1.0, observed_keep_rate))
        if self.keep_rate_ema is None:
            self.keep_rate_ema = observed_keep_rate
        else:
            alpha = self.KEEP_RATE_EMA_ALPHA
            self.keep_rate_ema = alpha * observed_keep_rate + (1.0 - alpha) * self.keep_rate_ema

        logger.info(
            "[adaptive_oversampling] observed_keep_rate=%.4f keep_rate_ema=%.4f alpha=%.2f",
            observed_keep_rate,
            self.keep_rate_ema,
            self.KEEP_RATE_EMA_ALPHA,
        )
        return self.keep_rate_ema
