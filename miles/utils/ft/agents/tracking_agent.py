from __future__ import annotations

import logging
import os

from miles.utils.ft.agents.controller_handle import ControllerHandleMixin

logger = logging.getLogger(__name__)


class FtTrackingAgent(ControllerHandleMixin):
    """Forwards training metrics to FtController via Ray fire-and-forget calls.

    Designed to be registered as a hook in tracking_utils.log(), so that all
    metrics logged to Wandb/TensorBoard also reach the fault-tolerance
    controller's MiniWandb store.
    """

    def __init__(self, rank: int, run_id: str | None = None) -> None:
        super().__init__()
        self._rank = rank
        self._run_id = run_id or os.environ.get("FT_TRAINING_RUN_ID", "")

    def log(self, *, metrics: dict[str, float], step: int) -> None:
        if not self._run_id:
            return

        try:
            controller = self._get_controller_handle()
            if controller is not None:
                controller.log_step.remote(
                    run_id=self._run_id,
                    rank=self._rank,
                    step=step,
                    metrics=metrics,
                )
        except Exception:
            logger.warning(
                "FtTrackingAgent.log() failed at step=%d", step, exc_info=True
            )
