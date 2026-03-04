import math
from abc import ABC, abstractmethod

from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import Decision


class BaseFaultDetector(ABC):
    @abstractmethod
    def evaluate(
        self,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        rank_placement: dict[int, str],
    ) -> Decision: ...

    def on_new_run(self, run_id: str) -> None:
        """Called when a new training run is registered.

        Subclasses should override to reset stateful internal state
        that should not carry over between runs.
        """


def _get_non_finite_loss(mini_wandb: MiniWandb) -> float | None:
    """Return the loss value if it is non-finite (NaN/Inf), otherwise None."""
    latest_loss = mini_wandb.latest("loss", rank=0)
    if latest_loss is not None and not math.isfinite(latest_loss):
        return latest_loss
    return None
