import logging

import wandb
from miles.utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.event_logger.models import MetricEvent
from miles.utils.tensorboard_utils import _TensorboardAdapter

from . import wandb_utils
from .prometheus_utils import get_prometheus, init_prometheus

logger = logging.getLogger(__name__)


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)

    if args.use_prometheus:
        init_prometheus(args, start_server=primary)


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])

    if args.use_prometheus:
        prom = get_prometheus()
        assert prom is not None, (
            "Prometheus collector is not initialized; ensure init_tracking(..., primary=...) ran on the "
            "driver and workers can resolve the miles_prometheus_collector Ray actor."
        )
        prom.update.remote(metrics)

    if is_event_logger_initialized():
        float_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if float_metrics:
            get_event_logger().log(MetricEvent, {"metrics": float_metrics}, print_log=False)
