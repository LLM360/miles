import logging
import time

import ray

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "miles_metric_"
_COLLECTOR_ACTOR_NAME = "miles_prometheus_collector"
_GET_ACTOR_TIMEOUT = 60
_GET_ACTOR_INTERVAL = 2

_collector_handle = None


def init_prometheus(args, start_server: bool = False):
    """Initialize the Prometheus metric collector.

    The driver process (``start_server=True``) creates a named Ray
    actor that holds the HTTP server and all gauges.  Actor processes
    (``start_server=False``) look up the existing named actor.  Ray
    remote calls transport metrics across nodes transparently.
    """
    global _collector_handle

    if start_server:
        _collector_handle = ray.remote(_PrometheusCollector).options(name=_COLLECTOR_ACTOR_NAME).remote(args)
        ray.get(_collector_handle.ping.remote())
        logger.info("Prometheus collector actor created")
    else:
        deadline = time.monotonic() + _GET_ACTOR_TIMEOUT
        while True:
            try:
                _collector_handle = ray.get_actor(_COLLECTOR_ACTOR_NAME)
                break
            except ValueError:
                if time.monotonic() >= deadline:
                    logger.warning(
                        "Prometheus collector actor not found "
                        f"after {_GET_ACTOR_TIMEOUT}s, "
                        "metrics will not be reported"
                    )
                    return
                time.sleep(_GET_ACTOR_INTERVAL)


def get_prometheus():
    """Return the collector actor handle, or ``None``."""
    return _collector_handle


class _PrometheusCollector:
    """Ray actor that owns the Prometheus HTTP server and gauges.

    Runs on the driver node.  All processes push metrics here via
    ``handle.update.remote(metrics)`` — works across nodes because
    Ray handles the RPC transparently.
    """

    def __init__(self, args):
        from prometheus_client import Gauge, start_http_server

        self._Gauge = Gauge
        self._gauges: dict = {}
        self._run_name = (
            getattr(args, "prometheus_run_name", None) or getattr(args, "wandb_group", None) or "miles_training"
        )
        self._label_keys = ["run_name"]
        self._label_vals = [self._run_name]

        self._heartbeat_counter = 0
        self._ft_training_heartbeat = Gauge(
            "miles_training_heartbeat",
            "Monotonically increasing counter, bumped on each training step and phase change",
            ["run_name"],
        )
        self._ft_training_phase = Gauge(
            "miles_training_phase",
            "Training phase: 0=idle, 1=training, 2=checkpoint_saving",
            ["run_name"],
        )

        port = args.prometheus_port
        start_http_server(port)
        logger.info("Prometheus metrics server started on port %d, run_name=%s", port, self._run_name)

    def update(self, metrics: dict):
        """Set gauge values for all numeric metrics."""
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            safe_key = _METRIC_PREFIX + (key.replace("/", "_").replace("-", "_").replace("@", "_at_"))
            if safe_key not in self._gauges:
                self._gauges[safe_key] = self._Gauge(
                    safe_key,
                    key,
                    self._label_keys,
                )
            self._gauges[safe_key].labels(*self._label_vals).set(value)

    def set_training_phase(self, phase: int):
        """Set phase gauge and bump heartbeat."""
        self._ft_training_phase.labels(self._run_name).set(phase)
        self._heartbeat_counter += 1
        self._ft_training_heartbeat.labels(self._run_name).set(self._heartbeat_counter)

    def bump_heartbeat(self):
        self._heartbeat_counter += 1
        self._ft_training_heartbeat.labels(self._run_name).set(self._heartbeat_counter)

    def ping(self):
        return True
