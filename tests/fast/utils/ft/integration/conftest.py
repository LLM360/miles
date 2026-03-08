from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator

import pytest
import ray

from miles.utils.ft.models.controller import ControllerStatus

logger = logging.getLogger(__name__)

_TIMEOUT_SCALE = float(os.environ.get("FT_TEST_TIMEOUT_SCALE", "1.0"))
FAST_TIMEOUT = 30.0 * _TIMEOUT_SCALE
RECOVERY_TIMEOUT = 60.0 * _TIMEOUT_SCALE
LONG_RECOVERY_TIMEOUT = 120.0 * _TIMEOUT_SCALE


@pytest.fixture(scope="module")
def local_ray() -> Generator[None, None, None]:
    if ray.is_initialized():
        ray.shutdown()
    ray.init(address="local", num_cpus=32, num_gpus=0, include_dashboard=False)
    yield
    ray.shutdown()


def _kill_named_actor(name: str) -> None:
    try:
        handle = ray.get_actor(name)
        ray.kill(handle, no_restart=True)
    except ValueError:
        pass
    except Exception:
        logger.warning("Failed to kill actor %s", name, exc_info=True)


def get_status(handle: ray.actor.ActorHandle, timeout: float = 5) -> ControllerStatus:
    return ray.get(handle.get_status.remote(), timeout=timeout)


def poll_for_run_id(
    handle: ray.actor.ActorHandle,
    timeout: float = 10.0,
    interval: float = 0.2,
) -> str:
    """Poll get_status until active_run_id is set, return it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = get_status(handle)
        if status.active_run_id is not None:
            return status.active_run_id
        time.sleep(interval)
    raise TimeoutError("active_run_id not set within timeout")
