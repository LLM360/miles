"""E2E: Training hang via SIGSTOP → detection → recovery.

Slowest E2E test due to the hang detection timeout (~5-10 min).
Uses scenario_hang_detection_and_recovery which handles the full
detection → recovery → verify-training-resumes pipeline.
"""

from __future__ import annotations

import pytest
import ray
from tests.e2e.ft.conftest import E2eFaultInjector, FaultInjectorFactory, wait_for_training_stable
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import scenario_hang_detection_and_recovery

_HANG_TIMEOUT_MINUTES = 10
_DETECTION_BUFFER_SECONDS = 60
_MAX_DETECTION_SECONDS = _HANG_TIMEOUT_MINUTES * 60 + _DETECTION_BUFFER_SECONDS


@pytest.mark.timeout(900)
async def test_hang_detection_and_recovery(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    await wait_for_training_stable(
        handle=ft_controller_handle,
        n_iterations=3,
        timeout=180.0,
    )

    fault = E2eFaultInjector(
        injector_handle=fault_injector.deploy_to(node_id=target_node),
        target_node=target_node,
    )

    await scenario_hang_detection_and_recovery(
        handle=ft_controller_handle,
        injector=fault,
        hang_timeout=720.0,
        recovery_timeout=720.0,
        max_detection_seconds=_MAX_DETECTION_SECONDS,
        post_recovery_iterations=5,
        post_recovery_timeout=300.0,
    )
