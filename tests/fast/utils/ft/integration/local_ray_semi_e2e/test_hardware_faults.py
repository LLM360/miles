"""Semi-E2E: hardware faults — GPU lost, NaN loss, fault during recovery."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.models.metric_names import GPU_AVAILABLE
from miles.utils.ft.models.metrics import GaugeSample
from miles.utils.ft.models.recovery import ControllerMode, RecoveryPhase

from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import (
    E2EEnv,
    NodeSpec,
    _FAST_SCRAPE,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    assert_phase_path_contains,
    get_status,
    wait_for_mode,
    wait_for_recovery_complete,
    wait_for_training_stable,
)


class TestHardwareAlert:
    async def test_gpu_lost_triggers_direct_eviction(
        self, e2e_full_detector_env: E2EEnv,
    ) -> None:
        """GPU_AVAILABLE=0 → HighConfidenceHardwareDetector → MARK_BAD_AND_RESTART."""
        env = e2e_full_detector_env

        # Step 1: inject GPU unavailable metric
        env.set_collector_metrics("e2efd-node-0", [
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "e2efd-node-0", "gpu": "0"},
                value=0.0,
            ),
        ])

        # Step 2: with phase chaining, MARK_BAD_AND_RESTART may complete
        # recovery in a single tick (CHECK_ALERTS → EVICT_AND_RESTART → DONE),
        # so RECOVERY mode may be too transient to observe. Poll until
        # phase_history contains the expected eviction path instead.
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and RecoveryPhase.EVICT_AND_RESTART in status.phase_history:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("Recovery with EVICT_AND_RESTART not observed within 60s")

        assert_phase_path_contains(status, [
            RecoveryPhase.EVICT_AND_RESTART,
            RecoveryPhase.DONE,
        ])


class TestNanLoss:
    async def test_nan_loss_triggers_recovery(
        self, e2e_full_detector_env: E2EEnv,
    ) -> None:
        """loss=NaN via custom log metrics → NanLossDetector → ENTER_RECOVERY."""
        env = e2e_full_detector_env

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: inject NaN loss
        await env.injector.inject_nan_loss()

        # Step 2: wait for recovery
        status = await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )
        assert status.mode == ControllerMode.RECOVERY


class TestFaultDuringRecovery:
    async def test_hardware_fault_during_reattempting_escalates(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Crash → REATTEMPTING → inject GPU_AVAILABLE=0 → critical detector finds → EVICT_AND_RESTART."""
        env = make_e2e_env(
            ft_id="e2efdr",
            nodes=[NodeSpec(
                node_id="e2efdr-node-0",
                use_remote_collector=True,
            )],
            detectors=build_detector_chain(),
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=3, timeout=30.0)

        # Step 1: first crash → enters recovery
        await env.injector.crash_training()
        await wait_for_mode(
            env.controller,
            target_mode=ControllerMode.RECOVERY,
            timeout=30.0,
        )

        # Step 2: while in recovery, inject hardware fault
        env.set_collector_metrics("e2efdr-node-0", [
            GaugeSample(
                name=GPU_AVAILABLE,
                labels={"node_id": "e2efdr-node-0", "gpu": "0"},
                value=0.0,
            ),
        ])

        # Step 3: wait for recovery to complete with eviction
        final = await wait_for_recovery_complete(env.controller, timeout=90.0)
        assert_phase_path_contains(final, [RecoveryPhase.EVICT_AND_RESTART])
