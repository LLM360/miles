"""Semi-E2E: network faults — ephemeral NIC down, sustained NIC down."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import timedelta

from miles.utils.ft.controller.detectors.network import NetworkAlertDetector, NetworkAlertDetectorConfig
from miles.utils.ft.models.metric_names import NODE_NETWORK_UP
from miles.utils.ft.models.metrics import GaugeSample
from miles.utils.ft.models.recovery import ControllerMode

from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import (
    E2EEnv,
    NodeSpec,
    _FAST_SCRAPE,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    get_status,
    wait_for_training_stable,
)


class TestEphemeralNic:
    async def test_ephemeral_nic_fault_goes_to_reattempting(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """NIC up→down transition → NetworkAlertDetector → MARK_BAD_AND_RESTART.

        MARK_BAD_AND_RESTART evicts directly without entering recovery mode,
        so we detect the action by observing run_id change.
        """
        env = make_e2e_env(
            ft_id="e2enic",
            nodes=[NodeSpec(
                node_id="e2enic-node-0",
                use_remote_collector=True,
            )],
            detectors=[
                NetworkAlertDetector(config=NetworkAlertDetectorConfig(
                    alert_window_minutes=10 / 60,
                    alert_threshold=1,
                )),
            ],
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)

        # Step 1: inject NIC up so MiniPrometheus has a baseline
        env.set_collector_metrics("e2enic-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enic-node-0", "device": "eth0"},
                value=1.0,
            ),
        ])
        await asyncio.sleep(_FAST_SCRAPE * 3)

        old_run_id = get_status(env.controller).active_run_id

        # Step 2: inject NIC down to create up→down transition
        env.set_collector_metrics("e2enic-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enic-node-0", "device": "eth0"},
                value=0.0,
            ),
        ])

        # Step 3: MARK_BAD_AND_RESTART evicts and restarts without entering
        # recovery mode; poll until active_run_id changes.
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.active_run_id != old_run_id:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("active_run_id did not change within 60s")

        assert status.mode == ControllerMode.MONITORING


class TestNetworkAlert:
    async def test_sustained_nic_down_triggers_eviction(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Sustained NIC up→down → NetworkAlertDetector → MARK_BAD_AND_RESTART.

        MARK_BAD_AND_RESTART evicts directly without entering recovery mode.
        """
        env = make_e2e_env(
            ft_id="e2enet",
            nodes=[NodeSpec(
                node_id="e2enet-node-0",
                use_remote_collector=True,
            )],
            detectors=[
                NetworkAlertDetector(config=NetworkAlertDetectorConfig(
                    alert_window_minutes=10 / 60,
                    alert_threshold=1,
                )),
            ],
            scrape_interval_seconds=_FAST_SCRAPE,
        )

        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)

        # Step 1: inject NIC up baseline
        env.set_collector_metrics("e2enet-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enet-node-0", "device": "eth0"},
                value=1.0,
            ),
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enet-node-0", "device": "eth1"},
                value=1.0,
            ),
        ])
        await asyncio.sleep(_FAST_SCRAPE * 3)

        old_run_id = get_status(env.controller).active_run_id

        # Step 2: inject sustained NIC down to create up→down transitions
        env.set_collector_metrics("e2enet-node-0", [
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enet-node-0", "device": "eth0"},
                value=0.0,
            ),
            GaugeSample(
                name=NODE_NETWORK_UP,
                labels={"node_id": "e2enet-node-0", "device": "eth1"},
                value=0.0,
            ),
        ])

        # Step 3: poll until active_run_id changes (MARK_BAD_AND_RESTART
        # evicts and restarts without entering recovery mode)
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.active_run_id != old_run_id:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("active_run_id did not change within 60s")

        assert status.mode == ControllerMode.MONITORING
