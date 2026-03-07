"""Semi-E2E: resilience — agent without controller, fire-and-forget after death."""
from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from unittest.mock import patch

import ray

from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector
from miles.utils.ft.protocols.platform import ft_controller_actor_name

from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import (
    E2EEnv,
    NodeSpec,
)
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    wait_for_training_stable,
)


class TestAgentWithoutController:
    async def test_rank_agent_graceful_degrade(self, local_ray: None) -> None:
        """Creating FtTrainingRankAgent when controller doesn't exist → no crash."""
        os.environ["MILES_FT_ID"] = "nonexistent"
        os.environ["MILES_FT_TRAINING_RUN_ID"] = "fake-run"

        from miles.utils.ft.agents.core.training_rank_agent import FtTrainingRankAgent

        with patch("socket.gethostname", return_value="test-node"):
            agent = FtTrainingRankAgent(rank=0, world_size=1)

        agent.step(1)
        agent.shutdown()


class TestFireAndForget:
    async def test_log_step_after_controller_death(
        self, make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Kill controller → worker continues log_step → doesn't crash/block."""
        env = make_e2e_env(
            ft_id="e2eff",
            nodes=[NodeSpec(node_id="e2eff-node-0")],
            detectors=[TrainingCrashDetector()],
        )

        await wait_for_training_stable(env.controller, n_iterations=2, timeout=30.0)

        # Step 1: kill controller
        controller_name = ft_controller_actor_name(env.ft_id)
        try:
            ray.get(env.controller.shutdown.remote(), timeout=5)
        except Exception:
            pass
        try:
            ray.kill(ray.get_actor(controller_name), no_restart=True)
        except (ValueError, Exception):
            pass

        # Step 2: wait a bit - worker should continue running without crashing
        await asyncio.sleep(2.0)

        # Step 3: verify workers are still alive
        for worker in env.workers:
            iteration = ray.get(worker.get_iteration.remote(), timeout=5)
            assert iteration > 0
