"""Integration tests for FtController: end-to-end data flows."""

import pytest

from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType, Decision
from tests.fast.utils.ft.conftest import make_test_controller


class _AlwaysMarkBadDetector(BaseFaultDetector):
    def evaluate(
        self,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
    ) -> Decision:
        return Decision(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=["node-1"],
            reason="test fault",
        )


class TestEmptyDetectorChainMultipleTicks:
    @pytest.mark.asyncio
    async def test_three_ticks_succeed(self) -> None:
        controller, _, _, _, _ = make_test_controller()

        for _ in range(3):
            await controller._tick()

        assert controller._tick_count == 3


class TestRegisterRankLogStepQuery:
    @pytest.mark.asyncio
    async def test_end_to_end_data_flow(self) -> None:
        controller, _, _, _, mini_wandb = make_test_controller()
        run_id = "integ-run-1"

        await controller.register_rank(
            run_id=run_id, rank=0, world_size=4,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        await controller.register_rank(
            run_id=run_id, rank=1, world_size=4,
            node_id="node-1", exporter_address="http://node-1:9090",
        )

        await controller.log_step(
            run_id=run_id, rank=0, step=1,
            metrics={"loss": 3.5, "grad_norm": 1.2},
        )
        await controller.log_step(
            run_id=run_id, rank=1, step=1,
            metrics={"loss": 3.4, "grad_norm": 1.1},
        )

        assert mini_wandb.latest(metric_name="loss", rank=0) == 3.5
        assert mini_wandb.latest(metric_name="loss", rank=1) == 3.4

        result_rank0 = mini_wandb.query_last_n_steps(
            metric_name="grad_norm", rank=0, last_n=10,
        )
        assert len(result_rank0) == 1
        assert result_rank0[0] == (1, 1.2)


class TestRunIdIsolation:
    @pytest.mark.asyncio
    async def test_new_run_id_clears_old_data(self) -> None:
        controller, _, _, _, mini_wandb = make_test_controller()

        await controller.register_rank(
            run_id="run-1", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )
        await controller.log_step(
            run_id="run-1", rank=0, step=10,
            metrics={"loss": 2.0},
        )
        assert mini_wandb.latest(metric_name="loss", rank=0) == 2.0

        await controller.register_rank(
            run_id="run-2", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9090",
        )

        assert mini_wandb.latest(metric_name="loss", rank=0) is None
        assert controller._active_run_id == "run-2"
        assert controller._rank_placement == {0: "node-0"}


class TestTrainingJobStatusInjection:
    @pytest.mark.asyncio
    async def test_tick_makes_status_queryable(self) -> None:
        controller, _, _, metric_store, _ = make_test_controller()

        await controller._tick()

        df = metric_store.instant_query("training_job_status")
        assert not df.is_empty()
        assert df["value"][0] == 1.0


class TestCustomDetectorInTick:
    @pytest.mark.asyncio
    async def test_detector_returns_mark_bad(self) -> None:
        detector = _AlwaysMarkBadDetector()
        controller, _, _, _, _ = make_test_controller(detectors=[detector])

        await controller._tick()

        decision = controller._evaluate_detectors()
        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert decision.bad_node_ids == ["node-1"]
