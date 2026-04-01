import asyncio
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import ray
from tests.fast.ray.train.dummy_actor import DummyTrainActor

from miles.ray.train.group import RayTrainGroup

pytestmark = pytest.mark.asyncio


def _make_mock_args(*, indep_dp: bool = True) -> MagicMock:
    args = MagicMock()
    args.indep_dp = indep_dp
    args.trainer_heartbeat_interval = 30.0
    args.trainer_heartbeat_timeout = 10.0
    args.trainer_heartbeat_staleness = 90.0
    args.trainer_heartbeat_first_wait = 300.0
    return args


def _make_group(
    *,
    num_cells: int = 3,
    actor_count_per_cell: int = 1,
    rollout_manager: object | None = None,
) -> RayTrainGroup:
    """Create a RayTrainGroup through real __init__ with mocked pg and actor factory."""
    with patch(
        "miles.ray.train.group.compute_megatron_world_size_except_dp",
        return_value=1,
    ), patch(
        "miles.ray.train.group.allocate_gpus_for_actor",
        side_effect=lambda **kwargs: [DummyTrainActor.remote() for _ in range(actor_count_per_cell)],
    ):
        group = RayTrainGroup(
            args=_make_mock_args(indep_dp=True),
            num_nodes=1,
            num_gpus_per_node=num_cells,
            pg=(MagicMock(), list(range(num_cells)), list(range(num_cells))),
            role="actor",
            with_ref=False,
            rollout_manager=rollout_manager,
        )
    return group


async def _init_group(group: RayTrainGroup) -> None:
    """Call init and wait for all cells to become alive."""
    await group.init()


async def _make_alive_group(*, num_cells: int = 3, **kwargs) -> RayTrainGroup:
    """Create a group and init all cells to alive."""
    group = _make_group(num_cells=num_cells, **kwargs)
    await _init_group(group)
    return group


class TestInit:
    def test_creates_correct_number_of_cells(self):
        group = _make_group(num_cells=3)

        assert len(group._cells) == 3
        assert [c.cell_index for c in group._cells] == [0, 1, 2]

    def test_cells_are_allocated_after_init(self):
        group = _make_group(num_cells=2)

        for cell in group._cells:
            assert cell.is_allocated
            assert not cell.is_alive

    def test_each_cell_has_own_actors(self):
        group = _make_group(num_cells=3, actor_count_per_cell=2)

        handles_per_cell = [cell._get_actor_handles() for cell in group._cells]
        assert all(len(h) == 2 for h in handles_per_cell)

        all_handles = [h for handles in handles_per_cell for h in handles]
        assert len(set(id(h) for h in all_handles)) == 6

    def test_single_cell_no_tcp_store(self):
        with patch(
            "miles.ray.train.group.compute_megatron_world_size_except_dp",
            return_value=1,
        ), patch(
            "miles.ray.train.group.allocate_gpus_for_actor",
            side_effect=lambda **kwargs: [DummyTrainActor.remote()],
        ):
            group = RayTrainGroup(
                args=_make_mock_args(indep_dp=False),
                num_nodes=1,
                num_gpus_per_node=1,
                pg=(MagicMock(), [0], [0]),
                role="actor",
                with_ref=False,
                rollout_manager=None,
            )

        assert len(group._cells) == 1
        assert group._indep_dp_store is None

    async def test_init_marks_all_cells_alive(self):
        group = _make_group(num_cells=3)

        await _init_group(group)

        for cell in group._cells:
            assert cell.is_alive
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]
            assert cell.indep_dp_info.alive_size == 3

        assert group._cells[0].indep_dp_info.alive_rank == 0
        assert group._cells[1].indep_dp_info.alive_rank == 1
        assert group._cells[2].indep_dp_info.alive_rank == 2


class TestStopStartCell:
    async def test_stop_cell_transitions_to_stopped(self):
        group = await _make_alive_group(num_cells=2)

        group.stop_cell(1)

        assert group._cells[1].is_stopped
        assert group._cells[0].is_alive

    async def test_start_cell_transitions_to_pending(self):
        group = await _make_alive_group(num_cells=2)
        group.stop_cell(1)

        group.start_cell(1)

        assert group._cells[1].is_pending


class TestExecuteFirstAlive:
    async def test_picks_first_alive_cell(self):
        group = await _make_alive_group(num_cells=3)

        await group._execute_first_alive("save_model", 42)

        for handle in group._cells[0]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "save_model" for c in calls)

        for cell in group._cells[1:]:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert not any(c[0] == "save_model" for c in calls)

    async def test_skips_stopped_picks_next(self):
        group = await _make_alive_group(num_cells=2)
        group._cells[0].stop()

        await group._execute_first_alive("update_weights")

        for handle in group._cells[1]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "update_weights" for c in calls)


class TestComputeIndepDPInfo:
    def test_all_alive(self):
        group = _make_group(num_cells=3)

        info = group._compute_indep_dp_info(cell_index=2, alive_cell_indices=[0, 1, 2])

        assert info.alive_rank == 2
        assert info.alive_size == 3
        assert info.cell_index == 2

    def test_with_gap(self):
        group = _make_group(num_cells=3)

        info = group._compute_indep_dp_info(cell_index=2, alive_cell_indices=[0, 2])

        assert info.alive_rank == 1
        assert info.alive_size == 2


class TestBroadcastAlive:
    async def test_skips_stopped_cells(self):
        group = await _make_alive_group(num_cells=2)
        group._cells[1].stop()

        await group._broadcast_alive("train")

        for handle in group._cells[0]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "train" for c in calls)

    async def test_asserts_on_no_alive_cells(self):
        group = await _make_alive_group(num_cells=1)
        group._cells[0].stop()

        with pytest.raises(AssertionError, match="No alive cells"):
            await group._broadcast_alive("train")


class TestRefreshCellsReconfigure:
    async def test_reconfigure_triggers_on_alive_change(self):
        """When a cell is stopped, _refresh_cells reconfigures remaining alive cells."""
        group = await _make_alive_group(num_cells=3)

        # Step 1: Stop cell 1
        group.stop_cell(1)

        # Step 2: Refresh
        await group._refresh_cells()

        # Step 3: Quorum bumped (init was quorum 0, this is first reconfigure)
        assert group._indep_dp_quorum_id == 1

        # Step 4: Remaining alive cells have updated indep_dp_info
        assert group._cells[0].is_alive
        assert group._cells[0].indep_dp_info.alive_cell_indices == [0, 2]
        assert group._cells[0].indep_dp_info.alive_rank == 0
        assert group._cells[0].indep_dp_info.alive_size == 2

        assert group._cells[2].is_alive
        assert group._cells[2].indep_dp_info.alive_rank == 1

        # Step 5: Stopped cell untouched
        assert group._cells[1].is_stopped

        # Step 6: Actors received reconfigure_indep_dp
        for cell in [group._cells[0], group._cells[2]]:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "reconfigure_indep_dp" for c in calls)

    async def test_no_reconfigure_when_unchanged(self):
        group = await _make_alive_group(num_cells=2)

        await group._refresh_cells()

        assert group._indep_dp_quorum_id == 0


class TestRefreshCellsHealing:
    async def test_pending_cell_gets_healed(self):
        """A pending cell goes through allocate + healing with correct alive_rank."""
        group = await _make_alive_group(num_cells=3)

        # Step 1: Stop cell 2, then start it (pending)
        group.stop_cell(2)
        group.start_cell(2)

        # Step 2: Refresh heals the pending cell
        await group._refresh_cells()

        # Step 3: All 3 cells are now alive
        assert all(c.is_alive for c in group._cells)

        # Step 4: All cells have consistent indep_dp_info
        for cell in group._cells:
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]
            assert cell.indep_dp_info.alive_size == 3

        # Step 5: Healed cell's actors received init
        for handle in group._cells[2]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "init" for c in calls)

        # Step 6: Source cell sent ckpt to healed cell's alive_rank
        for handle in group._cells[0]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            send_calls = [c for c in calls if c[0] == "send_ckpt"]
            assert len(send_calls) == 1
            assert send_calls[0][2]["dst_rank"] == 2

    async def test_multiple_pending_cells_healed(self):
        """Multiple pending cells healed simultaneously."""
        group = await _make_alive_group(num_cells=3)
        group.stop_cell(1)
        group.stop_cell(2)
        group.start_cell(1)
        group.start_cell(2)

        await group._refresh_cells()

        assert all(c.is_alive for c in group._cells)
        for cell in group._cells:
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]

        # Source (cell 0) sent ckpt to both healed cells
        for handle in group._cells[0]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            send_calls = [c for c in calls if c[0] == "send_ckpt"]
            assert len(send_calls) == 2
            dst_ranks = sorted(c[2]["dst_rank"] for c in send_calls)
            assert dst_ranks == [1, 2]

    async def test_healed_cell_receives_set_rollout_manager(self):
        """Healed cell receives set_rollout_manager after init."""
        rollout_mgr = MagicMock()
        group = await _make_alive_group(num_cells=2, rollout_manager=rollout_mgr)
        group.stop_cell(1)
        group.start_cell(1)

        await group._refresh_cells()

        assert group._cells[1].is_alive
        for handle in group._cells[1]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "set_rollout_manager" for c in calls)

    async def test_pending_cell_with_stopped_cell(self):
        """Pending + stopped: only alive and pending participate, stopped excluded."""
        group = await _make_alive_group(num_cells=3)

        # cell 1 stopped (not restarted), cell 2 pending
        group.stop_cell(1)
        group.stop_cell(2)
        group.start_cell(2)

        await group._refresh_cells()

        assert group._cells[0].is_alive
        assert group._cells[1].is_stopped
        assert group._cells[2].is_alive

        assert group._cells[0].indep_dp_info.alive_cell_indices == [0, 2]
        assert group._cells[0].indep_dp_info.alive_size == 2
        assert group._cells[2].indep_dp_info.alive_rank == 1


class TestRefreshCellsNoOp:
    async def test_repeated_refresh_without_change_does_not_reconfigure(self):
        """Calling _refresh_cells multiple times without state changes dispatches no actor calls."""
        group = await _make_alive_group(num_cells=3)

        # Clear init calls by noting current call count
        init_call_counts = {}
        for cell in group._cells:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                init_call_counts[id(handle)] = len(calls)

        # Two refreshes — neither should change anything
        await group._refresh_cells()
        await group._refresh_cells()
        assert group._indep_dp_quorum_id == 0

        # No new calls dispatched
        for cell in group._cells:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert len(calls) == init_call_counts[id(handle)]

    async def test_refresh_after_reconfigure_is_noop_on_second_call(self):
        group = await _make_alive_group(num_cells=3)
        group.stop_cell(1)
        await group._refresh_cells()
        assert group._indep_dp_quorum_id == 1

        await group._refresh_cells()
        assert group._indep_dp_quorum_id == 1


class TestConsecutiveStopStartCycles:
    async def test_stop_train_stop_train_start_train(self):
        """Consecutive: stop 1 → refresh → stop 2 → refresh → start 1 → refresh."""
        group = await _make_alive_group(num_cells=3)

        # Step 1: Stop cell 1
        group.stop_cell(1)
        await group._refresh_cells()
        assert group._indep_dp_quorum_id == 1
        assert group._cells[0].indep_dp_info.alive_cell_indices == [0, 2]

        # Step 2: Stop cell 2 (only cell 0 alive)
        group.stop_cell(2)
        await group._refresh_cells()
        assert group._indep_dp_quorum_id == 2
        assert group._cells[0].indep_dp_info.alive_cell_indices == [0]
        assert group._cells[0].indep_dp_info.alive_size == 1

        # Step 3: Start cell 1 (cells 0 and 1 alive)
        group.start_cell(1)
        await group._refresh_cells()
        assert group._indep_dp_quorum_id == 3
        assert group._cells[0].is_alive
        assert group._cells[1].is_alive
        assert group._cells[2].is_stopped
        assert group._cells[0].indep_dp_info.alive_cell_indices == [0, 1]
        assert group._cells[1].indep_dp_info.alive_cell_indices == [0, 1]


class TestTrain:
    async def test_train_refreshes_and_dispatches(self):
        group = await _make_alive_group(num_cells=2)

        await group.train(rollout_id=0, rollout_data_ref="data")

        for cell in group._cells:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "train" for c in calls)

    async def test_train_with_stopped_cell_only_dispatches_to_alive(self):
        group = await _make_alive_group(num_cells=3)
        group.stop_cell(1)

        await group.train(rollout_id=0, rollout_data_ref="data")

        for cell in [group._cells[0], group._cells[2]]:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "train" for c in calls)

        assert group._cells[1].is_stopped

    async def test_consecutive_train_no_reconfigure_overhead(self):
        """Multiple train calls with no state changes — no reconfigure overhead."""
        group = await _make_alive_group(num_cells=3)

        # Note init call count
        init_counts = {}
        for cell in group._cells:
            for handle in cell._get_actor_handles():
                init_counts[id(handle)] = len(ray.get(handle.get_calls.remote()))

        for step in range(3):
            await group.train(rollout_id=step, rollout_data_ref="data")

        assert group._indep_dp_quorum_id == 0

        for cell in group._cells:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                new_calls = calls[init_counts[id(handle)] :]
                assert not any(c[0] == "reconfigure_indep_dp" for c in new_calls)
                train_calls = [c for c in new_calls if c[0] == "train"]
                assert len(train_calls) == 3

    async def test_rapid_stop_start_before_train(self):
        """Cell stopped and immediately started before next train — healed in one shot."""
        group = await _make_alive_group(num_cells=3)

        group.stop_cell(1)
        group.start_cell(1)

        await group.train(rollout_id=0, rollout_data_ref="data")

        assert all(c.is_alive for c in group._cells)
        for cell in group._cells:
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]

    async def test_full_lifecycle_through_train(self):
        """End-to-end: normal → degraded → steady degraded → healing → full."""
        group = await _make_alive_group(num_cells=3)

        # Step 1: Normal training (no reconfigure)
        await group.train(rollout_id=0, rollout_data_ref="data")
        assert group._indep_dp_quorum_id == 0

        # Step 2: Stop cell 2 → degraded (triggers reconfigure)
        group.stop_cell(2)
        await group.train(rollout_id=1, rollout_data_ref="data")
        assert group._indep_dp_quorum_id == 1
        assert group._cells[0].indep_dp_info.alive_cell_indices == [0, 1]

        # Step 3: Steady degraded (no reconfigure)
        await group.train(rollout_id=2, rollout_data_ref="data")
        assert group._indep_dp_quorum_id == 1

        # Step 4: Start cell 2 → healing (triggers reconfigure)
        group.start_cell(2)
        await group.train(rollout_id=3, rollout_data_ref="data")
        assert group._indep_dp_quorum_id == 2
        assert all(c.is_alive for c in group._cells)
        assert group._cells[2].indep_dp_info.alive_cell_indices == [0, 1, 2]

        # Step 5: Full training again (no reconfigure)
        await group.train(rollout_id=4, rollout_data_ref="data")
        assert group._indep_dp_quorum_id == 2


class TestPerCellErrorIsolation:
    async def test_one_cell_failure_marks_errored_others_ok(self):
        """One cell's actor fails during broadcast, that cell is errored, others complete normally."""
        group = await _make_alive_group(num_cells=3)

        # Step 1: Make cell 1's actors fail on train
        for handle in group._cells[1]._get_actor_handles():
            ray.get(handle.set_fail_methods.remote(["train"]))

        # Step 2: Broadcast train
        await group._broadcast_alive("train", 0, "data")

        # Step 3: Cell 1 is errored, others alive
        assert group._cells[0].is_alive
        assert group._cells[1].is_errored
        assert group._cells[2].is_alive

        # Step 4: Other cells received train call
        for cell_idx in [0, 2]:
            for handle in group._cells[cell_idx]._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "train" for c in calls)

    async def test_errored_cell_skipped_in_next_broadcast(self):
        """After marking a cell errored, subsequent broadcasts skip it."""
        group = await _make_alive_group(num_cells=2)

        # Step 1: Make cell 0 fail
        for handle in group._cells[0]._get_actor_handles():
            ray.get(handle.set_fail_methods.remote(["train"]))

        await group._broadcast_alive("train", 0, "data")
        assert group._cells[0].is_errored

        # Step 2: Next broadcast only goes to cell 1
        await group._broadcast_alive("train", 1, "data")

        for handle in group._cells[1]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            train_calls = [c for c in calls if c[0] == "train"]
            assert len(train_calls) == 2

    async def test_get_errored_cell_indices(self):
        group = await _make_alive_group(num_cells=3)

        # Step 1: Make cell 2 fail
        for handle in group._cells[2]._get_actor_handles():
            ray.get(handle.set_fail_methods.remote(["train"]))

        await group._broadcast_alive("train", 0, "data")

        # Step 2: Check errored indices
        assert group.get_errored_cell_indices() == [2]


class TestExecuteFirstAliveFallback:
    async def test_first_cell_fails_falls_back_to_next(self):
        """If the first alive cell fails, _execute_first_alive tries the next."""
        group = await _make_alive_group(num_cells=3)

        # Step 1: Make cell 0 fail on save_model
        for handle in group._cells[0]._get_actor_handles():
            ray.get(handle.set_fail_methods.remote(["save_model"]))

        # Step 2: Execute first alive
        await group._execute_first_alive("save_model", 42)

        # Step 3: Cell 0 errored, cell 1 handled it
        assert group._cells[0].is_errored
        assert group._cells[1].is_alive

        for handle in group._cells[1]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "save_model" for c in calls)

    async def test_all_cells_fail_raises_runtime_error(self):
        """If all cells fail, RuntimeError is raised."""
        group = await _make_alive_group(num_cells=2)

        for cell in group._cells:
            for handle in cell._get_actor_handles():
                ray.get(handle.set_fail_methods.remote(["save_model"]))

        with pytest.raises(RuntimeError, match="All cells failed for save_model"):
            await group._execute_first_alive("save_model", 42)


def _make_failing_actor_factory() -> Callable:
    """Create a factory that returns actors pre-configured to fail on init."""

    def factory():
        actor = DummyTrainActor.remote()
        ray.get(actor.set_fail_methods.remote(["init"]))
        return [actor]

    return factory


class TestRefreshCellsErrorHandling:
    async def test_healing_failure_stops_pending_cell_keeps_alive(self):
        """When healing init fails, pending cell is stopped, alive cells unaffected."""
        group = await _make_alive_group(num_cells=3)

        # Step 1: Stop cell 2 and start it (pending)
        group.stop_cell(2)
        group.start_cell(2)

        # Step 2: Replace actor factory so new actors fail on init
        group._cells[2]._actor_factory = _make_failing_actor_factory()

        # Step 3: Refresh — healing should fail, pending cell stopped
        await group._refresh_cells()

        # Step 4: Cell 2 stopped, cells 0 and 1 still alive
        assert group._cells[0].is_alive
        assert group._cells[1].is_alive
        assert group._cells[2].is_stopped


class TestHeartbeatMonitor:
    async def test_heartbeat_normal_does_not_mark_errored(self):
        """When heartbeat returns recent timestamp, cells stay alive."""
        group = await _make_alive_group(num_cells=2)

        group._heartbeat_monitor.check()

        assert all(c.is_alive for c in group._cells)

    async def test_heartbeat_stale_timestamp_marks_errored(self):
        """When heartbeat returns old timestamp, cell is marked errored."""
        group = await _make_alive_group(num_cells=2)
        group._heartbeat_monitor._staleness = 1.0
        group._heartbeat_monitor._timeout = 5.0

        # Step 1: Make cell 1's timestamp very old
        for handle in group._cells[1]._get_actor_handles():
            ray.get(handle.set_last_active_timestamp.remote(0.0))

        # Step 2: Check heartbeats
        group._heartbeat_monitor.check()

        # Step 3: Cell 1 errored, cell 0 alive
        assert group._cells[0].is_alive
        assert group._cells[1].is_errored

    async def test_heartbeat_timeout_marks_errored(self):
        """When heartbeat call fails (actor unresponsive), cell is marked errored."""
        group = await _make_alive_group(num_cells=2)
        group._heartbeat_monitor._staleness = 90.0
        group._heartbeat_monitor._timeout = 5.0

        # Step 1: Make cell 0's heartbeat fail
        for handle in group._cells[0]._get_actor_handles():
            ray.get(handle.set_heartbeat_fail.remote(True))

        # Step 2: Check heartbeats
        group._heartbeat_monitor.check()

        # Step 3: Cell 0 errored, cell 1 alive
        assert group._cells[0].is_errored
        assert group._cells[1].is_alive

    async def test_pause_resume_prevents_false_positive(self):
        """Paused heartbeat monitor does not check, so stale timestamps don't trigger errors."""
        group = await _make_alive_group(num_cells=2)

        # Step 1: Make cell 1's timestamp very old
        for handle in group._cells[1]._get_actor_handles():
            ray.get(handle.set_last_active_timestamp.remote(0.0))

        # Step 2: Pause heartbeat
        monitor = group._heartbeat_monitor
        assert monitor is not None
        monitor.pause()
        assert monitor._pause_event.is_set()

        # Step 3: Resume
        monitor.resume()
        assert not monitor._pause_event.is_set()
