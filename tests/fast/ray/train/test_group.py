import asyncio
from unittest.mock import MagicMock

import pytest
import ray
from tests.fast.ray.train.conftest import make_alive_cell, make_cell

from miles.ray.train.group import RayTrainGroup


def _make_group_with_cells(cells: list) -> RayTrainGroup:
    """Create a RayTrainGroup with pre-set cells, bypassing __init__."""
    group = object.__new__(RayTrainGroup)
    group._cells = cells
    group._indep_dp_quorum_id = 0
    return group


class TestStopStartCell:
    def test_stop_cell_transitions_to_stopped(self):
        cells = [make_alive_cell(0, alive_cell_indices=[0, 1]), make_alive_cell(1, alive_cell_indices=[0, 1])]
        group = _make_group_with_cells(cells)

        group.stop_cell(1)

        assert cells[1].is_stopped
        assert cells[0].is_alive

    def test_start_cell_transitions_to_pending(self):
        cells = [make_alive_cell(0, alive_cell_indices=[0, 1]), make_alive_cell(1, alive_cell_indices=[0, 1])]
        group = _make_group_with_cells(cells)
        group.stop_cell(1)

        group.start_cell(1)

        assert cells[1].is_pending


class TestAsyncExecuteFirstAlive:
    def test_picks_first_alive_cell(self):
        """_async_execute_first_alive dispatches to the first alive cell only."""
        cells = [make_alive_cell(i, alive_cell_indices=[0, 1, 2]) for i in range(3)]
        group = _make_group_with_cells(cells)

        refs = group._async_execute_first_alive("save_model", 42)
        ray.get(refs)

        # Only cell 0's actors should have received the call
        for handle in cells[0]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "save_model" for c in calls)

        # Cell 1 and 2 should NOT have received anything
        for cell in cells[1:]:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert not any(c[0] == "save_model" for c in calls)

    def test_skips_stopped_picks_next(self):
        """When cell 0 is stopped, picks next alive cell."""
        cells = [make_alive_cell(i, alive_cell_indices=[0, 1]) for i in range(2)]
        group = _make_group_with_cells(cells)
        cells[0].stop()

        refs = group._async_execute_first_alive("update_weights")
        ray.get(refs)

        for handle in cells[1]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "update_weights" for c in calls)


class TestComputeIndepDPInfo:
    def test_all_alive(self):
        group = _make_group_with_cells([make_cell(i) for i in range(3)])

        info = group._compute_indep_dp_info(cell_index=2, alive_cell_indices=[0, 1, 2])

        assert info.alive_rank == 2
        assert info.alive_size == 3
        assert info.cell_index == 2
        assert info.alive_cell_indices == [0, 1, 2]

    def test_with_gap(self):
        group = _make_group_with_cells([make_cell(i) for i in range(3)])

        info = group._compute_indep_dp_info(cell_index=2, alive_cell_indices=[0, 2])

        assert info.alive_rank == 1
        assert info.alive_size == 2


class TestAsyncExecuteAlive:
    def test_skips_stopped_cells(self):
        alive_cell = make_alive_cell(0, alive_cell_indices=[0])
        stopped_cell = make_cell(1)
        stopped_cell.stop()
        group = _make_group_with_cells([alive_cell, stopped_cell])

        refs = group._async_execute_alive("train")

        assert len(refs) == 1
        ray.get(refs)

        for handle in alive_cell._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "train" for c in calls)

    def test_asserts_on_no_alive_cells(self):
        stopped = make_cell(0)
        stopped.stop()
        group = _make_group_with_cells([stopped])

        with pytest.raises(AssertionError, match="No alive cells"):
            group._async_execute_alive("train")


class TestRefreshCellsReconfigure:
    def test_reconfigure_triggers_on_alive_change(self):
        """When a cell is stopped, _refresh_cells reconfigures remaining alive cells."""
        all_alive = [0, 1, 2]
        cells = [make_alive_cell(i, alive_cell_indices=all_alive) for i in range(3)]
        group = _make_group_with_cells(cells)
        initial_quorum = group._indep_dp_quorum_id

        # Step 1: Stop cell 1
        cells[1].stop()

        # Step 2: Refresh
        asyncio.get_event_loop().run_until_complete(group._refresh_cells())

        # Step 3: Quorum bumped
        assert group._indep_dp_quorum_id == initial_quorum + 1

        # Step 4: Remaining alive cells have updated indep_dp_info
        assert cells[0].is_alive
        assert cells[0].indep_dp_info.alive_cell_indices == [0, 2]
        assert cells[0].indep_dp_info.alive_rank == 0
        assert cells[0].indep_dp_info.alive_size == 2

        assert cells[2].is_alive
        assert cells[2].indep_dp_info.alive_cell_indices == [0, 2]
        assert cells[2].indep_dp_info.alive_rank == 1

        # Step 5: Stopped cell untouched
        assert cells[1].is_stopped

        # Step 6: Actors received reconfigure_indep_dp
        for cell in [cells[0], cells[2]]:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "reconfigure_indep_dp" for c in calls)

    def test_no_reconfigure_when_unchanged(self):
        all_alive = [0, 1]
        cells = [make_alive_cell(i, alive_cell_indices=all_alive) for i in range(2)]
        group = _make_group_with_cells(cells)
        initial_quorum = group._indep_dp_quorum_id

        asyncio.get_event_loop().run_until_complete(group._refresh_cells())

        assert group._indep_dp_quorum_id == initial_quorum


class TestRefreshCellsHealing:
    def test_pending_cell_gets_healed(self):
        """A pending cell goes through allocate + healing with correct alive_rank."""
        # Step 1: cells [0, 1] alive, cell 2 pending
        cells = [
            make_alive_cell(0, alive_cell_indices=[0, 1]),
            make_alive_cell(1, alive_cell_indices=[0, 1]),
        ]
        pending_cell = make_cell(2)
        pending_cell.stop()
        pending_cell.mark_as_pending()
        cells.append(pending_cell)

        group = _make_group_with_cells(cells)

        # Step 2: Refresh heals the pending cell
        asyncio.get_event_loop().run_until_complete(group._refresh_cells())

        # Step 3: All 3 cells are now alive
        assert all(c.is_alive for c in cells)

        # Step 4: All cells have consistent indep_dp_info
        for cell in cells:
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]
            assert cell.indep_dp_info.alive_size == 3

        assert cells[0].indep_dp_info.alive_rank == 0
        assert cells[1].indep_dp_info.alive_rank == 1
        assert cells[2].indep_dp_info.alive_rank == 2

        # Step 5: Healed cell's actors received init
        for handle in cells[2]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "init" for c in calls)

        # Step 6: Source cell (cell 0) sent ckpt to healed cell's alive_rank
        for handle in cells[0]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            send_calls = [c for c in calls if c[0] == "send_ckpt"]
            assert len(send_calls) == 1
            assert send_calls[0][2]["dst_rank"] == 2  # alive_rank of cell 2 in [0,1,2]

    def test_multiple_pending_cells_healed(self):
        """Multiple pending cells healed simultaneously with correct ranks."""
        cells = [make_alive_cell(0, alive_cell_indices=[0])]
        for i in [1, 2]:
            cell = make_cell(i)
            cell.stop()
            cell.mark_as_pending()
            cells.append(cell)

        group = _make_group_with_cells(cells)
        asyncio.get_event_loop().run_until_complete(group._refresh_cells())

        assert all(c.is_alive for c in cells)
        for cell in cells:
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]

        # Source (cell 0) sent ckpt to both pending cells
        for handle in cells[0]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            send_calls = [c for c in calls if c[0] == "send_ckpt"]
            assert len(send_calls) == 2
            dst_ranks = sorted(c[2]["dst_rank"] for c in send_calls)
            assert dst_ranks == [1, 2]

    def test_healed_cell_receives_set_rollout_manager(self):
        """Healed cell receives set_rollout_manager after init."""
        rollout_mgr = MagicMock()
        cells = [
            make_alive_cell(0, alive_cell_indices=[0]),
        ]
        pending_cell = make_cell(1, rollout_manager=rollout_mgr)
        pending_cell.stop()
        pending_cell.mark_as_pending()
        cells.append(pending_cell)

        group = _make_group_with_cells(cells)
        asyncio.get_event_loop().run_until_complete(group._refresh_cells())

        assert cells[1].is_alive
        for handle in cells[1]._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "set_rollout_manager" for c in calls)

    def test_pending_cell_with_stopped_cell(self):
        """Pending + stopped: only alive and pending participate, stopped excluded."""
        cells = [
            make_alive_cell(0, alive_cell_indices=[0]),
        ]
        stopped_cell = make_cell(1)
        stopped_cell.stop()
        cells.append(stopped_cell)

        pending_cell = make_cell(2)
        pending_cell.stop()
        pending_cell.mark_as_pending()
        cells.append(pending_cell)

        group = _make_group_with_cells(cells)

        asyncio.get_event_loop().run_until_complete(group._refresh_cells())

        # cell 0 and 2 alive, cell 1 still stopped
        assert cells[0].is_alive
        assert cells[1].is_stopped
        assert cells[2].is_alive

        # alive set is [0, 2]
        assert cells[0].indep_dp_info.alive_cell_indices == [0, 2]
        assert cells[0].indep_dp_info.alive_size == 2
        assert cells[2].indep_dp_info.alive_cell_indices == [0, 2]
        assert cells[2].indep_dp_info.alive_rank == 1


class TestRefreshCellsNoOp:
    def test_repeated_refresh_without_change_does_not_reconfigure(self):
        """Calling _refresh_cells multiple times without state changes does not dispatch any actor calls."""
        all_alive = [0, 1, 2]
        cells = [make_alive_cell(i, alive_cell_indices=all_alive) for i in range(3)]
        group = _make_group_with_cells(cells)

        # Step 1: First refresh — no change, should be no-op
        asyncio.get_event_loop().run_until_complete(group._refresh_cells())
        assert group._indep_dp_quorum_id == 0

        # Step 2: Second refresh — still no change
        asyncio.get_event_loop().run_until_complete(group._refresh_cells())
        assert group._indep_dp_quorum_id == 0

        # Step 3: No actor calls were dispatched at all
        for cell in cells:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert len(calls) == 0, f"cell {cell.cell_index} unexpectedly received calls: {calls}"

    def test_refresh_after_reconfigure_is_noop_on_second_call(self):
        """After a reconfigure, the next _refresh_cells with no further changes is a no-op."""
        all_alive = [0, 1, 2]
        cells = [make_alive_cell(i, alive_cell_indices=all_alive) for i in range(3)]
        group = _make_group_with_cells(cells)

        # Step 1: Stop cell 1 → triggers reconfigure
        cells[1].stop()
        asyncio.get_event_loop().run_until_complete(group._refresh_cells())
        assert group._indep_dp_quorum_id == 1

        # Step 2: Clear actor call records
        for cell in [cells[0], cells[2]]:
            for handle in cell._get_actor_handles():
                ray.get(handle.get_calls.remote())  # just to confirm calls exist

        # Step 3: Second refresh — nothing changed, should NOT bump quorum
        asyncio.get_event_loop().run_until_complete(group._refresh_cells())
        assert group._indep_dp_quorum_id == 1  # NOT 2


class TestConsecutiveStopStartCycles:
    def test_stop_train_stop_train_start_train(self):
        """Consecutive stop-start cycle: stop 1 → refresh → stop 2 → refresh → start 1 → refresh."""
        all_alive = [0, 1, 2]
        cells = [make_alive_cell(i, alive_cell_indices=all_alive) for i in range(3)]
        group = _make_group_with_cells(cells)

        # Step 1: Stop cell 1 → refresh
        cells[1].stop()
        asyncio.get_event_loop().run_until_complete(group._refresh_cells())

        assert group._indep_dp_quorum_id == 1
        assert cells[0].indep_dp_info.alive_cell_indices == [0, 2]
        assert cells[2].indep_dp_info.alive_cell_indices == [0, 2]

        # Step 2: Stop cell 2 → refresh (only cell 0 alive)
        cells[2].stop()
        asyncio.get_event_loop().run_until_complete(group._refresh_cells())

        assert group._indep_dp_quorum_id == 2
        assert cells[0].indep_dp_info.alive_cell_indices == [0]
        assert cells[0].indep_dp_info.alive_size == 1

        # Step 3: Start cell 1 → refresh (cells 0 and 1 alive)
        cells[1].mark_as_pending()
        asyncio.get_event_loop().run_until_complete(group._refresh_cells())

        assert group._indep_dp_quorum_id == 3
        assert cells[0].is_alive
        assert cells[1].is_alive
        assert cells[2].is_stopped
        assert cells[0].indep_dp_info.alive_cell_indices == [0, 1]
        assert cells[1].indep_dp_info.alive_cell_indices == [0, 1]
        assert cells[0].indep_dp_info.alive_rank == 0
        assert cells[1].indep_dp_info.alive_rank == 1


class TestAsyncTrain:
    def test_async_train_refreshes_and_dispatches(self):
        """async_train calls _refresh_cells then dispatches train to alive cells."""
        all_alive = [0, 1]
        cells = [make_alive_cell(i, alive_cell_indices=all_alive) for i in range(2)]
        group = _make_group_with_cells(cells)

        refs = group.async_train(rollout_id=0, rollout_data_ref="data")
        ray.get(refs)

        # Both cells should have received train call
        for cell in cells:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "train" for c in calls)

    def test_async_train_with_stopped_cell_only_dispatches_to_alive(self):
        """async_train with a stopped cell only dispatches train to alive cells."""
        all_alive = [0, 1, 2]
        cells = [make_alive_cell(i, alive_cell_indices=all_alive) for i in range(3)]
        group = _make_group_with_cells(cells)

        # Stop cell 1
        cells[1].stop()

        refs = group.async_train(rollout_id=0, rollout_data_ref="data")
        ray.get(refs)

        # Cell 0 and 2 received train, cell 1 did not
        for cell in [cells[0], cells[2]]:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "train" for c in calls)

        assert cells[1].is_stopped

    def test_consecutive_async_train_no_reconfigure_overhead(self):
        """Multiple async_train calls with no state changes should not trigger reconfigure."""
        all_alive = [0, 1, 2]
        cells = [make_alive_cell(i, alive_cell_indices=all_alive) for i in range(3)]
        group = _make_group_with_cells(cells)

        # Step 1: Three consecutive async_train calls
        for step in range(3):
            refs = group.async_train(rollout_id=step, rollout_data_ref="data")
            ray.get(refs)

        # Step 2: Quorum never bumped
        assert group._indep_dp_quorum_id == 0

        # Step 3: No reconfigure_indep_dp calls, only train calls
        for cell in cells:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert not any(c[0] == "reconfigure_indep_dp" for c in calls)
                train_calls = [c for c in calls if c[0] == "train"]
                assert len(train_calls) == 3

    def test_rapid_stop_start_before_async_train(self):
        """Cell stopped and immediately started before next async_train — healed in one refresh."""
        all_alive = [0, 1, 2]
        cells = [make_alive_cell(i, alive_cell_indices=all_alive) for i in range(3)]
        group = _make_group_with_cells(cells)

        # Stop and immediately start cell 1
        group.stop_cell(1)
        group.start_cell(1)

        # async_train heals cell 1 in one shot
        refs = group.async_train(rollout_id=0, rollout_data_ref="data")
        ray.get(refs)

        # All 3 cells alive and trained
        assert all(c.is_alive for c in cells)
        for cell in cells:
            assert cell.indep_dp_info.alive_cell_indices == [0, 1, 2]

    def test_full_lifecycle_through_async_train(self):
        """End-to-end: normal → degraded → steady degraded → healing → full."""
        all_alive = [0, 1, 2]
        cells = [make_alive_cell(i, alive_cell_indices=all_alive) for i in range(3)]
        group = _make_group_with_cells(cells)

        # Step 1: Normal training (no reconfigure)
        ray.get(group.async_train(rollout_id=0, rollout_data_ref="data"))
        assert group._indep_dp_quorum_id == 0

        # Step 2: Stop cell 2 → degraded training (triggers reconfigure)
        group.stop_cell(2)
        ray.get(group.async_train(rollout_id=1, rollout_data_ref="data"))
        assert group._indep_dp_quorum_id == 1
        assert cells[0].indep_dp_info.alive_cell_indices == [0, 1]

        # Step 3: Steady degraded (no reconfigure)
        ray.get(group.async_train(rollout_id=2, rollout_data_ref="data"))
        assert group._indep_dp_quorum_id == 1

        # Step 4: Start cell 2 → healing (triggers reconfigure)
        group.start_cell(2)
        ray.get(group.async_train(rollout_id=3, rollout_data_ref="data"))
        assert group._indep_dp_quorum_id == 2
        assert all(c.is_alive for c in cells)
        assert cells[2].indep_dp_info.alive_cell_indices == [0, 1, 2]

        # Step 5: Full training again (no reconfigure)
        ray.get(group.async_train(rollout_id=4, rollout_data_ref="data"))
        assert group._indep_dp_quorum_id == 2
