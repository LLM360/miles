import asyncio
from unittest.mock import MagicMock

import pytest
import ray

from miles.ray.train.cell import RayTrainCell
from miles.ray.train.group import RayTrainGroup
from miles.utils.indep_dp import IndepDPInfo
from tests.fast.ray.train.dummy_actor import DummyTrainActor


@pytest.fixture(scope="module", autouse=True)
def ray_env():
    ray.init(num_cpus=4, num_gpus=0, ignore_reinit_error=True)
    yield
    ray.shutdown()


def _make_dummy_factory(actor_count: int = 2):
    def factory():
        return [DummyTrainActor.remote() for _ in range(actor_count)]

    return factory


def _make_cell(cell_index: int, *, actor_count: int = 1) -> RayTrainCell:
    return RayTrainCell(
        args=MagicMock(),
        role="actor",
        with_ref=False,
        cell_index=cell_index,
        actor_factory=_make_dummy_factory(actor_count),
        rollout_manager=None,
    )


def _make_alive_cell(cell_index: int, *, alive_cell_indices: list[int], quorum_id: int = 0) -> RayTrainCell:
    """Create a cell and transition it to Alive state."""
    cell = _make_cell(cell_index)
    cell._mark_as_alive(
        indep_dp_info=IndepDPInfo(
            cell_index=cell_index,
            num_cells=3,
            alive_rank=alive_cell_indices.index(cell_index),
            alive_size=len(alive_cell_indices),
            quorum_id=quorum_id,
            alive_cell_indices=alive_cell_indices,
        )
    )
    return cell


def _make_group_with_cells(cells: list[RayTrainCell]) -> RayTrainGroup:
    """Create a RayTrainGroup with pre-set cells, bypassing __init__."""
    group = object.__new__(RayTrainGroup)
    group._cells = cells
    group._indep_dp_quorum_id = 0
    return group


class TestComputeIndepDPInfo:
    def test_all_alive(self):
        """Correct IndepDPInfo for all-alive scenario."""
        group = _make_group_with_cells([_make_cell(i) for i in range(3)])

        info = group._compute_indep_dp_info(cell_index=2, alive_cell_indices=[0, 1, 2])

        assert info.alive_rank == 2
        assert info.alive_size == 3
        assert info.cell_index == 2
        assert info.alive_cell_indices == [0, 1, 2]

    def test_with_gap(self):
        """When cell 1 is missing, cell 2 gets alive_rank=1."""
        group = _make_group_with_cells([_make_cell(i) for i in range(3)])

        info = group._compute_indep_dp_info(cell_index=2, alive_cell_indices=[0, 2])

        assert info.alive_rank == 1
        assert info.alive_size == 2


class TestAsyncExecuteAlive:
    def test_skips_stopped_cells(self):
        """_async_execute_alive only dispatches to alive cells."""
        alive_cell = _make_alive_cell(0, alive_cell_indices=[0])
        stopped_cell = _make_cell(1)
        stopped_cell.stop()
        group = _make_group_with_cells([alive_cell, stopped_cell])

        refs = group._async_execute_alive("train")

        assert len(refs) == 1
        ray.get(refs)

        # Verify alive cell's actor received the call
        for handle in alive_cell._get_actor_handles():
            calls = ray.get(handle.get_calls.remote())
            assert any(c[0] == "train" for c in calls)

    def test_asserts_on_no_alive_cells(self):
        """_async_execute_alive asserts when no cells are alive."""
        stopped = _make_cell(0)
        stopped.stop()
        group = _make_group_with_cells([stopped])

        with pytest.raises(AssertionError, match="No alive cells"):
            group._async_execute_alive("train")


class TestRefreshCellsReconfigure:
    def test_reconfigure_triggers_on_alive_change(self):
        """When a cell is stopped, _refresh_cells reconfigures remaining alive cells."""
        all_alive = [0, 1, 2]
        cells = [_make_alive_cell(i, alive_cell_indices=all_alive) for i in range(3)]
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

        # Step 6: Alive cells' actors received reconfigure_indep_dp
        for cell in [cells[0], cells[2]]:
            for handle in cell._get_actor_handles():
                calls = ray.get(handle.get_calls.remote())
                assert any(c[0] == "reconfigure_indep_dp" for c in calls)

    def test_no_reconfigure_when_unchanged(self):
        """When alive set has not changed, no reconfiguration happens."""
        all_alive = [0, 1]
        cells = [_make_alive_cell(i, alive_cell_indices=all_alive) for i in range(2)]
        group = _make_group_with_cells(cells)
        initial_quorum = group._indep_dp_quorum_id

        asyncio.get_event_loop().run_until_complete(group._refresh_cells())

        assert group._indep_dp_quorum_id == initial_quorum


class TestRefreshCellsHealing:
    def test_pending_cell_gets_healed(self):
        """A pending cell goes through allocate + healing with correct alive_rank."""
        # Step 1: cells [0, 1] alive, cell 2 pending
        cells = [
            _make_alive_cell(0, alive_cell_indices=[0, 1]),
            _make_alive_cell(1, alive_cell_indices=[0, 1]),
        ]
        pending_cell = _make_cell(2)
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

    def test_pending_cell_with_stopped_cell(self):
        """Pending + stopped: only alive and pending participate, stopped excluded."""
        cells = [
            _make_alive_cell(0, alive_cell_indices=[0]),
        ]
        stopped_cell = _make_cell(1)
        stopped_cell.stop()
        cells.append(stopped_cell)

        pending_cell = _make_cell(2)
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
