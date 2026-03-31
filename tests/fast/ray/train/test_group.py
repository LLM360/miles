import asyncio
from unittest.mock import MagicMock, patch

from miles.ray.train.group import RayTrainGroup
from miles.utils.indep_dp import IndepDPInfo


class _MockCell:
    def __init__(
        self,
        cell_index: int,
        *,
        is_alive: bool = True,
        alive_cell_indices: list[int] | None = None,
    ):
        self.cell_index = cell_index
        self._is_alive = is_alive
        self._alive_cell_indices = alive_cell_indices
        self._execute_calls: list[tuple[str, tuple, dict]] = []

    @property
    def is_alive(self) -> bool:
        return self._is_alive

    @property
    def is_pending(self) -> bool:
        return False

    @property
    def is_stopped(self) -> bool:
        return not self._is_alive

    @property
    def indep_dp_info(self) -> IndepDPInfo:
        assert self._is_alive
        assert self._alive_cell_indices is not None
        return IndepDPInfo(
            cell_index=self.cell_index,
            num_cells=3,
            alive_rank=self._alive_cell_indices.index(self.cell_index),
            alive_size=len(self._alive_cell_indices),
            quorum_id=0,
            alive_cell_indices=self._alive_cell_indices,
        )

    def async_execute(self, fn_name: str, *args, **kwargs) -> list:
        self._execute_calls.append((fn_name, args, kwargs))
        sentinel = MagicMock()
        return [sentinel]


def _make_group_with_cells(cells: list[_MockCell]) -> RayTrainGroup:
    """Create a RayTrainGroup with pre-set mock cells, bypassing __init__."""
    group = object.__new__(RayTrainGroup)
    group._cells = cells
    group._indep_dp_quorum_id = 0
    return group


class TestComputeIndepDPInfo:
    def test_all_alive(self):
        """3 cells all alive produce correct IndepDPInfo for each."""
        group = _make_group_with_cells([_MockCell(i) for i in range(3)])

        info_0 = group._compute_indep_dp_info(cell_index=0, alive_cell_indices=[0, 1, 2])
        info_2 = group._compute_indep_dp_info(cell_index=2, alive_cell_indices=[0, 1, 2])

        assert info_0.alive_rank == 0
        assert info_0.alive_size == 3
        assert info_2.alive_rank == 2
        assert info_2.alive_size == 3

    def test_with_gap(self):
        """When cell 1 is missing from alive set, cell 2 gets alive_rank=1."""
        group = _make_group_with_cells([_MockCell(i) for i in range(3)])

        info_2 = group._compute_indep_dp_info(cell_index=2, alive_cell_indices=[0, 2])

        assert info_2.alive_rank == 1
        assert info_2.alive_size == 2
        assert info_2.alive_cell_indices == [0, 2]


class TestAsyncExecuteAlive:
    @patch("miles.ray.train.group.ray")
    def test_skips_stopped_cells(self, mock_ray: MagicMock):
        """_async_execute_alive only dispatches to alive cells."""
        alive_cell = _MockCell(0, alive_cell_indices=[0])
        stopped_cell = _MockCell(1, is_alive=False)
        group = _make_group_with_cells([alive_cell, stopped_cell])

        group._async_execute_alive("train", 42, "data_ref")

        assert len(alive_cell._execute_calls) == 1
        assert alive_cell._execute_calls[0][0] == "train"
        assert stopped_cell._execute_calls == []

    def test_asserts_on_no_alive_cells(self):
        """_async_execute_alive asserts when no cells are alive."""
        stopped = _MockCell(0, is_alive=False)
        group = _make_group_with_cells([stopped])

        try:
            group._async_execute_alive("train")
            assert False, "Should have raised"
        except AssertionError as e:
            assert "No alive cells" in str(e)


class TestRefreshCells:
    @patch("miles.ray.train.group.ray")
    def test_reconfigure_triggers_on_alive_change(self, mock_ray: MagicMock):
        """When alive set changes, _refresh_cells triggers reconfiguration."""
        mock_ray.get.return_value = None

        all_alive = [0, 1, 2]
        cells = [
            _MockCell(0, alive_cell_indices=all_alive),
            _MockCell(1, alive_cell_indices=all_alive),
            _MockCell(2, alive_cell_indices=all_alive),
        ]
        group = _make_group_with_cells(cells)
        initial_quorum = group._indep_dp_quorum_id

        # Stop cell 1
        cells[1]._is_alive = False

        asyncio.get_event_loop().run_until_complete(group._refresh_cells())

        assert group._indep_dp_quorum_id == initial_quorum + 1

        # Verify reconfigure was called on alive cells only
        assert len(cells[0]._execute_calls) == 1
        assert cells[0]._execute_calls[0][0] == "reconfigure_indep_dp"
        assert len(cells[2]._execute_calls) == 1
        assert cells[2]._execute_calls[0][0] == "reconfigure_indep_dp"
        assert cells[1]._execute_calls == []

        # Verify IndepDPInfo passed to cell 0
        info_0 = cells[0]._execute_calls[0][2]["indep_dp_info"]
        assert isinstance(info_0, IndepDPInfo)
        assert info_0.cell_index == 0
        assert info_0.alive_rank == 0
        assert info_0.alive_size == 2
        assert info_0.alive_cell_indices == [0, 2]

        # Verify IndepDPInfo passed to cell 2
        info_2 = cells[2]._execute_calls[0][2]["indep_dp_info"]
        assert info_2.cell_index == 2
        assert info_2.alive_rank == 1
        assert info_2.alive_size == 2

    @patch("miles.ray.train.group.ray")
    def test_no_reconfigure_when_unchanged(self, mock_ray: MagicMock):
        """When alive set has not changed, no reconfiguration happens."""
        all_alive = [0, 1]
        cells = [
            _MockCell(0, alive_cell_indices=all_alive),
            _MockCell(1, alive_cell_indices=all_alive),
        ]
        group = _make_group_with_cells(cells)
        initial_quorum = group._indep_dp_quorum_id

        asyncio.get_event_loop().run_until_complete(group._refresh_cells())

        assert group._indep_dp_quorum_id == initial_quorum
        assert cells[0]._execute_calls == []
        assert cells[1]._execute_calls == []
