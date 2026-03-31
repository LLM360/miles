from unittest.mock import MagicMock, patch

import pytest

from miles.ray.train.group import RayTrainGroup
from miles.utils.indep_dp_group_info import IndepDPGroupInfo


class _MockCell:
    def __init__(self, cell_id: int, *, is_running: bool = True):
        self.cell_id = cell_id
        self._is_running = is_running
        self._execute_calls: list[tuple[str, tuple, dict]] = []

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_pending(self) -> bool:
        return False

    @property
    def is_stopped(self) -> bool:
        return not self._is_running

    def async_execute(self, fn_name: str, *args, **kwargs) -> list:
        self._execute_calls.append((fn_name, args, kwargs))
        sentinel = MagicMock()
        return [sentinel]


def _make_group_with_cells(cells: list[_MockCell]) -> RayTrainGroup:
    """Create a RayTrainGroup with pre-set mock cells, bypassing __init__."""
    group = object.__new__(RayTrainGroup)
    group._cells = cells
    group._indep_dp_quorum_id = 0
    group._alive_cell_ids = frozenset(c.cell_id for c in cells if c.is_running)
    return group


class TestComputeAliveMapping:
    def test_all_running(self):
        """3 cells all running produce identity mapping."""
        cells = [_MockCell(i) for i in range(3)]
        group = _make_group_with_cells(cells)
        assert group._compute_alive_mapping() == {0: 0, 1: 1, 2: 2}

    def test_with_stopped_cell(self):
        """3 cells with cell 1 stopped produce gap-free mapping."""
        cells = [_MockCell(0), _MockCell(1, is_running=False), _MockCell(2)]
        group = _make_group_with_cells(cells)
        assert group._compute_alive_mapping() == {0: 0, 2: 1}

    def test_single_alive(self):
        """3 cells with only cell 1 running."""
        cells = [
            _MockCell(0, is_running=False),
            _MockCell(1),
            _MockCell(2, is_running=False),
        ]
        group = _make_group_with_cells(cells)
        assert group._compute_alive_mapping() == {1: 0}


class TestAsyncExecuteSkipsStoppedCells:
    @patch("miles.ray.train.group.ray")
    def test_stopped_cell_not_called(self, mock_ray: MagicMock):
        """_async_execute only dispatches to running cells."""
        running_cell = _MockCell(0)
        stopped_cell = _MockCell(1, is_running=False)
        group = _make_group_with_cells([running_cell, stopped_cell])

        group._async_execute("train", 42, "data_ref")

        assert len(running_cell._execute_calls) == 1
        assert running_cell._execute_calls[0][0] == "train"
        assert stopped_cell._execute_calls == []


class TestReconfigureTriggeredOnAliveChange:
    @patch("miles.ray.train.group.ray")
    def test_reconfigure_triggers_on_change(self, mock_ray: MagicMock):
        """When a cell is stopped, _reconfigure_if_alive_changed triggers reconfiguration."""
        mock_ray.get.return_value = None

        cells = [_MockCell(0), _MockCell(1), _MockCell(2)]
        group = _make_group_with_cells(cells)
        initial_quorum = group._indep_dp_quorum_id

        # Stop cell 1
        cells[1]._is_running = False

        group._reconfigure_if_alive_changed()

        assert group._indep_dp_quorum_id == initial_quorum + 1
        assert group._alive_cell_ids == frozenset({0, 2})

        # Verify reconfigure was called on running cells
        assert len(cells[0]._execute_calls) == 1
        assert cells[0]._execute_calls[0][0] == "reconfigure_indep_dp"
        assert len(cells[2]._execute_calls) == 1
        assert cells[2]._execute_calls[0][0] == "reconfigure_indep_dp"
        assert cells[1]._execute_calls == []

        # Verify IndepDPGroupInfo passed to cell 0
        info_0 = cells[0]._execute_calls[0][2]["indep_dp_group_info"]
        assert isinstance(info_0, IndepDPGroupInfo)
        assert info_0.cell_index == 0
        assert info_0.alive_rank == 0
        assert info_0.alive_size == 2

        # Verify IndepDPGroupInfo passed to cell 2
        info_2 = cells[2]._execute_calls[0][2]["indep_dp_group_info"]
        assert info_2.cell_index == 2
        assert info_2.alive_rank == 1
        assert info_2.alive_size == 2

    @patch("miles.ray.train.group.ray")
    def test_reconfigure_skips_when_unchanged(self, mock_ray: MagicMock):
        """When alive set has not changed, no reconfiguration happens."""
        cells = [_MockCell(0), _MockCell(1)]
        group = _make_group_with_cells(cells)
        initial_quorum = group._indep_dp_quorum_id

        group._reconfigure_if_alive_changed()

        assert group._indep_dp_quorum_id == initial_quorum
        assert cells[0]._execute_calls == []
        assert cells[1]._execute_calls == []
