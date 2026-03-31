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
        is_pending: bool = False,
        alive_cell_indices: list[int] | None = None,
    ):
        self.cell_index = cell_index
        self._is_alive = is_alive
        self._is_pending = is_pending
        self._alive_cell_indices = alive_cell_indices
        self._indep_dp_info: IndepDPInfo | None = None
        self._execute_calls: list[tuple[str, tuple, dict]] = []
        self.allocate_for_pending_calls: int = 0
        self.prepare_alive_calls: list[dict] = []
        self.prepare_healing_calls: list[dict] = []

    @property
    def is_alive(self) -> bool:
        return self._is_alive

    @property
    def is_pending(self) -> bool:
        return self._is_pending

    @property
    def is_stopped(self) -> bool:
        return not self._is_alive and not self._is_pending

    @property
    def indep_dp_info(self) -> IndepDPInfo:
        assert self._is_alive
        if self._indep_dp_info is not None:
            return self._indep_dp_info
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

    def allocate_for_pending(self) -> None:
        self.allocate_for_pending_calls += 1
        self._is_pending = False
        self._is_alive = True

    async def prepare_indep_dp_mode_alive(self, *, indep_dp_info: IndepDPInfo, send_ckpt_dst_ranks: list[int]) -> None:
        self.prepare_alive_calls.append(dict(indep_dp_info=indep_dp_info, send_ckpt_dst_ranks=send_ckpt_dst_ranks))
        self._indep_dp_info = indep_dp_info

    async def prepare_indep_dp_mode_healing(self, *, indep_dp_info: IndepDPInfo, recv_ckpt_src_rank: int | None) -> None:
        self.prepare_healing_calls.append(dict(indep_dp_info=indep_dp_info, recv_ckpt_src_rank=recv_ckpt_src_rank))
        self._is_alive = True
        self._indep_dp_info = indep_dp_info


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

    def test_pending_cell_gets_healed(self):
        """A pending cell goes through allocate + prepare_healing with correct alive_rank."""
        # Step 1: cells [0, 1] alive, cell 2 pending (was stopped, then start() called)
        cells = [
            _MockCell(0, alive_cell_indices=[0, 1]),
            _MockCell(1, alive_cell_indices=[0, 1]),
            _MockCell(2, is_pending=True, is_alive=False),
        ]
        group = _make_group_with_cells(cells)

        asyncio.get_event_loop().run_until_complete(group._refresh_cells())

        # Step 2: cell 2 was allocated
        assert cells[2].allocate_for_pending_calls == 1

        # Step 3: alive cells got prepare_alive (reconfigure)
        assert len(cells[0].prepare_alive_calls) == 1
        assert len(cells[1].prepare_alive_calls) == 1

        # Step 4: pending cell got prepare_healing
        assert len(cells[2].prepare_healing_calls) == 1
        healing_info = cells[2].prepare_healing_calls[0]["indep_dp_info"]
        assert healing_info.cell_index == 2
        assert healing_info.alive_rank == 2
        assert healing_info.alive_size == 3
        assert healing_info.alive_cell_indices == [0, 1, 2]

        # Step 5: src cell (cell 0) sends ckpt to healing cell's alive_rank
        src_call = cells[0].prepare_alive_calls[0]
        assert src_call["send_ckpt_dst_ranks"] == [2]  # alive_rank of cell 2

        # Step 6: healing cell receives from src's alive_rank
        assert cells[2].prepare_healing_calls[0]["recv_ckpt_src_rank"] == 0  # alive_rank of cell 0

        # Step 7: non-src alive cell sends nothing
        assert cells[1].prepare_alive_calls[0]["send_ckpt_dst_ranks"] == []

    def test_pending_cell_with_stopped_cell(self):
        """Pending + stopped: only alive and pending cells participate, stopped excluded."""
        # cells [0] alive, cell 1 stopped, cell 2 pending
        cells = [
            _MockCell(0, alive_cell_indices=[0]),
            _MockCell(1, is_alive=False),  # stopped
            _MockCell(2, is_pending=True, is_alive=False),
        ]
        group = _make_group_with_cells(cells)

        asyncio.get_event_loop().run_until_complete(group._refresh_cells())

        # cell 2 allocated and healed
        assert cells[2].allocate_for_pending_calls == 1
        assert len(cells[2].prepare_healing_calls) == 1

        # cell 0 reconfigured with alive set [0, 2]
        assert len(cells[0].prepare_alive_calls) == 1
        alive_info = cells[0].prepare_alive_calls[0]["indep_dp_info"]
        assert alive_info.alive_cell_indices == [0, 2]
        assert alive_info.alive_size == 2

        # cell 1 untouched
        assert cells[1].allocate_for_pending_calls == 0
        assert cells[1].prepare_alive_calls == []
        assert cells[1].prepare_healing_calls == []
