from unittest.mock import MagicMock

import pytest
import ray

from miles.ray.train.cell import RayTrainCell
from miles.utils.indep_dp import IndepDPInfo
from tests.fast.ray.train.dummy_actor import DummyTrainActor


@pytest.fixture(scope="module", autouse=True)
def ray_env():
    ray.init(num_cpus=4, num_gpus=0, ignore_reinit_error=True)
    yield
    ray.shutdown()


def make_indep_dp_info(
    *,
    cell_index: int = 0,
    alive_cell_indices: list[int] | None = None,
    quorum_id: int = 1,
) -> IndepDPInfo:
    if alive_cell_indices is None:
        alive_cell_indices = [0]
    return IndepDPInfo(
        cell_index=cell_index,
        num_cells=3,
        alive_rank=alive_cell_indices.index(cell_index),
        alive_size=len(alive_cell_indices),
        quorum_id=quorum_id,
        alive_cell_indices=alive_cell_indices,
    )


def make_cell(cell_index: int = 0, *, actor_count: int = 2) -> RayTrainCell:
    def factory():
        return [DummyTrainActor.remote() for _ in range(actor_count)]

    return RayTrainCell(
        args=MagicMock(),
        role="actor",
        with_ref=False,
        cell_index=cell_index,
        actor_factory=factory,
        rollout_manager=None,
    )


def make_alive_cell(cell_index: int, *, alive_cell_indices: list[int], quorum_id: int = 0) -> RayTrainCell:
    """Create a cell and transition it to Alive state."""
    cell = make_cell(cell_index)
    cell._mark_as_alive(indep_dp_info=make_indep_dp_info(
        cell_index=cell_index,
        alive_cell_indices=alive_cell_indices,
        quorum_id=quorum_id,
    ))
    return cell
