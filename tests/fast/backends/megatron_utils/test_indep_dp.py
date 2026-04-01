import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.megatron_utils.indep_dp import _collective_bool_and


def _worker_collective_bool_and(
    rank: int, world_size: int, port: int, *, value_by_rank: dict[int, bool], expected: bool
) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        result = _collective_bool_and(value=value_by_rank[rank], group=group)
        assert result is expected, f"rank {rank}: expected {expected}, got {result}"
    finally:
        dist.destroy_process_group()


class TestCollectiveBoolAnd:
    def test_all_true(self) -> None:
        def _worker(rank: int, world_size: int, port: int) -> None:
            _worker_collective_bool_and(rank, world_size, port, value_by_rank={0: True, 1: True}, expected=True)

        run_multiprocess(_worker)

    def test_all_false(self) -> None:
        def _worker(rank: int, world_size: int, port: int) -> None:
            _worker_collective_bool_and(rank, world_size, port, value_by_rank={0: False, 1: False}, expected=False)

        run_multiprocess(_worker)

    def test_mixed_returns_false(self) -> None:
        def _worker(rank: int, world_size: int, port: int) -> None:
            _worker_collective_bool_and(rank, world_size, port, value_by_rank={0: True, 1: False}, expected=False)

        run_multiprocess(_worker)
