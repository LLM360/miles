"""Real CPU collectives with disjoint domains and optional metric columns."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, world_size, rendezvous, output_dir):
    from miles.backends.training_utils import log_utils

    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=40)
    )
    try:
        groups = [dist.group.WORLD]
        if world_size == 4:
            # Two CP/intra groups crossed with two independent-replica groups.
            groups = []
            for members in ([0, 1], [2, 3], [0, 2], [1, 3]):
                group = dist.new_group(members, backend="gloo", timeout=timedelta(seconds=40))
                if rank in members:
                    groups.append(group)
        state = SimpleNamespace(
            cp=SimpleNamespace(size=1),
            effective_dp_cp=SimpleNamespace(
                size=world_size,
                gloo_groups_inner_to_outer=groups,
                groups_inner_to_outer=groups,
            ),
        )
        log_utils.get_parallel_state = lambda: state
        domain = ["math", "code", "science", None][rank]
        value = [2.0, 8.0, 6.0, 5.0][rank]
        metrics = {"loss": value}
        if domain is not None:
            metrics[f"loss/{domain}"] = value
        if rank == 1:
            metrics["ref_kl/code"] = 0.4
        row = {"keys": list(metrics), "values": torch.tensor([1.0, *metrics.values()])}
        result = log_utils.aggregate_train_losses([row])
        (Path(output_dir) / f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_disjoint_domains_across_workers_and_independent_replicas(tmp_path, world_size):
    mp.spawn(_worker, args=(world_size, f"file://{tmp_path}/rendezvous", str(tmp_path)), nprocs=world_size, join=True)
    expected = {"loss": 5.0, "loss/math": 1.0, "loss/code": 4.0, "ref_kl/code": 0.2}
    if world_size == 4:
        expected = {"loss": 5.25, "loss/math": 0.5, "loss/code": 2.0, "loss/science": 1.5, "ref_kl/code": 0.1}
    for rank in range(world_size):
        result = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert result == pytest.approx(expected)
