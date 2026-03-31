import logging
from datetime import timedelta

import torch.distributed as dist

from miles.utils.indep_dp_group_info import IndepDPGroupInfo
from miles.utils.process_group_utils import GroupInfo

from ..training_utils.parallel import ParallelState

logger = logging.getLogger(__name__)


def create_indep_dp_group(
    store_addr: str | None,
    indep_dp_group_info: IndepDPGroupInfo,
    megatron_rank: int,
    megatron_world_size: int,
    indep_dp_quorum_id: int,
) -> GroupInfo:
    info = indep_dp_group_info

    if info.alive_size <= 1:
        return GroupInfo(rank=0, size=1, group=None)

    from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL

    def _create(pg_cls: type, backend_name: str) -> dist.ProcessGroup:
        pg = pg_cls(timeout=timedelta(seconds=60))
        pg.configure(
            store_addr=f"{store_addr}/indep_dp/{backend_name}/{indep_dp_quorum_id}/{megatron_rank}",
            replica_id=str(info.cell_id),
            rank=info.alive_rank,
            world_size=info.alive_size,
            quorum_id=indep_dp_quorum_id,
            group_rank=megatron_rank,
            group_world_size=megatron_world_size,
        )
        return pg

    nccl_pg = _create(ProcessGroupNCCL, "nccl")
    gloo_pg = _create(ProcessGroupGloo, "gloo")
    logger.info(
        f"Configured independent DP PG: cell_id={info.cell_id}, alive_rank={info.alive_rank}, "
        f"alive_size={info.alive_size}, num_cells={info.num_cells}, "
        f"megatron_rank={megatron_rank}, megatron_world_size={megatron_world_size}, "
        f"indep_dp_quorum_id={indep_dp_quorum_id}"
    )
    return GroupInfo(rank=info.alive_rank, size=info.alive_size, group=nccl_pg, gloo_group=gloo_pg)


def reconfigure_indep_dp_group(
    parallel_state: ParallelState,
    store_addr: str | None,
    indep_dp_group_info: IndepDPGroupInfo,
    megatron_rank: int,
    megatron_world_size: int,
    indep_dp_quorum_id: int,
) -> None:
    """Shutdown old indep_dp PGs and create new ones with a fresh quorum_id."""
    old = parallel_state.indep_dp
    for g in [old.group, old.gloo_group]:
        if g is not None:
            g.shutdown()

    parallel_state.indep_dp = create_indep_dp_group(
        store_addr=store_addr,
        indep_dp_group_info=indep_dp_group_info,
        megatron_rank=megatron_rank,
        megatron_world_size=megatron_world_size,
        indep_dp_quorum_id=indep_dp_quorum_id,
    )
    logger.info(f"Reconfigured indep_dp PG with indep_dp_quorum_id={indep_dp_quorum_id}")
