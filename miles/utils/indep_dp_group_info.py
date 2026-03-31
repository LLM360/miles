from dataclasses import dataclass


@dataclass(frozen=True)
class IndepDPGroupInfo:
    cell_id: int
    num_cells: int
    alive_rank: int
    alive_size: int
