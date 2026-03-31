from dataclasses import dataclass


@dataclass(frozen=True)
class IndepDPInfo:
    cell_index: int
    num_cells: int
    alive_rank: int
    alive_size: int
    quorum_id: int
    alive_cell_indices: list[int]
