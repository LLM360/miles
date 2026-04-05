import logging
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from miles.utils.event_logger.logger import get_event_logger
from miles.utils.event_logger.models import WitnessSnapshotParamEvent
from miles.utils.witness.allocator import WitnessInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install_witness(model: nn.Module, *, buffer_size: int) -> None:
    model.head_witness = _DataWitness(buffer_size=buffer_size)
    model.tail_witness = _DataWitness(buffer_size=buffer_size)


def witness_dump_and_clear_stale(
    *,
    model: Sequence[nn.Module],
    witness_info: WitnessInfo,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Log nonzero witness param rows, then clear stale ring buffer entries."""
    for chunk_index, chunk in enumerate(model):
        for attr in _WITNESS_ATTRS:
            witness: _DataWitness = getattr(chunk.module, attr)
            _record_and_log_witness_param(
                witness=witness,
                instance_id=f"pp{chunk_index}." + attr.replace("_witness", ""),
                stale_ids=witness_info.stale_ids,
            )

    _clear_witness_stale_rows(model=model, stale_ids=witness_info.stale_ids, optimizer=optimizer)


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


class _DataWitness(nn.Module):
    def __init__(self, buffer_size: int) -> None:
        super().__init__()
        self.buffer_size = buffer_size
        self.witness = nn.Embedding(num_embeddings=buffer_size, embedding_dim=1)
        nn.init.zeros_(self.witness.weight)

    def forward(self, input_ids: Tensor, witness_ids: Tensor) -> Tensor:
        assert input_ids.shape == witness_ids.shape
        w = self.witness(witness_ids)  # (*, 1)
        out = w - w.detach()  # forward: bitwise 0 (for finite w), backward: d/dw = I
        return out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


_WITNESS_ATTRS = ("head_witness", "tail_witness")


def _clear_witness_stale_rows(
    *,
    model: Sequence[nn.Module],
    stale_ids: list[int],
    optimizer: torch.optim.Optimizer,
) -> None:
    if not stale_ids:
        return

    witnesses = list(_get_all_witnesses_in_model(model))
    for witness in witnesses:
        idx = torch.tensor(stale_ids, dtype=torch.long, device=witness.witness.weight.device)
        _zero_witness_rows(witness=witness, idx=idx, optimizer=optimizer)


def _get_all_witnesses_in_model(model_chunks: Sequence[nn.Module]) -> list[_DataWitness]:
    witnesses: list[_DataWitness] = []
    for chunk in model_chunks:
        for attr in _WITNESS_ATTRS:
            witnesses.append(getattr(chunk.module, attr))
    return witnesses


def _zero_witness_rows(*, witness: _DataWitness, idx: Tensor, optimizer: torch.optim.Optimizer) -> None:
    model_weight = witness.witness.weight
    model_weight.data[idx] = 0.0

    opt_weight: Tensor = getattr(model_weight, "main_param", model_weight)
    if opt_weight is not model_weight:
        opt_weight.data[idx] = 0.0

    if opt_weight in optimizer.state:
        state = optimizer.state[opt_weight]
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                state[key][idx] = 0.0


def _record_and_log_witness_param(
    *,
    witness: _DataWitness,
    instance_id: str,
    stale_ids: list[int],
) -> None:
    weight = witness.witness.weight.data
    nonzero_witness_ids: list[int] = weight.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

    get_event_logger().log(
        WitnessSnapshotParamEvent,
        dict(
            instance_id=instance_id,
            nonzero_witness_ids=nonzero_witness_ids,
            stale_ids=stale_ids,
        ),
        print_log=False,
    )
