"""Agree on metric column names before positional training collectives."""

import json

import torch

from miles.utils.ft_utils.process_group_utils import GeneralPGUtil


def synchronize_metric_keys(keys, groups):
    """Share only names; numerical values keep the existing reduction path.

    Every rank participates, even if its shard has no labeled domains. Repeating
    the union over the inner/outer groups also covers independent replicas with
    different data. GeneralPGUtil preserves torchft's waited collectives.
    """
    keys = sorted(set(keys))
    for group in groups:
        util = GeneralPGUtil.create(group)
        size = util.get_size(group)
        if size == 1:
            continue
        is_root = util.get_rank(group) == 0
        gathered = [None] * size if is_root else None
        util.gather_object(keys, gathered, group)
        payload = json.dumps(sorted({key for row in gathered for key in row})).encode() if is_root else b""
        length = torch.tensor([len(payload)], dtype=torch.int64)
        util.broadcast(length, group)
        buffer = (
            torch.tensor(list(payload), dtype=torch.uint8)
            if is_root
            else torch.empty(length.item(), dtype=torch.uint8)
        )
        util.broadcast(buffer, group)
        keys = json.loads(bytes(buffer.tolist()))
    return keys


def sum_aligned_metrics(losses_reduced, keys, *, device=None):
    """Sum microbatches by name; absent optional/domain metrics contribute zero."""
    if losses_reduced:
        values = losses_reduced[0]["values"].new_zeros(len(keys) + 1)
    else:
        values = torch.zeros(len(keys) + 1, device=device)
    positions = {key: i + 1 for i, key in enumerate(keys)}
    for row in losses_reduced:
        row_keys, row_values = row["keys"], row["values"]
        if len(row_keys) != len(set(row_keys)) or len(row_keys) + 1 != row_values.numel():
            raise ValueError("Training metric keys must be unique and match their value tensor")
        index = torch.tensor([0, *[positions[key] for key in row_keys]], device=values.device)
        values.index_add_(0, index, row_values.to(values))
    return values
