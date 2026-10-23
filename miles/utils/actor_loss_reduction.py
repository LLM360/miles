"""Actor-only token mean using the full optimizer batch, not each microbatch."""
import torch
import torch.distributed as dist

SCALE_KEY = "_actor_loss_token_scale"


def validate_actor_reduction(args):
    if getattr(args, "actor_loss_agg_mode", None) != "token-mean":
        return
    if (getattr(args, "train_backend", "megatron") != "megatron"
            or getattr(args, "context_parallel_size", 1) != 1
            or getattr(args, "pipeline_model_parallel_size", 1) != 1):
        raise ValueError("Actor-only token mean currently requires Megatron, CP=PP=1")
    if (getattr(args, "calculate_per_token_loss", False)
            or getattr(args, "loss_agg_mode", None) not in (None, "sample-mean")
            or getattr(args, "loss_type", "policy_loss") != "policy_loss"):
        raise ValueError("Actor-only token mean requires the native sample-mean critic/policy loss path")
    for key in ("use_tis", "get_mismatch_metrics", "use_opsm", "use_gspo",
                "custom_pg_loss_reducer_function_path", "custom_tis_function_path"):
        if getattr(args, key, False):
            raise ValueError(f"Actor-only token mean does not support {key}: token membership/reduction may change")


def attach_token_scales(args, data, parallel):
    """One DP reduction per rollout; a separate denominator per optimizer step.

    Iterators partition each contiguous local optimizer batch into microbatches.
    The scale B/N cancels the native outer /B while leaving critic scaling intact.
    Full masks are counted before packing. CP=1 prevents replicated-token counts.
    """
    data.pop(SCALE_KEY, None)
    if getattr(args, "actor_loss_agg_mode", None) != "token-mean":
        return
    validate_actor_reduction(args)
    if parallel.cp.size != 1:
        raise ValueError("Actor token denominator requires CP=1")
    global_batch = data.get("dynamic_global_batch_size", args.global_batch_size)
    dp_size = parallel.intra_dp.size
    if global_batch <= 0 or global_batch % dp_size:
        raise ValueError("Global batch must divide evenly across DP ranks")
    local_batch = global_batch // dp_size
    masks = data["loss_masks"]
    if not masks or len(masks) % local_batch:
        raise ValueError("Incomplete optimizer batch: cannot compute token denominator")
    counts = torch.stack([torch.stack([x.detach().double().sum() for x in masks[i:i+local_batch]]).sum()
                          for i in range(0, len(masks), local_batch)])
    if dp_size > 1:
        dist.all_reduce(counts, group=parallel.intra_dp.group)
    if not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Every optimizer batch needs a finite, positive valid-token count")
    data[SCALE_KEY] = [global_batch / n for n in counts.cpu().tolist() for _ in range(local_batch)]


def actor_reducer(args, batch, sequence_reducer):
    if getattr(args, "actor_loss_agg_mode", None) != "token-mean":
        return sequence_reducer
    validate_actor_reduction(args)
    scales = batch.get(SCALE_KEY)
    if not scales or any(s != scales[0] for s in scales):
        raise ValueError("Missing token denominator or microbatch crosses optimizer-step boundary")
    mask = torch.cat(batch["loss_masks"])
    scale = scales[0]
    def reduce(values):
        if values.shape != mask.shape:
            raise ValueError("Actor values and full response masks must align")
        return (values * mask.to(values)).sum() * scale
    return reduce
