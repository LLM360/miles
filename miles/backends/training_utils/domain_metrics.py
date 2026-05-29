"""Domain contributions to training metrics, using each parent metric's weights."""

import torch

from miles.backends.training_utils.cp_utils import get_local_response_loss_masks, get_sum_of_sample_mean


@torch.no_grad()
def compute_domain_metrics(args, batch, per_token, *, loss_masks, mismatch_metrics=None):
    """Return detached contributions, not means over only the named domain.

    The outer training metric reducer supplies the global sample/token count.
    Missing domains contribute zero. Preserve the parent denominator even when
    selecting a subset of tokens, including after rejection sampling.
    """
    domains = batch.get("domains")
    all_domains = batch.get("all_domains")
    if domains is None or len(domains) == 0 or all_domains is None or len(all_domains) == 0:
        return {}

    result = _reduce_domain_tensors(args, batch, domains, all_domains, loss_masks, per_token)
    if mismatch_metrics:
        result.update(_reduce_domain_tensors(args, batch, domains, all_domains, batch["loss_masks"], mismatch_metrics))

    custom_reducer = args.custom_pg_loss_reducer_function_path is not None
    for domain in all_domains:
        if custom_reducer:
            # An arbitrary custom reducer need not be additive over domains.
            result[f"standard_pg_loss/{domain}"] = result.pop(f"pg_loss/{domain}")
            continue
        total = result[f"pg_loss/{domain}"]
        if args.entropy_coef != 0:
            total = total - args.entropy_coef * result[f"entropy_loss/{domain}"]
        if args.use_kl_loss and args.kl_loss_coef != 0:
            total = total + args.kl_loss_coef * result[f"kl_loss/{domain}"]
        result[f"loss/{domain}"] = total
    return result


def _reduce_domain_tensors(args, batch, domains, all_domains, masks, per_token):
    denominators = batch.get("rollout_mask_sums")
    if denominators is None:
        denominators = [mask.sum() for mask in masks]
    result = {}
    for domain in all_domains:
        selected_masks = [
            mask if label == domain else torch.zeros_like(mask) for label, mask in zip(domains, masks, strict=True)
        ]
        local_mask = torch.cat(
            get_local_response_loss_masks(
                batch["total_lengths"],
                batch["response_lengths"],
                selected_masks,
                args.qkv_format,
                batch.get("max_seq_lens"),
            )
        ).bool()
        reducer = get_sum_of_sample_mean(
            batch["total_lengths"],
            batch["response_lengths"],
            selected_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            batch.get("max_seq_lens"),
            denominators=denominators,
            loss_agg_mode=getattr(args, "loss_agg_mode", None),
        )
        for name, values in per_token.items():
            # 0 * NaN is NaN: remove inactive values before the weighted sum.
            selected = torch.where(local_mask.to(values.device), values.detach(), values.new_zeros(()))
            result[f"{name}/{domain}"] = reducer(selected).detach()
    return result
