"""Opt-in K2 diagnostics. Equal response weights; complete groups only; CP=1."""
import math
import os
from collections import defaultdict

import torch
import torch.distributed as dist


def capture_advantages(data, advantages, args=None):
    if getattr(args, "cb_diagnostics", False) or os.environ.get("MILES_CB_DIAGNOSTICS") == "1":
        # Native whitening is out-of-place. Keep detached references until logging.
        data["_cb_diag_raw_advantages"] = [a.detach() for a in advantages]


@torch.no_grad()
def response_rows(data):
    required = ("sample_indices", "rewards", "loss_masks", "values",
                "_cb_diag_raw_advantages", "advantages", "returns")
    if not all(k in data for k in required):
        return []
    rows, tensors = [], []
    for i, (index, reward, mask, value, raw, used, target) in enumerate(zip(*(data[k] for k in required), strict=True)):
        mask = mask.bool()
        v, a, u = (x.detach().float()[mask] for x in (value, raw, used))
        if not a.numel():
            continue
        r = float(reward)
        v0 = value.detach().float()[0]
        stats = [a.mean(), a.square().mean(), (a > 0).float().mean(),
                 u.mean(), u.square().mean(), (u > 0).float().mean(),
                 (u < 0).float().mean(), (u == 0).float().mean(),
                 (v-r).mean(), (v-r).square().mean(), ((v < 0) | (v > 1)).float().mean(),
                 v0, v0-r, (v0-r).square()]
        reference = None
        if "_cb_diag_shift" in data:
            reference = a - float(data["_cb_diag_shift"][i])
        elif "group_baseline" in data and data.get("group_baseline_valid", [False]*len(data["rewards"]))[i]:
            # Only explicitly enabled for native cb_* modes by the caller.
            reference = data["returns"][i].detach().float()[mask] - float(data["group_baseline"][i])
        zero = a.new_zeros(())
        stats += [zero if reference is None else (a-reference).mean(),
                  zero if reference is None else ((reference <= 0) & (a > 0)).float().mean()]
        target = target.detach().float()[mask]
        error = v-target
        stats += [error.square().mean(), error.mean()]
        # Relative response-position quartiles. Use original token positions,
        # not compressed positions after masking; carry counts for global pooling.
        positions = torch.arange(mask.numel(), device=mask.device)[mask]
        bins = torch.clamp(4*positions//mask.numel(), max=3)
        for b in range(4):
            selected = bins == b
            stats += [selected.sum().float(), error[selected].square().sum(), error[selected].sum()]
        tensors.append(torch.stack(stats))
        rows.append(dict(index=int(index), reward=r, tokens=a.numel(), reference=reference is not None,
                         response_length=int(data.get("response_lengths", [len(x) for x in data["values"]])[i]),
                         truncated=float(data["truncated"][i]) if "truncated" in data else None))
    if tensors:
        for row, stats in zip(rows, torch.stack(tensors).cpu().tolist(), strict=True):
            row["stats"] = stats
    return rows


def summarize_rows(rows, group_size):
    """Reduce global per-response summaries, never averages of per-rank rates."""
    finite = [r for r in rows if math.isfinite(r["reward"]) and all(map(math.isfinite, r["stats"]))]
    groups = defaultdict(list)
    for row in finite:
        groups[row["index"] // group_size].append(row)
    complete = {g: rs for g, rs in groups.items() if len(rs) == group_size
                and len({r["index"] for r in rs}) == group_size
                and all(r["reward"] in (0., 1.) for r in rs)}
    outcome = {g: sum(r["reward"] for r in rs) for g, rs in complete.items()}
    metrics = {"responses": len(finite), "nonfinite_responses": len(rows)-len(finite),
               "complete_groups": len(complete)}
    subsets = {"all": finite,
               "wrong": [r for r in finite if r["reward"] == 0],
               "correct": [r for r in finite if r["reward"] == 1]}
    for name, predicate in (("all_wrong", lambda n: n == 0),
                            ("mixed", lambda n: 0 < n < group_size),
                            ("all_correct", lambda n: n == group_size)):
        ids = {g for g, n in outcome.items() if predicate(n)}
        metrics[f"groups/{name}"] = len(ids)
        subsets[name] = [r for r in finite if r["index"] // group_size in ids]
    names = ("raw_mean", "raw_second_moment", "raw_positive_frac",
             "used_mean", "used_second_moment", "used_positive_frac", "used_negative_frac", "used_zero_frac",
             "value_bias", "value_mse", "value_outside_01_frac", "v0_mean", "v0_bias", "v0_mse")
    for label, subset in subsets.items():
        prefix = f"{label}/"
        metrics[prefix+"responses"] = len(subset)
        # Empty strata have support zero; omit undefined rates instead of inventing zeros.
        if not subset:
            continue
        means = [sum(r["stats"][i] for r in subset)/len(subset) for i in range(len(names))]
        metrics.update({prefix+k: v for k, v in zip(names, means, strict=True)})
        for offset, kind in ((0, "raw"), (3, "used")):
            metrics[prefix+kind+"_std"] = math.sqrt(max(0., means[offset+1]-means[offset]**2))
        reference = [r for r in subset if r["reference"]]
        metrics[prefix+"reference_responses"] = len(reference)
        if reference:
            metrics[prefix+"shift_mean"] = sum(r["stats"][14] for r in reference)/len(reference)
            metrics[prefix+"nonpositive_to_positive_frac"] = sum(r["stats"][15] for r in reference)/len(reference)
        tokens = sum(r["tokens"] for r in subset)
        metrics[prefix+"valid_tokens"] = tokens
        metrics[prefix+"response_length_mean"] = sum(r["response_length"] for r in subset)/len(subset)
        truncated = [r["truncated"] for r in subset if r["truncated"] is not None]
        if truncated:
            metrics[prefix+"truncation_rate"] = sum(truncated)/len(truncated)
        # Legacy value_mse/bias are equal-response, against outcome reward.
        for idx, name in ((8, "value_bias"), (9, "value_mse"), (16, "critic_target_mse"), (17, "critic_target_bias")):
            metrics[prefix+name+"_sequence_mean"] = sum(r["stats"][idx] for r in subset)/len(subset)
            metrics[prefix+name+"_token_mean"] = sum(r["stats"][idx]*r["tokens"] for r in subset)/tokens
        for b in range(4):
            support = sum(r["stats"][18+3*b] for r in subset)
            metrics[prefix+f"position_q{b+1}/tokens"] = support
            if support:
                metrics[prefix+f"position_q{b+1}/critic_target_mse"] = sum(r["stats"][19+3*b] for r in subset)/support
                metrics[prefix+f"position_q{b+1}/critic_target_bias"] = sum(r["stats"][20+3*b] for r in subset)/support
    if complete:
        # Finite-sample empirical prompt calibration; do not call this true value error.
        metrics["prompt/group_mean_mse"] = sum((sum(r["stats"][11] for r in rs)/group_size
            - outcome[g]/group_size)**2 for g, rs in complete.items())/len(complete)
        metrics["prompt/groups"] = len(complete)
    return {"diag/cb/"+k: float(v) for k, v in metrics.items()}


def collect_diagnostics(args, data, parallel):
    """Called on TP0/PP-last, like the native logger. One compact DP gather."""
    if not getattr(args, "cb_diagnostics", False) and os.environ.get("MILES_CB_DIAGNOSTICS") != "1":
        return {}
    try:
        if parallel.cp.size != 1:
            return {"diag/cb/supported": 0.}
        # Value mode's reference is V_t, not the sibling group. DGCV has its own estimator.
        view = data
        if getattr(args, "ppo_baseline", "value") == "value":
            view = {k: v for k, v in data.items() if k != "group_baseline"}
        rows = response_rows(view)
        pg = parallel.intra_dp_cp
        if pg.size > 1:
            shards = [None] * pg.size
            dist.all_gather_object(shards, rows, group=pg.gloo_group if pg.gloo_group is not None else pg.group)
            rows = [row for shard in shards for row in shard]
        # Every rank gets the same result; native mean reduction preserves it exactly.
        return {"diag/cb/supported": 1., **summarize_rows(rows, args.n_samples_per_prompt)}
    finally:
        data.pop("_cb_diag_raw_advantages", None)
        data.pop("_cb_diag_shift", None)
