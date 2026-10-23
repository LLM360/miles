"""Explicit future-experiment profile; no job dispatch or model substitution."""
CONTEXT_LENGTH = 16384


def apply_cb_profile(args):
    if not getattr(args, "cb_token_mean_16k", False):
        return
    if getattr(args, "custom_generate_function_path", None) != "miles360.dgcv.recipe_data.generate":
        raise ValueError("The CB 16k profile requires the source-recipe generator with dynamic context budgeting")
    if getattr(args, "gamma", 1.) != 1. or getattr(args, "lambd", 1.) != 1.:
        raise ValueError("Keep gamma=lambda=1 in this reduction/context ablation")
    context = int(getattr(args, "cb_total_context_len", CONTEXT_LENGTH))
    reserved = int(getattr(args, "cb_sampler_reserved_tokens", 0))
    if context <= 2048 or reserved < 0:
        raise ValueError("Context must exceed prompt limit and reserved slots must be nonnegative")
    args.actor_loss_agg_mode = "token-mean"
    args.cb_diagnostics = True
    args.training_stage_timing = True
    args.dgcv_compact_calibration = True
    if getattr(args, "dgcv_query_chunk", None) is None:
        args.dgcv_query_chunk = 2048
    args.rollout_max_prompt_len = args.eval_max_prompt_len = 2048
    args.rollout_max_response_len = args.eval_max_response_len = context
    args.rollout_max_context_len = args.eval_max_context_len = context
    args.sglang_context_length = context + reserved
    args.seq_length = max(getattr(args, "seq_length", 0) or 0, context)
    args.max_position_embeddings = max(getattr(args, "max_position_embeddings", 0) or 0, context)
    budget = getattr(args, "max_tokens_per_gpu", None)
    if budget is not None and budget < context:
        raise ValueError(f"Training token budget must fit a complete {context}-token sequence")


def generation_budget(context_length, prompt_tokens, response_cap):
    if prompt_tokens < 0 or response_cap <= 0:
        raise ValueError("Invalid prompt length or response cap")
    budget = response_cap if context_length is None else min(response_cap, context_length-prompt_tokens)
    if budget <= 0:
        raise ValueError("Prompt exhausts the context; no response token fits")
    return budget


def protocol(args):
    return dict(version=1, actor_loss="global-optimizer-batch-token-mean",
                critic_loss="sequence-mean", gamma=args.gamma, lambd=args.lambd,
                prompt_limit=args.rollout_max_prompt_len,
                context_limit=args.rollout_max_context_len,
                response_rule="min(response_cap, context_limit - actual_prompt_tokens)",
                response_cap=args.rollout_max_response_len,
                eval_context_limit=args.eval_max_context_len,
                eval_response_cap=args.eval_max_response_len)
