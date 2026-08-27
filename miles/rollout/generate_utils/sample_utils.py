import math
import re
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import fields
from typing import Any

from miles.utils.types import Sample


_DIAGNOSTIC_LABEL_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


def _safe_diagnostic_label(value: Any, default: str) -> str:
    """Return a bounded internal label that cannot leak prompt/response text."""

    if hasattr(value, "value"):
        value = value.value
    if not isinstance(value, str) or not value:
        return default
    value = _DIAGNOSTIC_LABEL_RE.sub("_", value).strip("_")
    return value[:64] or default


def _invalid_eval_signature(sample: Sample, failure: str) -> str:
    """Describe an invalid sample using only bounded operational metadata."""

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    multi_attempt = metadata.get("multi_attempt")
    multi_attempt = multi_attempt if isinstance(multi_attempt, dict) else {}

    reason = _safe_diagnostic_label(
        multi_attempt.get("invalid_reason") or multi_attempt.get("stop_reason"),
        failure,
    )
    status = _safe_diagnostic_label(sample.status, "unknown")

    finish_reason = None
    attempts = multi_attempt.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
        last_attempt = attempts[-1]
        finish_reason = last_attempt.get("engine_finish_reason") or last_attempt.get("finish_reason")
    finish = _safe_diagnostic_label(finish_reason, "unknown")
    return f"reason={reason},status={status},finish={finish}"


def collect_eval_rewards(samples: list[Sample], reward_key: str | None) -> list[float]:
    """Collect valid eval rewards or fail the evaluation without biasing it.

    An aborted sample has no trustworthy correctness label. Treating it as
    zero would turn scorer or transport failures into model failures, so the
    entire evaluation must be retried after the underlying failure is fixed.
    """

    rewards = []
    invalid: list[tuple[Sample, str]] = []
    for sample in samples:
        if sample.reward is None:
            invalid.append((sample, "missing_reward"))
            continue
        if reward_key:
            if not isinstance(sample.reward, dict) or reward_key not in sample.reward:
                invalid.append((sample, "missing_reward_channel"))
                continue
            reward = sample.reward[reward_key]
        else:
            reward = sample.reward
        try:
            reward_value = float(reward)
        except (TypeError, ValueError, OverflowError):
            invalid.append((sample, "non_numeric_reward"))
            continue
        if not math.isfinite(reward_value):
            invalid.append((sample, "non_finite_reward"))
            continue
        rewards.append(reward_value)
    if invalid:
        indices = [sample.index for sample, _failure in invalid]
        preview = indices[:10]
        suffix = "..." if len(invalid) > len(preview) else ""
        diagnostics = Counter(_invalid_eval_signature(sample, failure) for sample, failure in invalid)
        diagnostic_summary = "; ".join(
            f"{signature}:{count}" for signature, count in sorted(diagnostics.items())
        )
        raise RuntimeError(
            f"evaluation contains {len(invalid)} invalid or unlabeled samples "
            f"(indices={preview}{suffix}; diagnostics={{{diagnostic_summary}}}); "
            "refusing to report them as incorrect"
        )
    return rewards


def finalize_eval_rewards(
    data: dict[str, dict[str, Any]], reward_key: str | None
) -> dict[str, dict[str, Any]]:
    """Validate raw evaluation samples after their durable debug save.

    Standard rollout functions deliberately return ``rewards=None``.  The
    rollout manager first persists the raw samples and only then calls this
    function, so a failed scorer or generation boundary remains diagnosable.
    Custom evaluation functions that already provide rewards retain their
    existing behavior.
    """

    for dataset_name, info in data.items():
        if info.get("rewards") is not None:
            continue
        samples = info.get("samples")
        if not isinstance(samples, list):
            raise RuntimeError(
                f"evaluation dataset {dataset_name!r} deferred reward validation "
                "without a raw sample list"
            )
        info["rewards"] = collect_eval_rewards(samples, reward_key)
    return data


def persist_then_finalize_eval_rewards(
    data: dict[str, dict[str, Any]],
    reward_key: str | None,
    persist_raw_samples: Callable[[dict[str, dict[str, Any]]], None],
) -> dict[str, dict[str, Any]]:
    """Establish the evaluation evidence-before-aggregation ordering."""

    persist_raw_samples(data)
    return finalize_eval_rewards(data, reward_key)


def drop_samples_after_first_non_completed(samples: list[Sample]) -> tuple[list[Sample], int]:
    """Keep turns up to and including the first non-COMPLETED one.

    A turn that ended early (engine abort during rollout shutdown, per-turn
    length limit) means every later turn was conditioned on incomplete
    output, so those turns are invalid training data. Dropping them
    establishes the ``merge_samples`` invariant that only the final sample
    may be non-COMPLETED.

    Returns the kept prefix and the number of dropped samples.
    """
    for i, sample in enumerate(samples[:-1]):
        if sample.status != Sample.Status.COMPLETED:
            return samples[: i + 1], len(samples) - i - 1
    return samples, 0


def merge_samples(samples: list[Sample], tokenizer) -> Sample:
    acc = samples[0]
    for sample in samples[1:]:
        acc = _merge_sample_pair(acc, sample, tokenizer=tokenizer)
    return acc


def _merge_sample_pair(a: Sample, b: Sample, tokenizer) -> Sample:
    """Merge two samples generated from sibling inference engine calls."""
    a, b = deepcopy(a), deepcopy(b)

    def _merge_equal_value(field):
        x = getattr(a, field)
        y = getattr(b, field)
        assert x == y, f"{field} mismatch: a.{field}={x}, b.{field}={y}"
        return x

    def _fill_defaults(sample: Sample):
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = [0.0] * sample.response_length

    _fill_defaults(a)
    _fill_defaults(b)

    obs_len = len(b.tokens) - len(a.tokens) - b.response_length
    obs_tokens = b.tokens[len(a.tokens) : len(a.tokens) + obs_len]
    # TODO: is this acceptable?
    obs_text = tokenizer.decode(obs_tokens)

    try:
        a.validate()
        b.validate()
        assert _startswith(short=a.prompt, long=b.prompt), "b.prompt must start with a.prompt"
        assert _startswith(short=a.tokens, long=b.tokens), "b.tokens must start with a.tokens"
        assert obs_len > 0, f"obs_len must be > 0, got {obs_len}"
        if a.rollout_routed_experts is not None:
            assert a.rollout_routed_experts.shape[0] <= b.rollout_routed_experts.shape[0]
        assert a.status == Sample.Status.COMPLETED, f"a.status must be COMPLETED, got {a.status}"

        return _create_with_all_fields(
            Sample,
            group_index=_merge_equal_value("group_index"),
            index=_merge_equal_value("index"),
            prompt=b.prompt,
            tokens=b.tokens,
            multimodal_inputs=_merge_equal_value("multimodal_inputs"),
            multimodal_train_inputs=_merge_equal_value("multimodal_train_inputs"),
            response=a.response + obs_text + b.response,
            response_length=a.response_length + obs_len + b.response_length,
            label=_merge_equal_value("label"),
            reward=_merge_equal_value("reward"),
            loss_mask=a.loss_mask + [0] * obs_len + b.loss_mask,
            weight_versions=a.weight_versions + b.weight_versions,
            rollout_log_probs=a.rollout_log_probs + [0.0] * obs_len + b.rollout_log_probs,
            rollout_routed_experts=b.rollout_routed_experts,
            remove_sample=_merge_equal_value("remove_sample"),
            status=b.status,
            metadata=_merge_equal_value("metadata"),
            generate_function_path=_merge_equal_value("generate_function_path"),
            train_metadata=_merge_equal_value("train_metadata"),
            session_id=_merge_equal_value("session_id"),
            non_generation_time=_merge_equal_value("non_generation_time"),
            spec_info=_merge_spec_info(a.spec_info, b.spec_info),
            prefix_cache_info=_merge_prefix_cache_info(a.prefix_cache_info, b.prefix_cache_info),
        )
    except AssertionError as e:
        e.add_note(f"{a=} {b=}")
        raise


def _merge_spec_info(a: Sample.SpecInfo, b: Sample.SpecInfo) -> Sample.SpecInfo:
    def _merge_plus_value(field):
        return getattr(a, field) + getattr(b, field)

    return _create_with_all_fields(
        Sample.SpecInfo,
        spec_accept_token_num=_merge_plus_value("spec_accept_token_num"),
        spec_draft_token_num=_merge_plus_value("spec_draft_token_num"),
        spec_verify_ct=_merge_plus_value("spec_verify_ct"),
        completion_token_num=_merge_plus_value("completion_token_num"),
    )


def _merge_prefix_cache_info(a: Sample.PrefixCacheInfo, b: Sample.PrefixCacheInfo) -> Sample.PrefixCacheInfo:
    def _merge_plus_value(field):
        return getattr(a, field) + getattr(b, field)

    return _create_with_all_fields(
        Sample.PrefixCacheInfo,
        cached_tokens=_merge_plus_value("cached_tokens"),
        total_prompt_tokens=_merge_plus_value("total_prompt_tokens"),
    )


def _create_with_all_fields(cls, **kwargs):
    expected = {f.name for f in fields(cls)}
    actual = set(kwargs.keys())
    assert (
        expected == actual
    ), f"{cls.__name__} field mismatch. Missing: {expected - actual}, Extra: {actual - expected}"
    return cls(**kwargs)


def _startswith(*, short, long) -> bool:
    if isinstance(short, str) and isinstance(long, str):
        return long.startswith(short)
    if isinstance(short, list) and isinstance(long, list):
        return (len(long) >= len(short)) and (long[: len(short)] == short)
    raise NotImplementedError
