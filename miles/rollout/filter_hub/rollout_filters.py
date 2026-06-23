from miles.rollout.filter_hub.dynamic_sampling_filters import _flatten_samples

__all__ = ["mask_truncated", "mask_truncated_and_llm_judge_failed"]


def mask_truncated(args, samples: list[object]) -> None:
    for sample in _flatten_samples(samples):
        status = getattr(sample, "status", None)
        if getattr(status, "value", status) == "truncated":
            sample.remove_sample = True


def mask_truncated_and_llm_judge_failed(args, samples: list[object]) -> None:
    for sample in _flatten_samples(samples):
        status = getattr(sample, "status", None)
        if getattr(status, "value", status) == "truncated" or (sample.train_metadata or {}).get("llm_judge_failed"):
            sample.remove_sample = True
