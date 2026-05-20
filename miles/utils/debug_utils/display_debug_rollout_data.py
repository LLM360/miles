import json
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import typer

from miles.ray.rollout.metrics import _compute_perf_metrics_from_samples
from miles.utils.rollout_dump import find_rollout_dump, load_rollout_dump
from miles.utils.types import Sample

_WHITELIST_KEYS = [
    "group_index",
    "index",
    "prompt",
    "response",
    "response_length",
    "label",
    "reward",
    "status",
    "metadata",
]


def main(
    # Deliberately make this name consistent with main training arguments
    load_debug_rollout_data: Annotated[str, typer.Option()],
    show_metrics: bool = True,
    show_samples: bool = True,
    category: list[str] = None,
    rollout_time: Annotated[
        float | None, typer.Option(help="Measured rollout duration in seconds, required for speed metrics.")
    ] = None,
    rollout_num_gpus: Annotated[
        int | None, typer.Option(help="Rollout GPU count, required for per-GPU speed metrics.")
    ] = None,
):
    if rollout_time is not None and rollout_time <= 0:
        raise typer.BadParameter("--rollout-time must be positive")
    if rollout_num_gpus is not None and rollout_num_gpus <= 0:
        raise typer.BadParameter("--rollout-num-gpus must be positive")
    if category is None:
        category = ["train", "eval"]
    for rollout_id, path in _get_rollout_dump_paths(load_debug_rollout_data, category):
        print("-" * 80)
        print(f"{rollout_id=} {path=}")
        print("-" * 80)

        pack = load_rollout_dump(path)
        sample_dicts = pack["samples"]

        if show_metrics:
            if rollout_time is None:
                print("Speed metrics unavailable: supply --rollout-time with the measured duration in seconds.")
            elif not sample_dicts:
                print("Speed metrics unavailable: the dump contains no samples.")
            else:
                args = SimpleNamespace(rollout_num_gpus=rollout_num_gpus)
                sample_objects = [Sample.from_dict(s) for s in sample_dicts]
                metrics = _compute_perf_metrics_from_samples(args, sample_objects, rollout_time)
                print("metrics", metrics)

        if show_samples:
            for sample in sample_dicts:
                print(json.dumps({k: v for k, v in sample.items() if k in _WHITELIST_KEYS}))


def _get_rollout_dump_paths(load_debug_rollout_data: str, categories: list[str]):
    # may improve later
    for rollout_id in range(1000):
        for category in categories:
            prefix = {
                "train": "",
                "eval": "eval_",
            }[category]
            path = find_rollout_dump(Path(load_debug_rollout_data.format(rollout_id=f"{prefix}{rollout_id}")))
            if path.exists():
                yield rollout_id, path


if __name__ == "__main__":
    """python -m miles.utils.debug_utils.display_debug_rollout_data --load-debug-rollout-data ..."""
    typer.run(main)
