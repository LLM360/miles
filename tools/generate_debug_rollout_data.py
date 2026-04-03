"""Generate synthetic debug rollout data for FT e2e tests.

Creates .pt files with the format expected by --load-debug-rollout-data:
  {"rollout_id": int, "samples": [Sample.to_dict(), ...]}

Usage:
  python tools/generate_debug_rollout_data.py \
    --model-path /root/models/Qwen3-30B-A3B-5layer \
    --output-dir /tmp/debug_rollout_data \
    --num-rollouts 10 \
    --samples-per-rollout 8
"""

import random
from pathlib import Path
from typing import Annotated

import torch
import typer
from transformers import AutoTokenizer

from miles.utils.types import Sample


def _generate_sample(
    tokenizer: AutoTokenizer,
    group_index: int,
    index: int,
    rng: random.Random,
) -> Sample:
    prompt_text = f"What is {rng.randint(1, 100)} + {rng.randint(1, 100)}?"
    response_text = f"The answer is {rng.randint(1, 200)}."

    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    all_tokens = prompt_tokens + response_tokens

    response_length = len(response_tokens)
    loss_mask = [0] * len(prompt_tokens) + [1] * response_length

    rollout_log_probs = [rng.uniform(-5.0, -0.1) for _ in range(response_length)]

    reward = rng.uniform(0.0, 1.0)

    return Sample(
        group_index=group_index,
        index=index,
        prompt=prompt_text,
        tokens=all_tokens,
        response=response_text,
        response_length=response_length,
        label=str(rng.randint(1, 200)),
        reward=reward,
        loss_mask=loss_mask,
        weight_versions=[],
        rollout_log_probs=rollout_log_probs,
        status=Sample.Status.COMPLETED,
    )


def main(
    model_path: Annotated[str, typer.Option(help="Path to HF model (for tokenizer)")],
    output_dir: Annotated[str, typer.Option(help="Output directory for .pt files")],
    num_rollouts: Annotated[int, typer.Option(help="Number of rollout files to generate")] = 10,
    samples_per_rollout: Annotated[int, typer.Option(help="Samples per rollout file")] = 8,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
) -> None:
    rng = random.Random(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for rollout_id in range(num_rollouts):
        samples = []
        for i in range(samples_per_rollout):
            sample = _generate_sample(
                tokenizer=tokenizer,
                group_index=i,
                index=i,
                rng=rng,
            )
            samples.append(sample.to_dict())

        data = {"rollout_id": rollout_id, "samples": samples}
        file_path = output_path / f"{rollout_id}.pt"
        torch.save(data, file_path)
        print(f"Generated {file_path} with {len(samples)} samples")

    print(f"Done. Generated {num_rollouts} rollout files in {output_dir}")


if __name__ == "__main__":
    typer.run(main)
