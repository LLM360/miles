"""Prepare a shared YaRN checkpoint view for Qwen3 training and rollout.

Only the experiment-local config is written. Weights and tokenizer files are
symlinked to the source checkpoint, so existing experiments retain their config.
"""

import json
from pathlib import Path

CONTEXT_LENGTH = 131072
# SGLang reserves one slot in max_req_len and one more in max_new_tokens.
GENERATION_RESERVE = 2
ROPE_SCALING = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}


def megatron_yarn_args() -> str:
    """Match the Hugging Face/SGLang YaRN recipe in the dense Megatron model."""
    return (
        f"--seq-length {CONTEXT_LENGTH} --position-embedding-type yarn --rotary-base 1000000 "
        f"--rotary-scaling-factor {ROPE_SCALING['factor']} "
        f"--yarn-original-max-position-embeddings {ROPE_SCALING['original_max_position_embeddings']} "
        "--yarn-beta-fast 32 --yarn-beta-slow 1 --mscale 1 --mscale-all-dim 0 "
        "--yarn-correction-range-round-to-int "
    )


def prepare_checkpoint(source: Path, destination: Path, *, context_length: int) -> None:
    source = source.resolve()
    if destination.resolve() == source:
        raise ValueError("The long-context checkpoint view must differ from the source checkpoint")
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        raise ValueError("The long-context recipe supports Qwen3 dense checkpoints only")
    if context_length != CONTEXT_LENGTH:
        raise ValueError(f"The Qwen3 YaRN recipe uses context_length={CONTEXT_LENGTH}")
    config.update(max_position_embeddings=context_length, rope_scaling=ROPE_SCALING)
    # Transformers 5 also accepts rope_parameters; avoid competing RoPE definitions.
    config.pop("rope_parameters", None)
    target_config = destination / "config.json"
    if target_config.exists() and json.loads(target_config.read_text()) != config:
        raise ValueError(f"Existing long-context config differs: {target_config}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.name.startswith(".") or path.name == "config.json" or not path.is_file():
            continue
        target = destination / path.name
        if target.is_symlink() and target.resolve() == path.resolve():
            continue
        if target.exists() or target.is_symlink():
            raise ValueError(f"Existing checkpoint-view file differs: {target}")
        target.symlink_to(path)
    if not target_config.exists():
        target_config.write_text(json.dumps(config, indent=2) + "\n")


def validate_prompt_budget(
    dataset_path: Path,
    checkpoint: Path,
    *,
    response_length: int,
    enable_thinking: bool | None,
    expected_prompts: int | None = None,
) -> int:
    # Tokenization is only needed during long-context preparation, not launcher import.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    lengths = []
    with dataset_path.open() as stream:
        for line in stream:
            record = json.loads(line)
            prompt = record["prompt"]
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, **template_kwargs
            )
            lengths.append(len(tokenizer.encode(formatted, add_special_tokens=False)))
    if not lengths or (expected_prompts is not None and len(lengths) != expected_prompts):
        raise ValueError(f"Expected {expected_prompts or 'nonempty'} prompts; found {len(lengths)}")
    longest = max(lengths)
    if longest + response_length + GENERATION_RESERVE > CONTEXT_LENGTH:
        raise ValueError(
            f"Longest formatted prompt ({longest}) + response ({response_length}) + "
            f"engine reserve ({GENERATION_RESERVE}) exceeds {CONTEXT_LENGTH}; "
            "the dataset will not be silently filtered or the response budget reduced"
        )
    return longest
