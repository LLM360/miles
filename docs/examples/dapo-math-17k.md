---
title: "Qwen3-8B on DAPO Math 17K"
description: "Qwen3-8B Megatron GRPO with Math-Verify rewards and optional eight-GPU 128K-response presets."
# Generated from examples/dapo_math_17k/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Train Qwen3-8B with Megatron GRPO and binary Math-Verify correctness rewards.
The launcher supports ordinary response lengths and presets using eight-GPU model groups for
128,000-token responses. Training and SGLang rollout share the GPUs.

## Setup

Use a Miles training environment with PyTorch, Megatron-LM, Transformer Engine,
Transformers, SGLang and Ray, then install the reward dependency:

```bash
python -m pip install -r examples/dapo_math_17k/requirements.txt
```

`math-verify[antlr4_9_3]==0.9.0` uses the ANTLR runtime compatible with
OmegaConf. Missing model weights are downloaded from `--model-repo`; an
existing local checkpoint can be selected with `--hf-checkpoint`. This recipe
supports dense Qwen3-4B and Qwen3-8B. Set `--model-repo` to the matching model
size when supplying a local checkpoint.

Point `--megatron-path` at the compatible Megatron checkout. The 128K recipe
requires dense-model YaRN support, including `--yarn-original-max-position-embeddings`
and the mapping into `TransformerConfig.yarn_*`; an older Megatron checkout
may need updating.

The full dataset is bundled at `dapo-math-17k.jsonl`: 17,398 records from
`zhuzilin/dapo-math-17k`, revision
`2e65612930298bde4c5d58fd97b3f23a483aaff9`, SHA-256
`cc9c39c2aa19177abe9464741e121cf4cac90fd25484ef3cdf86535101e3a5b6`.
Each record contains a chat conversation in `prompt` and an integer answer
string in `label`. The long-context presets use the bundled fixed subset
`dapo-math-640-seed42.jsonl`. Use `--dataset-path` for another training JSONL.

## Basic training

Run from the repository root inside a GPU allocation:

```bash
python examples/dapo_math_17k/run_dapo_math_17k.py \
  --model-repo Qwen/Qwen3-8B \
  --model-dir /path/to/models \
  --megatron-path /path/to/Megatron-LM \
  --output-dir /path/to/run \
  --num-gpus-per-node 2
```

The default model is `Qwen/Qwen3-8B`. The default recipe uses 32 prompts × 8
responses, global batch 256, a 4,096-token response limit, and 3,000 rollouts.
Sampling uses temperature 0.7, top-p 0.8 and top-k 20. Training uses BF16,
Megatron FlashAttention, full-layer activation recomputation, sample-mean GRPO
loss and no KL or entropy penalty. The default layout is TP=2, CP=1, PP=1.
Attention and hidden dropout are zero.
SGLang CUDA graphs are disabled in the base recipe; the 128K presets enable
decode graphs. GPU memory requirements depend on the model, hardware and
sequence lengths.

Preparation converts the starting weights to Megatron `torch_dist` under
`<output-dir>/megatron_init/<model-name>_torch_dist`. A completed conversion
is reused for the same run directory. Use a fresh output directory when changing
the source weights. `--load-path` skips conversion and selects an existing
Megatron checkpoint; an FSDP checkpoint cannot be resumed directly. SGLang and
the tokenizer use the Hugging Face checkpoint.

For a small three-rollout check, add:

```bash
  --num-rollout 3 \
  --rollout-batch-size 2 \
  --over-sampling-batch-size 2 \
  --n-samples-per-prompt 4 \
  --global-batch-size 8 \
  --save-interval 3
```

| Option | Purpose |
| --- | --- |
| `--enable-thinking` / `--no-enable-thinking` | Explicitly sets the Qwen3 chat-template thinking switch; unset uses the tokenizer default. |
| `--seed`, `--rollout-seed` | Training and rollout seeds. |
| `--lr`, `--min-lr`, `--lr-decay-style`, `--lr-decay-iters` | Learning-rate schedule; decay iterations include warmup. |
| `--lr-warmup-init`, `--lr-warmup-iters` | Warmup starting rate and update count. |
| `--load-path`, `--save-dir` | Megatron checkpoint load/resume and save location. |
| `--tensor-model-parallel-size`, `--context-parallel-size` | GPUs sharing each layer and each sequence; their product must divide the training GPU count. |
| `--max-tokens-per-gpu` | Per-GPU training token budget; CP times this budget must cover the full context in long-context mode. |
| `--dump-details` | Saves rollout/train batches under the run's `debug` directory. |
| `--wandb-project`, `--wandb-run-name` | Optional project and run naming overrides. |

W&B engages when `WANDB_API_KEY` is present. An explicit `--wandb-project`
requires that variable; its value is forwarded to Ray workers and redacted
from echoed submission commands. For accuracy, inspect `rollout/raw_reward`;
the normalized `rollout/rewards` mean is usually close to zero.

## 128K-response presets

```bash
python examples/dapo_math_17k/run_dapo_math_17k.py \
  --v1-experiment qwen3-8b_warmup_thinking_seed42 \
  --model-dir /path/to/models \
  --megatron-path /path/to/Megatron-LM \
  --output-dir /path/to/long-context-runs
```

Experiment names combine `qwen3-4b` or `qwen3-8b`, `warmup` or `nowarmup`,
and `thinking` or `nothinking`, followed by `_seed42`. Each invocation launches
one experiment in a dedicated subdirectory under `--output-dir`. For an
eight-node job, start Ray on all eight allocated nodes, set
`MILES_SCRIPT_EXTERNAL_RAY=1`, and pass `--num-nodes 8`. The global batch
remains 512 responses, with one optimizer update per rollout and 100 updates
in total; the warmup variants use ten warmup updates followed by 90 decay updates.

The preset pins:

- Eight GPUs per node: tensor parallelism of two and context parallelism of
  four, with pipeline parallelism of one. Node count comes from `--num-nodes`
  or `SLURM_JOB_NUM_NODES` (default one). Eight nodes provide eight
  data-parallel model groups on 64 GPUs. Sequence parallelism and `a2a` CP
  communication are enabled.
- The fixed 640-prompt subset, 100 rollouts, 64 prompts × 8 responses per
  rollout, and global batch 512.
- 128,000 response tokens within a 131,072-token context window.
- A padded training budget of 32,768 tokens per GPU, log-probability chunks
  of 512 tokens, and loss recomputation.
- Eight single-GPU rollout engines per node with up to eight concurrent requests each.
- Round-robin routing spreads requests evenly across the rollout engines.
- FA3 attention, 64-token KV-cache pages, 8,192-token prefill chunks, and FP8 E4M3
  KV cache. Full decode CUDA graphs capture batches up to eight. Model weights
  and Megatron training remain BF16. These presets require a compatible Hopper
  SGLang build with the per-phase CUDA graph options.
- FP8 cache uses SGLang's default scale of 1.0 when the checkpoint supplies no
  scales. Check reward/accuracy when comparing with BF16 cache; concurrency is
  a ceiling and available cache memory still limits admission at long contexts.
- Temperature 1.0, top-p 0.95, top-k 20, and training/rollout seeds of 42.
- Learning rate 5e-6 decaying linearly to 2e-6 over 100 updates. Warmup variants
  start at 1.414e-6 for ten updates; no-warmup variants start at 5e-6.
- Checkpoints every 50 updates (one update per rollout), binary correctness rewards, and sample-mean
  GRPO loss. These presets do not enable dynamic filtering or length penalties.

Preparation validates every formatted prompt against the response budget,
including two positions reserved by SGLang. It creates `hf_long_context` with
a YaRN factor of 4 and original context of 32,768. Weight/tokenizer files are
symlinked from the source checkpoint, so that checkpoint must stay available.
SGLang and the tokenizer load the prepared view. Megatron converts its weights
and receives matching YaRN settings explicitly: factor 4, original context
32,768, rotary base 1,000,000, beta fast/slow 32/1, mscale 1, mscale-all-dim 0,
and integer-rounded correction bounds. Its sequence length is 131,072.
The Qwen3 model definitions use the canonical `--position-embedding-type rope`
default so the launcher can override it with `yarn`. The legacy
`--use-rotary-position-embeddings` flag would force ordinary RoPE.

Megatron supplies context parallelism and packed `thd` batches. The launcher
checks the GPU layout, KV-head divisibility for `a2a`, and token budget before
submission. Engine readiness checks the actual KV-cache capacity; insufficient
capacity fails validation instead of silently reducing the response budget.
The eight-GPU settings are a starting configuration: full 128K training and
rollout still require validation on the target hardware.

`--extra-args` is rejected for a fixed preset. Existing preset checkpoints
require `--load-path` or a fresh output directory. W&B project naming remains
configurable. For custom long-context experiments, use `--qwen3-long-context`,
`--tensor-model-parallel-size`, `--context-parallel-size`,
`--max-tokens-per-gpu`, and `--log-probs-chunk-size` without a preset.

## Reward behavior

The reward function parses the gold label and the model response with
Math-Verify and returns 1.0 for a mathematically equivalent answer, otherwise
0.0. It accepts both individual samples and batches, preserving sample order.
It scores the full response, including reasoning; this recipe does not require
a final answer after `</think>`.

Each score runs in a spawned subprocess with five-second parsing/verification
timeouts and a 20-second parent timeout. Failed or timed-out scores return
zero. Concurrency is capped at 16, and oversized finite symbolic sums are
rejected before verification.

## Validation

CPU checks cover the default submission, all eight presets, conversion/resume
selection, YaRN preparation, prompt budgets, GPU-layout validation and rewards.

```bash
python -m pytest tests/fast/examples/dapo_math_17k
```

Before a full 128K experiment, run a small end-to-end job on eight GPUs with a
fresh output directory. This exercises conversion, generation, rewards, training,
weight updates and checkpoint saving:

```bash
python examples/dapo_math_17k/run_dapo_math_17k.py \
  --model-repo Qwen/Qwen3-8B \
  --model-dir /path/to/models \
  --megatron-path /path/to/Megatron-LM \
  --output-dir /path/to/128k-smoke \
  --num-gpus-per-node 8 \
  --qwen3-long-context \
  --tensor-model-parallel-size 2 --context-parallel-size 4 \
  --max-tokens-per-gpu 32768 --log-probs-chunk-size 512 \
  --rollout-max-response-len 128000 --sglang-max-running-requests 1 \
  --num-rollout 3 --rollout-batch-size 1 --over-sampling-batch-size 1 \
  --n-samples-per-prompt 2 --global-batch-size 2 --save-interval 1
```

Generated responses may stop early, so a successful smoke run alone does not
prove a full-length 128K forward/backward pass fits in GPU memory.
