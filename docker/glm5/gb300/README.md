# GLM5 on GB300 (Blackwell, CUDA 13, ARM64)

## Files

| File | Description |
|---|---|
| `Dockerfile_gb300` | Final image Dockerfile (base image + setup layers) |
| `build_gb300.sh` | Build & push to Docker Hub via K8s builder pod |
| `build_flashmla_gb300.sh` | Standalone FlashMLA sm103a rebuild script |
| `../Dockerfile_glm5` | Original Dockerfile (reference) |
| `../transformers.patch` | Transformers patch for GLM5 |

## Architecture

The image is built in two stages:

1. **Base image** (`msc/glm5_gb300:{hash}`) — built by `msc` from `build_commands` in `.msc/preset.py`. Contains: sglang, Megatron-LM, flash-attn, transformer-engine, apex, etc.

2. **Final image** (`yuemingy/miles:glm5-gb300`) — built by `build_gb300.sh` from `Dockerfile_gb300`. Adds: miles install, patches, FlashMLA GB300 fix, CLI wrappers.

## Build Steps

### Prerequisites

- `kubectl` configured with context `gcp-radixark-02`
- Base image already built on cluster (via `msc exec` with `glm5_gb300` flavor)
- Repos synced to PVC (via `msc exec 'echo done'`)

### 1. Build base image (one-time, ~1 hour)

```bash
cd ~/radixark/glm5
msc exec 'echo done'
```

This builds the base image from `build_commands` in `.msc/preset.py` and caches it on the K8s node.

### 2. Build & push final image (~15 min)

```bash
export DOCKERHUB_TOKEN="your_docker_hub_access_token"
bash miles/docker/glm5/gb300/build_gb300.sh
```

Pushes to `docker.io/yuemingy/miles:glm5-gb300`.

### 3. Interactive development

```bash
cd ~/radixark/glm5
msc shell   # enter container with synced repos
```

## Key Fixes for GB300

- **FlashMLA sm103a**: Base sglang image has FlashMLA compiled for sm100a only. `build_flashmla_gb300.sh` cherry-picks [sgl-project/sglang@8acd4d7](https://github.com/sgl-project/sglang/commit/8acd4d7d7e6f436601ef3ae51678f130fdc04d25) to patch `flashmla_utils.h` and `cutlass/arch/config.h`, then rebuilds with `TORCH_CUDA_ARCH_LIST="9.0a;10.0a;10.3a"`.

- **CCCL symlink**: CUDA 13 needs `ln -sf /usr/local/cuda/include/cccl/cuda /usr/local/cuda/include/cuda` for cutlass headers.

- **Missing CLI entry points**: Base sglang image installs packages without pip entry points. Wrapper scripts created for `torchrun`, `huggingface-cli`, `hf`.

- **Helm YAML double-quote constraint**: `setup_commands` are injected into Helm YAML inline arrays — no double quotes allowed. Use single quotes or avoid quotes entirely.
