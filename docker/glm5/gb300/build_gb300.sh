#!/bin/bash
# Build and push GLM5 GB300 Docker image via K8s builder pod (buildctl)
#
# Prerequisites:
#   - kubectl context set to gcp-radixark-02
#   - msc sync completed (repos on PVC at /shared-data/synced/glm5/)
#   - Base image msc/glm5_gb300:{hash} already built on cluster
#
# Usage:
#   bash miles/docker/glm5/gb300/build_gb300.sh
#
# What this builds:
#   Base: lmsysorg/sglang:v0.5.7-cu130-runtime
#     + build_commands from .msc/preset.py (cached as msc/glm5_gb300:{hash})
#     + setup: requirements, patches, FlashMLA GB300, CLI wrappers, miles install
#   → yuemingy/miles:glm5-gb300

set -e

# ── Config ──
BUILDER_POD="yueming-glm5-gb300-glm5-builder"
DOCKERHUB_USER="yuemingy"
IMAGE_TAG="docker.io/yuemingy/miles:glm5-gb300"
BASE_IMAGE="docker.io/msc/glm5_gb300:5ff88c1f4fb3"
PVC_WORKSPACE="/shared-data/synced/glm5"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 1. Docker Hub auth ──
echo "==> Setting up Docker Hub auth..."
kubectl exec -it $BUILDER_POD -- sh -c "
mkdir -p /root/.docker
echo '{\"auths\":{\"https://index.docker.io/v1/\":{\"username\":\"$DOCKERHUB_USER\",\"password\":\"$DOCKERHUB_TOKEN\"}}}' > /root/.docker/config.json
"

# ── 2. Prepare build context ──
echo "==> Copying repos to build context..."
kubectl exec $BUILDER_POD -- sh -c "
rm -rf /scratch/ctx && mkdir -p /scratch/ctx/workspace
cp -r $PVC_WORKSPACE/miles /scratch/ctx/workspace/miles
cp -r $PVC_WORKSPACE/sglang /scratch/ctx/workspace/sglang
"

# ── 3. Write Dockerfile ──
echo "==> Writing Dockerfile..."
kubectl exec $BUILDER_POD -- sh -c 'cat > /scratch/ctx/Dockerfile << '\''DEOF'\''
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY workspace/miles /workspace/miles
COPY workspace/sglang /workspace/sglang

RUN pip install -r /workspace/miles/requirements.txt

RUN cp /workspace/miles/docker/patch/latest/megatron.patch /root/Megatron-LM/ && \
    cd /root/Megatron-LM && git update-index --refresh && \
    git apply megatron.patch --3way && rm megatron.patch

RUN cd $(python -c "import transformers, os; print(os.path.dirname(os.path.dirname(transformers.__file__)))") && \
    patch -p2 < /workspace/miles/docker/glm5/transformers.patch

RUN printf '\''#!/bin/bash\npython -m torch.distributed.run "$@"\n'\'' > /usr/local/bin/torchrun && chmod +x /usr/local/bin/torchrun && \
    printf '\''#!/bin/bash\npython -m huggingface_hub.commands.huggingface_cli "$@"\n'\'' > /usr/local/bin/huggingface-cli && chmod +x /usr/local/bin/huggingface-cli && \
    ln -sf /usr/local/bin/huggingface-cli /usr/local/bin/hf

RUN pip install "huggingface-hub<1.0,>=0.34.0"

RUN bash /workspace/sglang/sgl-kernel/build_flashmla_gb300.sh

RUN pip install -e /workspace/miles --no-deps
DEOF
'

# ── 4. Build and push ──
echo "==> Building image (this takes ~10 min for FlashMLA CUDA compilation)..."
kubectl exec $BUILDER_POD -- buildctl build \
    --frontend=dockerfile.v0 \
    --local context=/scratch/ctx \
    --local dockerfile=/scratch/ctx \
    --opt build-arg:BASE_IMAGE=$BASE_IMAGE \
    --output "type=image,name=$IMAGE_TAG,push=true"

# ── 5. Cleanup ──
kubectl exec $BUILDER_POD -- rm -rf /scratch/ctx

echo "==> Done! Image pushed to $IMAGE_TAG"
