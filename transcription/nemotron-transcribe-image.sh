#!/usr/bin/env bash
# Build and push the prebuilt image for nemotron-transcribe.py, inside an HF Job.
#
# The recipe installs vLLM 0.32.0 + NeMo Speech main at runtime and FlashInfer
# JIT-builds kernels on first start. This script does all of that once, on the
# target GPU, and pushes the result as one layer on top of the vLLM image:
#
#   - /opt/nemotron-transcribe/nemo-speech (source + locked venv + kernels)
#   - FlashInfer JIT cache, vLLM torch.compile cache and the Triton cubins it
#     references (warm for the flavor
#     the build ran on; other GPUs rebuild their kernels on first start)
#
# The model checkpoint is NOT baked in (it downloads in ~35 s on Jobs).
# No Docker daemon is needed: `crane append` adds the layer and pushes.
#
# Run (the container repo must exist; HF_TOKEN needs write on its org):
#
#   hf jobs run --flavor rtx-pro-6000 -s HF_TOKEN --timeout 1h \
#       -v hf://datasets/uv-scripts/transcription:/recipe:ro \
#       vllm/vllm-openai:v0.32.0 -- \
#       bash /recipe/nemotron-transcribe-image.sh cr.hf.co/uv-scripts/nemotron-transcribe
set -euo pipefail

REPO="${1:?usage: nemotron-transcribe-image.sh <cr.hf.co/namespace/name>}"
RECIPE_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="vllm/vllm-openai:v0.32.0"
WORKDIR=/opt/nemotron-transcribe

pip install -q uv

# 1. Install + warm up: run the recipe on a short silent clip.
mkdir -p /tmp/warmup-in /tmp/warmup-out
python3 -c "import wave; w = wave.open('/tmp/warmup-in/silence.wav', 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b'\0' * 64000); w.close()"
uv run "$RECIPE_DIR/nemotron-transcribe.py" /tmp/warmup-in /tmp/warmup-out --workdir "$WORKDIR"

NEMO_SHA="$(cat "$WORKDIR/nemo-speech/.nemo-ref")"
TAG="v0.32.0-nemo-${NEMO_SHA:0:12}"

# 2. One layer: the serving stack and the warm caches, not the checkpoint.
tar -C / -cf /tmp/layer.tar \
    --exclude="${WORKDIR#/}/model" --exclude="${WORKDIR#/}/nemo-speech.tar.gz" \
    "${WORKDIR#/}" root/.cache/flashinfer root/.cache/vllm \
    tmp/torchinductor_root  # Triton cubins the vLLM compile cache points at
ls -lh /tmp/layer.tar

# 3. Append it to the base image and push (also tag :latest).
curl -sL https://github.com/google/go-containerregistry/releases/download/v0.22.1/go-containerregistry_Linux_x86_64.tar.gz \
    | tar -xz -C /usr/local/bin crane
crane auth login cr.hf.co -u "$(python3 -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>/dev/null || echo user)" -p "$HF_TOKEN"
crane append --platform linux/amd64 -b "$BASE" -f /tmp/layer.tar -t "$REPO:$TAG"
crane tag "$REPO:$TAG" latest
echo "PUSHED $REPO:$TAG"
