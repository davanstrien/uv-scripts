#!/usr/bin/env bash
# Build cr.hf.co/<namespace>/hpd-parsing-vllm: Paddle's HPD-Parsing image plus uv.
#
# Paddle's image (vendor vLLM fork 0.17.1+hpdparsing) has no `uv`, which
# `hf jobs uv run --image <image>` needs. This adds one layer, the uv and uvx binaries in /usr/local/bin, to the
# vendor image pinned by digest. Nothing else changes: same user (hpd), same venv,
# same entrypoint. No Docker daemon is needed: `crane append` adds the layer and
# pushes. Runs on a CPU Job (the base layers stream from Baidu's registry to cr.hf.co).
#
# Run (the container repo must exist; HF_TOKEN needs write on its namespace):
#
#   hf jobs run --flavor cpu-upgrade -s HF_TOKEN --timeout 1h \
#       -v hf://datasets/uv-scripts/ocr:/recipe:ro \
#       python:3.12 -- bash /recipe/images/hpd-parsing/build.sh cr.hf.co/<namespace>/hpd-parsing-vllm
set -euo pipefail

REPO="${1:?usage: build.sh <cr.hf.co/namespace/name>}"
BASE_DIGEST="sha256:87496aa5dd702a7df6c70bb00c29d5bf8d1a0f0505d66b613fafa6f71cd72de2"
BASE="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm@${BASE_DIGEST}"
UV_VERSION="0.13.6"
CRANE_VERSION="v0.22.1"
TAG="${BASE_DIGEST:7:8}-uv${UV_VERSION}"

# 1. The layer: uv + uvx from the official release, checksum-verified.
cd /tmp
UV_TGZ="uv-x86_64-unknown-linux-gnu.tar.gz"
curl -sSfLO "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${UV_TGZ}"
curl -sSfLO "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${UV_TGZ}.sha256"
sha256sum -c "${UV_TGZ}.sha256"
tar -xzf "$UV_TGZ"
mkdir -p layer/usr/local/bin
install -m 755 uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx layer/usr/local/bin/
tar -C layer --owner=0 --group=0 -cf layer.tar usr
tar -tvf layer.tar

# 2. Append it to the vendor image and push.
curl -sSfL "https://github.com/google/go-containerregistry/releases/download/${CRANE_VERSION}/go-containerregistry_Linux_x86_64.tar.gz" \
    | tar -xz -C /usr/local/bin crane
crane auth login cr.hf.co -u "${HF_USER:-user}" -p "$HF_TOKEN"
crane append --platform linux/amd64 -b "$BASE" -f layer.tar -t "$REPO:$TAG"
crane tag "$REPO:$TAG" latest
echo "PUSHED $REPO:$TAG $(crane digest "$REPO:$TAG")"
