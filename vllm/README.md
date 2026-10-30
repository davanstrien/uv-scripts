---
viewer: false
tags: [uv-script, vllm, gpu, inference, hf-jobs]
---

# vLLM Inference Scripts

Ready-to-run UV scripts for GPU-accelerated inference using [vLLM](https://github.com/vllm-project/vllm).

These scripts use [UV's inline script metadata](https://docs.astral.sh/uv/guides/scripts/) to automatically manage dependencies - just run with `uv run` and everything installs automatically!

## 📋 Available Scripts

### vlm-classify.py

Vision Language Model (VLM) image classification with structured output constraints.

**Features:**

- 🖼️ Process images through state-of-the-art VLMs (Qwen2-VL)
- 🎯 Structured classification using vLLM's `GuidedDecodingParams`
- 📐 Automatic image resizing to optimize token usage
- 💾 Memory-efficient lazy batch processing
- 🏷️ Simple CLI interface for defining classes
- 🤗 Direct integration with Hugging Face datasets

**Usage:**

```bash
# Basic classification
uv run vlm-classify.py \
    username/input-dataset \
    username/output-dataset \
    --classes "document,photo,diagram,other"

# With custom prompt and image resizing
uv run vlm-classify.py \
    username/input-dataset \
    username/output-dataset \
    --classes "index-card,manuscript,title-page,other" \
    --prompt "What type of historical document is this?" \
    --max-size 768

# Quick test with sample limit
uv run vlm-classify.py \
    davanstrien/sloane-index-cards \
    username/test-output \
    --classes "index,content,other" \
    --max-samples 10
```

**HF Jobs execution:**

```bash
hf jobs uv run \
    --flavor l40sx1 \
    --image vllm/vllm-openai:v0.32.0 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/vllm/raw/main/vlm-classify.py \
    username/input-dataset \
    username/output-dataset \
    --classes "title-page,content,index,other" \
    --max-size 768
```

**Key Parameters:**

- `--classes`: Comma-separated list of classification categories (required)
- `--prompt`: Custom classification prompt (optional, auto-generated if not provided)
- `--max-size`: Maximum image dimension in pixels for resizing (reduces token count)
- `--model`: VLM model to use (default: Qwen/Qwen2-VL-7B-Instruct)
- `--batch-size`: Number of images to process at once (default: 8)
- `--max-samples`: Limit number of samples for testing

### classify-dataset.py

Batch text classification using BERT-style encoder models (e.g., BERT, RoBERTa, DeBERTa, ModernBERT) with vLLM's optimized inference engine.

**Note**: This script is specifically for encoder-only classification models, not generative LLMs.

**Features:**

- 🚀 High-throughput batch processing
- 🏷️ Automatic label mapping from model config
- 📊 Confidence scores for predictions
- 🤗 Direct integration with Hugging Face Hub

**Usage:**

```bash
# Local execution (requires GPU)
uv run classify-dataset.py \
    davanstrien/ModernBERT-base-is-new-arxiv-dataset \
    username/input-dataset \
    username/output-dataset \
    --inference-column text \
    --batch-size 10000
```

**HF Jobs execution:**

```bash
hf jobs uv run \
    --flavor l4x1 \
    --image vllm/vllm-openai:v0.32.0 \
    https://huggingface.co/datasets/uv-scripts/vllm/resolve/main/classify-dataset.py \
    davanstrien/ModernBERT-base-is-new-arxiv-dataset \
    username/input-dataset \
    username/output-dataset \
    --inference-column text \
    --batch-size 100000
```

### generate-responses.py

Generate responses for prompts using generative LLMs (e.g., Llama, Qwen, Mistral) with vLLM's high-performance inference engine.

**Features:**

- 💬 Automatic chat template application
- 📝 Support for both chat messages and plain text prompts
- 🔀 Multi-GPU tensor parallelism support
- 📏 Smart filtering for prompts exceeding context length
- 📊 Comprehensive dataset cards with generation metadata
- ⚡ HF Transfer enabled for fast model downloads
- 🎛️ Full control over sampling parameters
- 🎯 Sample limiting with `--max-samples` for testing

**Usage:**

```bash
# With chat-formatted messages (default)
uv run generate-responses.py \
    username/input-dataset \
    username/output-dataset \
    --messages-column messages \
    --max-tokens 1024

# With plain text prompts (NEW!)
uv run generate-responses.py \
    username/input-dataset \
    username/output-dataset \
    --prompt-column question \
    --max-tokens 1024 \
    --max-samples 100

# With custom model and parameters
uv run generate-responses.py \
    username/input-dataset \
    username/output-dataset \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --prompt-column text \
    --temperature 0.9 \
    --top-p 0.95 \
    --max-model-len 8192
```

**HF Jobs execution (multi-GPU):**

```bash
hf jobs uv run \
    --flavor l4x4 \
    --image vllm/vllm-openai:v0.32.0 \
    -e UV_PRERELEASE=if-necessary \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/vllm/raw/main/generate-responses.py \
    davanstrien/cards_with_prompts \
    davanstrien/test-generated-responses \
    --model-id Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --gpu-memory-utilization 0.9 \
    --max-tokens 600 \
    --max-model-len 8000
```

### Multi-GPU Tensor Parallelism

- Auto-detects available GPUs by default
- Use `--tensor-parallel-size` to manually specify
- Required for models larger than single GPU memory (e.g., 30B+ models)

### Handling Long Contexts

The generate-responses.py script includes smart prompt filtering:

- **Default behavior**: Skips prompts exceeding max_model_len
- **Use `--max-model-len`**: Limit context to reduce memory usage
- **Use `--no-skip-long-prompts`**: Fail on long prompts instead of skipping
- Skipped prompts receive empty responses and are logged

## 📚 About vLLM

vLLM is a high-throughput inference engine optimized for:

- Fast model serving with PagedAttention
- Efficient batch processing
- Support for various model architectures
- Seamless integration with Hugging Face models

## 🔧 Technical Details

### UV Script Benefits

- **Zero setup**: Dependencies install automatically on first run
- **Reproducible**: Locked dependencies ensure consistent behavior
- **Self-contained**: Everything needed is in the script file
- **Direct execution**: Run from local files or URLs

### Dependencies

Scripts use UV's inline metadata for automatic dependency management:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "flashinfer-python",
#     "huggingface-hub[hf_transfer]",
#     "torch",
#     "transformers",
#     "vllm",
# ]
# ///
```

For bleeding-edge features, use the `UV_PRERELEASE=if-necessary` environment variable to allow pre-release versions when needed.

### Docker Image

For HF Jobs, we recommend the official vLLM Docker image: `vllm/vllm-openai`

This image includes:

- Pre-installed CUDA libraries
- vLLM and all dependencies
- UV package manager
- Optimized for GPU inference

### Environment Variables

- `HF_TOKEN`: Your Hugging Face authentication token (auto-detected if logged in)
- `UV_PRERELEASE=if-necessary`: Allow pre-release packages when required
- `HF_HUB_ENABLE_HF_TRANSFER=1`: Automatically enabled for faster downloads

## 🔗 Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [UV Documentation](https://docs.astral.sh/uv/)
- [UV Scripts Organization](https://huggingface.co/uv-scripts)
