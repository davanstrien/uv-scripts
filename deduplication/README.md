---
viewer: false
tags: [uv-script, deduplication, semantic-similarity, data-processing, hf_jobs]
---

<a href="https://huggingface.co/uv-scripts"><img src="https://huggingface.co/spaces/uv-scripts/README/resolve/main/lockup.svg" alt="uv scripts" width="180"></a>

# Semantic Deduplication UV Script

<a href="https://huggingface.co/uv-scripts"><picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md-dark.svg"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md.svg" alt="Follow uv-scripts on Hugging Face"></picture></a>

Remove duplicate / near-duplicate text samples from a Hugging Face dataset by semantic similarity — clean training data and prevent train/test leakage. Uses [SemHash](https://github.com/MinishLab/semhash) with Model2Vec embeddings: **CPU-optimized, no GPU required.**

## Quick start

```bash
# CPU is enough — run on Hugging Face Jobs
hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/deduplication/raw/main/semantic-dedupe.py \
    your-input-dataset text your-output-dataset

# or locally
uv run https://huggingface.co/datasets/uv-scripts/deduplication/raw/main/semantic-dedupe.py \
    your-input-dataset text your-output-dataset
```

Arguments are `INPUT_DATASET TEXT_COLUMN OUTPUT_DATASET`. Tune with `--threshold` (similarity cutoff) and `--max-samples` (for testing); run with `--help` for all options.
