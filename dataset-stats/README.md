---
viewer: false
tags:
  - uv-script
  - dataset-statistics
  - polars
  - temporal-analysis
  - hf_jobs
license: apache-2.0
---

<a href="https://huggingface.co/uv-scripts"><img src="https://huggingface.co/spaces/uv-scripts/README/resolve/main/lockup.svg" alt="uv scripts" width="180"></a>

# Dataset Statistics

<a href="https://huggingface.co/uv-scripts"><picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md-dark.svg"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md.svg" alt="Follow uv-scripts on Hugging Face"></picture></a>

UV scripts for analyzing HuggingFace datasets using streaming mode.

## Scripts

### `finepdfs-stats.py` - Temporal Educational Quality Analysis

Analyze educational quality trends across CommonCrawl dumps using Polars streaming. Answers: **"Is the web getting more educational over time?"**

**Features:**
- Polars streaming (no download of 300GB+ dataset)
- Temporal analysis across 106 CommonCrawl dumps (2013-2025)
- ASCII chart visualizations
- Uploads results to HF Hub with auto-generated dataset card
- Supports single language or all 70+ languages

**Quick Examples:**

```bash
# Quick test (10K samples)
uv run https://huggingface.co/datasets/uv-scripts/dataset-stats/raw/main/finepdfs-stats.py \
    --limit 10000 --show-plan

# Analyze English PDFs
uv run https://huggingface.co/datasets/uv-scripts/dataset-stats/raw/main/finepdfs-stats.py \
    --output-repo username/my-stats

# Analyze ALL 70+ languages
uv run https://huggingface.co/datasets/uv-scripts/dataset-stats/raw/main/finepdfs-stats.py \
    --all-languages --output-repo username/my-stats
```

**Run on HF Jobs (recommended for full dataset):**

```bash
hf jobs uv run \
    -s HF_TOKEN \
    -e HF_XET_HIGH_PERFORMANCE=1 \
    https://huggingface.co/datasets/uv-scripts/dataset-stats/raw/main/finepdfs-stats.py \
    -- --all-languages --output-repo username/finepdfs-temporal-stats
```

**Example output:** [davanstrien/finepdfs-temporal-stats-all](https://huggingface.co/datasets/davanstrien/finepdfs-temporal-stats-all)

**Performance:**
- 50M docs in ~14 minutes (~60K docs/sec)
- Single scan using [Polars HF Hub integration](https://huggingface.co/docs/hub/datasets-polars)
- Works on HF Jobs CPU instances

## Related Scripts

Check out other scripts in the [uv-scripts organization](https://huggingface.co/uv-scripts):
- **dataset-creation**: Create datasets from PDFs and other formats
- **vllm**: GPU-accelerated classification and inference
- **ocr**: Document OCR using vision-language models

## Why UV Scripts?

UV scripts are self-contained Python scripts that:
- Run with a single `uv run` command (no setup required)
- Include all dependencies in PEP 723 inline metadata
- Work seamlessly on both local machines and HF Jobs

Learn more about UV: https://docs.astral.sh/uv/

