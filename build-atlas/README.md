---
viewer: false
tags: [uv-script, hf-jobs, hf-buckets]
---

<a href="https://huggingface.co/uv-scripts"><img src="https://huggingface.co/spaces/uv-scripts/README/resolve/main/lockup.svg" alt="uv scripts" width="180"></a>

# Atlas Export

<a href="https://huggingface.co/uv-scripts"><picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md-dark.svg"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md.svg" alt="Follow uv-scripts on Hugging Face"></picture></a>

[Embedding Atlas](https://github.com/apple/embedding-atlas) is an open-source library from Apple for creating interactive, browser-based visualizations of embedding spaces. It renders millions of data points with WebGPU acceleration, supports real-time search and filtering, and automatically generates cluster labels.

These scripts wrap Embedding Atlas to make it easy to go from a HuggingFace dataset to a deployed visualization. See [open-library-atlas](https://huggingface.co/spaces/davanstrien/atlas-bucket-test) for a live example (2M books).

![Example Visualization](https://huggingface.co/datasets/huggingface/documentation-images/resolve/58e8cf44ec5d74712775b7049b1203f9b0b15ebb/hub/atlas-dataset-library-screenshot.png)

## Scripts

| Script | Description | Best for |
|--------|-------------|----------|
| **`atlas-e2e.py`** | End-to-end: dataset to deployed Space (experimental) | One-command pipeline |
| **`atlas-build-gpu.py`** | GPU atlas build with cuml.accel UMAP | Running as an HF Job |
| **`atlas-deploy.py`** | Deploy a Space from bucket data | Deploying existing builds |
| `atlas-export.py` | All-in-one static export | Small datasets (<10GB) |
| `atlas-export-remote.py` | Static export with remote data | Medium datasets |

## Bucket Pipeline (Recommended)

The new pipeline uses [Storage Buckets](https://huggingface.co/docs/hub/storage-buckets) + [Jobs](https://huggingface.co/docs/hub/jobs) + [Spaces](https://huggingface.co/docs/hub/spaces) for large-scale visualizations:

```
atlas-e2e.py (local orchestrator)
  │
  ├─ 1. Creates Storage Bucket
  │
  ├─ 2. Submits GPU Job (atlas-build-gpu.py)
  │     • Embeds text/images on GPU
  │     • UMAP via cuml.accel (~50x faster)
  │     • Writes parquet to bucket
  │
  └─ 3. Deploys Docker Space (atlas-deploy.py)
        • Mounts bucket (read-only)
        • Serves embedding-atlas in server mode
        • Server-side DuckDB (not browser WASM)
```

### Quick Start

```bash
# One command: dataset → deployed Space
uv run atlas-e2e.py stanfordnlp/imdb \
    --text text --split train \
    --name imdb-atlas --sample 50000
```

### Step by Step

```bash
# 1. Build atlas on GPU (runs as HF Job)
# (bucket mounts at /output — Jobs reserves /data for the script artifact)
hf jobs uv run --flavor a100-large \
    -v hf://buckets/user/atlas-data:/output \
    -s HF_TOKEN --timeout 2h \
    atlas-build-gpu.py my-org/my-dataset \
    --text text --name my-atlas --sample 1000000

# 2. Deploy Space from bucket
uv run atlas-deploy.py \
    --name my-atlas \
    --bucket user/atlas-data \
    --space-id user/my-atlas-viz
```

### With a Prep Step (for filtering/enrichment)

```bash
# Prep: filter and add categories with DuckDB
hf jobs uv run --flavor cpu-upgrade \
    -v hf://buckets/user/atlas-data:/output \
    -s HF_TOKEN \
    open-library-prep.py --output /output/books/books.parquet

# Build: embed the prepped data
hf jobs uv run --flavor a100-large \
    -v hf://buckets/user/atlas-data:/output \
    -s HF_TOKEN --timeout 2h \
    atlas-build-gpu.py /output/books/books.parquet \
    --text title --name books-atlas --sample 2000000

# Deploy
uv run atlas-deploy.py --name books-atlas --bucket user/atlas-data
```

### GPU UMAP Performance

The build script uses [cuml.accel](https://docs.rapids.ai/api/cuml/stable/cuml-accel/) for zero-code-change GPU acceleration of UMAP. No changes to embedding-atlas needed — just an environment variable.

| Dataset | Rows | Time (A100) | Cost |
|---------|------|-------------|------|
| IMDB | 5K | ~1 min | $0.04 |
| TinyStories | 1M | ~20 min | $0.83 |
| Open Library | 2M | ~40 min | $1.67 |

For comparison: CPU UMAP on 250K rows took >2 hours.

## Legacy Scripts

### Basic (data embedded in Space)

```bash
uv run atlas-export.py stanfordnlp/imdb --space-name my-imdb-viz
```

### Remote data

```bash
uv run atlas-export-remote.py stanfordnlp/imdb \
    --space-name my-imdb-viz \
    --data-repo my-imdb-data
```

## Examples

### Text Datasets

```bash
# Custom embedding model with sampling
uv run atlas-export-remote.py wikipedia \
    --space-name wiki-viz \
    --data-repo wiki-atlas-data \
    --model nomic-ai/nomic-embed-text-v1.5 \
    --text-column text \
    --sample 50000
```

### Image Datasets

```bash
# Visualize image datasets with CLIP
uv run atlas-export-remote.py food101 \
    --space-name food-atlas \
    --data-repo food-atlas-data \
    --image-column image \
    --text-column label \
    --sample 5000
```

### Pre-computed Embeddings

```bash
# If you already have embeddings in your dataset
uv run atlas-export.py my-dataset-with-embeddings \
    --space-name my-viz \
    --no-compute-embeddings \
    --x-column umap_x \
    --y-column umap_y
```

### From an Existing Export

```bash
# Use an atlas export ZIP you already have
uv run atlas-export-remote.py \
    --from-export atlas_export.zip \
    --space-name my-viz \
    --data-repo my-data
```

### Sharded Datasets (Glob)

```bash
# Use a glob pattern to combine multiple parquet shards
hf jobs uv run --flavor a10g-small -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/build-atlas/raw/main/atlas-export-remote.py \
    --glob "hf://datasets/HuggingFaceFW/finephrase/faq/*.parquet" \
    --glob-max-shards 10 \
    --space-name finephrase-atlas --data-repo finephrase-data \
    --text-column text --sample 50000
```

### Multiple Parquet URLs

```bash
# Pass several parquet URLs directly as inputs
hf jobs uv run --flavor t4-small -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/build-atlas/raw/main/atlas-export-remote.py \
    https://huggingface.co/datasets/my-org/my-data/resolve/main/shard-0.parquet \
    https://huggingface.co/datasets/my-org/my-data/resolve/main/shard-1.parquet \
    https://huggingface.co/datasets/my-org/my-data/resolve/main/shard-2.parquet \
    --space-name my-atlas --data-repo my-data \
    --text-column text --sample 50000
```

### Lance Format Datasets

```bash
# Lance datasets work out of the box (pylance is included)
hf jobs uv run --flavor a100-large -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/build-atlas/raw/main/atlas-export-remote.py \
    librarian-bots/arxiv-cs-papers-lance \
    --space-name arxiv-atlas --data-repo arxiv-data \
    --text-column abstract
```

### GPU Acceleration (HF Jobs)

```bash
# Run on HF Jobs with GPU — the recommended way for large datasets
hf jobs uv run --flavor t4-small -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/build-atlas/raw/main/atlas-export-remote.py \
    stanfordnlp/imdb \
    --space-name imdb-viz \
    --data-repo imdb-atlas-data \
    --sample 10000

# With a bigger GPU for faster processing
hf jobs uv run --flavor a10g-large -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/build-atlas/raw/main/atlas-export-remote.py \
    your-dataset \
    --space-name your-atlas \
    --data-repo your-atlas-data \
    --text-column output \
    --sample 50000
```

Available GPU flavors: `t4-small`, `t4-medium`, `l4x1`, `a10g-small`, `a10g-large`, `a100-large`.
For large datasets, add `--timeout 7200` (2 hours) to the `hf jobs` command.

## Key Options

### atlas-export.py

| Option | Description | Default |
|--------|-------------|---------|
| `dataset_id` | HuggingFace dataset to visualize | Required |
| `--space-name` | Name for your Space | Required |
| `--model` | Embedding model to use | Auto-selected |
| `--text-column` | Column containing text | "text" |
| `--image-column` | Column containing images | None |
| `--sample` | Number of samples to visualize | All |
| `--batch-size` | Batch size for embedding generation | 32 (text), 16 (images) |
| `--split` | Dataset split to use | "train" |

### atlas-export-remote.py

| Option | Description | Default |
|--------|-------------|---------|
| `inputs` | Dataset ID(s) or parquet URL(s) — positional, supports multiple | Required* |
| `--space-name` | Name for your Space | Required |
| `--data-repo` | Name for the HF dataset repo (stores parquet) | Required |
| `--glob` | HF glob pattern for parquet shards | None |
| `--glob-max-shards` | Randomly sample N shards from glob matches | All |
| `--model` | Embedding model to use | Auto-selected |
| `--text-column` | Column containing text | "text" |
| `--image-column` | Column containing images | None |
| `--sample` | Number of samples to visualize | All |
| `--split` | Dataset split to use | "train" |
| `--trust-remote-code` | Trust remote code in datasets/models | False |
| `--from-export` | Use an existing atlas export ZIP | None |
| `--organization` | HF org for repos (default: your username) | None |
| `--private` | Make both Space and dataset private | False |
| `--private-space` | Make only the Space private | False |
| `--private-data` | Make only the dataset private | False |
| `--hf-token` | Explicit HF token (or set HF_TOKEN env) | Auto |
| `--output-dir` | Local output directory | Temp dir |
| `--local-only` | Prepare locally without deploying | False |

*Either `inputs`, `--glob`, or `--from-export` is required.

Run either script without arguments to see all options.

## How It Works

### atlas-export.py
1. Loads dataset from HuggingFace Hub
2. Generates embeddings (or uses pre-computed)
3. Creates static web app with embedded data
4. Deploys to HF Space

### atlas-export-remote.py
1. Loads dataset and generates embeddings
2. Exports viewer with `--export-metadata` pointing to the remote parquet URL
3. Uploads parquet to a HF dataset repo
4. Deploys the lightweight viewer (~100MB) to a HF Space
5. The viewer fetches data on-demand via HTTP range requests

## Tips

- **Re-running** with the same `--space-name`/`--data-repo` updates in place (no need to delete first)
- **Lance format** is supported out of the box — `pylance` is bundled as a dependency
- **Source datasets** are automatically linked in the Space metadata
- **More shards = more RAM** — use `a10g-large` or `a100-large` for many shards
- Keep `--glob-max-shards` reasonable (5-30) to avoid OOM on smaller GPU flavors

## Live Examples

- [open-library-atlas](https://huggingface.co/spaces/davanstrien/atlas-bucket-test) — 2M books with category coloring (bucket pipeline)
- [nemotron-v3-atlas](https://huggingface.co/spaces/davanstrien/nemotron-v3-atlas) — 250K NVIDIA training data samples
- [arxiv-cs-atlas](https://huggingface.co/spaces/librarian-bots/arxiv-cs-atlas) — 215K arXiv CS papers (lance format)

## Credits

Built on [Embedding Atlas](https://github.com/apple/embedding-atlas) by Apple (>= 0.19.1). GPU UMAP via [cuML](https://docs.rapids.ai/api/cuml/stable/cuml-accel/).

---

Part of the [UV Scripts](https://huggingface.co/uv-scripts) collection
