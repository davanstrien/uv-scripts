---
viewer: false
tags: [uv-script, object-detection]
---

<a href="https://huggingface.co/uv-scripts"><img src="https://huggingface.co/spaces/uv-scripts/README/resolve/main/lockup.svg" alt="uv scripts" width="180"></a>

# Object Detection Dataset Scripts

<a href="https://huggingface.co/uv-scripts"><picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md-dark.svg"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md.svg" alt="Follow uv-scripts on Hugging Face"></picture></a>

8 scripts to **create**, convert, review, validate, inspect, diff, and sample object detection datasets on the Hub. Supports 6 bbox formats — no setup required.

Start from nothing: `falcon-perception.py` generates a first-pass detection dataset for any class you can name, zero-shot, with no labelling and no training. The other six then convert, check, and measure it.
This repository is inspired by [panlabel](https://github.com/strickvl/panlabel)

## Quick Start

Convert bounding box formats without cloning anything:

```bash
# Convert COCO-style bboxes to YOLO normalized format
uv run convert-hf-dataset.py merve/coco-dataset merve/coco-yolo \
    --from coco_xywh --to yolo --max-samples 100
```

That's it! The script will:

- Load the dataset from the Hub
- Convert all bounding boxes in-place
- Push the result to a new dataset repo
- View results at: `https://huggingface.co/datasets/merve/coco-yolo`

## Scripts

| Script | Description |
|--------|-------------|
| `falcon-perception.py` | **Create** a detection dataset zero-shot from any image dataset — name a class, get boxes + masks (runs on Apple Silicon too) |
| `falcon-perception-bucket.py` | Same, reading images from an HF bucket, resumable across restarts |
| `review-detections.py` | **Review** a detection dataset in your browser — keyboard accept/reject per image or per box, acceptance + missed rates, pushes a `review` column |
| `convert-hf-dataset.py` | Convert between 6 bbox formats and push to Hub |
| `validate-hf-dataset.py` | Check annotations for errors (invalid bboxes, duplicates, bounds) |
| `stats-hf-dataset.py` | Compute statistics (counts, label histogram, area, co-occurrence) |
| `diff-hf-datasets.py` | Compare two datasets semantically (IoU-based annotation matching) |
| `sample-hf-dataset.py` | Create subsets (random or stratified) and push to Hub |
| `embed-bucket-images.py` | Join image bytes back into a bucket teacher pass — gold excluded + asserted, one final-schema write |
| `materialize-coco.py` | Generate a canonical COCO 2017 directory tree from the parquet, in-job, on ephemeral disk |
| `render-detections.py` | Render box/mask overlays and pixel-verify they actually drew (blank renders exit nonzero) |
| `smoke-test.py` | Free local check (~30 s) that the plumbing chain (`embed-bucket-images.py` → `materialize-coco.py` → `render-detections.py`) still produces correct output — run before any job depends on it |

## Supported Bbox Formats

All scripts support these 6 bounding box formats, matching the [panlabel](https://github.com/strickvl/panlabel) Rust CLI:

| Format | Encoding | Coordinate Space |
|--------|----------|------------------|
| `coco_xywh` | `[x, y, width, height]` | Pixels |
| `xyxy` | `[xmin, ymin, xmax, ymax]` | Pixels |
| `voc` | `[xmin, ymin, xmax, ymax]` | Pixels (alias for `xyxy`) |
| `yolo` | `[center_x, center_y, width, height]` | Normalized 0–1 |
| `tfod` | `[xmin, ymin, xmax, ymax]` | Normalized 0–1 |
| `label_studio` | `[x, y, width, height]` | Percentage 0–100 |

Conversions go through XYXY pixel-space as the intermediate representation, so any format can be converted to any other format.

## Common Options

All scripts accept flexible column mapping. Datasets can store annotations as flat columns or nested under an `objects` dict — both layouts are handled automatically.

| Option | Description |
|--------|-------------|
| `--bbox-column` | Column containing bboxes (default: `bbox`) |
| `--category-column` | Column containing category labels (default: `category`) |
| `--width-column` | Column for image width (default: `width`) |
| `--height-column` | Column for image height (default: `height`) |
| `--split` | Dataset split (default: `train`) |
| `--max-samples` | Limit number of samples (useful for testing) |
| `--hf-token` | HF API token (or set `HF_TOKEN` env var) |
| `--private` | Make output dataset private |

Every script supports `--help` to see all available options:

```bash
uv run convert-hf-dataset.py --help
```

## Convert (`convert-hf-dataset.py`)

Convert bounding boxes between any of the 6 supported formats:

```bash
# COCO -> XYXY
uv run convert-hf-dataset.py merve/license-plates merve/license-plates-voc \
    --from coco_xywh --to voc

# YOLO -> COCO
uv run convert-hf-dataset.py merve/license-plates merve/license-plates-yolo \
    --from coco_xywh --to yolo

# TFOD (normalized xyxy) -> COCO
uv run convert-hf-dataset.py merve/license-plates-tfod merve/license-plates-coco \
    --from tfod --to coco_xywh

# Label Studio (percentage xywh) -> XYXY
uv run convert-hf-dataset.py merve/ls-dataset merve/ls-xyxy \
    --from label_studio --to xyxy

# Test on 10 samples first
uv run convert-hf-dataset.py merve/dataset merve/converted \
    --from xyxy --to yolo --max-samples 10

# Shuffle before converting a subset
uv run convert-hf-dataset.py merve/dataset merve/converted \
    --from coco_xywh --to tfod --max-samples 500 --shuffle
```

| Option | Description |
|--------|-------------|
| `--from` | Source bbox format (required) |
| `--to` | Target bbox format (required) |
| `--batch-size` | Batch size for map (default: 1000) |
| `--create-pr` | Push as PR instead of direct commit |
| `--shuffle` | Shuffle dataset before processing |
| `--seed` | Random seed for shuffling (default: 42) |

## Validate (`validate-hf-dataset.py`)

Check annotations for common issues:

```bash
# Basic validation
uv run validate-hf-dataset.py merve/coco-dataset

# Validate YOLO-format dataset
uv run validate-hf-dataset.py merve/yolo-dataset --bbox-format yolo

# Validate TFOD-format dataset
uv run validate-hf-dataset.py merve/tfod-dataset --bbox-format tfod

# Strict mode (warnings become errors)
uv run validate-hf-dataset.py merve/dataset --strict

# JSON report
uv run validate-hf-dataset.py merve/dataset --report json

# Stream large datasets without full download
uv run validate-hf-dataset.py merve/huge-dataset --streaming --max-samples 5000

# Push validation report to Hub
uv run validate-hf-dataset.py merve/dataset --output-dataset merve/validation-report
```

**Issue Codes:**

| Code | Level | Description |
|------|-------|-------------|
| E001 | Error | Bbox/category count mismatch |
| E002 | Error | Invalid bbox (missing values) |
| E003 | Error | Non-finite coordinates (NaN/Inf) |
| E004 | Error | xmin > xmax |
| E005 | Error | ymin > ymax |
| W001 | Warning | No annotations in example |
| W002 | Warning | Zero or negative area |
| W003 | Warning | Bbox before image origin |
| W004 | Warning | Bbox beyond image bounds |
| W005 | Warning | Empty category label |
| W006 | Warning | Duplicate file name |

## Stats (`stats-hf-dataset.py`)

Compute rich statistics for a dataset:

```bash
# Basic stats
uv run stats-hf-dataset.py merve/coco-dataset

# Top 20 label histogram, JSON output
uv run stats-hf-dataset.py merve/dataset --top 20 --report json

# Stats for TFOD-format dataset
uv run stats-hf-dataset.py merve/dataset --bbox-format tfod

# Stream large datasets
uv run stats-hf-dataset.py merve/huge-dataset --streaming --max-samples 10000

# Push stats report to Hub
uv run stats-hf-dataset.py merve/dataset --output-dataset merve/stats-report
```

Reports include: summary counts, label distribution, annotation density, bbox area/aspect ratio distributions, per-category area stats, category co-occurrence pairs, and image resolution distribution.

## Diff (`diff-hf-datasets.py`)

Compare two datasets semantically using IoU-based annotation matching:

```bash
# Basic diff
uv run diff-hf-datasets.py merve/dataset-v1 merve/dataset-v2

# Stricter matching
uv run diff-hf-datasets.py merve/old merve/new --iou-threshold 0.7

# Per-annotation change details
uv run diff-hf-datasets.py merve/old merve/new --detail

# JSON report
uv run diff-hf-datasets.py merve/old merve/new --report json
```

Reports include: shared/unique images, shared/unique categories, matched/added/removed/modified annotations.

## Sample (`sample-hf-dataset.py`)

Create random or stratified subsets:

```bash
# Random 500 samples
uv run sample-hf-dataset.py merve/dataset merve/subset -n 500

# 10% fraction
uv run sample-hf-dataset.py merve/dataset merve/subset --fraction 0.1

# Stratified sampling (preserves class distribution)
uv run sample-hf-dataset.py merve/dataset merve/subset \
    -n 200 --strategy stratified

# Filter by categories
uv run sample-hf-dataset.py merve/dataset merve/subset \
    -n 100 --categories "cat,dog,bird"

# Reproducible sampling
uv run sample-hf-dataset.py merve/dataset merve/subset \
    -n 500 --seed 42
```

| Option | Description |
|--------|-------------|
| `-n` | Number of samples to select |
| `--fraction` | Fraction of dataset (0.0–1.0) |
| `--strategy` | `random` (default) or `stratified` |
| `--categories` | Comma-separated list of categories to filter by |
| `--category-mode` | `images` (default) or `annotations` |

## Run Locally

```bash
# Clone and run
git clone https://huggingface.co/datasets/uv-scripts/panlabel
cd panlabel
uv run convert-hf-dataset.py input-dataset output-dataset --from coco_xywh --to yolo

# Or run directly from URL
uv run https://huggingface.co/datasets/uv-scripts/panlabel/raw/main/convert-hf-dataset.py \
    input-dataset output-dataset --from coco_xywh --to yolo
```

Works with any Hugging Face dataset containing object detection annotations — COCO, YOLO, VOC, TFOD, or Label Studio format.

## Making a dataset from scratch

The other scripts assume you already have annotations. `falcon-perception.py` is where they can come from — [Falcon-Perception](https://huggingface.co/tiiuae/Falcon-Perception) finds every instance of a class you name, with no label set and no training:

```bash
# 1. does the model do the thing? (your laptop — no GPU needed)
uv run https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/falcon-perception.py --image page.jpg --query illustration --preview

# 2. does it work on YOUR data? (first rows of the real corpus)
uv run https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/falcon-perception.py --dataset biglam/british-library-book-images \
    --config plates --limit 3 --preview

# 3. the whole corpus, on a GPU
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/falcon-perception.py \
    --dataset biglam/british-library-book-images --config plates \
    --id-col fname --query illustration --out you/plates-illustrations

# 4. it is already in `yolo` format — the rest of this directory just works
uv run https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/validate-hf-dataset.py you/plates-illustrations --bbox-format yolo
uv run https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/stats-hf-dataset.py    you/plates-illustrations --bbox-format yolo
```

Falcon emits boxes as normalised centre x,y + w,h, which *is* the `yolo` format above, so no conversion step is needed.

**The correction loop.** A zero-shot first pass is a starting point, not ground truth. Convert it for human review, correct it, then diff the two to find out how good the first pass actually was:

```bash
uv run https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/convert-hf-dataset.py you/plates-illustrations you/for-review --from yolo --to label_studio
#  ... correct in Label Studio, push as you/corrected ...
uv run https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/diff-hf-datasets.py you/plates-illustrations you/corrected   # IoU match = zero-shot accuracy
```

**Runs without a CUDA GPU.** Unlike most recipes in this repo, `falcon-perception.py` selects the MLX backend on Apple Silicon automatically. It is slower there (about 6 s/img vs 0.4 on an A10G), which is the right trade for step 1 and 2 above — checking your class name works before spending GPU hours.

### Known limits

Measured, not guessed — see the script docstrings for the failure each one came from.

| Limit | What to do |
|---|---|
| `--query` is a **class name**, not an instruction | `illustration` works; `the illustration, excluding captions` returns nothing |
| **One class per run** | A combined query returned 6 instances where three single-class runs found 24. N classes = N runs, then concatenate |
| **No confidence scores** — the model has no score token | Sort review by the emitted `rectangularity` (mask area ÷ bbox area, measured 0.34–1.00) and apply an area floor |
| `a10g-small` gets OOMKilled | The engine's auto-config sizes from the GPU and ignores host RAM — use `a10g-large` |

### Just want the numbers?

`--out` takes a file path as readily as a repo id — no Hub push, nothing to clean up:

```bash
uv run https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/falcon-perception.py --image page.jpg --query illustration --out results.json
uv run https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/falcon-perception.py --image "scans/*.jpg" --query illustration --out results.jsonl
uv run https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/falcon-perception.py --image page.jpg --query illustration --json | jq '.[0].objects.bbox'
```

Anything ending `.json`, `.jsonl` or `.parquet` is written locally; anything else is treated as a Hub dataset repo id.

### Bucket runs

`falcon-perception-bucket.py` reads images from an HF bucket and writes resumable parquet parts back to a bucket — kill it and re-run the same command, done keys are skipped. Publish once at the end to use the rest of this directory:

```python
from datasets import ClassLabel, Sequence, load_dataset
ds = load_dataset("parquet", data_files="hf://buckets/<namespace>/<bucket>/part-*.parquet",
                  split="train")
feats = ds.features.copy()  # parquet stores category as bare ints; name the class
feats["objects"]["category"] = Sequence(ClassLabel(names=[ds[0]["query"]]))
ds.cast(feats).push_to_hub("<namespace>/<dataset>")  # a dataset repo, distinct from the bucket
```

### Output columns

`objects.bbox` (`yolo`), `objects.category` (a `ClassLabel` named after the query), `objects.area`, `objects.rectangularity`, plus `image`, `image_id` (int64 — COCO-style trainers require an integer id), `source_id` (the original key), `width`, `height`, `n_instances`, and `masks_rle` (COCO RLE — segmentation rides along; the bbox scripts ignore it).

### Train on the output

Convert to COCO pixel boxes, then fine-tune a compact Apache-2.0 detector (D-FINE, RT-DETRv2 — not ultralytics/YOLO, which is AGPL). The [`SKILL.md`](SKILL.md) in this directory walks an agent (or you) through the whole loop — teacher labels → validate → convert → train → honest eval:

```bash
uv run https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/convert-hf-dataset.py you/plates-illustrations you/plates-coco --from yolo --to coco_xywh
```
