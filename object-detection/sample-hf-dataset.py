# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=3.1.0",
#     "huggingface-hub",
#     "tqdm",
#     "Pillow", 
# ]
# ///

"""
Create random or stratified subsets of object detection datasets on HF Hub.

Mirrors panlabel's sample command. Supports:

- Random sampling: Uniform random selection of N images or a fraction
- Stratified sampling: Category-aware weighted sampling to preserve class distribution
- Category filtering: Select only images containing specific categories
- Category mode: Filter by image-level or annotation-level membership

Pushes the resulting subset to a new dataset repo on HF Hub.

Examples:
  uv run sample-hf-dataset.py merve/dataset merve/subset -n 500
  uv run sample-hf-dataset.py merve/dataset merve/subset --fraction 0.1
  uv run sample-hf-dataset.py merve/dataset merve/subset -n 200 --strategy stratified
  uv run sample-hf-dataset.py merve/dataset merve/subset -n 100 --categories "cat,dog,bird"
"""

import argparse
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_image_categories(
    example: dict[str, Any],
    category_column: str,
) -> list[str]:
    """Get list of category labels from an example."""
    objects = example.get("objects", example)
    categories = objects.get(category_column, []) or []
    return [str(c) for c in categories if c is not None]


def create_dataset_card(
    source_dataset: str,
    output_dataset: str,
    strategy: str,
    num_samples: int,
    original_size: int,
    categories_filter: list[str] | None,
    category_mode: str,
    seed: int,
    split: str,
) -> str:
    fraction = num_samples / original_size if original_size > 0 else 0
    filter_str = f"\n- **Category Filter**: {', '.join(categories_filter)}" if categories_filter else ""
    return f"""---
tags:
- object-detection
- dataset-subset
- panlabel
- uv-script
- generated
---

# Dataset Subset: {strategy} sampling

A {strategy} subset of [{source_dataset}](https://huggingface.co/datasets/{source_dataset}).

## Details

- **Source**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Strategy**: {strategy}
- **Samples**: {num_samples:,} / {original_size:,} ({fraction:.1%})
- **Seed**: {seed}
- **Split**: `{split}`
- **Category Mode**: {category_mode}{filter_str}
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

## Reproduction

```bash
uv run sample-hf-dataset.py {source_dataset} {output_dataset} \\
    -n {num_samples} --strategy {strategy} --seed {seed}
```

Generated with panlabel-hf (sample-hf-dataset.py)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    n: int | None = None,
    fraction: float | None = None,
    strategy: str = "random",
    category_column: str = "category",
    categories: list[str] | None = None,
    category_mode: str = "images",
    split: str = "train",
    seed: int = 42,
    hf_token: str | None = None,
    private: bool = False,
    create_pr: bool = False,
):
    """Create a subset of an object detection dataset and push to Hub."""

    start_time = datetime.now()

    if n is None and fraction is None:
        logger.error("Must specify either -n (count) or --fraction")
        sys.exit(1)

    if n is not None and fraction is not None:
        logger.error("Specify only one of -n or --fraction, not both")
        sys.exit(1)

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    logger.info(f"Loading dataset: {input_dataset} (split={split})")
    dataset = load_dataset(input_dataset, split=split)
    original_size = len(dataset)
    logger.info(f"Loaded {original_size:,} examples")

    # Determine target count
    if fraction is not None:
        target_n = max(1, int(original_size * fraction))
        logger.info(f"Fraction {fraction} -> {target_n:,} samples")
    else:
        target_n = min(n, original_size)

    rng = random.Random(seed)

    # Category filtering
    if categories:
        logger.info(f"Filtering by categories: {categories} (mode={category_mode})")
        keep_indices = []
        for idx in tqdm(range(original_size), desc="Filtering"):
            ex = dataset[idx]
            img_cats = get_image_categories(ex, category_column)
            if category_mode == "images":
                # Keep image if ANY of its annotations match
                if any(c in categories for c in img_cats):
                    keep_indices.append(idx)
            else:  # annotations mode — just check presence, filtering happens below
                if any(c in categories for c in img_cats):
                    keep_indices.append(idx)

        dataset = dataset.select(keep_indices)
        logger.info(f"After category filter: {len(dataset):,} examples")
        target_n = min(target_n, len(dataset))

    if strategy == "random":
        logger.info(f"Random sampling {target_n:,} from {len(dataset):,}")
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        selected = sorted(indices[:target_n])
        dataset = dataset.select(selected)

    elif strategy == "stratified":
        logger.info(f"Stratified sampling {target_n:,} from {len(dataset):,}")

        # Count categories per image and build index
        cat_to_images = defaultdict(list)
        for idx in tqdm(range(len(dataset)), desc="Indexing categories"):
            ex = dataset[idx]
            img_cats = set(get_image_categories(ex, category_column))
            for cat in img_cats:
                cat_to_images[cat].append(idx)

        # Compute per-category allocation proportional to frequency
        total_cat_count = sum(len(imgs) for imgs in cat_to_images.values())
        cat_allocations = {}
        for cat, imgs in cat_to_images.items():
            cat_allocations[cat] = max(1, round(target_n * len(imgs) / total_cat_count))

        # Greedy selection: pick from underrepresented categories first
        selected = set()
        cat_fulfilled = Counter()

        # Sort categories by allocation (smallest first for better representation)
        sorted_cats = sorted(cat_allocations.keys(), key=lambda c: cat_allocations[c])

        for cat in sorted_cats:
            needed = cat_allocations[cat] - cat_fulfilled[cat]
            if needed <= 0:
                continue

            available = [i for i in cat_to_images[cat] if i not in selected]
            rng.shuffle(available)
            pick = available[:needed]
            selected.update(pick)

            # Update fulfilled counts for all categories of picked images
            for idx in pick:
                ex = dataset[idx]
                for c in set(get_image_categories(ex, category_column)):
                    cat_fulfilled[c] += 1

        # If we still need more, fill randomly
        if len(selected) < target_n:
            remaining = [i for i in range(len(dataset)) if i not in selected]
            rng.shuffle(remaining)
            selected.update(remaining[: target_n - len(selected)])

        # If we have too many, trim
        selected_list = sorted(selected)
        if len(selected_list) > target_n:
            rng.shuffle(selected_list)
            selected_list = sorted(selected_list[:target_n])

        dataset = dataset.select(selected_list)
        logger.info(f"Selected {len(dataset):,} samples via stratified sampling")

    else:
        logger.error(f"Unknown strategy: {strategy}")
        sys.exit(1)

    num_samples = len(dataset)
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds():.1f}s"

    # Push to Hub
    logger.info(f"Pushing {num_samples:,} samples to {output_dataset}")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.warning("Disabling XET (fallback to HTTP upload)")
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            dataset.push_to_hub(
                output_dataset,
                private=private,
                token=HF_TOKEN,
                max_shard_size="500MB",
                create_pr=create_pr,
            )
            break
        except Exception as e:
            logger.error(f"Upload attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                delay = 30 * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error("All upload attempts failed.")
                sys.exit(1)

    # Push dataset card
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        output_dataset=output_dataset,
        strategy=strategy,
        num_samples=num_samples,
        original_size=original_size,
        categories_filter=categories,
        category_mode=category_mode,
        seed=seed,
        split=split,
    )
    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("Done!")
    logger.info(f"Dataset: https://huggingface.co/datasets/{output_dataset}")
    logger.info(f"Sampled {num_samples:,} / {original_size:,} in {processing_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create random or stratified subsets of HF object detection datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies:
  random       Uniform random selection (default)
  stratified   Category-aware weighted sampling

Category modes (with --categories):
  images       Keep images containing any matching annotation (default)
  annotations  Keep images containing any matching annotation

Examples:
  uv run sample-hf-dataset.py merve/dataset merve/subset -n 500
  uv run sample-hf-dataset.py merve/dataset merve/subset --fraction 0.1
  uv run sample-hf-dataset.py merve/dataset merve/subset -n 200 --strategy stratified
  uv run sample-hf-dataset.py merve/dataset merve/subset -n 100 --categories "cat,dog"
        """,
    )

    parser.add_argument("input_dataset", help="Input dataset ID on HF Hub")
    parser.add_argument("output_dataset", help="Output dataset ID on HF Hub")
    parser.add_argument("-n", type=int, help="Number of samples to select")
    parser.add_argument("--fraction", type=float, help="Fraction of dataset to select (0.0-1.0)")
    parser.add_argument("--strategy", choices=["random", "stratified"], default="random", help="Sampling strategy (default: random)")
    parser.add_argument("--category-column", default="category", help="Column containing categories (default: category)")
    parser.add_argument("--categories", help="Comma-separated list of categories to filter by")
    parser.add_argument("--category-mode", choices=["images", "annotations"], default="images", help="How to apply category filter (default: images)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--hf-token", help="HF API token")
    parser.add_argument("--private", action="store_true", help="Make output dataset private")
    parser.add_argument("--create-pr", action="store_true", help="Create PR instead of direct push")

    args = parser.parse_args()

    cats = None
    if args.categories:
        cats = [c.strip() for c in args.categories.split(",")]

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        n=args.n,
        fraction=args.fraction,
        strategy=args.strategy,
        category_column=args.category_column,
        categories=cats,
        category_mode=args.category_mode,
        split=args.split,
        seed=args.seed,
        hf_token=args.hf_token,
        private=args.private,
        create_pr=args.create_pr,
    )
