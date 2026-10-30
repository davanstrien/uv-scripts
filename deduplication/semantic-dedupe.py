# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "semhash",
#     "datasets",
#     "huggingface-hub",
#     
# ]
# ///

"""
Semantic deduplication for Hugging Face datasets using SemHash.

This script removes duplicate or near-duplicate text samples from datasets based on
semantic similarity, helping to clean training data and prevent train/test leakage.

SemHash is CPU-optimized and uses Model2Vec embeddings that are 500x faster on CPU
than traditional transformers. No GPU required!

Example usage:
    # Basic deduplication
    uv run semantic-dedupe.py username/dataset text username/dataset-deduped

    # With custom threshold and max samples for testing
    uv run semantic-dedupe.py username/dataset text username/dataset-deduped \\
        --threshold 0.85 --max-samples 1000

    # Using HF Jobs (CPU is sufficient)
    hf jobs uv run --flavor cpu-xl \\
        -e HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; print(get_token())") \\
        https://huggingface.co/datasets/uv-scripts/deduplication/raw/main/semantic-dedupe.py \\
        username/dataset text username/dataset-deduped
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

from datasets import Dataset, load_dataset
from huggingface_hub import DatasetCard, login
from semhash import SemHash




def parse_args():
    parser = argparse.ArgumentParser(
        description="Deduplicate a dataset using semantic similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run semantic-dedupe.py imdb text imdb-deduped

  # With options
  uv run semantic-dedupe.py squad question squad-deduped --threshold 0.85 --method duplicates

  # Test with small sample
  uv run semantic-dedupe.py large-dataset text test-dedup --max-samples 100
        """,
    )
    
    parser.add_argument("dataset", help="Input dataset ID (e.g., 'imdb' or 'username/dataset')")
    parser.add_argument("column", help="Text column to deduplicate on")
    parser.add_argument("output_repo", help="Output dataset repository name")
    
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--method",
        choices=["duplicates", "outliers", "representatives"],
        default="duplicates",
        help="Deduplication method (default: duplicates)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for duplicates (default: 0.9)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for processing (default: 64)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private dataset repository",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face API token (defaults to HF_TOKEN env var)",
    )
    
    return parser.parse_args()


def create_dataset_card(
    original_dataset: str,
    column: str,
    method: str,
    threshold: float,
    original_size: int,
    deduped_size: int,
) -> str:
    """Create a dataset card with deduplication information."""
    reduction_pct = ((original_size - deduped_size) / original_size) * 100
    
    return f"""---
viewer: false
tags:
- deduplication
- semhash
- uv-script
---

# Deduplicated {original_dataset}

This dataset is a deduplicated version of [{original_dataset}](https://huggingface.co/datasets/{original_dataset}).

## Deduplication Details

- **Method**: {method}
- **Column**: `{column}`
- **Threshold**: {threshold}
- **Original size**: {original_size:,} samples
- **Deduplicated size**: {deduped_size:,} samples
- **Reduction**: {reduction_pct:.1f}%
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

## Method Description

{get_method_description(method)}

## How to Use

```python
from datasets import load_dataset

# Load the deduplicated dataset
dataset = load_dataset("{os.environ.get('HF_USERNAME', 'username')}/{os.path.basename(original_dataset)}-deduped")
```

## Reproduce

This dataset was created using the [uv-scripts/deduplication](https://huggingface.co/datasets/uv-scripts/deduplication) tool:

```bash
uv run https://huggingface.co/datasets/uv-scripts/deduplication/raw/main/semantic-dedupe.py \\
    {original_dataset} {column} {os.path.basename(original_dataset)}-deduped \\
    --method {method} --threshold {threshold}
```

Generated with 🤖 UV Scripts
"""


def get_method_description(method: str) -> str:
    """Get description for deduplication method."""
    descriptions = {
        "duplicates": "Removes semantic duplicates by finding samples with high similarity scores above the threshold.",
        "outliers": "Removes outlier samples that have low similarity to other samples in the dataset.",
        "representatives": "Keeps only representative samples, removing both duplicates and outliers.",
    }
    return descriptions.get(method, "Unknown method")


def main():
    args = parse_args()
    
    # Authenticate
    if args.hf_token:
        login(args.hf_token)
    else:
        print("Warning: No HF token provided. Using cached credentials or anonymous mode.")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)
    
    # Apply max samples if specified
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} samples for testing")
    
    original_size = len(dataset)
    
    # Check if column exists
    if args.column not in dataset.column_names:
        print(f"Error: Column '{args.column}' not found in dataset")
        print(f"Available columns: {', '.join(dataset.column_names)}")
        sys.exit(1)
    
    # Convert dataset to records (preserves all columns)
    print("Converting dataset to records...")
    records = [dict(row) for row in dataset]
    
    # Initialize SemHash
    print("Initializing SemHash (CPU-optimized)...")
    semhash = SemHash.from_records(records=records, columns=[args.column])
    
    # Perform deduplication
    print(f"Performing {args.method} deduplication on '{args.column}' column...")
    
    if args.method == "duplicates":
        result = semhash.self_deduplicate(threshold=args.threshold)
    elif args.method == "outliers":
        result = semhash.self_filter_outliers()
    elif args.method == "representatives":
        result = semhash.self_find_representative()
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Get deduplicated records (all columns preserved)
    deduplicated_records = result.selected
    
    # Convert back to HF Dataset
    result_dataset = Dataset.from_list(deduplicated_records)
    deduped_size = len(result_dataset)
    
    # Print statistics
    print(f"\nDeduplication complete!")
    print(f"Original size: {original_size:,}")
    print(f"Deduplicated size: {deduped_size:,}")
    print(f"Removed: {original_size - deduped_size:,} ({((original_size - deduped_size) / original_size) * 100:.1f}%)")
    print("\nNote: SemHash processes ~20,000 sentences/second on CPU")
    
    # Create dataset card
    card = create_dataset_card(
        args.dataset,
        args.column,
        args.method,
        args.threshold,
        original_size,
        deduped_size,
    )
    
    # Push to hub
    print(f"\nPushing to hub: {args.output_repo}")
    result_dataset.push_to_hub(
        args.output_repo,
        private=args.private,
        commit_message=f"Deduplicated using {args.method} method",
    )
    
    # Create and push dataset card
    dataset_card = DatasetCard(card)
    dataset_card.push_to_hub(args.output_repo)
    
    print(f"✅ Dataset successfully pushed to: https://huggingface.co/datasets/{args.output_repo}")


if __name__ == "__main__":
    main()