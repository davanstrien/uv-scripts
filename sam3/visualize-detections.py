#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "matplotlib",
#     "pillow",
# ]
# ///

"""
Visualize object detection predictions from a HuggingFace dataset.

This script loads a dataset with object detection predictions and visualizes
the bounding boxes on sample images.

Examples:
    # Visualize the first sample with detections
    uv run visualize-detections.py my-username/detected-objects --first-with-detections

    # Visualize a specific sample
    uv run visualize-detections.py my-username/detected-objects --index 0

    # Visualize multiple random samples
    uv run visualize-detections.py my-username/detected-objects --num-samples 5

    # Save visualizations to files instead of displaying
    uv run visualize-detections.py my-username/detected-objects --num-samples 3 --output-dir ./visualizations

    # Visualize specific split
    uv run visualize-detections.py my-username/detected-objects --split train --num-samples 5
"""

import argparse
import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from datasets import load_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize object detection predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "dataset_id", help="HuggingFace dataset ID (e.g., 'username/dataset')"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Index of sample to visualize (default: random)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to visualize (default: 1)",
    )
    parser.add_argument(
        "--first-with-detections",
        action="store_true",
        help="Find and visualize the first sample with detections",
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split to use (default: 'train')"
    )
    parser.add_argument(
        "--image-column",
        default="image",
        help="Name of the image column (default: 'image')",
    )
    parser.add_argument(
        "--objects-column",
        default="objects",
        help="Name of the objects column (default: 'objects')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: show interactively)",
    )
    parser.add_argument(
        "--figsize-width",
        type=int,
        default=15,
        help="Figure width in inches (default: 15)",
    )
    parser.add_argument(
        "--figsize-height",
        type=int,
        default=20,
        help="Figure height in inches (default: 20)",
    )
    parser.add_argument(
        "--bbox-color",
        default="red",
        help="Color for bounding boxes (default: 'red')",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        default=True,
        help="Show confidence scores on bounding boxes",
    )

    return parser.parse_args()


def visualize_sample(
    sample,
    image_column="image",
    objects_column="objects",
    figsize=(15, 20),
    bbox_color="red",
    show_scores=True,
    title=None,
):
    """Visualize a single sample with bounding boxes."""
    image = sample[image_column]
    objects = sample[objects_column]

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image, cmap="gray" if image.mode == "L" else None)

    # Draw bounding boxes
    num_detections = len(objects["bbox"])
    for i in range(num_detections):
        bbox = objects["bbox"][i]
        score = objects["score"][i]

        x, y, w, h = bbox
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor=bbox_color, facecolor="none"
        )
        ax.add_patch(rect)

        if show_scores:
            label = f"{score:.2f}"
            ax.text(
                x,
                y - 5,
                label,
                color=bbox_color,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7),
            )

    # Set title
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    else:
        ax.set_title(f"Detections: {num_detections}", fontsize=14, pad=20)

    ax.axis("off")
    plt.tight_layout()

    return fig, ax


def main():
    args = parse_args()

    # Load dataset
    print(f"📂 Loading dataset: {args.dataset_id} (split: {args.split})")
    dataset = load_dataset(args.dataset_id, split=args.split)
    print(f"✅ Loaded {len(dataset)} samples")

    # Determine indices to visualize
    if args.index is not None:
        indices = [args.index]
    elif args.first_with_detections:
        # Find first sample with detections
        print("🔍 Finding first sample with detections...")
        first_idx = None
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if len(sample[args.objects_column]["bbox"]) > 0:
                first_idx = idx
                break

        if first_idx is None:
            print("❌ No samples with detections found in dataset")
            return

        print(f"✅ Found first sample with detections at index {first_idx}")
        indices = [first_idx]
    else:
        # Select random samples
        indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    # Create output directory if saving
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving visualizations to: {output_path}")

    # Visualize samples
    figsize = (args.figsize_width, args.figsize_height)

    for idx in indices:
        sample = dataset[idx]
        num_detections = len(sample[args.objects_column]["bbox"])

        print(f"\n🖼️  Sample {idx}: {num_detections} detections")

        # Create visualization
        title = f"Sample {idx} - {num_detections} detections"
        fig, ax = visualize_sample(
            sample,
            image_column=args.image_column,
            objects_column=args.objects_column,
            figsize=figsize,
            bbox_color=args.bbox_color,
            show_scores=args.show_scores,
            title=title,
        )

        # Save or show
        if args.output_dir:
            output_file = output_path / f"sample_{idx}.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"   Saved: {output_file}")
            plt.close(fig)
        else:
            plt.show()

    if args.output_dir:
        print(f"\n✅ Saved {len(indices)} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
