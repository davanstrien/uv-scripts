# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=3.1.0",
#     "huggingface-hub",
#     "tqdm",
#     "toolz",
#     "Pillow", 
# ]
# ///

"""
Convert bounding box formats in a Hugging Face object detection dataset.

Mirrors panlabel's convert command. Converts between:
  - COCO xywh: [x, y, width, height] in pixels
  - XYXY: [xmin, ymin, xmax, ymax] in pixels
  - VOC: [xmin, ymin, xmax, ymax] in pixels (alias for xyxy)
  - YOLO: [center_x, center_y, width, height] normalized 0-1
  - TFOD: [xmin, ymin, xmax, ymax] normalized 0-1
  - Label Studio: [x, y, width, height] percentage 0-100

Reads from HF Hub, converts bboxes in-place, and pushes the result to a new
(or the same) dataset repo on HF Hub.

Examples:
  uv run convert-hf-dataset.py merve/coco-dataset merve/coco-xyxy --from coco_xywh --to xyxy
  uv run convert-hf-dataset.py merve/yolo-dataset merve/yolo-coco --from yolo --to coco_xywh
  uv run convert-hf-dataset.py merve/dataset merve/converted --from xyxy --to yolo --max-samples 100
  uv run convert-hf-dataset.py merve/dataset merve/converted --from tfod --to coco_xywh
  uv run convert-hf-dataset.py merve/dataset merve/converted --from label_studio --to xyxy
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

from datasets import load_dataset
from huggingface_hub import DatasetCard, login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BBOX_FORMATS = ["coco_xywh", "xyxy", "voc", "yolo", "tfod", "label_studio"]


def convert_bbox(
    bbox: list[float],
    from_fmt: str,
    to_fmt: str,
    img_w: float = 1.0,
    img_h: float = 1.0,
) -> list[float]:
    """Convert a single bbox between formats via XYXY pixel-space intermediate."""
    # Step 1: to XYXY pixel space
    if from_fmt == "coco_xywh":
        x, y, w, h = bbox[:4]
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
    elif from_fmt in ("xyxy", "voc"):
        xmin, ymin, xmax, ymax = bbox[:4]
    elif from_fmt == "yolo":
        cx, cy, w, h = bbox[:4]
        xmin = (cx - w / 2) * img_w
        ymin = (cy - h / 2) * img_h
        xmax = (cx + w / 2) * img_w
        ymax = (cy + h / 2) * img_h
    elif from_fmt == "tfod":
        xmin_n, ymin_n, xmax_n, ymax_n = bbox[:4]
        xmin = xmin_n * img_w
        ymin = ymin_n * img_h
        xmax = xmax_n * img_w
        ymax = ymax_n * img_h
    elif from_fmt == "label_studio":
        x_pct, y_pct, w_pct, h_pct = bbox[:4]
        xmin = x_pct / 100.0 * img_w
        ymin = y_pct / 100.0 * img_h
        xmax = (x_pct + w_pct) / 100.0 * img_w
        ymax = (y_pct + h_pct) / 100.0 * img_h
    else:
        raise ValueError(f"Unknown source format: {from_fmt}")

    # Step 2: from XYXY pixel space to target
    if to_fmt in ("xyxy", "voc"):
        return [xmin, ymin, xmax, ymax]
    elif to_fmt == "coco_xywh":
        return [xmin, ymin, xmax - xmin, ymax - ymin]
    elif to_fmt == "yolo":
        if img_w <= 0 or img_h <= 0:
            raise ValueError("YOLO format requires positive image dimensions")
        w = xmax - xmin
        h = ymax - ymin
        cx = (xmin + w / 2) / img_w
        cy = (ymin + h / 2) / img_h
        return [cx, cy, w / img_w, h / img_h]
    elif to_fmt == "tfod":
        if img_w <= 0 or img_h <= 0:
            raise ValueError("TFOD format requires positive image dimensions")
        return [xmin / img_w, ymin / img_h, xmax / img_w, ymax / img_h]
    elif to_fmt == "label_studio":
        if img_w <= 0 or img_h <= 0:
            raise ValueError("Label Studio format requires positive image dimensions")
        x_pct = xmin / img_w * 100.0
        y_pct = ymin / img_h * 100.0
        w_pct = (xmax - xmin) / img_w * 100.0
        h_pct = (ymax - ymin) / img_h * 100.0
        return [x_pct, y_pct, w_pct, h_pct]
    else:
        raise ValueError(f"Unknown target format: {to_fmt}")


def convert_example(
    example: dict[str, Any],
    bbox_column: str,
    from_fmt: str,
    to_fmt: str,
    width_column: str | None,
    height_column: str | None,
) -> dict[str, Any]:
    """Convert bboxes in a single example."""
    objects = example.get("objects")
    is_nested = objects is not None and isinstance(objects, dict)

    if is_nested:
        bboxes = objects.get(bbox_column, []) or []
    else:
        bboxes = example.get(bbox_column, []) or []

    img_w = 1.0
    img_h = 1.0
    if width_column:
        img_w = float(example.get(width_column, 1.0) or 1.0)
    if height_column:
        img_h = float(example.get(height_column, 1.0) or 1.0)

    converted = []
    for bbox in bboxes:
        if bbox is None or len(bbox) < 4:
            converted.append(bbox)
            continue
        converted.append(convert_bbox(bbox, from_fmt, to_fmt, img_w, img_h))

    if is_nested:
        new_objects = dict(objects)
        new_objects[bbox_column] = converted
        return {"objects": new_objects}
    else:
        return {bbox_column: converted}


def create_dataset_card(
    source_dataset: str,
    output_dataset: str,
    from_fmt: str,
    to_fmt: str,
    num_samples: int,
    processing_time: str,
    split: str,
) -> str:
    return f"""---
tags:
- object-detection
- bbox-conversion
- panlabel
- uv-script
- generated
---

# Bbox Format Conversion: {from_fmt} -> {to_fmt}

Bounding boxes converted from `{from_fmt}` to `{to_fmt}` format.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Conversion**: `{from_fmt}` -> `{to_fmt}`
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}
- **Split**: `{split}`

## Bbox Formats

| Format | Description |
|--------|-------------|
| `coco_xywh` | `[x, y, width, height]` in pixels |
| `xyxy` | `[xmin, ymin, xmax, ymax]` in pixels |
| `voc` | `[xmin, ymin, xmax, ymax]` in pixels (alias for xyxy) |
| `yolo` | `[center_x, center_y, width, height]` normalized 0-1 |
| `tfod` | `[xmin, ymin, xmax, ymax]` normalized 0-1 |
| `label_studio` | `[x, y, width, height]` percentage 0-100 |

## Reproduction

```bash
uv run convert-hf-dataset.py {source_dataset} {output_dataset} --from {from_fmt} --to {to_fmt}
```

Generated with panlabel-hf (convert-hf-dataset.py)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    from_fmt: str,
    to_fmt: str,
    bbox_column: str = "bbox",
    width_column: str | None = "width",
    height_column: str | None = "height",
    split: str = "train",
    max_samples: int | None = None,
    batch_size: int = 1000,
    hf_token: str | None = None,
    private: bool = False,
    create_pr: bool = False,
    shuffle: bool = False,
    seed: int = 42,
):
    """Convert bbox format in a HF dataset and push to Hub."""

    if from_fmt == to_fmt:
        logger.error(f"Source and target formats are the same: {from_fmt}")
        sys.exit(1)

    start_time = datetime.now()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    logger.info(f"Loading dataset: {input_dataset} (split={split})")
    dataset = load_dataset(input_dataset, split=split)

    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    total_samples = len(dataset)
    logger.info(f"Converting {total_samples:,} samples: {from_fmt} -> {to_fmt}")

    # Convert using map
    dataset = dataset.map(
        lambda example: convert_example(
            example, bbox_column, from_fmt, to_fmt, width_column, height_column
        ),
        desc=f"Converting {from_fmt} -> {to_fmt}",
    )

    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds():.1f}s"

    # Add conversion metadata
    conversion_info = json.dumps({
        "source_format": from_fmt,
        "target_format": to_fmt,
        "source_dataset": input_dataset,
        "timestamp": datetime.now().isoformat(),
        "script": "convert-hf-dataset.py",
    })

    if "conversion_info" not in dataset.column_names:
        dataset = dataset.add_column(
            "conversion_info", [conversion_info] * len(dataset)
        )

    # Push to Hub
    logger.info(f"Pushing to {output_dataset}")
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
        from_fmt=from_fmt,
        to_fmt=to_fmt,
        num_samples=total_samples,
        processing_time=processing_time_str,
        split=split,
    )
    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("Done!")
    logger.info(f"Dataset: https://huggingface.co/datasets/{output_dataset}")
    logger.info(f"Converted {total_samples:,} samples in {processing_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert bbox formats in a HF object detection dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Bbox formats:
  coco_xywh     [x, y, width, height] in pixels
  xyxy          [xmin, ymin, xmax, ymax] in pixels
  voc           [xmin, ymin, xmax, ymax] in pixels (alias for xyxy)
  yolo          [cx, cy, w, h] normalized 0-1
  tfod          [xmin, ymin, xmax, ymax] normalized 0-1
  label_studio  [x, y, w, h] percentage 0-100

Examples:
  uv run convert-hf-dataset.py merve/coco merve/coco-xyxy --from coco_xywh --to xyxy
  uv run convert-hf-dataset.py merve/yolo merve/yolo-coco --from yolo --to coco_xywh
  uv run convert-hf-dataset.py merve/tfod merve/tfod-coco --from tfod --to coco_xywh
        """,
    )

    parser.add_argument("input_dataset", help="Input dataset ID on HF Hub")
    parser.add_argument("output_dataset", help="Output dataset ID on HF Hub")
    parser.add_argument("--from", dest="from_fmt", required=True, choices=BBOX_FORMATS, help="Source bbox format")
    parser.add_argument("--to", dest="to_fmt", required=True, choices=BBOX_FORMATS, help="Target bbox format")
    parser.add_argument("--bbox-column", default="bbox", help="Column containing bboxes (default: bbox)")
    parser.add_argument("--width-column", default="width", help="Column for image width (default: width)")
    parser.add_argument("--height-column", default="height", help="Column for image height (default: height)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--max-samples", type=int, help="Max samples to process")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for map (default: 1000)")
    parser.add_argument("--hf-token", help="HF API token")
    parser.add_argument("--private", action="store_true", help="Make output dataset private")
    parser.add_argument("--create-pr", action="store_true", help="Create PR instead of direct push")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before processing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        from_fmt=args.from_fmt,
        to_fmt=args.to_fmt,
        bbox_column=args.bbox_column,
        width_column=args.width_column,
        height_column=args.height_column,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        hf_token=args.hf_token,
        private=args.private,
        create_pr=args.create_pr,
        shuffle=args.shuffle,
        seed=args.seed,
    )
