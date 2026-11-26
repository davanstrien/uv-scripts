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
Validate object detection annotations in a Hugging Face dataset.

Streams a HF dataset and checks for common annotation issues, mirroring
panlabel's validate command. Checks include:

- Duplicate image file names
- Missing or empty bounding boxes
- Bounding box ordering (xmin <= xmax, ymin <= ymax)
- Bounding boxes out of image bounds
- Non-finite coordinates (NaN/Inf)
- Zero-area bounding boxes
- Empty or missing category labels
- Category ID consistency

Supports COCO-style (xywh), XYXY/VOC, YOLO (normalized center xywh),
TFOD (normalized xyxy), and Label Studio (percentage xywh) bbox formats.
Outputs a validation report as text or JSON.

Examples:
  uv run validate-hf-dataset.py merve/test-coco-dataset
  uv run validate-hf-dataset.py merve/test-coco-dataset --bbox-format xyxy --strict
  uv run validate-hf-dataset.py merve/test-coco-dataset --bbox-format tfod --report json
  uv run validate-hf-dataset.py merve/test-coco-dataset --report json --max-samples 1000
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from datasets import load_dataset
from huggingface_hub import login
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BBOX_FORMATS = ["coco_xywh", "xyxy", "voc", "yolo", "tfod", "label_studio"]


def to_xyxy(bbox: list[float], fmt: str, img_w: float = 1.0, img_h: float = 1.0) -> tuple[float, float, float, float]:
    """Convert any bbox format to (xmin, ymin, xmax, ymax) in pixel space."""
    if fmt == "coco_xywh":
        x, y, w, h = bbox
        return (x, y, x + w, y + h)
    elif fmt in ("xyxy", "voc"):
        return tuple(bbox[:4])
    elif fmt == "yolo":
        cx, cy, w, h = bbox
        xmin = (cx - w / 2) * img_w
        ymin = (cy - h / 2) * img_h
        xmax = (cx + w / 2) * img_w
        ymax = (cy + h / 2) * img_h
        return (xmin, ymin, xmax, ymax)
    elif fmt == "tfod":
        xmin_n, ymin_n, xmax_n, ymax_n = bbox
        return (xmin_n * img_w, ymin_n * img_h, xmax_n * img_w, ymax_n * img_h)
    elif fmt == "label_studio":
        x_pct, y_pct, w_pct, h_pct = bbox
        return (
            x_pct / 100.0 * img_w,
            y_pct / 100.0 * img_h,
            (x_pct + w_pct) / 100.0 * img_w,
            (y_pct + h_pct) / 100.0 * img_h,
        )
    else:
        raise ValueError(f"Unknown bbox format: {fmt}")


def is_finite(val: float) -> bool:
    return not (math.isnan(val) or math.isinf(val))


def validate_example(
    example: dict[str, Any],
    idx: int,
    bbox_column: str,
    category_column: str,
    bbox_format: str,
    image_column: str,
    width_column: str | None,
    height_column: str | None,
    tolerance: float = 0.5,
) -> list[dict]:
    """Validate a single example. Returns a list of issue dicts."""
    issues = []

    def add_issue(level: str, code: str, message: str, ann_idx: int | None = None):
        issue = {"level": level, "code": code, "message": message, "example_idx": idx}
        if ann_idx is not None:
            issue["annotation_idx"] = ann_idx
        issues.append(issue)

    # Get objects container — handle nested dict (objects column) or flat lists
    objects = example.get("objects", example)
    bboxes = objects.get(bbox_column, [])
    categories = objects.get(category_column, [])

    if bboxes is None:
        bboxes = []
    if categories is None:
        categories = []

    # Image dimensions (if available)
    img_w = None
    img_h = None
    if width_column and width_column in example:
        img_w = example[width_column]
    elif width_column and objects and width_column in objects:
        img_w = objects[width_column]
    if height_column and height_column in example:
        img_h = example[height_column]
    elif height_column and objects and height_column in objects:
        img_h = objects[height_column]

    if not bboxes and not categories:
        add_issue("warning", "W001", "No annotations found in this example")
        return issues

    if len(bboxes) != len(categories):
        add_issue(
            "error",
            "E001",
            f"Bbox count ({len(bboxes)}) != category count ({len(categories)})",
        )

    for ann_idx, bbox in enumerate(bboxes):
        if bbox is None or len(bbox) < 4:
            add_issue("error", "E002", f"Invalid bbox (need 4 values, got {bbox})", ann_idx)
            continue

        # Check finite
        if not all(is_finite(v) for v in bbox[:4]):
            add_issue("error", "E003", f"Non-finite bbox coordinates: {bbox}", ann_idx)
            continue

        # Convert to xyxy
        w_for_conv = img_w if img_w else 1.0
        h_for_conv = img_h if img_h else 1.0
        xmin, ymin, xmax, ymax = to_xyxy(bbox[:4], bbox_format, w_for_conv, h_for_conv)

        # Check ordering
        if xmin > xmax:
            add_issue("error", "E004", f"xmin ({xmin}) > xmax ({xmax})", ann_idx)
        if ymin > ymax:
            add_issue("error", "E005", f"ymin ({ymin}) > ymax ({ymax})", ann_idx)

        # Check zero area
        area = (xmax - xmin) * (ymax - ymin)
        if area <= 0:
            add_issue("warning", "W002", f"Zero or negative area bbox: {bbox}", ann_idx)

        # Check bounds (only if image dimensions available)
        if img_w is not None and img_h is not None:
            if xmin < -tolerance or ymin < -tolerance:
                add_issue(
                    "warning",
                    "W003",
                    f"Bbox extends before image origin: ({xmin}, {ymin})",
                    ann_idx,
                )
            if xmax > img_w + tolerance or ymax > img_h + tolerance:
                add_issue(
                    "warning",
                    "W004",
                    f"Bbox extends beyond image bounds: ({xmax}, {ymax}) > ({img_w}, {img_h})",
                    ann_idx,
                )

    # Check categories
    for ann_idx, cat in enumerate(categories):
        if cat is None or (isinstance(cat, str) and cat.strip() == ""):
            add_issue("warning", "W005", "Empty category label", ann_idx)

    return issues


def main(
    input_dataset: str,
    bbox_column: str = "bbox",
    category_column: str = "category",
    bbox_format: str = "coco_xywh",
    image_column: str = "image",
    width_column: str | None = "width",
    height_column: str | None = "height",
    split: str = "train",
    max_samples: int | None = None,
    streaming: bool = False,
    strict: bool = False,
    report_format: str = "text",
    tolerance: float = 0.5,
    hf_token: str | None = None,
    output_dataset: str | None = None,
    private: bool = False,
):
    """Validate an object detection dataset from HF Hub."""

    start_time = datetime.now()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    logger.info(f"Loading dataset: {input_dataset} (split={split}, streaming={streaming})")
    dataset = load_dataset(input_dataset, split=split, streaming=streaming)

    all_issues = []
    file_names = []
    total_annotations = 0
    total_examples = 0
    category_counts = Counter()
    error_count = 0
    warning_count = 0

    iterable = dataset
    if max_samples:
        if streaming:
            iterable = dataset.take(max_samples)
        else:
            iterable = dataset.select(range(min(max_samples, len(dataset))))

    for idx, example in enumerate(tqdm(iterable, desc="Validating", total=max_samples)):
        total_examples += 1

        issues = validate_example(
            example=example,
            idx=idx,
            bbox_column=bbox_column,
            category_column=category_column,
            bbox_format=bbox_format,
            image_column=image_column,
            width_column=width_column,
            height_column=height_column,
            tolerance=tolerance,
        )
        all_issues.extend(issues)

        # Count stats
        objects = example.get("objects", example)
        bboxes = objects.get(bbox_column, []) or []
        categories = objects.get(category_column, []) or []
        total_annotations += len(bboxes)
        for cat in categories:
            if cat is not None:
                category_counts[str(cat)] += 1

        # Track file names for duplicate check
        fname = example.get("file_name") or example.get("image_id") or str(idx)
        file_names.append(fname)

    # Check duplicate file names
    fname_counts = Counter(file_names)
    duplicates = {k: v for k, v in fname_counts.items() if v > 1}
    for fname, count in duplicates.items():
        all_issues.append({
            "level": "warning",
            "code": "W006",
            "message": f"Duplicate file name '{fname}' appears {count} times",
            "example_idx": None,
        })

    for issue in all_issues:
        if issue["level"] == "error":
            error_count += 1
        else:
            warning_count += 1

    processing_time = datetime.now() - start_time

    # Build report
    report = {
        "dataset": input_dataset,
        "split": split,
        "total_examples": total_examples,
        "total_annotations": total_annotations,
        "unique_categories": len(category_counts),
        "errors": error_count,
        "warnings": warning_count,
        "duplicate_filenames": len(duplicates),
        "issues": all_issues,
        "processing_time_seconds": processing_time.total_seconds(),
        "timestamp": datetime.now().isoformat(),
        "valid": error_count == 0 and (not strict or warning_count == 0),
    }

    if report_format == "json":
        print(json.dumps(report, indent=2))
    else:
        print("\n" + "=" * 60)
        print(f"Validation Report: {input_dataset}")
        print("=" * 60)
        print(f"  Examples:      {total_examples:,}")
        print(f"  Annotations:   {total_annotations:,}")
        print(f"  Categories:    {len(category_counts):,}")
        print(f"  Errors:        {error_count}")
        print(f"  Warnings:      {warning_count}")
        if duplicates:
            print(f"  Duplicate IDs: {len(duplicates)}")
        print(f"  Processing:    {processing_time.total_seconds():.1f}s")
        print()

        if all_issues:
            print("Issues:")
            # Group by code
            by_code = defaultdict(list)
            for issue in all_issues:
                by_code[issue["code"]].append(issue)

            for code in sorted(by_code.keys()):
                code_issues = by_code[code]
                level = code_issues[0]["level"].upper()
                sample = code_issues[0]["message"]
                print(f"  [{level}] {code}: {sample}")
                if len(code_issues) > 1:
                    print(f"         ... and {len(code_issues) - 1} more")
            print()

        status = "VALID" if report["valid"] else "INVALID"
        mode = " (strict)" if strict else ""
        print(f"Result: {status}{mode}")
        print("=" * 60)

    # Optionally push validation report as a dataset
    if output_dataset:
        from datasets import Dataset as HFDataset

        report_ds = HFDataset.from_dict({
            "report": [json.dumps(report)],
            "dataset": [input_dataset],
            "valid": [report["valid"]],
            "errors": [error_count],
            "warnings": [warning_count],
            "total_examples": [total_examples],
            "total_annotations": [total_annotations],
            "timestamp": [datetime.now().isoformat()],
        })

        logger.info(f"Pushing validation report to {output_dataset}")
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    os.environ["HF_HUB_DISABLE_XET"] = "1"
                report_ds.push_to_hub(
                    output_dataset,
                    private=private,
                    token=HF_TOKEN,
                )
                break
            except Exception as e:
                logger.error(f"Upload attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(30 * (2 ** (attempt - 1)))
                else:
                    logger.error("All upload attempts failed.")
                    sys.exit(1)

        logger.info(f"Report pushed to: https://huggingface.co/datasets/{output_dataset}")

    if not report["valid"]:
        sys.exit(1 if strict else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate object detection annotations in a HF dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Bbox formats:
  coco_xywh     [x, y, width, height] in pixels (default)
  xyxy          [xmin, ymin, xmax, ymax] in pixels
  voc           [xmin, ymin, xmax, ymax] in pixels (alias for xyxy)
  yolo          [cx, cy, w, h] normalized 0-1
  tfod          [xmin, ymin, xmax, ymax] normalized 0-1
  label_studio  [x, y, w, h] percentage 0-100

Issue codes:
  E001  Bbox/category count mismatch
  E002  Invalid bbox (missing values)
  E003  Non-finite coordinates (NaN/Inf)
  E004  xmin > xmax
  E005  ymin > ymax
  W001  No annotations in example
  W002  Zero or negative area
  W003  Bbox before image origin
  W004  Bbox beyond image bounds
  W005  Empty category label
  W006  Duplicate file name

Examples:
  uv run validate-hf-dataset.py merve/coco-dataset
  uv run validate-hf-dataset.py merve/coco-dataset --bbox-format xyxy --strict
  uv run validate-hf-dataset.py merve/coco-dataset --streaming --max-samples 500
        """,
    )

    parser.add_argument("input_dataset", help="Input dataset ID on HF Hub")
    parser.add_argument("--bbox-column", default="bbox", help="Column containing bboxes (default: bbox)")
    parser.add_argument("--category-column", default="category", help="Column containing categories (default: category)")
    parser.add_argument(
        "--bbox-format",
        choices=BBOX_FORMATS,
        default="coco_xywh",
        help="Bounding box format (default: coco_xywh)",
    )
    parser.add_argument("--image-column", default="image", help="Column containing images (default: image)")
    parser.add_argument("--width-column", default="width", help="Column for image width (default: width)")
    parser.add_argument("--height-column", default="height", help="Column for image height (default: height)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--max-samples", type=int, help="Max samples to validate")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (no full download)")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument("--report", choices=["text", "json"], default="text", help="Report format (default: text)")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Out-of-bounds tolerance in pixels (default: 0.5)")
    parser.add_argument("--hf-token", help="HF API token")
    parser.add_argument("--output-dataset", help="Push validation report to this HF dataset")
    parser.add_argument("--private", action="store_true", help="Make output dataset private")

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        bbox_column=args.bbox_column,
        category_column=args.category_column,
        bbox_format=args.bbox_format,
        image_column=args.image_column,
        width_column=args.width_column,
        height_column=args.height_column,
        split=args.split,
        max_samples=args.max_samples,
        streaming=args.streaming,
        strict=args.strict,
        report_format=args.report,
        tolerance=args.tolerance,
        hf_token=args.hf_token,
        output_dataset=args.output_dataset,
        private=args.private,
    )
