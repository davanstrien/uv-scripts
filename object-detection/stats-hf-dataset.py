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
Generate rich statistics for object detection datasets on Hugging Face Hub.

Mirrors panlabel's stats command. Computes:

- Summary counts (images, annotations, categories)
- Label distribution histogram (top-N)
- Bounding box statistics (area, aspect ratio, out-of-bounds)
- Annotation density per image
- Per-category bbox statistics
- Category co-occurrence pairs
- Image resolution distribution

Supports COCO-style (xywh), XYXY/VOC, YOLO (normalized center xywh),
TFOD (normalized xyxy), and Label Studio (percentage xywh) bbox formats.
Supports streaming for large datasets. Outputs text or JSON.

Examples:
  uv run stats-hf-dataset.py merve/test-coco-dataset
  uv run stats-hf-dataset.py merve/test-coco-dataset --top 20 --report json
  uv run stats-hf-dataset.py merve/test-coco-dataset --bbox-format tfod
  uv run stats-hf-dataset.py merve/test-coco-dataset --streaming --max-samples 5000
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
        return (cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h
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


def percentile(sorted_vals: list[float], p: float) -> float:
    """Compute percentile from sorted values."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def main(
    input_dataset: str,
    bbox_column: str = "bbox",
    category_column: str = "category",
    bbox_format: str = "coco_xywh",
    width_column: str | None = "width",
    height_column: str | None = "height",
    split: str = "train",
    max_samples: int | None = None,
    streaming: bool = False,
    top: int = 10,
    report_format: str = "text",
    tolerance: float = 0.5,
    hf_token: str | None = None,
    output_dataset: str | None = None,
    private: bool = False,
):
    """Compute statistics for an object detection dataset."""

    start_time = datetime.now()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    logger.info(f"Loading dataset: {input_dataset} (split={split}, streaming={streaming})")
    dataset = load_dataset(input_dataset, split=split, streaming=streaming)

    # Accumulators
    total_images = 0
    total_annotations = 0
    category_counts = Counter()
    annotations_per_image = []
    areas = []
    aspect_ratios = []
    widths = []
    heights = []
    out_of_bounds_count = 0
    zero_area_count = 0
    per_category_areas = defaultdict(list)
    co_occurrence_pairs = Counter()
    images_without_annotations = 0

    iterable = dataset
    if max_samples:
        if streaming:
            iterable = dataset.take(max_samples)
        else:
            iterable = dataset.select(range(min(max_samples, len(dataset))))

    for idx, example in enumerate(tqdm(iterable, desc="Computing stats", total=max_samples)):
        total_images += 1

        objects = example.get("objects", example)
        bboxes = objects.get(bbox_column, []) or []
        categories = objects.get(category_column, []) or []

        # Image dimensions
        img_w = None
        img_h = None
        if width_column:
            img_w = example.get(width_column) or (objects.get(width_column) if isinstance(objects, dict) else None)
        if height_column:
            img_h = example.get(height_column) or (objects.get(height_column) if isinstance(objects, dict) else None)

        if img_w is not None and img_h is not None:
            widths.append(img_w)
            heights.append(img_h)

        num_anns = len(bboxes)
        annotations_per_image.append(num_anns)
        total_annotations += num_anns

        if num_anns == 0:
            images_without_annotations += 1
            continue

        # Track categories and co-occurrences
        image_cats = set()
        for ann_idx, bbox in enumerate(bboxes):
            cat = categories[ann_idx] if ann_idx < len(categories) else None
            cat_str = str(cat) if cat is not None else "<unknown>"
            category_counts[cat_str] += 1
            image_cats.add(cat_str)

            if bbox is None or len(bbox) < 4:
                continue
            if not all(math.isfinite(v) for v in bbox[:4]):
                continue

            w_for_conv = img_w if img_w else 1.0
            h_for_conv = img_h if img_h else 1.0
            xmin, ymin, xmax, ymax = to_xyxy(bbox[:4], bbox_format, w_for_conv, h_for_conv)

            bw = xmax - xmin
            bh = ymax - ymin
            area = bw * bh

            if area <= 0:
                zero_area_count += 1
            else:
                areas.append(area)
                per_category_areas[cat_str].append(area)

            if bh > 0:
                aspect_ratios.append(bw / bh)

            # Out of bounds check
            if img_w is not None and img_h is not None:
                if xmin < -tolerance or ymin < -tolerance or xmax > img_w + tolerance or ymax > img_h + tolerance:
                    out_of_bounds_count += 1

        # Co-occurrence pairs
        sorted_cats = sorted(image_cats)
        for i in range(len(sorted_cats)):
            for j in range(i + 1, len(sorted_cats)):
                co_occurrence_pairs[(sorted_cats[i], sorted_cats[j])] += 1

    processing_time = datetime.now() - start_time

    # Compute distribution stats
    areas.sort()
    aspect_ratios.sort()
    annotations_per_image.sort()

    def dist_stats(vals: list[float]) -> dict:
        if not vals:
            return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "p25": 0, "p75": 0}
        return {
            "count": len(vals),
            "min": round(vals[0], 2),
            "max": round(vals[-1], 2),
            "mean": round(sum(vals) / len(vals), 2),
            "median": round(percentile(vals, 50), 2),
            "p25": round(percentile(vals, 25), 2),
            "p75": round(percentile(vals, 75), 2),
        }

    # Top-N categories
    top_categories = category_counts.most_common(top)

    # Top co-occurrence pairs
    top_cooccurrences = co_occurrence_pairs.most_common(top)

    # Per-category bbox area stats
    per_cat_stats = {}
    for cat, cat_areas in sorted(per_category_areas.items(), key=lambda x: -len(x[1])):
        cat_areas.sort()
        per_cat_stats[cat] = dist_stats(cat_areas)

    report = {
        "dataset": input_dataset,
        "split": split,
        "summary": {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "unique_categories": len(category_counts),
            "images_without_annotations": images_without_annotations,
            "out_of_bounds_bboxes": out_of_bounds_count,
            "zero_area_bboxes": zero_area_count,
        },
        "label_distribution": {cat: count for cat, count in top_categories},
        "annotation_density": dist_stats([float(x) for x in annotations_per_image]),
        "bbox_area": dist_stats(areas),
        "bbox_aspect_ratio": dist_stats(aspect_ratios),
        "image_resolution": {
            "width": dist_stats([float(w) for w in sorted(widths)]) if widths else {},
            "height": dist_stats([float(h) for h in sorted(heights)]) if heights else {},
        },
        "per_category_area": {cat: per_cat_stats[cat] for cat in list(per_cat_stats)[:top]},
        "co_occurrence_pairs": [
            {"pair": list(pair), "count": count} for pair, count in top_cooccurrences
        ],
        "processing_time_seconds": processing_time.total_seconds(),
        "timestamp": datetime.now().isoformat(),
    }

    if report_format == "json":
        print(json.dumps(report, indent=2))
    else:
        print("\n" + "=" * 60)
        print(f"Dataset Statistics: {input_dataset}")
        print("=" * 60)

        s = report["summary"]
        print(f"\n  Images:           {s['total_images']:,}")
        print(f"  Annotations:      {s['total_annotations']:,}")
        print(f"  Categories:       {s['unique_categories']:,}")
        print(f"  Empty images:     {s['images_without_annotations']:,}")
        print(f"  Out-of-bounds:    {s['out_of_bounds_bboxes']:,}")
        print(f"  Zero-area bboxes: {s['zero_area_bboxes']:,}")

        if total_images > 0:
            print(f"\n  Annotations/image: {total_annotations / total_images:.1f} avg")

        d = report["annotation_density"]
        if d["count"]:
            print(f"    min={d['min']}, median={d['median']}, max={d['max']}")

        print(f"\n  Label Distribution (top {top}):")
        for cat, count in top_categories:
            pct = 100.0 * count / total_annotations if total_annotations else 0
            bar = "#" * int(pct / 2)
            print(f"    {cat:30s} {count:>8,} ({pct:5.1f}%) {bar}")

        a = report["bbox_area"]
        if a["count"]:
            print("\n  Bbox Area:")
            print(f"    min={a['min']}, median={a['median']}, mean={a['mean']}, max={a['max']}")

        ar = report["bbox_aspect_ratio"]
        if ar["count"]:
            print("\n  Bbox Aspect Ratio (w/h):")
            print(f"    min={ar['min']}, median={ar['median']}, mean={ar['mean']}, max={ar['max']}")

        if top_cooccurrences:
            print(f"\n  Category Co-occurrence (top {top}):")
            for pair, count in top_cooccurrences:
                print(f"    {pair[0]} + {pair[1]}: {count:,}")

        print(f"\n  Processing time: {processing_time.total_seconds():.1f}s")
        print("=" * 60)

    # Optionally push stats report as a dataset
    if output_dataset:
        from datasets import Dataset as HFDataset

        report_ds = HFDataset.from_dict({
            "report_json": [json.dumps(report)],
            "dataset": [input_dataset],
            "total_images": [total_images],
            "total_annotations": [total_annotations],
            "unique_categories": [len(category_counts)],
            "timestamp": [datetime.now().isoformat()],
        })

        logger.info(f"Pushing stats report to {output_dataset}")
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    os.environ["HF_HUB_DISABLE_XET"] = "1"
                report_ds.push_to_hub(output_dataset, private=private, token=HF_TOKEN)
                break
            except Exception as e:
                logger.error(f"Upload attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(30 * (2 ** (attempt - 1)))
                else:
                    logger.error("All upload attempts failed.")
                    sys.exit(1)

        logger.info(f"Stats pushed to: https://huggingface.co/datasets/{output_dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate statistics for object detection datasets on HF Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Bbox formats:
  coco_xywh     [x, y, width, height] in pixels (default)
  xyxy          [xmin, ymin, xmax, ymax] in pixels
  voc           [xmin, ymin, xmax, ymax] in pixels (alias for xyxy)
  yolo          [cx, cy, w, h] normalized 0-1
  tfod          [xmin, ymin, xmax, ymax] normalized 0-1
  label_studio  [x, y, w, h] percentage 0-100

Examples:
  uv run stats-hf-dataset.py merve/coco-dataset
  uv run stats-hf-dataset.py merve/coco-dataset --top 20 --report json
  uv run stats-hf-dataset.py merve/coco-dataset --streaming --max-samples 5000
        """,
    )

    parser.add_argument("input_dataset", help="Input dataset ID on HF Hub")
    parser.add_argument("--bbox-column", default="bbox", help="Column containing bboxes (default: bbox)")
    parser.add_argument("--category-column", default="category", help="Column containing categories (default: category)")
    parser.add_argument("--bbox-format", choices=BBOX_FORMATS, default="coco_xywh", help="Bbox format (default: coco_xywh)")
    parser.add_argument("--width-column", default="width", help="Column for image width (default: width)")
    parser.add_argument("--height-column", default="height", help="Column for image height (default: height)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--max-samples", type=int, help="Max samples to process")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    parser.add_argument("--top", type=int, default=10, help="Top-N items for histograms (default: 10)")
    parser.add_argument("--report", choices=["text", "json"], default="text", help="Report format (default: text)")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Out-of-bounds tolerance in pixels (default: 0.5)")
    parser.add_argument("--hf-token", help="HF API token")
    parser.add_argument("--output-dataset", help="Push stats report to this HF dataset")
    parser.add_argument("--private", action="store_true", help="Make output dataset private")

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        bbox_column=args.bbox_column,
        category_column=args.category_column,
        bbox_format=args.bbox_format,
        width_column=args.width_column,
        height_column=args.height_column,
        split=args.split,
        max_samples=args.max_samples,
        streaming=args.streaming,
        top=args.top,
        report_format=args.report,
        tolerance=args.tolerance,
        hf_token=args.hf_token,
        output_dataset=args.output_dataset,
        private=args.private,
    )
