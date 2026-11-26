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
Semantic diff between two object detection datasets on Hugging Face Hub.

Mirrors panlabel's diff command. Compares two dataset versions and reports:

- Images shared / only-in-A / only-in-B
- Categories shared / only-in-A / only-in-B
- Annotations added / removed / modified
- Bbox geometry changes (IoU-based matching)

Matching strategies:
  - ID-based: Match images by file_name or image_id column
  - For annotations within shared images, match by IoU threshold

Examples:
  uv run diff-hf-datasets.py merve/dataset-v1 merve/dataset-v2
  uv run diff-hf-datasets.py merve/old merve/new --iou-threshold 0.7 --detail
  uv run diff-hf-datasets.py merve/old merve/new --report json
"""

import argparse
import json
import logging
import math
import os
from datetime import datetime
from typing import Any

from datasets import load_dataset
from huggingface_hub import login
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BBOX_FORMATS = ["coco_xywh", "xyxy", "voc", "yolo", "tfod", "label_studio"]


def to_xyxy(bbox: list[float], fmt: str, img_w: float = 1.0, img_h: float = 1.0) -> tuple[float, float, float, float]:
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


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute IoU between two XYXY boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def extract_annotations(
    example: dict[str, Any],
    bbox_column: str,
    category_column: str,
    bbox_format: str,
    width_column: str | None,
    height_column: str | None,
) -> list[dict]:
    """Extract annotations from example as list of {bbox_xyxy, category}."""
    objects = example.get("objects", example)
    bboxes = objects.get(bbox_column, []) or []
    categories = objects.get(category_column, []) or []

    img_w = 1.0
    img_h = 1.0
    if width_column:
        img_w = float(example.get(width_column, 1.0) or 1.0)
    if height_column:
        img_h = float(example.get(height_column, 1.0) or 1.0)

    anns = []
    for i, bbox in enumerate(bboxes):
        if bbox is None or len(bbox) < 4:
            continue
        if not all(math.isfinite(v) for v in bbox[:4]):
            continue
        xyxy = to_xyxy(bbox[:4], bbox_format, img_w, img_h)
        cat = str(categories[i]) if i < len(categories) else "<unknown>"
        anns.append({"bbox_xyxy": xyxy, "category": cat})

    return anns


def match_annotations_iou(
    anns_a: list[dict],
    anns_b: list[dict],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy IoU matching. Returns (matched_pairs, unmatched_a, unmatched_b)."""
    if not anns_a or not anns_b:
        return [], list(range(len(anns_a))), list(range(len(anns_b)))

    # Compute all pairwise IoUs
    pairs = []
    for i, a in enumerate(anns_a):
        for j, b in enumerate(anns_b):
            iou = compute_iou(a["bbox_xyxy"], b["bbox_xyxy"])
            if iou >= iou_threshold:
                pairs.append((iou, i, j))

    pairs.sort(reverse=True)

    matched_a = set()
    matched_b = set()
    matches = []

    for iou, i, j in pairs:
        if i not in matched_a and j not in matched_b:
            matches.append((i, j, iou))
            matched_a.add(i)
            matched_b.add(j)

    unmatched_a = [i for i in range(len(anns_a)) if i not in matched_a]
    unmatched_b = [j for j in range(len(anns_b)) if j not in matched_b]

    return matches, unmatched_a, unmatched_b


def get_image_key(example: dict, id_column: str) -> str:
    """Get a unique key for an image example."""
    val = example.get(id_column)
    if val is not None:
        return str(val)
    return str(example.get("file_name", ""))


def main(
    dataset_a: str,
    dataset_b: str,
    bbox_column: str = "bbox",
    category_column: str = "category",
    bbox_format: str = "coco_xywh",
    id_column: str = "image_id",
    width_column: str | None = "width",
    height_column: str | None = "height",
    split: str = "train",
    max_samples: int | None = None,
    iou_threshold: float = 0.5,
    detail: bool = False,
    report_format: str = "text",
    hf_token: str | None = None,
):
    """Compare two object detection datasets semantically."""

    start_time = datetime.now()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    logger.info(f"Loading dataset A: {dataset_a}")
    ds_a = load_dataset(dataset_a, split=split)
    logger.info(f"Loading dataset B: {dataset_b}")
    ds_b = load_dataset(dataset_b, split=split)

    if max_samples:
        ds_a = ds_a.select(range(min(max_samples, len(ds_a))))
        ds_b = ds_b.select(range(min(max_samples, len(ds_b))))

    # Index by image key
    logger.info("Indexing images...")
    index_a = {}
    for i in tqdm(range(len(ds_a)), desc="Indexing A"):
        ex = ds_a[i]
        key = get_image_key(ex, id_column)
        index_a[key] = ex

    index_b = {}
    for i in tqdm(range(len(ds_b)), desc="Indexing B"):
        ex = ds_b[i]
        key = get_image_key(ex, id_column)
        index_b[key] = ex

    keys_a = set(index_a.keys())
    keys_b = set(index_b.keys())

    shared_keys = keys_a & keys_b
    only_a_keys = keys_a - keys_b
    only_b_keys = keys_b - keys_a

    # Collect all categories
    cats_a = set()
    cats_b = set()
    total_added = 0
    total_removed = 0
    total_modified = 0
    total_matched = 0
    detail_records = []

    logger.info(f"Comparing {len(shared_keys)} shared images...")
    for key in tqdm(sorted(shared_keys), desc="Diffing"):
        ex_a = index_a[key]
        ex_b = index_b[key]

        anns_a = extract_annotations(ex_a, bbox_column, category_column, bbox_format, width_column, height_column)
        anns_b = extract_annotations(ex_b, bbox_column, category_column, bbox_format, width_column, height_column)

        for a in anns_a:
            cats_a.add(a["category"])
        for b in anns_b:
            cats_b.add(b["category"])

        matches, unmatched_a, unmatched_b = match_annotations_iou(anns_a, anns_b, iou_threshold)

        total_matched += len(matches)
        total_removed += len(unmatched_a)
        total_added += len(unmatched_b)

        # Check for category changes in matched pairs
        for i, j, iou in matches:
            if anns_a[i]["category"] != anns_b[j]["category"]:
                total_modified += 1
                if detail:
                    detail_records.append({
                        "image": key,
                        "type": "modified",
                        "from_category": anns_a[i]["category"],
                        "to_category": anns_b[j]["category"],
                        "iou": round(iou, 3),
                    })

        if detail:
            for idx in unmatched_a:
                detail_records.append({
                    "image": key,
                    "type": "removed",
                    "category": anns_a[idx]["category"],
                    "bbox": list(anns_a[idx]["bbox_xyxy"]),
                })
            for idx in unmatched_b:
                detail_records.append({
                    "image": key,
                    "type": "added",
                    "category": anns_b[idx]["category"],
                    "bbox": list(anns_b[idx]["bbox_xyxy"]),
                })

    # Count annotations in only-A and only-B images
    anns_only_a = 0
    for key in only_a_keys:
        anns = extract_annotations(index_a[key], bbox_column, category_column, bbox_format, width_column, height_column)
        anns_only_a += len(anns)
        for a in anns:
            cats_a.add(a["category"])

    anns_only_b = 0
    for key in only_b_keys:
        anns = extract_annotations(index_b[key], bbox_column, category_column, bbox_format, width_column, height_column)
        anns_only_b += len(anns)
        for b in anns:
            cats_b.add(b["category"])

    shared_cats = cats_a & cats_b
    only_a_cats = cats_a - cats_b
    only_b_cats = cats_b - cats_a

    processing_time = datetime.now() - start_time

    report = {
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "split": split,
        "iou_threshold": iou_threshold,
        "images": {
            "in_a": len(keys_a),
            "in_b": len(keys_b),
            "shared": len(shared_keys),
            "only_in_a": len(only_a_keys),
            "only_in_b": len(only_b_keys),
        },
        "categories": {
            "in_a": len(cats_a),
            "in_b": len(cats_b),
            "shared": len(shared_cats),
            "only_in_a": sorted(only_a_cats),
            "only_in_b": sorted(only_b_cats),
        },
        "annotations": {
            "matched": total_matched,
            "modified": total_modified,
            "added_in_shared_images": total_added,
            "removed_in_shared_images": total_removed,
            "in_only_a_images": anns_only_a,
            "in_only_b_images": anns_only_b,
        },
        "processing_time_seconds": processing_time.total_seconds(),
    }

    if detail:
        report["details"] = detail_records

    if report_format == "json":
        print(json.dumps(report, indent=2))
    else:
        print("\n" + "=" * 60)
        print("Dataset Diff")
        print(f"  A: {dataset_a}")
        print(f"  B: {dataset_b}")
        print("=" * 60)

        img = report["images"]
        print("\n  Images:")
        print(f"    A: {img['in_a']:,}  |  B: {img['in_b']:,}")
        print(f"    Shared: {img['shared']:,}")
        print(f"    Only in A: {img['only_in_a']:,}")
        print(f"    Only in B: {img['only_in_b']:,}")

        cat = report["categories"]
        print("\n  Categories:")
        print(f"    A: {cat['in_a']}  |  B: {cat['in_b']}  |  Shared: {cat['shared']}")
        if cat["only_in_a"]:
            print(f"    Only in A: {', '.join(cat['only_in_a'][:10])}")
        if cat["only_in_b"]:
            print(f"    Only in B: {', '.join(cat['only_in_b'][:10])}")

        ann = report["annotations"]
        print(f"\n  Annotations (IoU >= {iou_threshold}):")
        print(f"    Matched:  {ann['matched']:,}")
        print(f"    Modified: {ann['modified']:,} (category changed)")
        print(f"    Added:    {ann['added_in_shared_images']:,} (in shared images)")
        print(f"    Removed:  {ann['removed_in_shared_images']:,} (in shared images)")
        if ann["in_only_a_images"]:
            print(f"    In A-only images: {ann['in_only_a_images']:,}")
        if ann["in_only_b_images"]:
            print(f"    In B-only images: {ann['in_only_b_images']:,}")

        if detail and detail_records:
            print(f"\n  Detail ({len(detail_records)} changes):")
            for rec in detail_records[:20]:
                if rec["type"] == "modified":
                    print(f"    [{rec['image']}] {rec['from_category']} -> {rec['to_category']} (IoU={rec['iou']})")
                elif rec["type"] == "added":
                    print(f"    [{rec['image']}] + {rec['category']}")
                elif rec["type"] == "removed":
                    print(f"    [{rec['image']}] - {rec['category']}")
            if len(detail_records) > 20:
                print(f"    ... and {len(detail_records) - 20} more")

        print(f"\n  Processing time: {processing_time.total_seconds():.1f}s")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic diff between two object detection datasets on HF Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run diff-hf-datasets.py merve/dataset-v1 merve/dataset-v2
  uv run diff-hf-datasets.py merve/old merve/new --iou-threshold 0.7 --detail
  uv run diff-hf-datasets.py merve/old merve/new --report json
        """,
    )

    parser.add_argument("dataset_a", help="First dataset ID (A)")
    parser.add_argument("dataset_b", help="Second dataset ID (B)")
    parser.add_argument("--bbox-column", default="bbox", help="Column containing bboxes (default: bbox)")
    parser.add_argument("--category-column", default="category", help="Column containing categories (default: category)")
    parser.add_argument("--bbox-format", choices=BBOX_FORMATS, default="coco_xywh", help="Bbox format (default: coco_xywh)")
    parser.add_argument("--id-column", default="image_id", help="Column to match images by (default: image_id)")
    parser.add_argument("--width-column", default="width", help="Column for image width (default: width)")
    parser.add_argument("--height-column", default="height", help="Column for image height (default: height)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--max-samples", type=int, help="Max samples per dataset")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching (default: 0.5)")
    parser.add_argument("--detail", action="store_true", help="Show per-annotation changes")
    parser.add_argument("--report", choices=["text", "json"], default="text", help="Report format (default: text)")
    parser.add_argument("--hf-token", help="HF API token")

    args = parser.parse_args()

    main(
        dataset_a=args.dataset_a,
        dataset_b=args.dataset_b,
        bbox_column=args.bbox_column,
        category_column=args.category_column,
        bbox_format=args.bbox_format,
        id_column=args.id_column,
        width_column=args.width_column,
        height_column=args.height_column,
        split=args.split,
        max_samples=args.max_samples,
        iou_threshold=args.iou_threshold,
        detail=args.detail,
        report_format=args.report,
        hf_token=args.hf_token,
    )
