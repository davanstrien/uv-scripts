# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "paddlepaddle-gpu>=3.0.0",
#     "paddleocr>=3.0.0",
#     "opencv-contrib-python-headless",
#     "datasets>=4.0.0",
#     "huggingface-hub>=1.6.0",
#     "pyarrow>=15.0",
#     "pillow",
#     "numpy",
#     "tqdm",
# ]
#
# [tool.uv]
# # PaddleOCR/PaddleX pull in opencv-contrib-python (full) which needs system
# # libGL.so.1 — not present in the slim uv-on-bookworm image used by HF Jobs.
# # Swap to the headless cv2 variant (same `import cv2`, no GUI deps).
# override-dependencies = [
#     "opencv-contrib-python ; python_version < '0'",
#     "opencv-python ; python_version < '0'",
# ]
#
# [[tool.uv.index]]
# name = "paddle"
# url = "https://www.paddlepaddle.org.cn/packages/stable/cu126/"
# explicit = true
#
# [tool.uv.sources]
# paddlepaddle-gpu = { index = "paddle" }
#
# [tool.hf-jobs]
# flavor = "t4-small"
# secrets = ["HF_TOKEN"]
# ///

"""
Detect document layout regions (text/title/table/figure/formula/...) with PP-DocLayout-L.

Runs PaddleOCR's PP-DocLayout-L (or M / S / plus-L variant) over an image source
and emits per-image bounding-box predictions. Unlike the OCR scripts in this repo
this does NOT extract text — it only locates and classifies regions.

Source can be:
  - HF dataset repo (default): "namespace/dataset"
  - HF bucket of image files: "hf://buckets/namespace/bucket/optional/prefix"

Sink can be:
  - HF dataset repo (default): "namespace/dataset" (one push at end + dataset card)
  - HF bucket: "hf://buckets/namespace/bucket/run-name" (incremental parquet
    shards, resumable, no git overhead)

Output schema (column `layout` is a JSON string):
  [{"bbox": [x1, y1, x2, y2], "label": "text", "score": 0.97, "cls_id": 2}, ...]

Coordinates are in the original input-image pixel space.

Example commands (the [tool.hf-jobs] header sets flavor + secrets; needs `hf` CLI 1.32+):

  # Dataset -> dataset (smoke test)
  hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/pp-doclayout.py \\
      davanstrien/ufo-ColPali pp-doclayout-smoke \\
      --max-samples 3 --shuffle --seed 42 --private

  # Dataset -> bucket (incremental shards, resumable)
  hf buckets create davanstrien/pp-doclayout-scratch --exist-ok
  hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/pp-doclayout.py \\
      davanstrien/ufo-ColPali \\
      hf://buckets/davanstrien/pp-doclayout-scratch/run1 \\
      --max-samples 20 --shard-size 5

  # Bucket of images -> dataset
  hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/pp-doclayout.py \\
      hf://buckets/davanstrien/pp-doclayout-images \\
      pp-doclayout-from-bucket --private
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MODELS = [
    "PP-DocLayout-L",
    "PP-DocLayout-M",
    "PP-DocLayout-S",
    "PP-DocLayout_plus-L",
]

MODEL_SIZES = {
    "PP-DocLayout-L": "~123M params (RT-DETR-L backbone)",
    "PP-DocLayout-M": "~22M params (PicoDet-M)",
    "PP-DocLayout-S": "~4M params (PicoDet-S)",
    "PP-DocLayout_plus-L": "~123M params, 20-class plus variant",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".jp2", ".j2k",
}

BUCKET_PREFIX = "hf://buckets/"


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def is_bucket_url(s: str) -> bool:
    return s.startswith(BUCKET_PREFIX)


def parse_bucket_url(url: str) -> Tuple[str, str]:
    """Split `hf://buckets/ns/bucket/path/in/bucket` into (`ns/bucket`, `path/in/bucket`)."""
    if not is_bucket_url(url):
        raise ValueError(f"Not a bucket URL: {url}")
    rest = url[len(BUCKET_PREFIX) :].strip("/")
    parts = rest.split("/", 2)
    if len(parts) < 2:
        raise ValueError(
            f"Bucket URL must include namespace and bucket name: {url}"
        )
    bucket_id = f"{parts[0]}/{parts[1]}"
    prefix = parts[2] if len(parts) > 2 else ""
    return bucket_id, prefix


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def to_pil(image: Union[Image.Image, Dict[str, Any], str, bytes]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict) and "bytes" in image:
        return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image)}")


def pil_to_array(pil_img: Image.Image) -> np.ndarray:
    """RGB PIL -> uint8 ndarray. PaddleOCR's predict() accepts numpy arrays directly."""
    return np.asarray(pil_img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------


def extract_detections(result: Any) -> List[Dict[str, Any]]:
    """Pull a clean list of detections out of a paddleocr LayoutDetection result."""
    payload = result.json
    res = payload.get("res", payload) if isinstance(payload, dict) else {}
    boxes = res.get("boxes", []) if isinstance(res, dict) else []
    detections = []
    for box in boxes:
        coord = box.get("coordinate") or box.get("bbox") or []
        coord = [float(x) for x in coord]
        detections.append(
            {
                "bbox": coord,
                "label": box.get("label"),
                "score": float(box.get("score", 0.0)),
                "cls_id": int(box.get("cls_id", -1)),
            }
        )
    return detections


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@dataclass
class SourceItem:
    key: str  # stable identifier per image (used for dedup/resume)
    image: Optional[Image.Image]  # None for an unreadable row (placeholder)
    extras: Dict[str, Any]  # original row fields (only populated for dataset source)


def iter_dataset_images(
    dataset_id: str,
    image_column: str,
    split: str,
    shuffle: bool,
    seed: int,
    max_samples: Optional[int],
):
    """Iterate (key, PIL) pairs from an HF dataset repo.

    Returns: (iterator, total, dataset_reference). The dataset reference is the
    post-shuffle/post-select Dataset, kept around so the dataset-repo sink can
    `add_column("layout", ...)` and preserve the original schema (especially
    Image-type columns).
    """
    from datasets import load_dataset

    logger.info(f"Loading dataset: {dataset_id} (split={split})")
    ds = load_dataset(dataset_id, split=split)

    if image_column not in ds.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {ds.column_names}"
        )

    if shuffle:
        logger.info(f"Shuffling with seed {seed}")
        ds = ds.shuffle(seed=seed)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
        logger.info(f"Limited to {len(ds)} samples")

    total = len(ds)

    def gen() -> Iterator[SourceItem]:
        failed = 0
        for i in range(total):
            try:
                row = ds[i]
                image = to_pil(row[image_column])
            except (UnidentifiedImageError, OSError) as e:
                # Still yield a placeholder so the output row stays aligned with
                # the source row (the dataset sink writes layouts positionally).
                failed += 1
                logger.warning(
                    f"Unreadable image at row {i}: "
                    f"{type(e).__name__}: {e} — writing empty layout"
                )
                yield SourceItem(
                    key=f"row-{i:08d}",
                    image=None,
                    extras={"failed": True},
                )
                continue
            yield SourceItem(
                key=f"row-{i:08d}",
                image=image,
                extras={},  # original schema is preserved by the sink via the dataset ref
            )
        if failed:
            logger.info(f"{failed} unreadable image(s) written as empty layouts")

    return gen(), total, ds


SOURCE_PATHS_SNAPSHOT = "_source_paths.json"


def _bucket_snapshot_path(output_url: str) -> Tuple[str, str]:
    """Return (bucket_id, key) for the source-paths snapshot inside an output bucket."""
    out_bucket_id, out_prefix = parse_bucket_url(output_url)
    snapshot_key = (
        f"{out_prefix}/{SOURCE_PATHS_SNAPSHOT}".lstrip("/")
        if out_prefix
        else SOURCE_PATHS_SNAPSHOT
    )
    return out_bucket_id, snapshot_key


def iter_bucket_images(
    bucket_url: str,
    shuffle: bool,
    seed: int,
    max_samples: Optional[int],
    hf_token: Optional[str],
    output_url: Optional[str] = None,
) -> Tuple[Iterator[SourceItem], int]:
    """Glob image files under a bucket prefix and stream them via HfFileSystem.

    If `output_url` is a bucket, the resolved source-path list is snapshotted to
    `<output>/_source_paths.json` on first run. Subsequent runs against the same
    output prefix reuse that snapshot, so resume stays consistent even if the
    source bucket grows or `--shuffle`/`--max-samples` would otherwise pick a
    different subset on the second run.
    """
    from huggingface_hub import HfApi, HfFileSystem

    bucket_id, prefix = parse_bucket_url(bucket_url)
    fs = HfFileSystem(token=hf_token)
    base = f"{BUCKET_PREFIX}{bucket_id}/{prefix}".rstrip("/")

    snapshot_bucket_id: Optional[str] = None
    snapshot_key: Optional[str] = None
    cached_paths: Optional[List[str]] = None

    if output_url and is_bucket_url(output_url):
        snapshot_bucket_id, snapshot_key = _bucket_snapshot_path(output_url)
        snapshot_url = f"{BUCKET_PREFIX}{snapshot_bucket_id}/{snapshot_key}"
        try:
            with fs.open(snapshot_url, "rb") as f:
                snapshot = json.load(f)
            if snapshot.get("source_url") != bucket_url:
                logger.warning(
                    f"Output prefix already has a snapshot referencing a "
                    f"different source ({snapshot.get('source_url')!r} vs "
                    f"{bucket_url!r}). Ignoring and re-listing."
                )
            else:
                cached_paths = snapshot["paths"]
                logger.info(
                    f"Reusing existing snapshot of {len(cached_paths)} source paths "
                    f"(written {snapshot.get('created_at', 'unknown')})"
                )
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Could not read existing snapshot ({e}); re-listing.")

    if cached_paths is not None:
        all_paths = cached_paths
    else:
        logger.info(f"Listing images under {base}")
        all_paths = []
        try:
            for entry in fs.find(base, detail=False):
                ext = Path(entry).suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    all_paths.append(entry)
        except FileNotFoundError as e:
            raise ValueError(f"Bucket prefix not found: {base}") from e

        if not all_paths:
            raise ValueError(
                f"No image files (any of {sorted(IMAGE_EXTENSIONS)}) under {base}"
            )

        all_paths.sort()
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(all_paths)
        if max_samples:
            all_paths = all_paths[:max_samples]

        # Persist the chosen list so resume runs see exactly this set.
        if snapshot_bucket_id is not None and snapshot_key is not None:
            api = HfApi(token=hf_token)
            payload = {
                "source_url": bucket_url,
                "shuffle": shuffle,
                "seed": seed,
                "max_samples": max_samples,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "paths": all_paths,
            }
            api.batch_bucket_files(
                snapshot_bucket_id,
                add=[(json.dumps(payload).encode(), snapshot_key)],
                token=hf_token,
            )
            logger.info(
                f"Wrote source-path snapshot ({len(all_paths)} paths) to "
                f"hf://buckets/{snapshot_bucket_id}/{snapshot_key}"
            )

    total = len(all_paths)
    logger.info(f"Found {total} images in bucket")

    def key_for(path: str) -> str:
        # Use the full bucket path (`buckets/<id>/<rel>`) as returned by
        # fs.find. This is stable across reruns (so resume works), and the
        # stored value in `source_path` is fully addressable — open via
        # HfFileSystem directly with `hf://` re-prepended.
        return path

    def gen() -> Iterator[SourceItem]:
        skipped = 0
        for path in all_paths:
            try:
                with fs.open(path, "rb") as f:
                    data = f.read()
                image = to_pil(data)
            except (UnidentifiedImageError, OSError) as e:
                skipped += 1
                logger.warning(
                    f"Skipping unreadable image {path}: "
                    f"{type(e).__name__}: {e}"
                )
                continue
            yield SourceItem(
                key=key_for(path),
                image=image,
                extras={"__source_path": key_for(path)},
            )
        if skipped:
            logger.info(f"Skipped {skipped} unreadable image(s) total")

    return gen(), total


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


class DatasetRepoSink:
    """Buffer all results in memory, push once at end with dataset card + inference_info.

    Two modes:
      - `original_dataset` provided (dataset-repo source): preserve the source
        schema (including Image-type columns) and just `add_column("layout", ...)`.
      - `original_dataset` is None (bucket-image source): build a Dataset from
        collected rows containing __source_path + layout.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        hf_token: Optional[str],
        private: bool,
        config: Optional[str],
        create_pr: bool,
        source_id: str,
        original_dataset=None,
        overwrite: bool = False,
    ):
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.private = private
        self.config = config
        self.create_pr = create_pr
        self.source_id = source_id
        self.original_dataset = original_dataset
        self.overwrite = overwrite
        # Used when original_dataset is None: row-by-row buffer.
        self._rows: List[Dict[str, Any]] = []
        # Used when original_dataset is set: ordered layouts aligned with dataset rows.
        self._layouts: List[str] = []

    @property
    def kind(self) -> str:
        return "dataset"

    def already_done(self) -> set:
        return set()  # dataset sink does a single push, no resume

    def write(self, key: str, layout: List[Dict[str, Any]], extras: Dict[str, Any]) -> None:
        layout_json = json.dumps(layout, ensure_ascii=False)
        if self.original_dataset is not None:
            self._layouts.append(layout_json)
            return
        row = {"__source_key": key, "layout": layout_json}
        for k, v in extras.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[k] = v
        self._rows.append(row)

    def finalize(self, model_id: str, args_dict: Dict[str, Any]) -> None:
        from datasets import Dataset

        if self.original_dataset is not None:
            if len(self._layouts) != len(self.original_dataset):
                logger.warning(
                    f"Layout count ({len(self._layouts)}) != dataset rows "
                    f"({len(self.original_dataset)}); padding with empty layouts."
                )
                # Pad to keep add_column happy.
                while len(self._layouts) < len(self.original_dataset):
                    self._layouts.append("[]")
            base = self.original_dataset
            if "layout" in base.column_names:
                if not self.overwrite:
                    raise ValueError(
                        "Output column 'layout' already exists in the input dataset; "
                        "pass --overwrite to replace it."
                    )
                logger.warning("--overwrite: replacing existing column 'layout'")
                base = base.remove_columns("layout")
            ds = base.add_column("layout", self._layouts)
        else:
            if not self._rows:
                logger.warning("No rows produced; nothing to push.")
                return
            ds = Dataset.from_list(self._rows)
            if "__source_key" in ds.column_names:
                ds = ds.rename_column("__source_key", "source_path")

        inference_entry = build_inference_entry(model_id, args_dict)

        if "inference_info" in ds.column_names:
            logger.info("Updating existing inference_info column")

            def _update(example):
                try:
                    existing = (
                        json.loads(example["inference_info"])
                        if example["inference_info"]
                        else []
                    )
                except (json.JSONDecodeError, TypeError):
                    existing = []
                existing.append(inference_entry)
                return {"inference_info": json.dumps(existing)}

            ds = ds.map(_update)
        else:
            ds = ds.add_column(
                "inference_info", [json.dumps([inference_entry])] * len(ds)
            )

        logger.info(f"Pushing {len(ds)} rows to {self.repo_id}")
        push_kwargs = {
            "private": self.private,
            "token": self.hf_token,
            "max_shard_size": "500MB",
            "create_pr": self.create_pr,
            "commit_message": f"Add PP-DocLayout layout predictions ({len(ds)} samples)"
            + (f" [{self.config}]" if self.config else ""),
        }
        if self.config:
            push_kwargs["config_name"] = self.config

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    logger.warning("Disabling XET (fallback to HTTP upload)")
                    os.environ["HF_HUB_DISABLE_XET"] = "1"
                ds.push_to_hub(self.repo_id, **push_kwargs)
                break
            except Exception as e:
                logger.error(f"Upload attempt {attempt}/{max_retries} failed: {e}")
                if attempt == max_retries:
                    logger.error("All upload attempts failed.")
                    raise
                time.sleep(30 * (2 ** (attempt - 1)))

        # Dataset card
        from huggingface_hub import DatasetCard

        card = DatasetCard(
            create_dataset_card(
                source=self.source_id,
                model_name=args_dict["model_name"],
                num_samples=len(ds),
                processing_time=args_dict["processing_time"],
                output_column="layout",
                threshold=args_dict["threshold"],
                layout_nms=args_dict["layout_nms"],
            )
        )
        card.push_to_hub(self.repo_id, token=self.hf_token)
        logger.info(
            f"Done: https://huggingface.co/datasets/{self.repo_id}"
        )


class BucketShardSink:
    """Write incremental parquet shards to a bucket prefix. Resumable."""

    METADATA_FILE = "_metadata.json"
    SHARD_PATTERN = "shard-{:05d}.parquet"

    def __init__(
        self,
        bucket_url: str,
        *,
        hf_token: Optional[str],
        shard_size: int,
        include_images: bool,
        resume: bool,
        source_id: str,
    ):
        from huggingface_hub import HfApi, HfFileSystem, create_bucket

        self.bucket_url = bucket_url
        self.bucket_id, self.prefix = parse_bucket_url(bucket_url)
        self.hf_token = hf_token
        self.shard_size = shard_size
        self.include_images = include_images
        self.resume = resume
        self.source_id = source_id

        self._api = HfApi(token=hf_token)
        self._fs = HfFileSystem(token=hf_token)

        # Make sure the bucket exists. Path inside the bucket is created lazily on first write.
        try:
            create_bucket(self.bucket_id, exist_ok=True, token=hf_token)
        except Exception as e:
            # If we don't have create rights but the bucket already exists, that's fine.
            logger.warning(f"create_bucket('{self.bucket_id}') warning: {e}")

        self._buffer: List[Dict[str, Any]] = []
        self._next_shard_idx = self._discover_next_shard_idx()
        self._completed_keys = self._discover_completed_keys() if resume else set()
        if self._completed_keys:
            logger.info(
                f"Resume: found {len(self._completed_keys)} already-processed keys, will skip them"
            )

    @property
    def kind(self) -> str:
        return "bucket"

    def already_done(self) -> set:
        return self._completed_keys

    # --- internal helpers ---

    def _shard_path(self, idx: int) -> str:
        return self._join(self.SHARD_PATTERN.format(idx))

    def _join(self, name: str) -> str:
        return f"{self.prefix}/{name}".lstrip("/") if self.prefix else name

    def _list_existing_shards(self) -> List[str]:
        try:
            tree = self._api.list_bucket_tree(
                self.bucket_id, prefix=self.prefix or None, recursive=True
            )
        except Exception:
            return []
        shards: List[str] = []
        for item in tree:
            path = getattr(item, "path", None)
            ftype = getattr(item, "type", None)
            if not path or ftype not in (None, "file"):
                continue
            base = Path(path).name
            if base.startswith("shard-") and base.endswith(".parquet"):
                shards.append(path)
        return sorted(shards)

    def _discover_next_shard_idx(self) -> int:
        shards = self._list_existing_shards()
        max_idx = -1
        for s in shards:
            stem = Path(s).stem  # shard-00007
            try:
                max_idx = max(max_idx, int(stem.split("-")[-1]))
            except ValueError:
                continue
        return max_idx + 1

    def _discover_completed_keys(self) -> set:
        import pyarrow.parquet as pq

        keys: set = set()
        for shard_path in self._list_existing_shards():
            full = f"{BUCKET_PREFIX}{self.bucket_id}/{shard_path}"
            try:
                with self._fs.open(full, "rb") as f:
                    table = pq.read_table(f, columns=["__source_key"])
                keys.update(table.column("__source_key").to_pylist())
            except Exception as e:
                logger.warning(f"Could not read keys from {shard_path}: {e}")
        return keys

    def _flush(self) -> None:
        if not self._buffer:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Build a stable schema. Skip the image column if not requested.
        columns = ["__source_key", "layout"]
        if self.include_images:
            columns.append("__image_bytes")
        # Carry through any extra string-coercible fields (e.g. __source_path).
        extra_keys = sorted(
            {k for row in self._buffer for k in row.keys() if k not in columns}
        )
        columns.extend(extra_keys)

        table_dict = {c: [row.get(c) for row in self._buffer] for c in columns}
        # pyarrow infers types from python objects; strings/bytes/lists handled fine.
        table = pa.Table.from_pydict(table_dict)

        buf = io.BytesIO()
        pq.write_table(table, buf, compression="zstd")
        data = buf.getvalue()

        shard_remote = self._shard_path(self._next_shard_idx)
        logger.info(
            f"Writing shard {self._next_shard_idx} ({len(self._buffer)} rows, "
            f"{len(data) / 1024 / 1024:.1f} MiB) to {shard_remote}"
        )
        self._api.batch_bucket_files(
            self.bucket_id, add=[(data, shard_remote)], token=self.hf_token
        )
        self._next_shard_idx += 1
        self._buffer.clear()

    def write(self, key: str, layout: List[Dict[str, Any]], extras: Dict[str, Any]) -> None:
        row: Dict[str, Any] = {
            "__source_key": key,
            "layout": json.dumps(layout, ensure_ascii=False),
        }
        if self.include_images and "__image_bytes" in extras:
            row["__image_bytes"] = extras["__image_bytes"]
        # Pass through string/numeric extras (skip raw PIL Image objects which
        # the dataset source never injects directly into extras anyway).
        for k, v in extras.items():
            if k in row or k == "__image_bytes":
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[k] = v
        self._buffer.append(row)
        if len(self._buffer) >= self.shard_size:
            self._flush()

    def finalize(self, model_id: str, args_dict: Dict[str, Any]) -> None:
        # Flush trailing rows.
        self._flush()
        # Write/update the metadata file alongside the shards.
        meta = {
            "model_id": model_id,
            "model_name": args_dict["model_name"],
            "task_mode": "layout-detection",
            "source": self.source_id,
            "threshold": args_dict["threshold"],
            "layout_nms": args_dict["layout_nms"],
            "shard_size": args_dict["shard_size"],
            "include_images": self.include_images,
            "last_run_at": datetime.now(timezone.utc).isoformat(),
            "processing_time": args_dict.get("processing_time"),
        }
        meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
        meta_path = self._join(self.METADATA_FILE)
        self._api.batch_bucket_files(
            self.bucket_id, add=[(meta_bytes, meta_path)], token=self.hf_token
        )
        logger.info(
            f"Done: https://huggingface.co/buckets/{self.bucket_id}"
            + (f"/{self.prefix}" if self.prefix else "")
        )


# ---------------------------------------------------------------------------
# inference_info + dataset card
# ---------------------------------------------------------------------------


def build_inference_entry(model_id: str, args_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_id": "PaddlePaddle/" + args_dict["model_name"],
        "model_name": args_dict["model_name"],
        "model_size": MODEL_SIZES.get(args_dict["model_name"], "unknown"),
        "task_mode": "layout-detection",
        "column_name": "layout",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threshold": args_dict["threshold"],
        "layout_nms": args_dict["layout_nms"],
        "backend": "paddleocr",
    }


def create_dataset_card(
    source: str,
    model_name: str,
    num_samples: int,
    processing_time: str,
    output_column: str,
    threshold: float,
    layout_nms: bool,
) -> str:
    """Render the dataset card markdown for the dataset-repo sink."""
    if is_bucket_url(source):
        source_link = f"[{source}]({source})"
    else:
        source_link = f"[{source}](https://huggingface.co/datasets/{source})"

    return f"""---
tags:
- layout-detection
- document-processing
- paddleocr
- pp-doclayout
- uv-script
- generated
viewer: false
---

# Layout detection with {model_name}

Bounding-box layout predictions for images from {source_link}, produced by
PaddleOCR's [{model_name}](https://huggingface.co/PaddlePaddle/{model_name}).

## Processing details

- **Source**: {source_link}
- **Model**: PaddlePaddle/{model_name} ({MODEL_SIZES.get(model_name, "unknown")})
- **Samples**: {num_samples:,}
- **Processing time**: {processing_time}
- **Processing date**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
- **Confidence threshold**: {threshold}
- **Layout NMS**: {"on" if layout_nms else "off"}
- **Output column**: `{output_column}` (JSON-encoded list of detections)

## Schema

Each row contains the original columns plus:

- `{output_column}`: JSON string. List of detections:
  ```json
  [
    {{"bbox": [x1, y1, x2, y2], "label": "text", "score": 0.97, "cls_id": 2}},
    {{"bbox": [x1, y1, x2, y2], "label": "table", "score": 0.92, "cls_id": 5}}
  ]
  ```
  Coordinates are in **original input-image pixel space** (top-left origin,
  `[xmin, ymin, xmax, ymax]`).
- `inference_info`: JSON list tracking every model that has been applied to
  this dataset (appended on each run).

## Usage

```python
import json
from datasets import load_dataset

ds = load_dataset("{{output_dataset_id}}", split="train")
detections = json.loads(ds[0]["{output_column}"])
for det in detections:
    print(det["label"], det["score"], det["bbox"])
```

## Reproduction

```bash
hf jobs uv run \\
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/pp-doclayout.py \\
    {source} <output> --model-name {model_name}
```

Generated with [UV Scripts](https://huggingface.co/uv-scripts).
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def resolve_device(device: str) -> str:
    if device == "gpu":
        try:
            import paddle

            if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                logger.info(
                    f"GPU available: {paddle.device.cuda.device_count()} device(s)"
                )
                return "gpu"
            logger.warning("No CUDA GPU detected; falling back to CPU.")
            return "cpu"
        except Exception as e:
            logger.warning(f"GPU check failed ({e}); falling back to CPU.")
            return "cpu"
    return device


def main(args: argparse.Namespace) -> None:
    from huggingface_hub import login

    start_time = datetime.now()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    device = resolve_device(args.device)

    # ---------- source ----------
    original_dataset = None
    if is_bucket_url(args.input_source):
        src_iter, total = iter_bucket_images(
            args.input_source,
            shuffle=args.shuffle,
            seed=args.seed,
            max_samples=args.max_samples,
            hf_token=hf_token,
            output_url=args.output_target,
        )
    else:
        src_iter, total, original_dataset = iter_dataset_images(
            args.input_source,
            image_column=args.image_column,
            split=args.split,
            shuffle=args.shuffle,
            seed=args.seed,
            max_samples=args.max_samples,
        )
        # Fail fast, before minutes of inference, if the 'layout' output column would
        # collide with an existing input column. --overwrite opts in to replacing it.
        if original_dataset is not None and "layout" in original_dataset.column_names:
            if not args.overwrite:
                logger.error(
                    "Output column 'layout' already exists in the input dataset "
                    f"(columns: {original_dataset.column_names})."
                )
                logger.error("Pass --overwrite to replace it.")
                sys.exit(1)
            logger.warning("--overwrite: will replace existing column 'layout'")

    # ---------- sink ----------
    if is_bucket_url(args.output_target):
        sink: Union[BucketShardSink, DatasetRepoSink] = BucketShardSink(
            args.output_target,
            hf_token=hf_token,
            shard_size=args.shard_size,
            include_images=args.include_images,
            resume=not args.no_resume,
            source_id=args.input_source,
        )
    else:
        sink = DatasetRepoSink(
            args.output_target,
            hf_token=hf_token,
            private=args.private,
            config=args.config,
            create_pr=args.create_pr,
            source_id=args.input_source,
            original_dataset=original_dataset,
            overwrite=args.overwrite,
        )

    completed = sink.already_done()

    # ---------- model ----------
    if args.model_name not in VALID_MODELS:
        raise ValueError(
            f"Invalid model {args.model_name!r}. Choose from: {VALID_MODELS}"
        )
    logger.info(f"Loading PaddleOCR LayoutDetection model: {args.model_name} on {device}")
    # PaddleX gates `import cv2` at module load time on
    # `is_dep_available("opencv-contrib-python")`, which checks
    # `importlib.metadata.version(...)`. We ship `opencv-contrib-python-headless`
    # (same `cv2`, no system libGL.so.1 needed) — but that's a different
    # distribution name, so the gate fails and `cv2` is never bound, causing
    # NameErrors deep inside paddlex modules. Patch the metadata lookup to
    # alias the GUI cv2 distros to the headless variant before importing
    # paddleocr; this lets paddlex's own `import cv2` succeed naturally.
    import importlib.metadata as _metadata

    _orig_metadata_version = _metadata.version

    def _patched_metadata_version(dep_name):
        if dep_name in ("opencv-contrib-python", "opencv-python"):
            for headless_alias in (
                "opencv-contrib-python-headless",
                "opencv-python-headless",
            ):
                try:
                    return _orig_metadata_version(headless_alias)
                except _metadata.PackageNotFoundError:
                    continue
        return _orig_metadata_version(dep_name)

    _metadata.version = _patched_metadata_version

    from paddleocr import LayoutDetection

    model = LayoutDetection(model_name=args.model_name, device=device)

    # ---------- loop ----------
    processed = 0
    skipped = 0
    errors = 0
    pbar = tqdm(src_iter, total=total, desc=f"Layout {args.model_name}")
    for item in pbar:
        if item.key in completed:
            skipped += 1
            continue
        if item.extras.get("failed") or item.image is None:
            # Unreadable source image — write an empty layout in position so the
            # output stays row-aligned with the source dataset.
            sink.write(item.key, [], {})
            errors += 1
            processed += 1
            continue
        try:
            arr = pil_to_array(item.image)
            results = model.predict(
                arr,
                batch_size=args.batch_size,
                layout_nms=args.layout_nms,
            )
            if not results:
                detections: List[Dict[str, Any]] = []
            else:
                detections = extract_detections(results[0])
                if args.threshold and args.threshold > 0:
                    detections = [d for d in detections if d["score"] >= args.threshold]
        except Exception as e:
            logger.error(f"Error on {item.key}: {e}")
            detections = []
            errors += 1

        extras = dict(item.extras)
        if isinstance(sink, BucketShardSink) and args.include_images:
            buf = io.BytesIO()
            item.image.save(buf, format="PNG")
            extras["__image_bytes"] = buf.getvalue()

        sink.write(item.key, detections, extras)
        processed += 1

    duration = datetime.now() - start_time
    processing_time_str = f"{duration.total_seconds() / 60:.2f} min"
    logger.info(
        f"Processed {processed} (skipped {skipped}, errors {errors}) in {processing_time_str}"
    )

    args_dict = {
        "model_name": args.model_name,
        "threshold": args.threshold,
        "layout_nms": args.layout_nms,
        "shard_size": args.shard_size,
        "processing_time": processing_time_str,
    }
    sink.finalize(model_id=f"PaddlePaddle/{args.model_name}", args_dict=args_dict)

    if args.verbose:
        import importlib.metadata

        logger.info("--- Resolved package versions ---")
        for pkg in [
            "paddlepaddle",
            "paddlepaddle-gpu",
            "paddleocr",
            "huggingface-hub",
            "datasets",
            "pyarrow",
            "pillow",
            "numpy",
        ]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")
        logger.info("--- End versions ---")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_usage_banner() -> None:
    print("=" * 80)
    print("PP-DocLayout layout detection")
    print("=" * 80)
    print(
        "\nDetect document layout regions (text/title/table/figure/formula/...)"
    )
    print("with PaddleOCR's PP-DocLayout-L (or M / S / plus-L variant).")
    print("\nModels:")
    for m in VALID_MODELS:
        print(f"  {m:24s} {MODEL_SIZES.get(m, '')}")
    print("\nSources:")
    print("  - HF dataset repo:        namespace/dataset")
    print("  - HF bucket of images:    hf://buckets/namespace/bucket[/prefix]")
    print("\nSinks:")
    print("  - HF dataset repo (one push + dataset card):")
    print("      namespace/dataset")
    print("  - HF bucket (incremental shards, resumable):")
    print("      hf://buckets/namespace/bucket/run-name")
    print("\nExamples:")
    print("\n  # Smoke test (dataset -> dataset; header needs hf CLI 1.32+)")
    print("  hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/pp-doclayout.py \\")
    print("      davanstrien/ufo-ColPali pp-doclayout-smoke \\")
    print("      --max-samples 3 --shuffle --seed 42 --private")
    print("\n  # Dataset -> bucket (incremental shards)")
    print(
        "  hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/pp-doclayout.py \\"
    )
    print("      davanstrien/ufo-ColPali \\")
    print(
        "      hf://buckets/davanstrien/pp-doclayout-scratch/run1 \\"
    )
    print("      --max-samples 20 --shard-size 5")
    print("\n  # Bucket of images -> dataset")
    print(
        "  hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/pp-doclayout.py \\"
    )
    print(
        "      hf://buckets/davanstrien/pp-doclayout-images \\"
    )
    print("      pp-doclayout-from-bucket --private")
    print("\nFor full help, run: uv run pp-doclayout.py --help")
    print("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PP-DocLayout layout detection over an HF dataset or bucket.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input_source",
        help="HF dataset id (namespace/dataset) OR hf://buckets/ns/bucket[/prefix]",
    )
    p.add_argument(
        "output_target",
        help="HF dataset id (namespace/dataset) OR hf://buckets/ns/bucket/run-name",
    )
    p.add_argument(
        "--model-name",
        default="PP-DocLayout-L",
        choices=VALID_MODELS,
        help="PaddleOCR layout model variant (default: PP-DocLayout-L)",
    )
    p.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu"],
        help="Device for inference (default: gpu, falls back to cpu if CUDA missing)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-image batch size passed to model.predict (default: 1)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Drop detections below this confidence (default: 0.5; 0 disables)",
    )
    p.add_argument(
        "--layout-nms",
        dest="layout_nms",
        action="store_true",
        default=True,
        help="Enable layout NMS (default: on)",
    )
    p.add_argument(
        "--no-layout-nms",
        dest="layout_nms",
        action="store_false",
        help="Disable layout NMS",
    )
    # Dataset-source-specific
    p.add_argument(
        "--image-column",
        default="image",
        help="Column containing images (dataset-repo source only, default: image)",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split (dataset-repo source only, default: train)",
    )
    p.add_argument(
        "--max-samples", type=int, help="Limit number of samples (for testing)"
    )
    p.add_argument(
        "--shuffle", action="store_true", help="Shuffle source before processing"
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffle (default: 42)"
    )
    # Dataset-sink-specific
    p.add_argument(
        "--private", action="store_true", help="Private dataset output (dataset sink only)"
    )
    p.add_argument(
        "--config",
        help="Config/subset name when pushing to Hub (dataset sink only)",
    )
    p.add_argument(
        "--create-pr",
        action="store_true",
        help="Create PR instead of direct push (dataset sink only)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column if it already exists in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )
    # Bucket-sink-specific
    p.add_argument(
        "--shard-size",
        type=int,
        default=256,
        help="Rows per parquet shard for bucket sink (default: 256)",
    )
    p.add_argument(
        "--include-images",
        action="store_true",
        help="Embed source image bytes in bucket output shards (off by default)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume scan when writing to a bucket sink",
    )
    # Auth + diagnostics
    p.add_argument("--hf-token", help="Hugging Face API token (else uses HF_TOKEN env)")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions at the end",
    )
    return p


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _print_usage_banner()
        sys.exit(0)
    main(build_parser().parse_args())
