# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "toolz",
#     "tqdm",
# ]
#
# [tool.hf-jobs]
# # vllm, torch and transformers come from the image (PYTHONPATH). Do not list them above: uv
# # would install the newest PyPI vllm deps next to the image's vllm and break it.
# image = "vllm/vllm-openai:v0.32.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "a10g-small"
# secrets = ["HF_TOKEN"]
# timeout = "1h"
# ///

"""
Class-list object detection with WeVisOne-VL via vLLM.

Give it an image dataset and a list of class names; it returns boxes for every
instance of every class and pushes the dataset back to the Hub with a
`detections` column in the same shape as qwen3vl-detect.py, so
inspect-detections.py and the downstream labelling steps work unchanged.

WeVisOne-VL (tencent/WeVisOne-VL-4B and -2B, Apache-2.0) is a Qwen3-VL model
trained for grounding: open-vocabulary detection, referring, counting and
pointing. It loads in stock vLLM. Use it when you know your classes; use
qwen3vl-detect.py when you want a free-form instruction and descriptive
sub-labels.

Example (Hugging Face Jobs; the header above supplies image, flavor and secrets):

    hf jobs uv run \\
        https://huggingface.co/datasets/uv-scripts/vlm-object-detection/raw/main/wevisone-detect.py \\
        rishitdagli/cppe-5 your-username/cppe5-wevisone \\
        --classes "Coverall,Face_Shield,Gloves,Goggles,Mask" \\
        --split test --max-samples 20 --shuffle

Measured behaviour (20-image smokes, IoU >= 0.5 against ground truth):

  * The model answers each class on its own, so one object can come back under
    several classes (a fox as Red Fox, Grey Fox and Coyote). Every box is kept;
    each box's `alt_labels` lists the other classes given to the same region
    (IoU >= --duplicate-iou), so a reviewer sees the ambiguity.
  * Keeping every box costs box precision on multi-class lists: each extra class
    name on the same object is an extra box (ENA24: 0.47 vs 0.90 merged).
    --merge-duplicates folds overlapping boxes of different classes into one box,
    labelled with the class listed first in --classes (there are no scores to
    choose by). Labels get worse when that order is arbitrary: ENA24 species
    recall fell from 0.75 to 0.30. Merge for coarse or priority-ordered classes,
    or dedupe downstream before using the boxes as training data.
  * Class names go into the prompt as plain phrases ("Face_Shield" is sent as
    "Face Shield"); the output label is the name exactly as you passed it. The
    prompt separates classes with ". ", so names cannot contain "." or ",".
  * There are no confidence scores.
  * Compared with qwen3vl-detect.py on Qwen3.6-35B-A3B: higher box recall on small
    objects and dense pages (cigarette ends 0.88 vs 0.79, newspaper pages 0.26 vs
    0.15), similar box recall elsewhere, lower box precision by default (see
    above), and slightly weaker fine-grained labels (ENA24 species recall 0.75 vs 0.80).
"""

import argparse
import io
import json
import logging
import os
import re
import shlex
import sys
from datetime import datetime
from typing import Any

from datasets import Dataset, Features, List, Value, concatenate_datasets, load_dataset
from datasets import Image as ImageFeature
from huggingface_hub import DatasetCard, login
from PIL import Image
from toolz import partition_all
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "tencent/WeVisOne-VL-4B"
SCRIPT_URL = "https://huggingface.co/datasets/uv-scripts/vlm-object-detection/raw/main/wevisone-detect.py"
# The released checkpoint's own system prompt and detection template
# (wevisone_vl/engine.py and wevisone_vl/prompts.py in Tencent/WeVisOne-VL).
SYSTEM_PROMPT = "You are a helpful assistant."
# Answers look like <ref>cat</ref><box><12><40><380><990>,<500><42><880><991></box>; coordinates
# are bins on a 0-1000 grid. A truncated answer may lose its closing tag. Absent classes come back
# as <box>none</box>, which yields no boxes. Follows evaluation/utils/parser.py in the same repo,
# except that the next <ref> is a lookahead, so an answer missing a closing tag does not swallow
# the following class.
REF_RE = re.compile(
    r"<ref>(.*?)</ref>\s*<(box|point|polygon)>(.*?)(?:</(?:box|point|polygon)>|(?=<ref>)|$)",
    re.DOTALL,
)
BIN_RE = re.compile(r"<(\d+)>")

DETECTION_FEATURES = Features(
    {
        "detections": List(
            {
                "bbox": List(Value("int64")),
                "label": Value("string"),
                "sub_label": Value("string"),
                "alt_labels": List(Value("string")),
            }
        ),
        "raw_response": Value("string"),
        "inference_info": Value("string"),
    }
)


def build_prompt(prompt_names: list[str]) -> str:
    """WeVisOne's detection instruction: class phrases joined by '. '."""
    return (
        "Locate all the instances that match the following description: "
        + ". ".join(prompt_names)
        + "."
    )


def to_pil(image: Image.Image | dict | str | None) -> Image.Image | None:
    """Convert HF dataset image entries (PIL / dict-with-bytes / path) to PIL; None stays None."""
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict) and image.get("bytes"):
        return Image.open(io.BytesIO(image["bytes"]))
    if isinstance(image, dict) and image.get("path"):
        return Image.open(image["path"])
    if isinstance(image, str):
        return Image.open(image)
    raise ValueError(f"Unsupported image type: {type(image)}")


def parse_detections(
    text: str, width: int, height: int, label_for: dict[str, str]
) -> list[dict]:
    """Parse <ref>label</ref><box>...</box> answers into pixel-space xyxy boxes."""
    text = text.split("<|im_end|>")[0]
    detections = []
    for phrase, shape, body in REF_RE.findall(text):
        if shape != "box":
            continue
        phrase = phrase.strip().rstrip(".").strip()
        label = label_for.get(phrase.lower(), phrase)
        for segment in body.strip().rstrip(",").split(","):
            values = [int(v) for v in BIN_RE.findall(segment)]
            for k in range(0, len(values) - 3, 4):
                x1, y1, x2, y2 = values[k : k + 4]
                if x2 <= x1 or y2 <= y1:
                    continue
                box = [
                    max(0, min(width, round(x1 / 1000 * width))),
                    max(0, min(height, round(y1 / 1000 * height))),
                    max(0, min(width, round(x2 / 1000 * width))),
                    max(0, min(height, round(y2 / 1000 * height))),
                ]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue  # collapsed to zero width or height after scaling/clamping
                detections.append(
                    {"bbox": box, "label": label, "sub_label": "", "alt_labels": []}
                )
    return detections


def box_iou(a: list[int], b: list[int]) -> float:
    inter_w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def annotate_cross_class_duplicates(detections: list[dict], threshold: float) -> int:
    """Fill each box's `alt_labels` with the other classes the model gave the same region.

    Two boxes are the same region when their IoU is >= threshold. Every box is kept.
    Returns the number of boxes with at least one alternative label.
    """
    if threshold <= 0:
        return 0
    for box in detections:
        for other in detections:
            if (
                other is not box
                and other["label"] != box["label"]
                and other["label"] not in box["alt_labels"]
                and box_iou(box["bbox"], other["bbox"]) >= threshold
            ):
                box["alt_labels"].append(other["label"])
    return sum(1 for box in detections if box["alt_labels"])


def merge_cross_class_duplicates(
    detections: list[dict], class_rank: dict[str, int], threshold: float
) -> tuple[list[dict], int]:
    """Merge boxes of different classes that overlap at IoU >= threshold.

    The class that comes first in --classes keeps the box; the other labels are
    recorded in `alt_labels`. Boxes of the same class are left alone.
    Returns the kept detections and the number of boxes merged away.
    """
    if threshold <= 0:
        return detections, 0
    ordered = sorted(
        detections, key=lambda d: class_rank.get(d["label"], len(class_rank))
    )
    kept: list[dict] = []
    merged = 0
    for detection in ordered:
        duplicate = next(
            (
                k
                for k in kept
                if k["label"] != detection["label"]
                and detection["label"] not in k["alt_labels"]
                and box_iou(k["bbox"], detection["bbox"]) >= threshold
            ),
            None,
        )
        if duplicate is None:
            kept.append(detection)
        else:
            duplicate["alt_labels"].append(detection["label"])
            merged += 1
    return kept, merged


def jobs_flavor() -> str:
    """Return the HF Jobs hardware flavor this run is on, or "" when it cannot be told.

    HF Jobs sets ACCELERATOR to "cpu" or "gpu", not the flavor name, so the flavor is looked up
    in the public Jobs hardware list: first by GPU model and count, then by the CPU_CORES and
    MEMORY values that Jobs sets in the container (to tell a10g-small from a10g-large, or
    cpu-basic from cpu-upgrade). Any failure (no network, older huggingface_hub) returns "".
    """
    hardware = os.environ.get("ACCELERATOR") or ""
    if "-" in hardware or any(character.isdigit() for character in hardware):
        return hardware  # already a flavor name
    try:
        from huggingface_hub import HfApi

        gpu_name, gpu_count = "", 0
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
        except ImportError:
            pass
        if gpu_count == 0 and hardware == "gpu":
            return ""  # a GPU run whose GPU cannot be identified: do not guess a CPU flavor

        # Whole-word match, so "L4" does not match "NVIDIA L40S" ("NVIDIA A100-SXM4-80GB" -> A100).
        gpu_words = gpu_name.upper().replace("-", " ").split()
        candidates = []
        for flavor in HfApi(token=False).list_jobs_hardware():
            accelerator = flavor.accelerator
            if gpu_count == 0 and accelerator is None:
                candidates.append(flavor)
            elif (
                gpu_count > 0
                and accelerator is not None
                and accelerator.model.upper() in gpu_words
                and int(accelerator.quantity) == gpu_count
            ):
                candidates.append(flavor)
        if not candidates:
            return ""

        cores = float(os.environ.get("CPU_CORES") or 0)
        memory_gb = float((os.environ.get("MEMORY") or "0").upper().rstrip("GIB ") or 0)

        def distance(flavor) -> float:
            # Relative gap to the listed vCPUs and RAM; Jobs reports slightly different numbers.
            listed_cores = float(flavor.cpu.split()[0])
            listed_memory = float(flavor.ram.split()[0])
            return (
                abs(listed_cores - cores) / listed_cores
                + abs(listed_memory - memory_gb) / listed_memory
            )

        return min(candidates, key=distance).name
    except Exception:
        return ""


def push_card(
    args: argparse.Namespace, classes: list[str], stats: dict[str, Any]
) -> None:
    """Replace the card body with one carrying the repo's canonical provenance stamp (AGENTS.md).

    The README that push_to_hub wrote is loaded first so its dataset_info/configs metadata
    is kept; only the tags are extended and the body replaced.
    """
    on_jobs = os.environ.get("JOB_ID") is not None  # set by HF Jobs in-container
    hw = jobs_flavor()  # flavor name, e.g. "l4x1"; "" on CPU
    origin = (
        (
            "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
            + (f" (`{hw}`)" if hw else "")
        )
        if on_jobs
        else "Generated"
    )
    tags = (
        ["uv-script", "hf-jobs", "object-detection"]
        if on_jobs
        else ["uv-script", "object-detection"]
    )
    card = DatasetCard.load(args.output_dataset, repo_type="dataset")
    card.data["tags"] = sorted(set(card.data.get("tags") or []) | set(tags))
    card.text = build_card_body(args, classes, stats, origin, hw)
    card.push_to_hub(args.output_dataset)


def build_reproduce_args(args: argparse.Namespace) -> str:
    """Script arguments for the card's reproduce command. Never includes --hf-token."""
    parts = [shlex.quote(args.input_dataset), "<output-dataset>"]
    parts += ["--classes", shlex.quote(args.classes)]
    if args.split != "train":
        parts += ["--split", shlex.quote(args.split)]
    if args.model != DEFAULT_MODEL:
        parts += ["--model", shlex.quote(args.model)]
    if args.image_column != "image":
        parts += ["--image-column", shlex.quote(args.image_column)]
    if args.max_samples:
        parts += ["--max-samples", str(args.max_samples)]
    if args.shuffle:
        parts += ["--shuffle", "--seed", str(args.seed)]
    if args.duplicate_iou != 0.8:
        parts += ["--duplicate-iou", str(args.duplicate_iou)]
    if args.merge_duplicates:
        parts.append("--merge-duplicates")
    if args.max_image_tokens != 2025:
        parts += ["--max-image-tokens", str(args.max_image_tokens)]
    if args.max_model_len != 8192:
        parts += ["--max-model-len", str(args.max_model_len)]
    if args.max_tokens != 4096:
        parts += ["--max-tokens", str(args.max_tokens)]
    if args.private:
        parts.append("--private")
    return " ".join(parts)


def build_card_body(
    args: argparse.Namespace,
    classes: list[str],
    stats: dict[str, Any],
    origin: str,
    hw: str,
) -> str:
    class_list = ", ".join(f"`{c}`" for c in classes)
    if args.merge_duplicates:
        duplicate_note = (
            f"{stats['merged']} boxes were merged into a box of another class at IoU >= "
            f"{args.duplicate_iou} (`--merge-duplicates`; the class listed first was kept)."
        )
    else:
        duplicate_note = (
            f"{stats['with_alt_labels']} boxes share their region with a box of another class "
            "(see `alt_labels`)."
        )
    reproduce_args = build_reproduce_args(args)
    return f"""
# Class-list detection with WeVisOne-VL

Boxes for {class_list} on `{args.input_dataset}` (split `{args.split}`), from
[{args.model}](https://huggingface.co/{args.model}).
{stats["rows"]} images, {stats["rows_with_detections"]} with at least one detection,
{stats["detections"]} detections. {duplicate_note}

These are **zero-shot labels**: no human checked them. Treat them as a first pass for review.

## Columns added

- `detections`: list of `{{bbox: [x1, y1, x2, y2], label, sub_label, alt_labels}}`. `bbox` is in
  original-image pixels. `sub_label` is always empty (WeVisOne gives class labels only); it is
  there so the schema matches `qwen3vl-detect.py`. `alt_labels` lists other classes the model
  gave the same region (IoU >= {args.duplicate_iou}); WeVisOne answers each class on its own,
  so one object can carry several class names.
- `raw_response`: the model's raw answer, for re-parsing.
- `inference_info`: JSON with the model, prompt, classes, image size and settings.

## Reproduction

{origin} with the [`wevisone-detect.py`]({SCRIPT_URL}) recipe from [uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:

```bash
hf jobs uv run {f"--flavor {hw} " if hw else ""}{SCRIPT_URL} \\
    {reproduce_args}
```
"""


def main(args: argparse.Namespace) -> None:
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        sys.exit(
            "--classes is empty. Pass a comma-separated list, e.g. --classes 'cat,dog'."
        )
    with_period = [c for c in classes if "." in c]
    if with_period:
        sys.exit(
            f"Class names cannot contain '.', which the prompt uses to separate classes: "
            f"{with_period}. Rename them (e.g. 'St Bernard')."
        )
    if args.max_image_tokens + args.max_tokens + 256 > args.max_model_len:
        sys.exit(
            f"--max-image-tokens ({args.max_image_tokens}) + --max-tokens ({args.max_tokens}) "
            f"does not fit in --max-model-len ({args.max_model_len}). Raise --max-model-len."
        )
    # A split slice such as 'train[:100]' loads fine but is not a valid name to push to.
    output_split = args.split.split("[")[0]

    import torch

    if not torch.cuda.is_available():
        sys.exit(
            "CUDA is not available. This script needs a GPU (e.g. --flavor a10g-small)."
        )
    logger.info("GPU: %s", torch.cuda.get_device_name(0))
    # The CUDA check above touches the driver in this process, so vLLM's engine process
    # must be spawned, not forked ("Cannot re-initialize CUDA in forked subprocess").
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    if args.hf_token:  # HF_TOKEN in the environment is picked up without a login call
        login(token=args.hf_token)

    prompt_names = [c.replace("_", " ") for c in classes]
    label_for = {p.lower(): c for p, c in zip(prompt_names, classes, strict=True)}
    class_rank = {c: i for i, c in enumerate(classes)}
    prompt = build_prompt(prompt_names)
    logger.info("Prompt: %s", prompt)

    logger.info("Loading dataset: %s (split=%s)", args.input_dataset, args.split)
    dataset = load_dataset(args.input_dataset, split=args.split)
    if args.image_column not in dataset.column_names:
        sys.exit(
            f"Column '{args.image_column}' not found. Columns: {dataset.column_names}"
        )
    if not isinstance(dataset.features[args.image_column], ImageFeature):
        # Paths or URLs would be uploaded as plain strings and the pixels lost after the Job.
        logger.info("Casting column '%s' to Image", args.image_column)
        dataset = dataset.cast_column(args.image_column, ImageFeature())
    clashes = {"detections", "raw_response", "inference_info"} & set(
        dataset.column_names
    )
    if clashes:
        logger.warning("Replacing existing column(s): %s", sorted(clashes))
        dataset = dataset.remove_columns(sorted(clashes))
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    logger.info("Rows to process: %d", len(dataset))

    min_pixels = args.min_image_tokens * 32 * 32
    max_pixels = args.max_image_tokens * 32 * 32
    llm = LLM(
        model=args.model,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": min_pixels, "max_pixels": max_pixels},
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    # skip_special_tokens=False keeps the <ref>/<box>/<N> coordinate tokens in the text.
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=False,
        stop=["<|im_end|>"],
    )
    chat_prompt = processor.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    all_detections: list[list[dict]] = []
    all_raw: list[str] = []
    all_info: list[str] = []
    merged_total = 0
    with_alt_total = 0
    started = datetime.now()
    for batch_indices in tqdm(
        list(partition_all(args.batch_size, range(len(dataset)))),
        desc="WeVisOne detection",
    ):
        images = []
        for i in batch_indices:
            image = to_pil(dataset[i][args.image_column])
            images.append(image.convert("RGB") if image is not None else None)
        requests = [
            {"prompt": chat_prompt, "multi_modal_data": {"image": im}}
            for im in images
            if im is not None
        ]
        generated = iter(llm.generate(requests, sampling_params, use_tqdm=False))
        for image in images:
            if image is None:  # empty image cell: keep the row, with no detections
                all_detections.append([])
                all_raw.append("")
                all_info.append(
                    json.dumps(
                        {
                            "model_id": args.model,
                            "image_size": None,
                            "error": "missing image",
                        }
                    )
                )
                continue
            output = next(generated)
            text = output.outputs[0].text if output.outputs else ""
            detections = parse_detections(text, image.width, image.height, label_for)
            merged = 0
            if args.merge_duplicates:
                detections, merged = merge_cross_class_duplicates(
                    detections, class_rank, args.duplicate_iou
                )
            with_alt = annotate_cross_class_duplicates(detections, args.duplicate_iou)
            merged_total += merged
            with_alt_total += with_alt
            all_detections.append(detections)
            all_raw.append(text)
            all_info.append(
                json.dumps(
                    {
                        "model_id": args.model,
                        "prompt": prompt,
                        "classes": classes,
                        "image_size": [image.width, image.height],
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "duplicate_iou": args.duplicate_iou,
                        "merge_duplicates": args.merge_duplicates,
                        "merged_boxes": merged,
                        "boxes_with_alt_labels": with_alt,
                        "script": "wevisone-detect.py",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                )
            )
    elapsed_min = (datetime.now() - started).total_seconds() / 60
    logger.info("Processed %d images in %.1f min", len(dataset), elapsed_min)

    new_columns = Dataset.from_dict(
        {
            "detections": all_detections,
            "raw_response": all_raw,
            "inference_info": all_info,
        },
        features=DETECTION_FEATURES,
    )
    dataset = concatenate_datasets([dataset, new_columns], axis=1)

    stats = {
        "rows": len(all_detections),
        "rows_with_detections": sum(1 for d in all_detections if d),
        "detections": sum(len(d) for d in all_detections),
        "merged": merged_total,
        "with_alt_labels": with_alt_total,
    }
    logger.info("Pushing to %s", args.output_dataset)
    dataset.push_to_hub(
        args.output_dataset,
        split=output_split,
        private=args.private,
        create_pr=args.create_pr,
        commit_message=f"WeVisOne detection: {args.model} on {len(dataset)} rows",
    )
    if not args.create_pr:  # with --create-pr the card would land on main, not the PR
        try:
            push_card(args, classes, stats)
        except Exception as e:
            logger.warning("Could not push dataset card: %s", e)

    logger.info(
        "Done. Rows with >=1 detection: %d / %d. Detections: %d. "
        "With alt_labels: %d. Merged: %d.",
        stats["rows_with_detections"],
        stats["rows"],
        stats["detections"],
        stats["with_alt_labels"],
        stats["merged"],
    )
    logger.info("Output: https://huggingface.co/datasets/%s", args.output_dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Class-list object detection with WeVisOne-VL via vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dataset", help="Input dataset ID on the Hub")
    parser.add_argument("output_dataset", help="Output dataset ID on the Hub")
    parser.add_argument(
        "--classes",
        required=True,
        help="Comma-separated class names (e.g. 'Coverall,Gloves,Mask'); names cannot contain ',' or '.'. "
        "Order sets priority for --merge-duplicates",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--image-column", default="image", help="Image column (default: image)"
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Cap rows processed"
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle before --max-samples"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Shuffle seed (default: 42)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Images per vLLM call (default: 32)"
    )
    parser.add_argument(
        "--duplicate-iou",
        type=float,
        default=0.8,
        help="IoU at which boxes of different classes count as one region; 0 disables "
        "alt_labels and merging (default: 0.8)",
    )
    parser.add_argument(
        "--merge-duplicates",
        action="store_true",
        help="Fold overlapping boxes of different classes into one box, labelled with the class "
        "listed first in --classes (default: keep every box and fill alt_labels)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Output cap (default: 4096)"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192, help="Context (default: 8192)"
    )
    parser.add_argument(
        "--min-image-tokens",
        type=int,
        default=16,
        help="Lower image-size bound in 32x32 patches (default: 16, the release default)",
    )
    parser.add_argument(
        "--max-image-tokens",
        type=int,
        default=2025,
        help="Upper image-size bound in 32x32 patches (default: 2025, about 2 MP, the release default)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory share (default: 0.85)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Repetition penalty (default: 1.05)",
    )
    parser.add_argument("--hf-token", default=None, help="HF token (or set HF_TOKEN)")
    parser.add_argument(
        "--private", action="store_true", help="Make the output dataset private"
    )
    parser.add_argument(
        "--create-pr", action="store_true", help="Push as a PR to the output repo"
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
