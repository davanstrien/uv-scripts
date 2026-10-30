# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "toolz",
#     "torch",
#     "tqdm",
#     "transformers",
#     "vllm>=0.15.1",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.32.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "a100-large"
# secrets = ["HF_TOKEN"]
# timeout = "1h"
# ///

"""
Instruction-oriented object detection with Qwen3-VL via vLLM.

Takes an HF dataset with an image column, runs each image through Qwen3-VL with
a free-form detection prompt, parses bounding-box JSON from the response, and
pushes a labelled dataset back to the Hub. Designed as a VLM-as-labeller
primitive for bootstrapping object-detection datasets.

The prompt is free-form: "detect all components and identify their reference
designators with bbox_2d, label, sub_label" or "detect every car and its
colour". Qwen3-VL emits bbox JSON on a 0-1000 normalised scale; we extract,
denormalise to original-image pixel coords, and store as ints ready for
downstream labelling tools (Label Studio, FiftyOne, COCO conversion, etc.).

Sibling: `uv-scripts/sam3/detect-objects.py` does class-prompted detection.
This script is the instruction-prompted counterpart.

Output columns added:
- detections: list[{bbox: [x1,y1,x2,y2] in ORIGINAL-IMAGE PIXELS, label, sub_label}]
- raw_response: full model text (for QA, re-parsing, audit)
- inference_info: JSON with model, prompt, image_size, min/max pixels, timestamp
"""

import argparse
import io
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Optional, Union

import torch
from datasets import load_dataset
from huggingface_hub import login
from PIL import Image
from toolz import partition_all
from tqdm.auto import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"

# Qwen-VL bbox prompt template — used when no --prompt is supplied. The
# "bbox_2d / label / sub_label" schema matches the original demo (app.py) so
# the same prompts work here.
DEFAULT_PROMPT = (
    "Detect every distinct object in the image. For each object, output a JSON "
    "object with keys: bbox_2d (an array of four numbers [x1, y1, x2, y2]), "
    "label (the object category), and sub_label (a short descriptive attribute "
    'or "" if none applies). Return a JSON array of these objects. Example: '
    '[{"bbox_2d": [x1, y1, x2, y2], "label": "car", "sub_label": "red"}].'
)


def eval_pixel_string(s: str) -> int:
    """Evaluate a pixel expression like '1024*32*32' to an integer.

    Lifted from app.py:284-289. Lets users write `--min-image-tokens 1024`
    and have it become 1024*32*32 pixels (Qwen-VL uses 32x32 patches).
    """
    try:
        return int(eval(s.strip().replace(" ", ""), {"__builtins__": {}}))
    except Exception as e:
        raise ValueError(f"Invalid pixel expression: {s}") from e


def to_pil(image: Union[Image.Image, dict, str]) -> Image.Image:
    """Convert HF dataset image entries (PIL / dict-with-bytes / path) to PIL."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict) and "bytes" in image:
        return Image.open(io.BytesIO(image["bytes"]))
    if isinstance(image, str):
        return Image.open(image)
    raise ValueError(f"Unsupported image type: {type(image)}")


def parse_bboxes_from_response(text: str) -> list[dict[str, Any]]:
    """Extract bbox objects from a model response.

    Lifted from app.py:147-194. Tolerates fenced code blocks, comments,
    trailing commas, and malformed JSON. Falls back to regex bbox extraction
    if no parseable objects are found.
    """
    results: list[dict[str, Any]] = []

    pattern = r'\{[^{}]*"bbox_2d"\s*:\s*\[[\d\s.,\-]+\][^{}]*\}'
    for match in re.findall(pattern, text, re.DOTALL):
        try:
            obj = json.loads(match)
            if "bbox_2d" in obj:
                results.append(obj)
                continue
        except json.JSONDecodeError:
            pass
        try:
            cleaned = re.sub(r"#.*$", "", match, flags=re.MULTILINE).strip()
            cleaned = re.sub(r",\s*}", "}", cleaned)
            cleaned = re.sub(r",\s*\]", "]", cleaned)
            obj = json.loads(cleaned)
            if "bbox_2d" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    if not results:
        bbox_pattern = (
            r'"bbox_2d"\s*:\s*\[\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*,\s*'
            r"([\d.\-]+)\s*,\s*([\d.\-]+)\s*\]"
        )
        for i, (x1, y1, x2, y2) in enumerate(re.findall(bbox_pattern, text)):
            results.append(
                {
                    "bbox_2d": [float(x1), float(y1), float(x2), float(y2)],
                    "label": f"object_{i + 1}",
                    "sub_label": "",
                }
            )

    return results


def denormalize_bbox(bbox: list[float], width: int, height: int) -> list[int]:
    """Convert a Qwen-VL 0-1000 normalised bbox to original-image pixel coords.

    Empirically confirmed (smoke runs on cppe-5 and Beyond Words, May 2026): the
    instruction-tuned Qwen3-VL family emits `bbox_2d` values in the 0-1000
    normalised space regardless of input image size or smart-resize behaviour.
    Multiply by W/1000 and H/1000 to recover pixel coords on the original image,
    then round to ints and clip to image bounds for safety.
    """
    if len(bbox) != 4:
        return []
    sx, sy = width / 1000.0, height / 1000.0
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(width, round(x1 * sx))),
        max(0, min(height, round(y1 * sy))),
        max(0, min(width, round(x2 * sx))),
        max(0, min(height, round(y2 * sy))),
    ]


def normalise_detection(obj: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Coerce a parsed bbox dict to the canonical detection schema.

    Output: {bbox: list[int] length 4 in ORIGINAL-IMAGE PIXELS, label, sub_label}
    """
    raw = obj.get("bbox_2d", [])
    raw = [float(x) for x in raw] if isinstance(raw, list) and len(raw) == 4 else []
    return {
        "bbox": denormalize_bbox(raw, width, height),
        "label": str(obj.get("label", "")),
        "sub_label": str(obj.get("sub_label") or ""),
    }


def main(
    input_dataset: str,
    output_dataset: str,
    prompt: str,
    image_column: str = "image",
    model: str = DEFAULT_MODEL,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    split: str = "train",
    # Defaults below assume Qwen3.6-35B-A3B on a100-large (80 GB) via the
    # vllm/vllm-openai image. For smaller GPUs use Qwen/Qwen3.5-9B and lower
    # budgets, e.g.:
    #   --max-model-len 12288 --max-image-tokens 4096 --gpu-memory-utilization 0.92
    max_model_len: int = 32768,
    max_tokens: int = 8192,
    min_image_tokens: int = 1024,
    max_image_tokens: int = 9800,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: Optional[int] = None,
    temperature: float = 0.0,
    repetition_penalty: float = 1.05,
    grayscale: bool = False,
    hf_token: Optional[str] = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    create_pr: bool = False,
) -> None:
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("For cloud execution: hf jobs uv run --flavor a100-large ...")
        sys.exit(1)
    logger.info("CUDA OK — GPU: %s", torch.cuda.get_device_name(0))

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    start_time = datetime.now()

    min_pixels = eval_pixel_string(str(min_image_tokens)) * 32 * 32
    max_pixels = eval_pixel_string(str(max_image_tokens)) * 32 * 32
    logger.info(
        "Pixel budget: min=%d (%d tokens), max=%d (%d tokens)",
        min_pixels,
        min_image_tokens,
        max_pixels,
        max_image_tokens,
    )

    logger.info("Loading dataset: %s (split=%s)", input_dataset, split)
    dataset = load_dataset(input_dataset, split=split)

    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    if shuffle:
        logger.info("Shuffling dataset (seed=%d)", seed)
        dataset = dataset.shuffle(seed=seed)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info("Limited to %d samples", len(dataset))

    if tensor_parallel_size is None:
        tensor_parallel_size = torch.cuda.device_count()
        logger.info(
            "Auto-detected %d GPU(s) for tensor parallelism", tensor_parallel_size
        )

    logger.info("Loading vLLM model: %s", model)
    # Cap max_num_seqs to our batch_size: hybrid-Mamba architectures (Qwen3.6's
    # qwen3_5_moe uses Gated DeltaNet) allocate per-sequence SSM cache blocks
    # separately from KV cache; vLLM's default 256 can exceed available blocks
    # and crash CUDA graph capture. We never decode > batch_size concurrently.
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "limit_mm_per_prompt": {"image": 1},
        "mm_processor_kwargs": {
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
        "max_num_seqs": max(batch_size, 8),
    }
    if max_model_len:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)

    # repetition_penalty > 1.0 prevents the catastrophic loop where the model
    # emits the same detection JSON object hundreds of times until it hits the
    # token cap (observed on Qwen3-VL-30B-A3B-Instruct on cppe-5 row 3/5).
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )
    logger.info(
        "Sampling: temperature=%.2f  repetition_penalty=%.2f  max_tokens=%d",
        temperature,
        repetition_penalty,
        max_tokens,
    )

    logger.info("Processing %d images in batches of %d", len(dataset), batch_size)
    logger.info("Prompt: %s", prompt[:200] + ("..." if len(prompt) > 200 else ""))

    all_detections: list[list[dict[str, Any]]] = []
    all_raw: list[str] = []
    all_sizes: list[list[int]] = []

    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="Qwen3-VL detection",
    ):
        batch_indices = list(batch_indices)
        batch_inputs: list[dict[str, Any]] = []
        batch_sizes: list[list[int]] = []

        for idx in batch_indices:
            pil_img = to_pil(dataset[idx][image_column]).convert("RGB")
            if grayscale:
                # Convert to luminance and replicate to 3 channels; useful for
                # discoloured/sepia historical scans where the colour channel
                # is misleading noise rather than signal.
                pil_img = pil_img.convert("L").convert("RGB")
            batch_sizes.append([pil_img.width, pil_img.height])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            templated_prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            batch_inputs.append(
                {
                    "prompt": templated_prompt,
                    "multi_modal_data": {"image": pil_img},
                }
            )

        try:
            outputs = llm.generate(batch_inputs, sampling_params, use_tqdm=False)
        except Exception as e:
            logger.error("Batch failed: %s", e)
            for size in batch_sizes:
                all_detections.append([])
                all_raw.append(f"[ERROR] {e}")
                all_sizes.append(size)
            continue

        for output, size in zip(outputs, batch_sizes):
            text = output.outputs[0].text if output.outputs else ""
            parsed = parse_bboxes_from_response(text)
            W, H = size
            all_detections.append([normalise_detection(o, W, H) for o in parsed])
            all_raw.append(text)
            all_sizes.append(size)

    processing_duration = datetime.now() - start_time
    logger.info(
        "Processing complete in %.1f min", processing_duration.total_seconds() / 60
    )

    # Per-row inference_info so the image_size travels with the row (bbox
    # coordinate-frame work in v2 needs this).
    inference_info_rows: list[str] = []
    for size in all_sizes:
        inference_info_rows.append(
            json.dumps(
                {
                    "model_id": model,
                    "prompt": prompt,
                    "image_size": size,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "script": "qwen3vl-detect.py",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
        )

    logger.info("Adding columns to dataset")
    dataset = dataset.add_column("detections", all_detections)
    dataset = dataset.add_column("raw_response", all_raw)
    dataset = dataset.add_column("inference_info", inference_info_rows)

    logger.info("Pushing to %s", output_dataset)
    dataset.push_to_hub(
        output_dataset,
        private=private,
        token=HF_TOKEN,
        create_pr=create_pr,
        commit_message=(f"Qwen3-VL detection: {model} on {len(dataset)} samples"),
    )

    n_with_dets = sum(1 for d in all_detections if d)
    total_dets = sum(len(d) for d in all_detections)
    logger.info("Done.")
    logger.info("  Rows with >=1 detection: %d / %d", n_with_dets, len(all_detections))
    logger.info("  Total detections: %d", total_dets)
    logger.info("  Output: https://huggingface.co/datasets/%s", output_dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Instruction-oriented object detection with Qwen3-VL (vLLM batch).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generic detection
  uv run qwen3vl-detect.py \\
      input-dataset username/output-dataset \\
      --prompt "Detect all visible objects, return JSON with bbox_2d, label, sub_label"

  # Component detection (matches the demo's example 1)
  uv run qwen3vl-detect.py \\
      pcb-images username/labelled-pcbs \\
      --prompt-file prompts/components.txt \\
      --max-samples 20

  # HF Jobs (local script upload)
  hf jobs uv run --flavor a100-large \\
      -s HF_TOKEN \\
      ./qwen3vl-detect.py \\
      input-dataset username/output-dataset \\
      --prompt "detect all photographs" --max-samples 5

Notes:
  Default --model is Qwen/Qwen3.6-35B-A3B (needs A100/80GB via
  vllm/vllm-openai:v0.32.0 image). For smaller GPUs use Qwen/Qwen3.5-9B
  and lower --max-model-len / --max-image-tokens.
        """,
    )
    parser.add_argument("input_dataset", help="Input dataset ID on the Hub")
    parser.add_argument("output_dataset", help="Output dataset ID on the Hub")

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        default=None,
        help="Detection instruction (free-form). Defaults to a generic JSON-bbox prompt.",
    )
    prompt_group.add_argument(
        "--prompt-file",
        default=None,
        help="Path to a text file containing the detection prompt.",
    )

    parser.add_argument(
        "--image-column", default="image", help="Image column name (default: image)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"VLM (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Cap rows processed (for testing)"
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="vLLM max_model_len (default: 32768, tuned for A100/80GB). "
        "Lower to ~12288 on L4-class GPUs.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max generation tokens (default: 8192)",
    )
    parser.add_argument(
        "--min-image-tokens",
        type=int,
        default=1024,
        help="Min image tokens (32x32 patches); default 1024",
    )
    parser.add_argument(
        "--max-image-tokens",
        type=int,
        default=9800,
        help="Max image tokens (32x32 patches); default 9800 (A100/80GB). "
        "Lower to ~4096 on L4-class GPUs.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilisation (default: 0.90)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor-parallel GPUs (default: auto)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="vLLM repetition_penalty (default: 1.05). Prevents the duplicate-detection loop "
        "observed on Qwen3-VL-30B-A3B. Set to 1.0 to disable.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert each image to greyscale (L channel replicated to RGB) before "
        "inference. Useful for sepia/discoloured historical scans where the colour "
        "channel is noise rather than signal.",
    )
    parser.add_argument(
        "--hf-token", default=None, help="HF token (or set HF_TOKEN env)"
    )
    parser.add_argument(
        "--private", action="store_true", help="Push output dataset as private"
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle before processing"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Shuffle seed (default: 42)"
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Push as PR instead of direct commit (useful for parallel runs)",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        print("\n" + "=" * 60)
        print("Example HF Jobs command:")
        print("=" * 60)
        print(
            """
hf jobs uv run \\
    --image vllm/vllm-openai:v0.32.0 \\
    --flavor a100-large \\
    --python /usr/bin/python3 \\
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \\
    -s HF_TOKEN \\
    ./qwen3vl-detect.py \\
    INPUT_DATASET OUTPUT_DATASET \\
    --prompt "detect all visible objects" \\
    --max-samples 5
            """
        )
        sys.exit(0)

    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = DEFAULT_PROMPT

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        prompt=prompt,
        image_column=args.image_column,
        model=args.model,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        split=args.split,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        min_image_tokens=args.min_image_tokens,
        max_image_tokens=args.max_image_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        grayscale=args.grayscale,
        hf_token=args.hf_token,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        create_pr=args.create_pr,
    )
