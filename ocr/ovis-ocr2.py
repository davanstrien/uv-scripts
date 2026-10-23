# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "toolz",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.29.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "a10g-small"
# secrets = ["HF_TOKEN"]
# ///

"""
Convert document images to markdown using OvisOCR2 with vLLM.

OvisOCR2 is a compact 0.9B end-to-end document parsing model (post-trained from
Qwen3.5-0.8B with SFT + RL + OPD). It scores 96.58 on OmniDocBench v1.6 — the
first end-to-end model to top that leaderboard — and 75.06 Avg3 on PureDocBench.
Outputs a single Markdown document in natural reading order: LaTeX formulas,
HTML tables, and (optionally) HTML <img> tags marking chart/image regions with
bounding boxes scaled to [0, 1000).

Model: ATH-MaaS/OvisOCR2 (Apache-2.0)
vLLM: stock Qwen3_5ForConditionalGeneration arch, in stable vLLM >= 0.22.1
      (the version the model card installs); no trust_remote_code needed.

Run on HF Jobs (needs `hf` CLI 1.32+; the [tool.hf-jobs] header above sets the
vllm/vllm-openai:v0.29.0 image, a10g-small flavor and HF_TOKEN secret):

    hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ovis-ocr2.py \\
        input-dataset output-dataset --batch-size 16

Run on your own GPU (vLLM and torch come from the image on Jobs, so add them here):

    uv run --with vllm==0.29.0 ovis-ocr2.py input-dataset output-dataset

Features:
- 0.9B parameters (ultra-compact, runs on l4x1)
- Markdown output with LaTeX formulas + HTML tables
- Visual-region <img> tags filtered by default (upstream parser default);
  keep them with --keep-image-tags for downstream crop extraction
- Upstream trailing-repeat cleanup applied to each output
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Union

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
from toolz import partition_all

# Disable vLLM's FlashInfer sampler: it JIT-compiles a CUDA kernel needing nvcc, which the
# default uv-script image lacks (engine init then crashes). Greedy OCR doesn't use it; this
# lets the plain default-image command work. On the vllm/vllm-openai image it's a harmless no-op.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# DeepGEMM's init calls _find_cuda_home, which asserts on the nvcc-less base image — a
# non-fatal warning traceback that clutters the log (seen in the 2026-07-14 smoke run on
# vLLM 0.25.1). Greedy OCR doesn't need the DeepGEMM JIT path, so disable it explicitly.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = "ATH-MaaS/OvisOCR2"

# Fixed instruction prompt, verbatim from the model card (including the leading newline).
# The card warns outputs are tuned to this exact wording — don't "improve" it.
OCR_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    'represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_'
    '{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box '
    "coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as "
    "HTML: <table>...</table>. Transcribe all other text as standard Markdown. "
    "Preserve the original text without translation or paraphrasing."
)

# Image bounds from the model card's parser (passed per-request as images_kwargs).
DEFAULT_MIN_PIXELS = 448 * 448  # 200,704
DEFAULT_MAX_PIXELS = 2880 * 2880  # 8,294,400


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with a CUDA-capable GPU.")
        sys.exit(1)
    else:
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def ensure_output_columns_free(dataset, columns, overwrite=False):
    """Fail fast if an output column would collide with an existing input column.

    Adding a column that already exists silently overwrites it (e.g. a ground-truth
    `text`/`markdown` column) or crashes on push with a duplicate-column error only
    *after* inference has run. Catch it up front. With overwrite=True, drop the clashing
    column(s) here instead (logged) so the later add_column is clean.
    """
    clash = [c for c in columns if c in dataset.column_names]
    if not clash:
        return dataset
    if overwrite:
        logger.warning(f"--overwrite: replacing existing column(s) {clash}")
        return dataset.remove_columns(clash)
    logger.error(
        f"Output column(s) {clash} already exist in the input dataset "
        f"(columns: {dataset.column_names})."
    )
    logger.error("Choose a different --output-column, or pass --overwrite to replace them.")
    sys.exit(1)


def to_pil_image(image: Union[Image.Image, Dict[str, Any], str]) -> Image.Image:
    """Convert a dataset image cell (PIL image, bytes dict, or path) to RGB PIL."""
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    return pil_img.convert("RGB")


def clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """Trim degenerate trailing repetition (verbatim port of the model card's cleanup).

    Long outputs that hit max_tokens can end in a repeated unit (a char, phrase, or
    table row); this detects the shortest repeating tail unit and keeps one copy.
    """
    n = len(text)
    if n < min_text_len:
        return text

    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue

        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1

        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len

        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len :]

    return text


def filter_image_tags(text: str) -> str:
    """Drop visual-region <img> blocks (upstream parser's default behaviour)."""
    return "\n\n".join(
        block
        for block in text.split("\n\n")
        if not block.strip().startswith('<img src="images/bbox_')
    )


def postprocess_output(text: str, keep_image_tags: bool) -> str:
    text = text.strip()
    if not keep_image_tags:
        text = filter_image_tags(text)
    return clean_truncated_repeats(text)


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
            # Torch-free drivers (e.g. the -server recipes): ask nvidia-smi instead.
            import subprocess

            try:
                listed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10, check=True,
                ).stdout
                names = [line.strip() for line in listed.splitlines() if line.strip()]
                if names:
                    gpu_name, gpu_count = names[0], len(names)
            except (OSError, subprocess.SubprocessError):
                pass
        if gpu_count == 0 and hardware == "gpu":
            return ""  # a GPU run whose GPU cannot be identified: do not guess a CPU flavor

        candidates = []
        for flavor in HfApi(token=False).list_jobs_hardware():
            accelerator = flavor.accelerator
            if gpu_count == 0 and accelerator is None:
                candidates.append(flavor)
            elif (
                gpu_count > 0
                and accelerator is not None
                and accelerator.model in gpu_name
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
            return abs(listed_cores - cores) / listed_cores + abs(listed_memory - memory_gb) / listed_memory

        return min(candidates, key=distance).name
    except Exception:
        return ""


def create_dataset_card(
    source_dataset: str,
    model: str,
    num_samples: int,
    processing_time: str,
    batch_size: int,
    max_model_len: int,
    max_tokens: int,
    gpu_memory_utilization: float,
    keep_image_tags: bool,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    # Canonical provenance stamp (see AGENTS.md): Jobs claim gated on JOB_ID, set by HF Jobs in-container.
    on_jobs = os.environ.get("JOB_ID") is not None
    hw = jobs_flavor()  # flavor name, e.g. "a10g-small"; "" on CPU
    origin = (
        "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
        + (f" (`{hw}`)" if hw else "")
    ) if on_jobs else "Generated"
    jobs_tag = "\n- hf-jobs" if on_jobs else ""

    return f"""---
tags:
- ocr
- document-processing
- ovis-ocr2
- markdown
- uv-script
- generated{jobs_tag}
---

# Document OCR using {model_name}

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using OvisOCR2, a compact 0.9B document parsing model (96.58 on OmniDocBench v1.6).

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Dataset Split**: `{split}`
- **Batch Size**: {batch_size}
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **Temperature**: 0.0 (greedy, per model card)
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}
- **Visual-region image tags**: {"kept" if keep_image_tags else "filtered (default)"}

## Model Information

OvisOCR2 is a compact, high-performance document parsing model:
- 0.9B parameters (post-trained from Qwen3.5-0.8B with SFT + RL + OPD)
- 96.58 on OmniDocBench v1.6 (first end-to-end model to top the leaderboard)
- Markdown output in natural reading order
- LaTeX formula recognition, HTML table extraction
- Apache-2.0 licensed

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted text in markdown format
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Reproduction

{origin} with the [`ovis-ocr2.py`](https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ovis-ocr2.py) recipe from [uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:

```bash
hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ovis-ocr2.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --batch-size {batch_size}
```
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    max_model_len: int = 32768,
    max_tokens: int = 16384,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    gpu_memory_utilization: float = 0.8,
    keep_image_tags: bool = False,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    output_column: str = "markdown",
    overwrite: bool = False,
    verbose: bool = False,
    config: str = None,
    create_pr: bool = False,
):
    """Process images from HF dataset through OvisOCR2."""

    check_cuda_availability()

    start_time = datetime.now()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    logger.info(f"Using model: {MODEL}")

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)

    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    # Fail fast if the output column would collide with an existing input column
    dataset = ensure_output_columns_free(dataset, [output_column], overwrite=overwrite)

    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Initialize vLLM
    logger.info("Initializing vLLM with OvisOCR2")
    logger.info("This may take a few minutes on first run...")
    # gdn_prefill_backend="triton" is from the model card: Qwen3.5 uses gated-delta-net
    # linear attention, and the non-triton GDN prefill path needs a JIT/CUDA toolchain
    # the bare uv image lacks (same class of problem as the FLASHINFER guard above).
    llm = LLM(
        model=MODEL,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        gdn_prefill_backend="triton",
    )

    # The model card builds the prompt once via the chat template with
    # enable_thinking=False — Qwen3.5 templates can otherwise inject a thinking
    # preamble, which is wrong for OCR. Mirror it exactly.
    prompt = llm.get_tokenizer().apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Card-locked sampling: greedy, 16384 max tokens.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Output will be written to column: {output_column}")

    all_outputs = []
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    processed = 0

    for batch_num, batch_indices in enumerate(
        partition_all(batch_size, range(len(dataset))), 1
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        logger.info(
            f"Batch {batch_num}/{total_batches} "
            f"({processed}/{len(dataset)} images done)"
        )

        try:
            # Per-request inputs exactly as the model card's parser builds them
            # (PIL image + images_kwargs pixel bounds), via llm.generate not llm.chat
            # so the enable_thinking=False prompt above is used verbatim.
            vllm_inputs = [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": to_pil_image(img)},
                    "mm_processor_kwargs": {
                        "images_kwargs": {
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        }
                    },
                }
                for img in batch_images
            ]

            outputs = llm.generate(vllm_inputs, sampling_params)

            for output in outputs:
                text = output.outputs[0].text
                all_outputs.append(postprocess_output(text, keep_image_tags))

            processed += len(batch_images)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            all_outputs.extend(["[OCR ERROR]"] * len(batch_images))
            processed += len(batch_images)

    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Inference info tracking
    inference_entry = {
        "model_id": MODEL,
        "model_name": "OvisOCR2",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "keep_image_tags": keep_image_tags,
    }

    if "inference_info" in dataset.column_names:
        logger.info("Updating existing inference_info column")

        def update_inference_info(example):
            try:
                existing_info = (
                    json.loads(example["inference_info"])
                    if example["inference_info"]
                    else []
                )
            except (json.JSONDecodeError, TypeError):
                existing_info = []
            existing_info.append(inference_entry)
            return {"inference_info": json.dumps(existing_info)}

        dataset = dataset.map(update_inference_info)
    else:
        logger.info("Creating new inference_info column")
        inference_list = [json.dumps([inference_entry])] * len(dataset)
        dataset = dataset.add_column("inference_info", inference_list)

    # Push to hub with retry and XET fallback
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
                **({"config_name": config} if config else {}),
                create_pr=create_pr,
                commit_message=f"Add {MODEL} OCR results ({len(dataset)} samples)"
                + (f" [{config}]" if config else ""),
            )
            break
        except Exception as e:
            logger.error(f"Upload attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                delay = 30 * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error("All upload attempts failed. OCR results are lost.")
                sys.exit(1)

    # Create and push dataset card
    logger.info("Creating dataset card")
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        model=MODEL,
        num_samples=len(dataset),
        processing_time=processing_time_str,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        keep_image_tags=keep_image_tags,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("Done! OvisOCR2 processing complete.")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")
    logger.info(
        f"Processing speed: {len(dataset) / processing_duration.total_seconds():.2f} images/sec"
    )

    if verbose:
        import importlib.metadata

        logger.info("--- Resolved package versions ---")
        for pkg in ["vllm", "transformers", "torch", "datasets", "pyarrow", "pillow"]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")
        logger.info("--- End versions ---")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("OvisOCR2 Document Processing")
        print("=" * 70)
        print("\n0.9B document parsing model - 96.58 on OmniDocBench v1.6")
        print("\nOutputs markdown in natural reading order:")
        print("  - LaTeX formulas, HTML tables")
        print("  - Visual-region <img> tags filtered by default")
        print("    (--keep-image-tags to retain them)")
        print("\nExamples:")
        print("\n1. Basic OCR (own GPU):")
        print("   uv run --with vllm==0.29.0 ovis-ocr2.py input-dataset output-dataset")
        print("\n2. Keep visual-region image tags:")
        print("   uv run --with vllm==0.29.0 ovis-ocr2.py docs results --keep-image-tags")
        print("\n3. Test with small sample:")
        print("   uv run --with vllm==0.29.0 ovis-ocr2.py large-dataset test --max-samples 10 --shuffle")
        print("\n4. Running on HF Jobs (hf CLI 1.32+; image/flavor/secrets from the script header):")
        print("   hf jobs uv run \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ovis-ocr2.py \\"
        )
        print("       input-dataset output-dataset --batch-size 16")
        print("\nFor full help: uv run --with vllm==0.29.0 ovis-ocr2.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using OvisOCR2 (0.9B, 96.58 OmniDocBench v1.6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run --with vllm==0.29.0 ovis-ocr2.py my-docs analyzed-docs
  uv run --with vllm==0.29.0 ovis-ocr2.py docs results --keep-image-tags
  uv run --with vllm==0.29.0 ovis-ocr2.py large-dataset test --max-samples 50 --shuffle
        """,
    )

    parser.add_argument("input_dataset", help="Input dataset ID from Hugging Face Hub")
    parser.add_argument("output_dataset", help="Output dataset ID for Hugging Face Hub")
    parser.add_argument(
        "--image-column",
        default="image",
        help="Column containing images (default: image)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing (default: 16)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum model context length (default: 32768; model supports 262144)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens to generate (default: 16384, the model card value)",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=DEFAULT_MIN_PIXELS,
        help=f"Minimum image pixels for the processor (default: {DEFAULT_MIN_PIXELS}, "
        "= 448*448, the model card value)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help=f"Maximum image pixels for the processor; larger images are downscaled "
        f"internally (default: {DEFAULT_MAX_PIXELS}, = 2880*2880, the model card value)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
    )
    parser.add_argument(
        "--keep-image-tags",
        action="store_true",
        help="Keep visual-region <img src=\"images/bbox_...\"> tags in the output "
        "(default: filtered, matching the upstream parser)",
    )
    parser.add_argument("--hf-token", help="Hugging Face API token")
    parser.add_argument(
        "--split", default="train", help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make output dataset private"
    )
    parser.add_argument(
        "--config",
        help="Config/subset name when pushing to Hub (for benchmarking multiple models in one repo)",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request instead of pushing directly (for parallel benchmarking)",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle dataset before processing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--output-column",
        default="markdown",
        help="Column name for output text (default: markdown)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column if it already exists in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions after processing (useful for pinning deps)",
    )

    args = parser.parse_args()

    if args.max_tokens > args.max_model_len:
        parser.error(
            f"--max-tokens ({args.max_tokens}) must be <= --max-model-len ({args.max_model_len})"
        )

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        gpu_memory_utilization=args.gpu_memory_utilization,
        keep_image_tags=args.keep_image_tags,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        output_column=args.output_column,
        overwrite=args.overwrite,
        verbose=args.verbose,
        config=args.config,
        create_pr=args.create_pr,
    )
