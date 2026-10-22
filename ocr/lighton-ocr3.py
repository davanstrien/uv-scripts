# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "toolz",
#     # LightOn's official client, pinned to a commit (not on PyPI). Used for its
#     # parser, page resize and dollar-delimiter repair so outputs match upstream.
#     "lightonocr @ https://github.com/lightonai/LightOnOCR/archive/36755d461be079737860a5f03ae0c803501269e9.tar.gz",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.30.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "a10g-small"
# secrets = ["HF_TOKEN"]
# ///

"""
Convert document images to markdown (or layout blocks) using LightOnOCR-3 with vLLM.

LightOnOCR-3 is LightOn's third generation of end-to-end OCR models. The 4B and 0.8B
variants use the stock Qwen3.5 vision-language architecture. Besides plain
transcription, they have a grounding mode: every block of the page comes back with a
label (title, text, table, chart, image, ...) and a bounding box. Images get a short
description and charts become an HTML table of their data points.

Models (Apache-2.0):
  lightonai/LightOnOCR-3-4B    (default)  86.3 olmOCR-Bench, 75.1 ParseBench
  lightonai/LightOnOCR-3-0.8B             85.5 olmOCR-Bench, 74.6 ParseBench
LightOnOCR-3-1B is not supported here: it keeps the LightOnOCR-2 (Pixtral)
architecture and needs a different vLLM/transformers pin.

Run on HF Jobs (needs `hf` CLI 1.32+; the [tool.hf-jobs] header above sets the
vllm/vllm-openai:v0.30.0 image, a10g-small flavor and HF_TOKEN secret):

    hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr3.py \\
        input-dataset output-dataset --max-samples 10

    # Layout blocks with labels and boxes, smaller model
    hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr3.py \\
        input-dataset output-dataset --mode grounding --model lightonai/LightOnOCR-3-0.8B

Run on your own GPU (vLLM and torch come from the image on Jobs, so add them here):

    uv run --with vllm==0.30.0 lighton-ocr3.py input-dataset output-dataset

The image is pinned to vLLM 0.30.0 because LightOn's client reports that vLLM 0.27.1
makes the models loop and 0.30.0 is the version they test. Loosen the pin once a newer
vLLM is checked against the same outputs.

Modes:
- plain (default): the image alone -> markdown transcription. Same as LightOnOCR-2.
- grounding: the image + the text "grounding" -> one `![label](x1,y1,x2,y2)` block per
  region, coordinates normalised to 0-1000. The output column gets the block texts
  joined as markdown; the --blocks-column gets the parsed blocks as JSON
  ([{label, bbox, text, continues}, ...]). Grounding adds about 25% output tokens.
The card says other instructions are out of distribution, so there is no custom prompt.

Sampling matches the model card's vLLM example: temperature 0.2, top_p 0.9,
max_tokens 4096, thinking disabled. (LightOn's client defaults to top_p 1.0 for
LightOnOCR-3; pass --top-p 1.0 to match it.) Pre- and post-processing follow the
official client (lightonocr/render.py, client.py): pages are downscaled to 2048 px on
the longest edge (the card's rendering tip; the processor does no further resizing),
and escaped dollar delimiters are repaired (`\\$x\\$` -> `$x$`) on every output.
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
from lightonocr import (
    DEFAULT_LONGEST_EDGE,
    GROUNDING_PROMPT,
    fit,
    fix_escaped_dollars,
    parse_blocks,
    to_markdown,
)
from PIL import Image
from toolz import partition_all

# Disable vLLM's FlashInfer sampler: it JIT-compiles a CUDA kernel needing nvcc, which the
# default uv-script image lacks (engine init then crashes). On the vllm/vllm-openai image it's
# a harmless no-op.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# DeepGEMM's init asserts on a missing nvcc (non-fatal traceback); OCR doesn't need it.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = ["lightonai/LightOnOCR-3-4B", "lightonai/LightOnOCR-3-0.8B"]
DEFAULT_MODEL = MODELS[0]

# Sampling from the model card's vLLM example (temperature 0.2, top_p 0.9, max_tokens 4096).
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 4096


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


def build_prompt(llm: LLM, mode: str) -> str:
    """Chat-template prompt for one page, with thinking disabled.

    plain sends the image alone; grounding adds the text "grounding" after it
    (client.py). The 0.8B and 4B were trained without thinking, so the template is
    rendered with enable_thinking=False, as the card and client do.
    """
    content = [{"type": "image"}]
    if mode == "grounding":
        content.append({"type": "text", "text": GROUNDING_PROMPT})
    return llm.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def postprocess_output(text: str, mode: str, fix_dollars: bool):
    """Return (markdown, blocks_json or None) for one raw model output."""
    text = text.strip()
    if fix_dollars:
        text = fix_escaped_dollars(text)
    if mode != "grounding":
        return text, None
    blocks = parse_blocks(text)
    if not blocks:
        # No markers parsed: keep the raw text so nothing is lost.
        return text, json.dumps([])
    markdown = to_markdown(blocks)
    return markdown, json.dumps([b.to_dict() for b in blocks], ensure_ascii=False)


def create_dataset_card(
    source_dataset: str,
    model: str,
    mode: str,
    num_samples: int,
    processing_time: str,
    batch_size: int,
    max_model_len: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    longest_edge: int,
    gpu_memory_utilization: float,
    output_column: str,
    blocks_column: str,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    # Canonical provenance stamp (see AGENTS.md): Jobs claim gated on JOB_ID, set by HF Jobs in-container.
    on_jobs = os.environ.get("JOB_ID") is not None
    hw = os.environ.get("ACCELERATOR") or ""
    origin = (
        "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
        + (f" (`{hw}`)" if hw else "")
    ) if on_jobs else "Generated"
    jobs_tag = "\n- hf-jobs" if on_jobs else ""

    model_arg = f" \\\n    --model {model}" if model != DEFAULT_MODEL else ""
    mode_arg = " \\\n    --mode grounding" if mode == "grounding" else ""
    blocks_line = (
        f"- `{blocks_column}`: JSON list of layout blocks "
        "(`label`, `bbox` as x1,y1,x2,y2 normalised to 0-1000, `text`, `continues`)\n"
        if mode == "grounding"
        else ""
    )

    return f"""---
tags:
- ocr
- document-processing
- lighton-ocr-3
- markdown
- uv-script
- generated{jobs_tag}
---

# Document OCR using {model_name}

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using [{model_name}](https://huggingface.co/{model}) in **{mode}** mode.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Mode**: `{mode}`
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Dataset Split**: `{split}`
- **Batch Size**: {batch_size}
- **Longest Image Edge**: {longest_edge}px
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **Temperature**: {temperature}
- **Top P**: {top_p}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Dataset Structure

The dataset contains all original columns plus:
- `{output_column}`: The extracted text in markdown format
{blocks_line}- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Reproduction

{origin} with the [`lighton-ocr3.py`](https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr3.py) recipe from [uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:

```bash
hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr3.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --batch-size {batch_size}{model_arg}{mode_arg}
```
"""


def main(
    input_dataset: str,
    output_dataset: str,
    model: str = DEFAULT_MODEL,
    mode: str = "plain",
    image_column: str = "image",
    batch_size: int = 16,
    max_model_len: int = 16384,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    longest_edge: int = DEFAULT_LONGEST_EDGE,
    no_fix_dollars: bool = False,
    gpu_memory_utilization: float = 0.8,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    output_column: str = "markdown",
    blocks_column: str = "layout_blocks",
    overwrite: bool = False,
    verbose: bool = False,
    config: str = None,
    create_pr: bool = False,
):
    """Process images from HF dataset through LightOnOCR-3."""

    check_cuda_availability()

    start_time = datetime.now()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    logger.info(f"Using model: {model} ({mode} mode)")

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)

    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    # Fail fast if an output column would collide with an existing input column
    output_columns = [output_column] + ([blocks_column] if mode == "grounding" else [])
    dataset = ensure_output_columns_free(dataset, output_columns, overwrite=overwrite)

    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Initialize vLLM
    logger.info(f"Initializing vLLM with {model}")
    logger.info("This may take a few minutes on first run...")
    # Stock Qwen3_5ForConditionalGeneration arch, so no trust_remote_code. A 2048px page
    # is at most 64x64 = 4,096 image tokens (16px patches, 2x2 merge), so 16384 leaves
    # room for the card's 4096 output tokens. gdn_prefill_backend="triton" avoids the
    # Qwen3.5 gated-delta-net prefill path that needs a CUDA JIT toolchain (as ovis-ocr2.py).
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        gdn_prefill_backend="triton",
    )

    prompt = build_prompt(llm, mode)
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Output will be written to column: {output_column}")
    if mode == "grounding":
        logger.info(f"Layout blocks will be written to column: {blocks_column}")

    all_markdown = []
    all_blocks = []
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
            # Downscale to the training render size (never upscale), as the client does.
            vllm_inputs = [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": fit(to_pil_image(img), longest_edge)},
                }
                for img in batch_images
            ]

            outputs = llm.generate(vllm_inputs, sampling_params)

            for output in outputs:
                markdown, blocks = postprocess_output(
                    output.outputs[0].text, mode, fix_dollars=not no_fix_dollars
                )
                all_markdown.append(markdown)
                all_blocks.append(blocks)

            processed += len(batch_images)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            all_markdown.extend(["[OCR ERROR]"] * len(batch_images))
            all_blocks.extend([json.dumps([])] * len(batch_images))
            processed += len(batch_images)

    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_markdown)
    if mode == "grounding":
        logger.info(f"Adding '{blocks_column}' column to dataset")
        dataset = dataset.add_column(blocks_column, all_blocks)

    # Inference info tracking
    inference_entry = {
        "model_id": model,
        "model_name": model.split("/")[-1],
        "mode": mode,
        "column_name": output_column,
        "blocks_column": blocks_column if mode == "grounding" else None,
        "timestamp": datetime.now().isoformat(),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "longest_edge": longest_edge,
        "fix_dollars": not no_fix_dollars,
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
                commit_message=f"Add {model} OCR results ({len(dataset)} samples)"
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
        model=model,
        mode=mode,
        num_samples=len(dataset),
        processing_time=processing_time_str,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        longest_edge=longest_edge,
        gpu_memory_utilization=gpu_memory_utilization,
        output_column=output_column,
        blocks_column=blocks_column,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("Done! LightOnOCR-3 processing complete.")
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
        for pkg in ["vllm", "transformers", "torch", "datasets", "pyarrow", "pillow", "lightonocr"]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")
        logger.info("--- End versions ---")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("LightOnOCR-3 Document Processing")
        print("=" * 70)
        print("\n4B (default) or 0.8B end-to-end OCR, Qwen3.5 architecture")
        print("\nModes:")
        print("  plain      markdown transcription (default)")
        print("  grounding  labelled layout blocks with 0-1000 boxes")
        print("\nExamples:")
        print("\n1. Basic OCR (own GPU):")
        print("   uv run --with vllm==0.30.0 lighton-ocr3.py input-dataset output-dataset")
        print("\n2. Layout blocks with the 0.8B model:")
        print("   uv run --with vllm==0.30.0 lighton-ocr3.py docs results \\")
        print("       --mode grounding --model lightonai/LightOnOCR-3-0.8B")
        print("\n3. Running on HF Jobs (hf CLI 1.32+; image/flavor/secrets from the script header):")
        print("   hf jobs uv run \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr3.py \\"
        )
        print("       input-dataset output-dataset --max-samples 10")
        print("\nFor full help: uv run --with vllm==0.30.0 lighton-ocr3.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using LightOnOCR-3 (4B or 0.8B, plain or grounding mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run --with vllm==0.30.0 lighton-ocr3.py my-docs analyzed-docs
  uv run --with vllm==0.30.0 lighton-ocr3.py docs results --mode grounding
  uv run --with vllm==0.30.0 lighton-ocr3.py docs results --model lightonai/LightOnOCR-3-0.8B
        """,
    )

    parser.add_argument("input_dataset", help="Input dataset ID from Hugging Face Hub")
    parser.add_argument("output_dataset", help="Output dataset ID for Hugging Face Hub")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=MODELS,
        help=f"Model checkpoint (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--mode",
        default="plain",
        choices=["plain", "grounding"],
        help="plain = markdown transcription; grounding = labelled layout blocks "
        "with boxes, parsed into --blocks-column (default: plain)",
    )
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
        default=16384,
        help="Maximum model context length (default: 16384; model supports 262144). "
        "A 2048px page is at most 4,096 image tokens.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS}, the card's "
        "vLLM example). Raise it for very dense pages in grounding mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}, model card)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Top-p (default: {DEFAULT_TOP_P}, model card; LightOn's client uses 1.0)",
    )
    parser.add_argument(
        "--longest-edge",
        type=int,
        default=DEFAULT_LONGEST_EDGE,
        help=f"Downscale pages so the longest edge is at most this many pixels "
        f"(default: {DEFAULT_LONGEST_EDGE}, the training render size)",
    )
    parser.add_argument(
        "--no-fix-dollars",
        action="store_true",
        help="Keep the raw model output instead of repairing escaped dollar "
        "delimiters (the official client repairs them by default)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
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
        "--blocks-column",
        default="layout_blocks",
        help="Column name for parsed layout blocks in grounding mode (default: layout_blocks)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column(s) if they already exist in the input dataset "
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
        model=args.model,
        mode=args.mode,
        image_column=args.image_column,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        longest_edge=args.longest_edge,
        no_fix_dollars=args.no_fix_dollars,
        gpu_memory_utilization=args.gpu_memory_utilization,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        output_column=args.output_column,
        blocks_column=args.blocks_column,
        overwrite=args.overwrite,
        verbose=args.verbose,
        config=args.config,
        create_pr=args.create_pr,
    )
