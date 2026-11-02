# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "tqdm",
#     "toolz",
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
Convert document images to markdown using dots.mocr with vLLM.

dots.mocr is a 3B multilingual document parsing model with SOTA performance
on 100+ languages. It excels at converting structured graphics (charts, UI
layouts, scientific figures) directly into SVG code. Core capabilities include
grounding, recognition, semantic understanding, and interactive dialogue.

Features:
- Multilingual support (100+ languages)
- Table extraction and formatting
- Formula recognition
- Layout-aware text extraction
- Web screen parsing
- Scene text spotting
- SVG code generation (use --prompt-mode svg, or --model rednote-hilab/dots.mocr-svg for best results)

Model: rednote-hilab/dots.mocr
SVG variant: rednote-hilab/dots.mocr-svg
vLLM: Officially integrated since v0.11.0
GitHub: https://github.com/rednote-hilab/dots.mocr
Paper: https://arxiv.org/abs/2603.13032

Run on HF Jobs (the [tool.hf-jobs] header sets image, flavor and secrets;
needs the hf CLI 1.32+):

    hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/dots-mocr.py \
        input-dataset output-dataset

On your own GPU, vLLM is not in the dependencies (the Jobs image provides it):

    uv run --with vllm==0.30.0 dots-mocr.py input-dataset output-dataset
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Union

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image, UnidentifiedImageError
from toolz import partition_all
from tqdm.auto import tqdm
# Disable vLLM's FlashInfer sampler: it JIT-compiles a CUDA kernel needing nvcc, which the
# default uv-script image lacks (engine init then crashes). Greedy OCR doesn't use it; this
# lets the plain default-image command work. On the vllm/vllm-openai image it's a harmless no-op.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# dots.mocr Prompt Templates
# Source: https://github.com/rednote-hilab/dots.mocr/blob/master/dots_mocr/utils/prompts.py
# ────────────────────────────────────────────────────────────────

PROMPT_TEMPLATES = {
    "ocr": """Extract the text content from this image.""",
    "layout-all": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
""",
    # NOTE: Bboxes from layout-all/layout-only are in the resized image coordinate
    # space (Qwen2VLImageProcessor smart_resize: max_pixels=11289600, factor=28),
    # NOT original image coordinates. To map back, compute:
    #   resized_h, resized_w = smart_resize(orig_h, orig_w)
    #   scale_x, scale_y = orig_w / resized_w, orig_h / resized_h
    "layout-only": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",
    "web-parsing": """Parsing the layout info of this webpage image with format json:\n""",
    "scene-spotting": """Detect and recognize the text in the image.""",
    "grounding-ocr": """Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).\nBounding Box:\n""",
    # SVG code generation — {width} and {height} are replaced with actual image dimensions.
    # For best results, use --model rednote-hilab/dots.mocr-svg
    # Uses higher temperature (0.9) and top_p (1.0) per official recommendation.
    "svg": """Please generate the SVG code based on the image. viewBox="0 0 {width} {height}" """,
    "general": """ """,
}


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


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str = PROMPT_TEMPLATES["ocr"],
) -> List[Dict]:
    """Create chat message for OCR processing."""
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Convert to RGB
    pil_img = pil_img.convert("RGB")

    # For SVG mode, inject actual image dimensions into the prompt
    if "{width}" in prompt and "{height}" in prompt:
        prompt = prompt.replace("{width}", str(pil_img.width)).replace(
            "{height}", str(pil_img.height)
        )

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # Return message in vLLM format
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def create_dataset_card(
    source_dataset: str,
    model: str,
    num_samples: int,
    processing_time: str,
    batch_size: int,
    max_model_len: int,
    max_tokens: int,
    gpu_memory_utilization: float,
    image_column: str = "image",
    split: str = "train",
    prompt_mode: str = "ocr",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    return f"""---
tags:
- ocr
- document-processing
- dots-mocr
- multilingual
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using dots.mocr, a 3B multilingual model with SOTA document parsing and SVG generation.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `markdown`
- **Dataset Split**: `{split}`
- **Batch Size**: {batch_size}
- **Prompt Mode**: {prompt_mode}
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

dots.mocr is a 3B multilingual document parsing model that excels at:
- 100+ Languages — Multilingual document support
- Table extraction — Structured data recognition
- Formulas — Mathematical notation preservation
- Layout-aware — Reading order and structure preservation
- Web screen parsing — Webpage layout analysis
- Scene text spotting — Text detection in natural scenes
- SVG code generation — Charts, UI layouts, scientific figures to SVG

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted text in markdown format
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Usage

```python
from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("{{output_dataset_id}}", split="{split}")

# Access the markdown text
for example in dataset:
    print(example["markdown"])
    break

# View all OCR models applied to this dataset
inference_info = json.loads(dataset[0]["inference_info"])
for info in inference_info:
    print(f"Column: {{info['column_name']}} - Model: {{info['model_id']}}")
```

## Reproduction

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) dots.mocr script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/dots-mocr.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --batch-size {batch_size} \\
    --prompt-mode {prompt_mode} \\
    --max-model-len {max_model_len} \\
    --max-tokens {max_tokens} \\
    --gpu-memory-utilization {gpu_memory_utilization}
```

Generated with [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    model: str = "rednote-hilab/dots.mocr",
    max_model_len: int = 24000,
    max_tokens: int = 24000,
    gpu_memory_utilization: float = 0.9,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    prompt_mode: str = "ocr",
    custom_prompt: str = None,
    output_column: str = "markdown",
    overwrite: bool = False,
    config: str = None,
    create_pr: bool = False,
    temperature: float = 0.1,
    top_p: float = 0.9,
    verbose: bool = False,
):
    """Process images from HF dataset through dots.mocr model."""

    # Check CUDA availability first
    check_cuda_availability()

    # Track processing start time
    start_time = datetime.now()

    # Login to HF if token provided
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Determine prompt to use
    if custom_prompt:
        prompt = custom_prompt
        logger.info(f"Using custom prompt: {prompt[:50]}...")
    else:
        prompt = PROMPT_TEMPLATES.get(prompt_mode, PROMPT_TEMPLATES["ocr"])
        logger.info(f"Using prompt mode: {prompt_mode}")

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)

    # Validate image column
    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    # Fail fast if the output column would collide with an existing input column
    dataset = ensure_output_columns_free(dataset, [output_column], overwrite=overwrite)

    # Shuffle if requested
    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    # Limit samples if requested
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Initialize vLLM model
    logger.info(f"Initializing vLLM with model: {model}")
    logger.info("This may take a few minutes on first run...")
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # SVG mode uses higher temperature/top_p per official recommendation
    if prompt_mode == "svg" and temperature == 0.1 and top_p == 0.9:
        logger.info("SVG mode: using recommended temperature=0.9, top_p=1.0")
        temperature = 0.9
        top_p = 1.0

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Output will be written to column: {output_column}")

    # Process images in batches
    all_outputs = []

    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="dots.mocr processing",
    ):
        batch_indices = list(batch_indices)

        # Fetch images first, with per-batch fallback for unreadable files.
        # One corrupt image used to take down the entire run via the list
        # comprehension; now we mark the whole batch as skipped and continue.
        try:
            batch_images = [dataset[i][image_column] for i in batch_indices]
        except (UnidentifiedImageError, OSError) as e:
            logger.warning(
                f"Skipping batch of {len(batch_indices)} — unreadable image "
                f"in batch: {type(e).__name__}: {e}"
            )
            all_outputs.extend(
                ["[OCR SKIPPED — UNREADABLE IMAGE]"] * len(batch_indices)
            )
            continue

        try:
            # Create messages for batch
            batch_messages = [make_ocr_message(img, prompt) for img in batch_images]

            # Process with vLLM (dots.mocr needs "string" content format)
            outputs = llm.chat(
                batch_messages,
                sampling_params,
                chat_template_content_format="string",
            )

            # Extract outputs
            for output in outputs:
                text = output.outputs[0].text.strip()
                all_outputs.append(text)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add error placeholders for failed batch
            all_outputs.extend(["[OCR ERROR]"] * len(batch_images))

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    # Add output column to dataset
    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Handle inference_info tracking (for multi-model comparisons)
    inference_entry = {
        "model_id": model,
        "model_name": "dots.mocr",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "prompt_mode": prompt_mode if not custom_prompt else "custom",
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    if "inference_info" in dataset.column_names:
        # Append to existing inference info
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
        # Create new inference_info column
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
        num_samples=len(dataset),
        processing_time=processing_time_str,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        image_column=image_column,
        split=split,
        prompt_mode=prompt_mode if not custom_prompt else "custom",
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("dots.mocr processing complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")

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
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 80)
        print("dots.mocr Document Processing")
        print("=" * 80)
        print("\n3B multilingual OCR model with SVG generation")
        print("\nFeatures:")
        print("- Multilingual support (100+ languages)")
        print("- Fast processing with vLLM")
        print("- Table extraction and formatting")
        print("- Formula recognition")
        print("- Layout-aware text extraction")
        print("- Web screen parsing")
        print("- Scene text spotting")
        print("- SVG code generation (charts, UI, figures)")
        print("\nPrompt modes:")
        print("  ocr            - Text extraction (default)")
        print("  layout-all     - Layout + bboxes + text (JSON)")
        print("  layout-only    - Layout + bboxes only (JSON)")
        print("  web-parsing    - Webpage layout analysis (JSON)")
        print("  scene-spotting - Scene text detection")
        print("  grounding-ocr  - Text from bounding box region")
        print("  svg            - SVG code generation")
        print("  general        - Free-form (use with --custom-prompt)")
        print("\nExample usage:")
        print("\n1. Basic OCR:")
        print("   uv run --with vllm==0.30.0 dots-mocr.py input-dataset output-dataset")
        print("\n2. SVG generation:")
        print(
            "   uv run --with vllm==0.30.0 dots-mocr.py charts svg-output --prompt-mode svg --model rednote-hilab/dots.mocr-svg"
        )
        print("\n3. Web screen parsing:")
        print("   uv run --with vllm==0.30.0 dots-mocr.py screenshots parsed --prompt-mode web-parsing")
        print("\n4. Layout analysis with structure:")
        print("   uv run --with vllm==0.30.0 dots-mocr.py papers analyzed --prompt-mode layout-all")
        print("\n5. Running on HF Jobs:")
        print("   # [tool.hf-jobs] header sets image/flavor/secrets (hf CLI 1.32+)")
        print("   hf jobs uv run \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/dots-mocr.py \\"
        )
        print("       input-dataset output-dataset")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run --with vllm==0.30.0 dots-mocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using dots.mocr (3B multilingual model with SVG generation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prompt Modes (official dots.mocr prompts):
  ocr            - Simple text extraction (default)
  layout-all     - Layout analysis with bboxes, categories, and text (JSON output)
  layout-only    - Layout detection with bboxes and categories only (JSON output)
  web-parsing    - Webpage layout analysis (JSON output)
  scene-spotting - Scene text detection and recognition
  grounding-ocr  - Extract text from bounding box region
  svg            - SVG code generation (auto-injects image dimensions into viewBox)
  general        - Free-form QA (use with --custom-prompt)

SVG Code Generation:
  Use --prompt-mode svg for SVG output. For best results, combine with
  --model rednote-hilab/dots.mocr-svg (the SVG-optimized variant).
  SVG mode automatically uses temperature=0.9, top_p=1.0 unless overridden.

Examples:
  # Basic text OCR (default)
  uv run --with vllm==0.30.0 dots-mocr.py my-docs analyzed-docs

  # SVG generation with optimized variant
  uv run --with vllm==0.30.0 dots-mocr.py charts svg-out --prompt-mode svg --model rednote-hilab/dots.mocr-svg

  # Web screen parsing
  uv run --with vllm==0.30.0 dots-mocr.py screenshots parsed --prompt-mode web-parsing

  # Full layout analysis with structure
  uv run --with vllm==0.30.0 dots-mocr.py papers structured --prompt-mode layout-all

  # Random sampling for testing
  uv run --with vllm==0.30.0 dots-mocr.py large-dataset test --max-samples 50 --shuffle
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
        "--model",
        default="rednote-hilab/dots.mocr",
        help="Model to use (default: rednote-hilab/dots.mocr, or rednote-hilab/dots.mocr-svg for SVG)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=24000,
        help="Maximum model context length (default: 24000)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=24000,
        help="Maximum tokens to generate (default: 24000)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
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
        "--shuffle", action="store_true", help="Shuffle dataset before processing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=list(PROMPT_TEMPLATES.keys()),
        default="ocr",
        help=f"Prompt template to use: {', '.join(PROMPT_TEMPLATES.keys())} (default: ocr)",
    )
    parser.add_argument(
        "--custom-prompt",
        help="Custom prompt text (overrides --prompt-mode)",
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
        "--config",
        help="Config/subset name when pushing to Hub (for benchmarking multiple models in one repo)",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request instead of pushing directly (for parallel benchmarking)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1, per official recommendation)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9, per official recommendation)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions after processing (useful for pinning deps)",
    )

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        batch_size=args.batch_size,
        model=args.model,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        prompt_mode=args.prompt_mode,
        custom_prompt=args.custom_prompt,
        output_column=args.output_column,
        overwrite=args.overwrite,
        config=args.config,
        create_pr=args.create_pr,
        temperature=args.temperature,
        top_p=args.top_p,
        verbose=args.verbose,
    )
