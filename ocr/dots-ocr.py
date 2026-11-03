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
# # >=v0.30.0: ships huggingface_hub>=1.31, which datasets>=5.1 imports (httpx). On v0.29.0,
# # PYTHONPATH puts the image's older hub first and `import datasets` crashes.
# image = "vllm/vllm-openai:v0.30.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "a10g-small"
# secrets = ["HF_TOKEN"]
# ///

"""
Convert document images to markdown using DoTS.ocr with vLLM.

DoTS.ocr is a compact 1.7B multilingual document parsing model with SOTA performance
on 100+ languages. This script uses vLLM for efficient batch processing.

Features:
- 🌍 Multilingual support (100+ languages)
- 📊 Table extraction and formatting
- 📐 Formula recognition
- 📝 Layout-aware text extraction
- 🎯 Compact model (1.7B parameters)

Model: rednote-hilab/dots.ocr
vLLM: Officially tested with 0.9.1+ (native support via PR #24645)

Run on HF Jobs (the [tool.hf-jobs] header sets image, flavor and secrets;
needs `hf` CLI 1.32+):

    hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/dots-ocr.py \\
        input-dataset output-dataset

Run on your own GPU (vLLM and torch come from the image on Jobs, so add them):

    uv run --with vllm==0.30.0 dots-ocr.py input-dataset output-dataset
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
from typing import Any, Dict, List, Union
from datetime import datetime

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
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
# DoTS OCR Prompt Templates (from official dots.ocr repo)
# Source: https://github.com/rednote-hilab/dots.ocr/blob/master/dots_ocr/utils/prompts.py
# ────────────────────────────────────────────────────────────────

PROMPT_TEMPLATES = {
    "ocr": "Extract the text content from this image.",
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

5. Final Output: The entire output must be a single JSON object.""",
    "layout-only": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",
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
    prompt_mode: str = "general",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    return f"""---
tags:
- ocr
- document-processing
- dots-ocr
- multilingual
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using DoTS.ocr, a compact 1.7B multilingual model.

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

DoTS.ocr is a compact multilingual document parsing model that excels at:
- 🌍 **100+ Languages** - Multilingual document support
- 📊 **Table extraction** - Structured data recognition
- 📐 **Formulas** - Mathematical notation preservation
- 📝 **Layout-aware** - Reading order and structure preservation
- 🎯 **Compact** - Only 1.7B parameters

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

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) DoTS OCR script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/dots-ocr.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --batch-size {batch_size} \\
    --prompt-mode {prompt_mode} \\
    --max-model-len {max_model_len} \\
    --max-tokens {max_tokens} \\
    --gpu-memory-utilization {gpu_memory_utilization}
```

Generated with 🤖 [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    model: str = "rednote-hilab/dots.ocr",
    max_model_len: int = 32768,
    max_pixels: int = None,
    max_tokens: int = 8192,
    gpu_memory_utilization: float = 0.8,
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
):
    """Process images from HF dataset through DoTS.ocr model."""

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
    llm_kwargs = dict(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )
    if max_pixels is not None:
        logger.info(f"Capping input images to max_pixels={max_pixels}")
        llm_kwargs["mm_processor_kwargs"] = {"max_pixels": max_pixels}
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for OCR
        max_tokens=max_tokens,
    )

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Output will be written to column: {output_column}")

    # Process images in batches
    all_outputs = []

    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="DoTS.ocr processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        try:
            # Create messages for batch
            batch_messages = [make_ocr_message(img, prompt) for img in batch_images]

            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)

            # Extract outputs
            for output in outputs:
                text = output.outputs[0].text.strip()
                all_outputs.append(text)

        except Exception as e:
            logger.warning(
                f"Batch failed ({len(batch_images)} images), retrying individually: {e}"
            )
            for img in batch_images:
                try:
                    msg = make_ocr_message(img, prompt)
                    out = llm.chat([msg], sampling_params)
                    all_outputs.append(out[0].outputs[0].text.strip())
                except Exception as img_e:
                    logger.error(f"Image failed: {img_e}")
                    all_outputs.append("[OCR ERROR]")

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    # Add output column to dataset
    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Handle inference_info tracking (for multi-model comparisons)
    inference_entry = {
        "model_id": model,
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "prompt_mode": prompt_mode if not custom_prompt else "custom",
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

    # Push to hub
    logger.info(f"Pushing to {output_dataset}")
    dataset.push_to_hub(
        output_dataset,
        private=private,
        token=HF_TOKEN,
        **({"config_name": config} if config else {}),
        create_pr=create_pr,
        commit_message=f"Add {model} OCR results ({len(dataset)} samples)"
        + (f" [{config}]" if config else ""),
    )

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

    logger.info("✅ DoTS.ocr processing complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")


if __name__ == "__main__":
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 80)
        print("DoTS.ocr Document Processing")
        print("=" * 80)
        print("\nCompact 1.7B multilingual OCR model supporting 100+ languages")
        print("\nFeatures:")
        print("- 🌍 Multilingual support (100+ languages)")
        print("- ⚡ Fast processing with vLLM (2-3x speedup)")
        print("- 📊 Table extraction and formatting")
        print("- 📐 Formula recognition")
        print("- 📝 Layout-aware text extraction")
        print("\nExample usage:")
        print("\n1. Basic OCR:")
        print("   uv run --with vllm==0.30.0 dots-ocr.py input-dataset output-dataset")
        print("\n2. With custom settings:")
        print(
            "   uv run --with vllm==0.30.0 dots-ocr.py docs analyzed-docs --batch-size 20 --max-samples 100"
        )
        print("\n3. Layout analysis with structure:")
        print(
            "   uv run --with vllm==0.30.0 dots-ocr.py papers analyzed-structure --prompt-mode layout-all"
        )
        print("\n4. Layout detection only (no text):")
        print("   uv run --with vllm==0.30.0 dots-ocr.py docs layout-info --prompt-mode layout-only")
        print("\n5. Running on HF Jobs (header sets image/flavor/secrets; hf CLI 1.32+):")
        print("   hf jobs uv run \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/dots-ocr.py \\"
        )
        print("       input-dataset output-dataset")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run --with vllm==0.30.0 dots-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using DoTS.ocr (1.7B multilingual model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prompt Modes (official DoTS.ocr prompts):
  ocr         - Simple text extraction (default)
  layout-all  - Layout analysis with bboxes, categories, and text (JSON output)
  layout-only - Layout detection with bboxes and categories only (JSON output)

Examples:
  # Basic text OCR (default)
  uv run --with vllm==0.30.0 dots-ocr.py my-docs analyzed-docs

  # Full layout analysis with structure
  uv run --with vllm==0.30.0 dots-ocr.py papers structured --prompt-mode layout-all

  # Random sampling for testing
  uv run --with vllm==0.30.0 dots-ocr.py large-dataset test --max-samples 50 --shuffle
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
        help="Batch size for processing (default: 16, DoTS handles 16-30 well)",
    )
    parser.add_argument(
        "--model",
        default="rednote-hilab/dots.ocr",
        help="Model to use (default: rednote-hilab/dots.ocr)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help=(
            "Maximum model context length (default: 32768). dots.ocr does NOT resize "
            "input images, so a full page can reach ~14k image tokens (the model's "
            "11.29M-px processor cap); the old 8192 default rejected such requests and "
            "the row was written as '[OCR ERROR]'. Pair with --max-pixels to cap memory."
        ),
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help=(
            "Optional cap on input image pixels (width*height) passed to vLLM's "
            "mm_processor. Lower this (e.g. 4000000) to bound image tokens and GPU "
            "memory on very large scans. Default: model's own cap (~11.29M px)."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192)",
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

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        batch_size=args.batch_size,
        model=args.model,
        max_model_len=args.max_model_len,
        max_pixels=args.max_pixels,
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
    )
