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
# image = "vllm/vllm-openai:v0.22.1"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "a10g-small"
# timeout = "1h"
# secrets = ["HF_TOKEN"]
# ///

"""
Convert document images to markdown using LightOnOCR-2 with vLLM.

LightOnOCR-2 is a compact 1B multilingual OCR model optimized for production speed.
Combines Pixtral ViT encoder with Qwen3 language model for efficient document parsing.
Uses Reinforcement Learning with Verifiable Rewards (RLVR) for improved quality.

Run on HF Jobs. vLLM and torch come from the vllm/vllm-openai:v0.22.1 image declared
in the [tool.hf-jobs] header (`hf` CLI 1.32+), which also sets the hardware and the
HF_TOKEN secret. The tag is pinned: unpinned vLLM 0.29/0.30 with current transformers
fails to import LightOnOCR-2 (PixtralRotaryEmbedding). Pass --timeout for a long run:

  hf jobs uv run --timeout 1h \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr2.py \\
      <input-dataset> <output-dataset>

To run on your own GPU, add the engine: `uv run --with vllm==0.22.1 lighton-ocr2.py ...`.

Features:
- ⚡ Fastest: 42.8 pages/sec on H100 GPU (7× faster than v1)
- 🎯 High accuracy: 83.2 ± 0.9% on OlmOCR-Bench (+7.1% vs v1)
- 🧠 RLVR trained: Eliminates repetition loops and formatting errors
- 📚 Better training: 2.5× larger dataset with cleaner annotations
- 🌍 Multilingual with European language optimization
- 📐 LaTeX formula recognition
- 📊 Table extraction (markdown format)
- 📝 Document structure preservation
- 💪 Production-ready: Outperforms models 9× larger

Model: lightonai/LightOnOCR-2-1B
vLLM: vllm/vllm-openai:v0.22.1 image (see the header)
Performance: 83.2 ± 0.9% on OlmOCR-Bench
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


# LightOnOCR-2 model (single variant)
MODEL = "lightonai/LightOnOCR-2-1B"


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


def resize_image_to_target(image: Image.Image, target_size: int = 1540) -> Image.Image:
    """
    Resize image so longest dimension is target_size while maintaining aspect ratio.

    LightOnOCR-2 was trained with images at 1540px max resolution and 200 DPI.
    """
    width, height = image.size

    # If image is already smaller, don't upscale
    if max(width, height) <= target_size:
        return image

    # Calculate new dimensions maintaining aspect ratio
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    resize: bool = True,
    target_size: int = 1540,
) -> List[Dict]:
    """
    Create chat message for OCR processing.

    LightOnOCR-2 was trained with 1540px max resolution at 200 DPI for optimal results.
    Unlike v1, LightOnOCR-2 does NOT use an empty text prefix - just the image.
    """
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

    # Resize to optimal dimensions for LightOnOCR-2
    if resize:
        pil_img = resize_image_to_target(pil_img, target_size)
        logger.debug(f"Resized image to {pil_img.size}")

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # LightOnOCR-2 uses message format with ONLY the image (no text prefix)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
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
    temperature: float,
    top_p: float,
    target_size: int,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    return f"""---
tags:
- ocr
- document-processing
- lighton-ocr-2
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using LightOnOCR-2, a fast and compact 1B OCR model trained with RLVR.

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
- **Target Image Size**: {target_size}px (longest dimension)
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **Temperature**: {temperature}
- **Top P**: {top_p}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

LightOnOCR-2 is a next-generation fast, compact OCR model that excels at:
- ⚡ **Fastest Speed** - 42.8 pages/second on H100 GPU (7× faster than v1)
- 🎯 **High Accuracy** - 83.2 ± 0.9% on OlmOCR-Bench (+7.1% vs v1)
- 🧠 **RLVR Training** - Eliminates repetition loops and formatting errors
- 📚 **Better Dataset** - 2.5× larger training data with cleaner annotations
- 📐 **LaTeX formulas** - Mathematical notation in LaTeX format
- 📊 **Tables** - Extracted and formatted as markdown
- 📝 **Document structure** - Hierarchy and layout preservation
- 🌍 **Multilingual** - Optimized for European languages
- 💪 **Production-ready** - Outperforms models 9× larger

### Key Improvements over v1

- **7.5× faster**: 42.8 vs 5.71 pages/sec on H100
- **+7.1% accuracy**: 83.2% vs 76.1% on benchmarks
- **Better quality**: RLVR training eliminates common OCR errors
- **Cleaner output**: No repetition loops or formatting glitches
- **Simpler**: Single model (no vocabulary variants)

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted text in markdown format with LaTeX formulas
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

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) LightOnOCR-2 script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr2.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --batch-size {batch_size}
```

## Performance

- **Processing Speed**: ~{num_samples / (float(processing_time.split()[0]) * 60):.2f} images/second
- **Benchmark Score**: 83.2 ± 0.9% on OlmOCR-Bench
- **Training**: RLVR (Reinforcement Learning with Verifiable Rewards)

Generated with 🤖 [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    max_model_len: int = 8192,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    top_p: float = 0.9,
    gpu_memory_utilization: float = 0.8,
    target_size: int = 1540,
    no_resize: bool = False,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    output_column: str = "markdown",
    overwrite: bool = False,
    config: str = None,
    create_pr: bool = False,
    verbose: bool = False,
):
    """Process images from HF dataset through LightOnOCR-2 model."""

    # Check CUDA availability first
    check_cuda_availability()

    # Track processing start time
    start_time = datetime.now()

    # Login to HF if token provided
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    logger.info(f"Using model: {MODEL}")

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
    logger.info("Initializing vLLM with LightOnOCR-2")
    logger.info("This may take a few minutes on first run...")
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},  # One image per prompt
        enforce_eager=False,  # Use torch.compile for better performance
    )

    # LightOnOCR-2 recommended sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Output will be written to column: {output_column}")
    if not no_resize:
        logger.info(f"Images will be resized to {target_size}px (longest dimension)")

    # Process images in batches
    all_outputs = []

    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="LightOnOCR-2 processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        try:
            # Create messages for batch
            batch_messages = [
                make_ocr_message(img, resize=not no_resize, target_size=target_size)
                for img in batch_images
            ]

            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)

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
        "model_id": MODEL,
        "model_name": "LightOnOCR-2",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "target_size": target_size if not no_resize else "original",
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
        commit_message=f"Add {MODEL} OCR results ({len(dataset)} samples)"
        + (f" [{config}]" if config else ""),
    )

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
        temperature=temperature,
        top_p=top_p,
        target_size=target_size,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("✅ LightOnOCR-2 processing complete!")
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
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 80)
        print("LightOnOCR-2 Document Processing")
        print("=" * 80)
        print("\nNext-generation 1B OCR model with RLVR training")
        print("\nFeatures:")
        print("- ⚡ Fastest processing: 42.8 pages/sec on H100 (7× faster than v1)")
        print("- 🎯 High accuracy: 83.2 ± 0.9% on OlmOCR-Bench (+7.1% vs v1)")
        print("- 🧠 RLVR trained: No repetition loops or formatting errors")
        print("- 📚 Better training: 2.5× larger dataset with cleaner annotations")
        print("- 🌍 Multilingual with European language optimization")
        print("- 📐 LaTeX formula recognition")
        print("- 📊 Table extraction (markdown format)")
        print("- 💪 Production-ready: Outperforms models 9× larger")
        print("\nExample usage:")
        print("\n1. Basic OCR:")
        print("   uv run lighton-ocr2.py input-dataset output-dataset")
        print("\n2. Custom batch size for performance:")
        print("   uv run lighton-ocr2.py docs results --batch-size 32")
        print("\n3. Test with small sample:")
        print("   uv run lighton-ocr2.py large-dataset test --max-samples 50 --shuffle")
        print("\n4. Original image size (no resize):")
        print("   uv run lighton-ocr2.py docs output --no-resize")
        print("\n5. Running on HF Jobs:")
        print("   (image, hardware and HF_TOKEN come from the script's [tool.hf-jobs] header)")
        print("   hf jobs uv run \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr2.py \\"
        )
        print("       input-dataset output-dataset --batch-size 32")
        print("\n" + "=" * 80)
        print("\nKey Improvements over v1:")
        print("  - 7.5× faster processing speed")
        print("  - 7.1% higher accuracy on benchmarks")
        print("  - Eliminates repetition loops and formatting errors")
        print("  - Simpler: single model (no vocabulary variants)")
        print("\nFor full help, run: uv run lighton-ocr2.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using LightOnOCR-2 (next-gen 1B model with RLVR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Key Improvements over v1:
  - 7.5× faster: 42.8 vs 5.71 pages/sec on H100
  - +7.1% accuracy: 83.2% vs 76.1% on benchmarks
  - Better quality: RLVR training eliminates repetition loops
  - Cleaner output: No formatting glitches
  - Simpler: Single model (no vocabulary variants)

Examples:
  # Basic text OCR
  uv run lighton-ocr2.py my-docs analyzed-docs

  # Test with random sampling
  uv run lighton-ocr2.py large-dataset test --max-samples 50 --shuffle

  # Custom batch size for GPU optimization
  uv run lighton-ocr2.py dataset output --batch-size 32 --gpu-memory-utilization 0.9
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
        default=16384,
        help=(
            "Maximum model context length (default: 16384). A full page resized to "
            "1540px is ~6k image tokens; with --max-tokens 4096 output that overflows "
            "the old 8192 default at admission and vLLM rejects the request."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096, recommended for arXiv papers)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1540,
        help="Target size for longest image dimension in pixels (default: 1540, matching training)",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Don't resize images (use original size)",
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

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        target_size=args.target_size,
        no_resize=args.no_resize,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        output_column=args.output_column,
        overwrite=args.overwrite,
        config=args.config,
        create_pr=args.create_pr,
        verbose=args.verbose,
    )
