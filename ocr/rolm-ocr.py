# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "vllm>=0.15.1",
#     "tqdm",
#     "toolz",
#     "torch",  # Added for CUDA check
# ]
#
# ///

"""
UNSUPPORTED: Broken (2026-09-23): no KV cache room on a 24 GB L4 with the default settings. Use olmocr2-vllm.py. See models.json (`support`).

Extract text from document images using RolmOCR with vLLM.

This script processes images through the RolmOCR model to extract
plain text content, ideal for general-purpose OCR tasks.

Features:
- Fast and efficient text extraction
- General-purpose document OCR
- Based on Qwen2.5-VL-7B architecture
- Optimized for batch processing with vLLM
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
from typing import Any, Dict, List, Union

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
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    prompt: str = "Return the plain text representation of this document as if you were reading it naturally.\n",
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
    output_column: str,
    batch_size: int,
    max_model_len: int,
    max_tokens: int,
    gpu_memory_utilization: float,
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
viewer: false
tags:
- ocr
- text-extraction
- rolmocr
- uv-script
- generated{jobs_tag}
---

# OCR Text Extraction using {model_name}

This dataset contains extracted text from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using RolmOCR.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `{output_column}`
- **Dataset Split**: `{split}`
- **Batch Size**: {batch_size}
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

RolmOCR is a fast, general-purpose OCR model based on Qwen2.5-VL-7B architecture. It extracts plain text from document images with high accuracy and efficiency.

## Dataset Structure

The dataset contains all original columns plus:
- `{output_column}`: The extracted text from each image
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Usage

```python
from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("{{output_dataset_id}}", split="{split}")

# Access the extracted text
for example in dataset:
    print(example["{output_column}"])
    break

# View all OCR models applied to this dataset
inference_info = json.loads(dataset[0]["inference_info"])
for info in inference_info:
    print(f"Column: {{info['column_name']}} - Model: {{info['model_id']}}")
```

## Reproduction

{origin} with the [`rolm-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/raw/main/rolm-ocr.py) recipe from [uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:

```bash
hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/rolm-ocr.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --batch-size {batch_size} \\
    --max-model-len {max_model_len} \\
    --max-tokens {max_tokens} \\
    --gpu-memory-utilization {gpu_memory_utilization}
```

## Performance

- **Processing Speed**: ~{num_samples / (float(processing_time.split()[0]) * 60):.1f} images/second
- **GPU Configuration**: vLLM with {gpu_memory_utilization:.0%} GPU memory utilization
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    model: str = "reducto/RolmOCR",
    max_model_len: int = 16384,
    max_tokens: int = 8192,
    gpu_memory_utilization: float = 0.8,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    output_column: str = None,
    overwrite: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    verbose: bool = False,
):
    """Process images from HF dataset through OCR model."""

    # Check CUDA availability first
    check_cuda_availability()

    # Track processing start time
    start_time = datetime.now()

    # Enable high-performance Xet downloads
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    # Login to HF if token provided
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)

    # Default output column is 'markdown' for consistency across scripts
    if output_column is None:
        output_column = "markdown"

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

    # Initialize vLLM
    logger.info(f"Initializing vLLM with model: {model}")
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for OCR
        max_tokens=max_tokens,
    )

    # Process images in batches
    all_text = []

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")

    # Process in batches to avoid memory issues
    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="OCR processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        try:
            # Create messages for batch
            batch_messages = [make_ocr_message(img) for img in batch_images]

            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)

            # Extract text from outputs
            for output in outputs:
                text = output.outputs[0].text.strip()
                all_text.append(text)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add error placeholders for failed batch
            all_text.extend(["[OCR FAILED]"] * len(batch_images))

    # Add text column to dataset
    logger.info(f"Adding {output_column} column to dataset")
    dataset = dataset.add_column(output_column, all_text)

    # Handle inference_info tracking
    logger.info("Updating inference_info...")

    inference_entry = {
        "model_id": model,
        "model_name": "RolmOCR",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "script": "rolm-ocr.py",
        "script_url": "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/rolm-ocr.py",
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

    # Push to hub
    logger.info(f"Pushing to {output_dataset}")
    dataset.push_to_hub(output_dataset, private=private, token=HF_TOKEN)

    # Calculate processing time
    end_time = datetime.now()
    processing_duration = end_time - start_time
    processing_time = f"{processing_duration.total_seconds() / 60:.1f} minutes"

    # Create and push dataset card
    logger.info("Creating dataset card...")
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        model=model,
        num_samples=len(dataset),
        processing_time=processing_time,
        output_column=output_column,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)
    logger.info("✅ Dataset card created and pushed!")

    logger.info("✅ OCR conversion complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
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
        print("RolmOCR Document Text Extraction")
        print("=" * 80)
        print("\nThis script extracts plain text from document images using")
        print("the RolmOCR model with vLLM acceleration.")
        print("\nFeatures:")
        print("- Fast and efficient text extraction")
        print("- General-purpose document OCR")
        print("- Based on Qwen2.5-VL-7B architecture")
        print("- Optimized for batch processing")
        print("\nExample usage:")
        print("\n1. Basic OCR conversion:")
        print("   uv run rolm-ocr.py document-images extracted-text")
        print("\n2. With custom settings:")
        print("   uv run rolm-ocr.py scanned-docs ocr-output \\")
        print("       --image-column page \\")
        print("       --batch-size 8 \\")
        print("       --gpu-memory-utilization 0.9")
        print("\n3. Process a subset for testing:")
        print("   uv run rolm-ocr.py large-dataset test-output --max-samples 10")
        print("\n4. Random sample from ordered dataset:")
        print(
            "   uv run rolm-ocr.py ordered-dataset random-test --max-samples 50 --shuffle"
        )
        print("\n5. Running on HF Jobs:")
        print("   hf jobs uv run --flavor l4x1 \\")
        print(
            '     -e HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; print(get_token())") \\'
        )
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/rolm-ocr.py \\"
        )
        print("       your-document-dataset \\")
        print("       your-text-output")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run rolm-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="OCR images to text using RolmOCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run rolm-ocr.py my-images-dataset ocr-results

  # With specific image column
  uv run rolm-ocr.py documents extracted-text --image-column scan

  # Process subset for testing
  uv run rolm-ocr.py large-dataset test-output --max-samples 100

  # Random sample of 100 images
  uv run rolm-ocr.py ordered-dataset random-sample --max-samples 100 --shuffle

  # Custom output column name (default: markdown)
  uv run rolm-ocr.py images texts --output-column ocr_text
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
        default="reducto/RolmOCR",
        help="Model to use (default: reducto/RolmOCR)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Maximum model context length (default: 16384)",
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
        "--output-column",
        default=None,
        help="Name of the output column for extracted text (default: markdown)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column if it already exists in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before processing (useful for random sampling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
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
        output_column=args.output_column,
        overwrite=args.overwrite,
        shuffle=args.shuffle,
        seed=args.seed,
        verbose=args.verbose,
    )
