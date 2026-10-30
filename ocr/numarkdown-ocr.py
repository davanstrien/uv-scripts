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
Convert document images to markdown using NuMarkdown-8B-Thinking with vLLM.

This script processes images through the NuMarkdown model to extract
text with advanced reasoning capabilities, ideal for complex document understanding.

Features:
- Reasoning-based document analysis with thinking tokens
- Superior table extraction and formatting
- Complex layout understanding
- Mathematical formula recognition
- Clean markdown output generation
- Optional thinking trace inclusion
- Multi-GPU support with automatic detection
- Optimized token budget for reasoning models
"""

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from torch import cuda
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


def check_gpu_availability() -> int:
    """Check if CUDA is available and return the number of GPUs."""
    if not cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error(
            "Please run on a machine with NVIDIA GPU or use HF Jobs with GPU flavor."
        )
        sys.exit(1)

    num_gpus = cuda.device_count()
    for i in range(num_gpus):
        gpu_name = cuda.get_device_name(i)
        gpu_memory = cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"GPU {i}: {gpu_name} with {gpu_memory:.1f} GB memory")

    return num_gpus


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


def validate_and_resize_image(
    image: Image.Image,
    min_pixels: int = 100 * 28 * 28,
    max_pixels: int = 5000 * 28 * 28,
) -> Image.Image:
    """Validate and resize image to meet pixel constraints if necessary."""
    width, height = image.size
    total_pixels = width * height

    if total_pixels < min_pixels or total_pixels > max_pixels:
        # Calculate scaling factor
        if total_pixels < min_pixels:
            scale = (min_pixels / total_pixels) ** 0.5
        else:
            scale = (max_pixels / total_pixels) ** 0.5

        new_width = int(width * scale)
        new_height = int(height * scale)

        logger.debug(
            f"Resizing image from {width}x{height} to {new_width}x{new_height}"
        )
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def extract_answer_from_thinking(text: str, include_thinking: bool = False) -> str:
    """
    Extract the final answer from NuMarkdown's thinking output.

    The model generates output in format:
    <think>reasoning process...</think>
    <answer>final markdown output</answer>
    """
    if include_thinking:
        # Return the full output including thinking traces
        return text.strip()

    # Extract content between <answer> tags
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, text, re.DOTALL)

    if answer_match:
        return answer_match.group(1).strip()

    # If no answer tags found, check if the entire text is markdown
    # (sometimes the model might not use tags)
    if "<think>" not in text and "<answer>" not in text:
        return text.strip()

    # Fallback: return everything after </think> if present
    think_end = text.find("</think>")
    if think_end != -1:
        remaining = text[think_end + 8 :].strip()
        # Remove <answer> tags if present
        remaining = remaining.replace("<answer>", "").replace("</answer>", "").strip()
        return remaining

    # Last resort: return the full text
    logger.warning("Could not extract answer from thinking tokens, returning full text")
    return text.strip()


def make_numarkdown_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str = "Convert this document to markdown. Focus on preserving structure, tables, formulas, and all textual content.",
) -> List[Dict]:
    """Create chat message for NuMarkdown processing."""
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    elif isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Validate and resize if necessary
    pil_img = validate_and_resize_image(pil_img)

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # Return message in vLLM chat format
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
    include_thinking: bool,
    tensor_parallel_size: int,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    return f"""---
tags:
- ocr
- document-processing
- numarkdown
- markdown
- reasoning
- thinking-tokens
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains markdown-formatted OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using NuMarkdown-8B-Thinking.

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
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}
- **Tensor Parallel Size**: {tensor_parallel_size} GPU(s)
- **Thinking Traces**: {"Included" if include_thinking else "Excluded (only final answers)"}

## Model Information

NuMarkdown-8B-Thinking is a state-of-the-art reasoning-based document OCR model that excels at:
- 🧠 **Reasoning Process** - Analyzes document layout before generation
- 📊 **Complex Tables** - Superior table extraction and formatting
- 📐 **Mathematical Formulas** - Accurate LaTeX/math notation preservation
- 📝 **Document Structure** - Maintains hierarchical document organization
- 🔍 **Layout Analysis** - Understands complex multi-column layouts
- ✨ **Clean Output** - Generates well-formatted markdown

### Thinking Tokens

This model uses a unique "thinking" process where it:
1. Analyzes the document structure internally (`<think>` phase)
2. Generates the final markdown output (`<answer>` phase)

{"The dataset includes both thinking traces and final answers." if include_thinking else "Only the final answers are included (thinking traces removed)."}

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

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) NuMarkdown OCR script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/numarkdown-ocr.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --batch-size {batch_size} \\
    --max-model-len {max_model_len} \\
    --max-tokens {max_tokens} \\
    --gpu-memory-utilization {gpu_memory_utilization} \\
    {"--include-thinking" if include_thinking else ""}
```

## Performance

- **Processing Speed**: ~{num_samples / (float(processing_time.split()[0]) * 60):.1f} images/second
- **GPU Configuration**: {tensor_parallel_size} GPU(s) with {gpu_memory_utilization:.0%} memory utilization
- **Model Size**: 8.29B parameters

Generated with 🤖 [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    model: str = "numind/NuMarkdown-8B-Thinking",
    max_model_len: int = 16384,
    max_tokens: int = 16384,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: Optional[int] = None,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    include_thinking: bool = False,
    temperature: float = 0.0,
    custom_prompt: Optional[str] = None,
    output_column: str = "markdown",
    overwrite: bool = False,
    config: str = None,
    create_pr: bool = False,
    verbose: bool = False,
):
    """Process images from HF dataset through NuMarkdown model.

    The max_tokens parameter controls the total token budget for both
    thinking and answer phases. For complex documents with extensive
    reasoning, the default of 16384 tokens provides ample room for both
    the thinking process and the final markdown output.
    """

    # GPU check and configuration
    num_gpus = check_gpu_availability()
    if tensor_parallel_size is None:
        tensor_parallel_size = num_gpus
        logger.info(
            f"Auto-detected {num_gpus} GPU(s), using tensor_parallel_size={tensor_parallel_size}"
        )
    else:
        logger.info(f"Using specified tensor_parallel_size={tensor_parallel_size}")
        if tensor_parallel_size > num_gpus:
            logger.warning(
                f"Requested {tensor_parallel_size} GPUs but only {num_gpus} available"
            )

    # Track processing start time
    start_time = datetime.now()

    # Login to HF if token provided
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

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

    # Initialize vLLM with trust_remote_code for NuMarkdown
    logger.info(f"Initializing vLLM with model: {model}")
    logger.info(f"Using {tensor_parallel_size} GPU(s) for inference")
    llm = LLM(
        model=model,
        trust_remote_code=True,  # Required for NuMarkdown
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        limit_mm_per_prompt={"image": 1},
    )

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Use custom prompt if provided, otherwise use default
    prompt = (
        custom_prompt
        or "Convert this document to markdown. Focus on preserving structure, tables, formulas, and all textual content."
    )

    # Process images in batches
    all_markdown = []

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Including thinking traces: {include_thinking}")

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
            batch_messages = [
                make_numarkdown_message(img, prompt) for img in batch_images
            ]

            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)

            # Extract markdown from outputs
            for output in outputs:
                raw_text = output.outputs[0].text.strip()
                # Extract answer from thinking tokens
                markdown_text = extract_answer_from_thinking(raw_text, include_thinking)
                all_markdown.append(markdown_text)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add error placeholders for failed batch
            all_markdown.extend(["[OCR FAILED]"] * len(batch_images))

    # Add output column to dataset
    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_markdown)

    # Handle inference_info tracking
    inference_entry = {
        "model_id": model,
        "model_name": "NuMarkdown-8B-Thinking",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "include_thinking": include_thinking,
        "temperature": temperature,
        "max_tokens": max_tokens,
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

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time = f"{processing_duration.total_seconds() / 60:.1f} minutes"

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
    logger.info("Creating dataset card...")
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        model=model,
        num_samples=len(dataset),
        processing_time=processing_time,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        include_thinking=include_thinking,
        tensor_parallel_size=tensor_parallel_size,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("NuMarkdown processing complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time}")

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
        print("NuMarkdown-8B-Thinking OCR with Reasoning")
        print("=" * 80)
        print("\nThis script converts document images to markdown using")
        print("the NuMarkdown-8B-Thinking model with advanced reasoning capabilities.")
        print("\nFeatures:")
        print("- 🧠 Reasoning-based document analysis")
        print("- 📊 Superior table extraction and formatting")
        print("- 📐 Mathematical formula recognition")
        print("- 📝 Complex layout understanding")
        print("- ✨ Clean markdown generation")
        print("- 🔍 Optional thinking trace inclusion")
        print("\nExample usage:")
        print("\n1. Basic OCR conversion:")
        print("   uv run numarkdown-ocr.py document-images markdown-docs")
        print("\n2. Include thinking traces:")
        print(
            "   uv run numarkdown-ocr.py complex-docs analyzed-docs --include-thinking"
        )
        print("\n3. With custom settings:")
        print("   uv run numarkdown-ocr.py scientific-papers extracted-text \\")
        print("       --batch-size 8 \\")
        print("       --max-tokens 16384 \\")
        print("       --gpu-memory-utilization 0.9")
        print("\n4. Process a subset for testing:")
        print("   uv run numarkdown-ocr.py large-dataset test-output --max-samples 10")
        print("\n5. Custom prompt for specific needs:")
        print("   uv run numarkdown-ocr.py invoices invoice-data \\")
        print(
            '       --custom-prompt "Extract all invoice details including line items"'
        )
        print("\n6. Multi-GPU processing:")
        print(
            "   uv run numarkdown-ocr.py large-docs processed-docs --tensor-parallel-size 2"
        )
        print("\n7. Running on HF Jobs:")
        print("   hf jobs uv run --flavor a100x4 \\")
        print(
            '     -e HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; print(get_token())") \\'
        )
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/numarkdown-ocr.py \\"
        )
        print("       your-document-dataset \\")
        print("       your-markdown-output")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run numarkdown-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="OCR images to markdown using NuMarkdown-8B-Thinking with reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run numarkdown-ocr.py my-images-dataset ocr-results
  
  # Include thinking traces in output
  uv run numarkdown-ocr.py documents analyzed-docs --include-thinking
  
  # Process subset for testing
  uv run numarkdown-ocr.py large-dataset test-output --max-samples 100
  
  # Custom prompt for specific extraction
  uv run numarkdown-ocr.py forms form-data --custom-prompt "Extract all form fields and values"
  
  # Multi-GPU for large datasets
  uv run numarkdown-ocr.py large-dataset processed --tensor-parallel-size 4
  
  # Random sample from dataset
  uv run numarkdown-ocr.py ordered-dataset random-sample --max-samples 50 --shuffle
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
        help="Batch size for processing (default: 16, lower than others due to model size)",
    )
    parser.add_argument(
        "--model",
        default="numind/NuMarkdown-8B-Thinking",
        help="Model to use (default: numind/NuMarkdown-8B-Thinking)",
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
        default=16384,
        help="Maximum tokens to generate including thinking tokens (default: 16384)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization per GPU (default: 0.9)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Number of GPUs to use (default: auto-detect all available)",
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
        "--include-thinking",
        action="store_true",
        help="Include thinking traces in output (default: only final answers)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0 for deterministic)",
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        help="Custom prompt for the model (overrides default)",
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
        tensor_parallel_size=args.tensor_parallel_size,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        include_thinking=args.include_thinking,
        temperature=args.temperature,
        custom_prompt=args.custom_prompt,
        output_column=args.output_column,
        overwrite=args.overwrite,
        config=args.config,
        create_pr=args.create_pr,
        verbose=args.verbose,
    )
