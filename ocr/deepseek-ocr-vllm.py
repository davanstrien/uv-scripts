# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0,<5.1",  # 5.1 needs huggingface_hub>=1.31; the image ships an older 1.x on PYTHONPATH, which wins
#     "huggingface-hub",
#     "pillow",
#     "tqdm",
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
Convert document images to markdown using DeepSeek-OCR with vLLM.

This script processes images through the DeepSeek-OCR model to extract
text and structure as markdown, using vLLM for efficient batch processing.

Uses the official vLLM offline pattern: llm.generate() with PIL images
and NGramPerReqLogitsProcessor to prevent repetition on complex documents.
See: https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html

Run on HF Jobs. vLLM and torch come from the vllm/vllm-openai:v0.29.0 image declared
in the [tool.hf-jobs] header (`hf` CLI 1.32+), which also sets the hardware and the
HF_TOKEN secret. The tag is pinned: vLLM 0.30 breaks the DeepEncoder Triton kernel
(LOG2E NameError), and nightly wheels moved the logits-processor API. Pass --timeout
for a long run. To run on your own GPU: `uv run --with vllm==0.29.0 deepseek-ocr-vllm.py ...`.

Features:
- LaTeX equation recognition
- Table extraction and formatting
- Document structure preservation
- Image grounding and descriptions
- Multilingual support
- Batch processing with vLLM for better performance
"""

import argparse
import io
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Union

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
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt mode presets (from DeepSeek-OCR GitHub)
PROMPT_MODES = {
    "document": "<image>\n<|grounding|>Convert the document to markdown.",
    "image": "<image>\n<|grounding|>OCR this image.",
    "free": "<image>\nFree OCR.",
    "figure": "<image>\nParse the figure.",
    "describe": "<image>\nDescribe this image in detail.",
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


def to_pil(image: Union[Image.Image, Dict[str, Any], str]) -> Image.Image:
    """Convert various image formats to PIL Image."""
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, dict) and "bytes" in image:
        return Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        return Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


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
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    return f"""---
tags:
- ocr
- document-processing
- deepseek
- deepseek-ocr
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains markdown-formatted OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using DeepSeek-OCR.

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

## Model Information

DeepSeek-OCR is a state-of-the-art document OCR model that excels at:
- LaTeX equations - Mathematical formulas preserved in LaTeX format
- Tables - Extracted and formatted as HTML/markdown
- Document structure - Headers, lists, and formatting maintained
- Image grounding - Spatial layout and bounding box information
- Complex layouts - Multi-column and hierarchical structures
- Multilingual - Supports multiple languages

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted text in markdown format with preserved structure
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Usage

```python
from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("{{{{output_dataset_id}}}}", split="{split}")

# Access the markdown text
for example in dataset:
    print(example["markdown"])
    break

# View all OCR models applied to this dataset
inference_info = json.loads(dataset[0]["inference_info"])
for info in inference_info:
    print(f"Column: {{{{info['column_name']}}}} - Model: {{{{info['model_id']}}}}")
```

## Reproduction

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) DeepSeek OCR vLLM script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py \\\\
    {source_dataset} \\\\
    <output-dataset> \\\\
    --image-column {image_column}
```

## Performance

- **Processing Speed**: ~{num_samples / (float(processing_time.split()[0]) * 60):.1f} images/second
- **Processing Method**: Batch processing with vLLM (2-3x speedup over sequential)

Generated with [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 8,
    model: str = "deepseek-ai/DeepSeek-OCR",
    max_model_len: int = 8192,
    max_tokens: int = 8192,
    gpu_memory_utilization: float = 0.8,
    prompt_mode: str = "document",
    prompt: str = None,
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
    """Process images from HF dataset through DeepSeek-OCR model with vLLM."""

    # Check CUDA availability first
    check_cuda_availability()

    # Track processing start time
    start_time = datetime.now()

    # Login to HF if token provided
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Determine prompt
    if prompt is not None:
        final_prompt = prompt
        logger.info("Using custom prompt")
    elif prompt_mode in PROMPT_MODES:
        final_prompt = PROMPT_MODES[prompt_mode]
        logger.info(f"Using prompt mode: {prompt_mode}")
    else:
        raise ValueError(
            f"Invalid prompt mode '{prompt_mode}'. "
            f"Use one of {list(PROMPT_MODES.keys())} or specify --prompt"
        )

    logger.info(f"Prompt: {final_prompt}")

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

    # Initialize vLLM (matches official DeepSeek-OCR vLLM recipe)
    logger.info(f"Initializing vLLM with model: {model}")
    logger.info("This may take a few minutes on first run...")

    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        skip_special_tokens=False,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},
        ),
    )

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")

    # Process images in batches using llm.generate() with PIL images
    all_markdown = []

    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="DeepSeek-OCR vLLM processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        try:
            # Build model inputs with PIL images (official vLLM pattern)
            model_inputs = []
            for img in batch_images:
                pil_img = to_pil(img).convert("RGB")
                model_inputs.append(
                    {"prompt": final_prompt, "multi_modal_data": {"image": pil_img}}
                )

            # Process with vLLM generate API
            outputs = llm.generate(model_inputs, sampling_params)

            # Extract outputs
            for output in outputs:
                text = output.outputs[0].text.strip()
                all_markdown.append(text)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            all_markdown.extend(["[OCR FAILED]"] * len(batch_images))

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    # Add output column to dataset
    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_markdown)

    # Handle inference_info tracking
    logger.info("Updating inference_info...")

    inference_entry = {
        "model_id": model,
        "model_name": "DeepSeek-OCR",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "prompt_mode": prompt_mode if prompt is None else "custom",
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "script": "deepseek-ocr-vllm.py",
        "script_url": "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py",
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
    logger.info("Creating dataset card...")
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
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)
    logger.info("✅ Dataset card created and pushed!")

    logger.info("✅ OCR conversion complete!")
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
        print("DeepSeek-OCR to Markdown Converter (vLLM)")
        print("=" * 80)
        print("\nThis script converts document images to markdown using")
        print("DeepSeek-OCR with vLLM for efficient batch processing.")
        print("\nFeatures:")
        print("- LaTeX equation recognition")
        print("- Table extraction and formatting")
        print("- Document structure preservation")
        print("- Image grounding and spatial layout")
        print("- Multilingual support")
        print("- Fast batch processing with vLLM")
        print("\nExample usage:")
        print("\n1. Basic OCR conversion (document mode with grounding):")
        print("   uv run deepseek-ocr-vllm.py document-images markdown-docs")
        print("\n2. Parse figures from documents:")
        print(
            "   uv run deepseek-ocr-vllm.py scientific-papers figures --prompt-mode figure"
        )
        print("\n3. Free OCR without layout:")
        print("   uv run deepseek-ocr-vllm.py images text --prompt-mode free")
        print("\n4. Process a subset for testing:")
        print(
            "   uv run deepseek-ocr-vllm.py large-dataset test-output --max-samples 10"
        )
        print("\n5. Running on HF Jobs:")
        print("   (image, hardware and HF_TOKEN come from the script's [tool.hf-jobs] header)")
        print("   hf jobs uv run \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py \\"
        )
        print("       your-document-dataset \\")
        print("       your-markdown-output")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run deepseek-ocr-vllm.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="OCR images to markdown using DeepSeek-OCR (vLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prompt Modes:
  document  Convert document to markdown with grounding (default)
  image     OCR any image with grounding
  free      Free OCR without layout preservation
  figure    Parse figures from documents
  describe  Generate detailed image descriptions

Examples:
  # Basic usage
  uv run deepseek-ocr-vllm.py my-images-dataset ocr-results

  # Parse figures from a document dataset
  uv run deepseek-ocr-vllm.py scientific-papers figures --prompt-mode figure

  # Free OCR without layout
  uv run deepseek-ocr-vllm.py images text --prompt-mode free

  # Custom prompt for specific task
  uv run deepseek-ocr-vllm.py dataset output --prompt "<image>\\nExtract all table data."

  # With custom batch size for performance tuning
  uv run deepseek-ocr-vllm.py dataset output --batch-size 16 --max-model-len 16384

  # Running on HF Jobs
  hf jobs uv run \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py \\
      my-dataset my-output --max-samples 10
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
        default=8,
        help="Batch size for processing (default: 8, adjust based on GPU memory)",
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-OCR",
        help="Model to use (default: deepseek-ai/DeepSeek-OCR)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length (default: 8192)",
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
    parser.add_argument(
        "--prompt-mode",
        default="document",
        choices=list(PROMPT_MODES.keys()),
        help="Prompt mode preset (default: document). Use --prompt for custom prompts.",
    )
    parser.add_argument(
        "--prompt",
        help="Custom OCR prompt (overrides --prompt-mode)",
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
        "--output-column",
        default="markdown",
        help="Column name for the OCR output text (default: markdown)",
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
        model=args.model,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prompt_mode=args.prompt_mode,
        prompt=args.prompt,
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
