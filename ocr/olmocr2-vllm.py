# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "tqdm",
#     "toolz",
#     "pyyaml",  # For parsing YAML front matter
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.29.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "a10g-small"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# ///

"""
Convert document images to markdown using olmOCR-2 with vLLM.

This script processes images through the olmOCR-2-7B model to extract
text and structure as markdown, optimized for document understanding.

Features:
- LaTeX equation recognition
- HTML table extraction
- Document structure preservation (headers, lists, formatting)
- Rotation detection and correction metadata
- Figure and chart descriptions
- Natural reading order inference
- High-quality OCR for various document types

Model: allenai/olmOCR-2-7B-1025-FP8
Based on: Qwen2.5-VL-7B-Instruct fine-tuned on olmOCR-mix

Run on HF Jobs (the [tool.hf-jobs] header sets image, flavor and secrets;
needs `hf` CLI 1.32+):
    hf jobs uv run olmocr2-vllm.py input-dataset output-dataset

Run on your own GPU (vLLM and torch come from the Jobs image, so add vLLM):
    uv run --with vllm==0.29.0 olmocr2-vllm.py input-dataset output-dataset
"""

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Union

import torch
import yaml
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

# olmOCR no-anchoring prompt (from olmocr/prompts/prompts.py:build_no_anchoring_v4_yaml_prompt)
OLMOCR_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the "
    "primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
)


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


def parse_yaml_frontmatter(text: str) -> tuple[dict, str]:
    """
    Parse YAML front matter from olmOCR output.

    Expected format:
    ---
    primary_language: en
    is_rotation_valid: true
    rotation_correction: 0
    is_table: false
    is_diagram: false
    ---
    # Document content here...

    Returns:
        (metadata_dict, content_without_frontmatter)
    """
    # Match YAML front matter between --- markers
    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(pattern, text.strip(), re.DOTALL)

    if match:
        yaml_str = match.group(1)
        content = match.group(2)
        try:
            metadata = yaml.safe_load(yaml_str)
            return metadata or {}, content
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse YAML front matter: {e}")
            return {}, text
    else:
        # No front matter found, return empty metadata
        logger.warning("No YAML front matter found in output")
        return {}, text


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str = OLMOCR_PROMPT,
    target_longest_dim: int = 1288,
) -> List[Dict]:
    """Create chat message for olmOCR processing.

    Args:
        image: Input image (PIL Image, dict with bytes, or path)
        prompt: OCR prompt text
        target_longest_dim: Target size for longest image dimension (default 1288, matching olmOCR)
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

    # Resize image to target dimension (matching olmOCR pipeline default of 1288px)
    width, height = pil_img.size
    longest_side = max(width, height)
    if longest_side != target_longest_dim:
        scale = target_longest_dim / longest_side
        new_width = int(width * scale)
        new_height = int(height * scale)
        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # Return message in vLLM format (text before image, matching olmOCR pipeline)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
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
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    return f"""---
tags:
- ocr
- document-processing
- olmocr
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains markdown-formatted OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using olmOCR-2-7B.

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

olmOCR-2-7B is a high-quality document OCR model based on Qwen2.5-VL-7B-Instruct, fine-tuned on olmOCR-mix-1025 dataset and optimized with GRPO reinforcement learning.

Key features:
- 📐 **LaTeX equations** - Mathematical formulas in LaTeX format
- 📊 **HTML tables** - Structured table extraction
- 📝 **Document structure** - Headers, lists, formatting preserved
- 🖼️ **Figure descriptions** - Charts and figures labeled with descriptions
- 🔄 **Rotation detection** - Metadata about document orientation
- 📑 **Natural reading order** - Handles multi-column and complex layouts
- 🎯 **High accuracy** - Scores 82.4 ± 1.1 on olmOCR-Bench

## Output Format

Each row contains:
- Original image from source dataset
- `markdown`: Extracted document content in markdown format
- `olmocr_metadata`: JSON with document metadata (language, rotation, table/diagram flags)

## Columns

- `{image_column}`: Original document image
- `markdown`: Extracted text and structure in markdown
- `olmocr_metadata`: Document metadata (primary_language, is_rotation_valid, rotation_correction, is_table, is_diagram)
- `inference_info`: Processing metadata (model, script version, timestamp)

## Reproduction

```bash
# Using HF Jobs (recommended; the script header sets image/flavor/secrets, hf CLI 1.32+)
hf jobs uv run \\
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/olmocr2-vllm.py \\
  {source_dataset} \\
  your-username/output-dataset

# Local with GPU
uv run --with vllm==0.29.0 https://huggingface.co/datasets/uv-scripts/ocr/raw/main/olmocr2-vllm.py \\
  {source_dataset} \\
  your-username/output-dataset
```

## Citation

```bibtex
@misc{{olmocr,
      title={{{{olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models}}}},
      author={{Jake Poznanski and Jon Borchardt and Jason Dunkelberger and Regan Huff and Daniel Lin and Aman Rangapur and Christopher Wilhelm and Kyle Lo and Luca Soldaini}},
      year={{2025}},
      eprint={{2502.18443}},
      archivePrefix={{arXiv}},
      primaryClass={{cs.CL}},
      url={{https://arxiv.org/abs/2502.18443}},
}}
```

---
*Generated with [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr)*
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    output_column: str = "markdown",
    overwrite: bool = False,
    batch_size: int = 16,
    model: str = "allenai/olmOCR-2-7B-1025-FP8",
    max_model_len: int = 16384,
    max_tokens: int = 8192,
    temperature: float = 0.1,
    gpu_memory_utilization: float = 0.8,
    guided_decoding: bool = False,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    config: str = None,
    create_pr: bool = False,
):
    """
    Process a dataset of document images through olmOCR-2 to extract markdown.

    Args:
        input_dataset: HuggingFace dataset ID containing images
        output_dataset: HuggingFace dataset ID for output
        image_column: Column name containing images
        output_column: Column name for markdown output
        batch_size: Number of images to process at once
        model: HuggingFace model ID for olmOCR
        max_model_len: Maximum context length
        max_tokens: Maximum tokens to generate per image
        temperature: Sampling temperature (0.1 default, matches olmOCR)
        gpu_memory_utilization: Fraction of GPU memory to use
        guided_decoding: Enable guided decoding with regex for YAML front matter
        hf_token: HuggingFace token for authentication
        split: Dataset split to process
        max_samples: Limit number of samples (for testing)
        private: Make output dataset private
        shuffle: Shuffle dataset before processing
        seed: Random seed for shuffling
    """
    import time

    start_time = time.time()

    # Check CUDA availability
    check_cuda_availability()

    # Login to HuggingFace if token provided
    if hf_token:
        login(token=hf_token)
    elif "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    ds = load_dataset(input_dataset, split=split)

    # Fail fast if the output column would collide with an existing input column
    ds = ensure_output_columns_free(ds, [output_column], overwrite=overwrite)

    # Shuffle if requested
    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        ds = ds.shuffle(seed=seed)

    # Limit samples if requested
    if max_samples:
        logger.info(f"Limiting to {max_samples} samples")
        ds = ds.select(range(min(max_samples, len(ds))))

    logger.info(f"Processing {len(ds)} samples")
    logger.info(f"Output will be written to column: {output_column}")

    # Set column names - namespace metadata by output column to avoid conflicts
    metadata_column_name = f"{output_column}_metadata"
    inference_info_column = "inference_info"
    logger.info(f"Metadata will be written to column: {metadata_column_name}")

    # Initialize LLM
    logger.info(f"Initializing vLLM with model: {model}")
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )

    # Sampling parameters - olmOCR uses temperature 0.1 (transformers example)
    sampling_params_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": 1.05,  # Discourage repetitive output
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }

    # Add guided decoding if requested (enforces YAML front matter structure).
    # vLLM 0.22.x renamed this API: GuidedDecodingParams (guided_decoding=) →
    # StructuredOutputsParams (structured_outputs=). Import lazily + shim so the
    # default path (guided_decoding=False) never touches the moved symbol.
    if guided_decoding:
        logger.info("Enabling guided decoding with YAML front matter regex")
        front_matter_regex = r"---\nprimary_language: (?:[a-z]{2}|null)\nis_rotation_valid: (?:True|False|true|false)\nrotation_correction: (?:0|90|180|270)\nis_table: (?:True|False|true|false)\nis_diagram: (?:True|False|true|false)\n(?:---|---\n[\s\S]+)"
        try:
            from vllm.sampling_params import StructuredOutputsParams

            sampling_params_kwargs["structured_outputs"] = StructuredOutputsParams(
                regex=front_matter_regex
            )
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams

            sampling_params_kwargs["guided_decoding"] = GuidedDecodingParams(
                regex=front_matter_regex
            )

    sampling_params = SamplingParams(**sampling_params_kwargs)

    # Process in batches
    all_outputs = []
    all_metadata = []

    for batch in tqdm(
        list(partition_all(batch_size, ds)),
        desc="Processing batches",
    ):
        # Create messages for batch
        messages = [make_ocr_message(item[image_column]) for item in batch]

        # Run inference
        outputs = llm.chat(messages, sampling_params=sampling_params)

        # Extract text and parse YAML front matter
        for idx, output in enumerate(outputs):
            response_text = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason

            # Log warning if generation didn't finish naturally
            if finish_reason != "stop":
                logger.warning(
                    f"Generation did not finish naturally (reason: {finish_reason}), output may be incomplete"
                )

            metadata, content = parse_yaml_frontmatter(response_text)
            all_outputs.append(content)
            all_metadata.append(json.dumps(metadata))

    # Add results to dataset
    # Check if columns already exist and handle appropriately
    if output_column in ds.column_names:
        logger.warning(
            f"Column '{output_column}' already exists, it will be overwritten"
        )
        ds = ds.remove_columns([output_column])
    ds = ds.add_column(output_column, all_outputs)

    if metadata_column_name in ds.column_names:
        logger.warning(
            f"Column '{metadata_column_name}' already exists, it will be overwritten"
        )
        ds = ds.remove_columns([metadata_column_name])
    ds = ds.add_column(metadata_column_name, all_metadata)

    # Add inference information — standard schema: a JSON-string LIST of entries, each carrying
    # model_id (used as the leaderboard label) and column_name (the output text column). The
    # script's own extras (version, timestamp, sampling params) ride along as extra keys.
    inference_entry = {
        "model_id": model,
        "column_name": output_column,
        "script": "olmocr2-vllm.py",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Handle existing inference_info column (append to the list; create a fresh list otherwise)
    if inference_info_column in ds.column_names:
        logger.info("Updating existing inference_info column")

        def update_inference_info(example):
            try:
                existing_info = (
                    json.loads(example[inference_info_column])
                    if example[inference_info_column]
                    else []
                )
            except (json.JSONDecodeError, TypeError):
                existing_info = []
            existing_info.append(inference_entry)
            return {inference_info_column: json.dumps(existing_info)}

        ds = ds.map(update_inference_info)
    else:
        logger.info("Creating new inference_info column")
        inference_list = [json.dumps([inference_entry])] * len(ds)
        ds = ds.add_column(inference_info_column, inference_list)

    # Calculate processing time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    processing_time = f"{hours}h {minutes}m {seconds}s"

    # Create and save dataset card
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        model=model,
        num_samples=len(ds),
        processing_time=processing_time,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        image_column=image_column,
        split=split,
    )

    # Push to hub
    logger.info(f"Pushing to HuggingFace Hub: {output_dataset}")
    commit_info = ds.push_to_hub(
        output_dataset,
        private=private,
        **({"config_name": config} if config else {}),
        create_pr=create_pr,
        commit_message=f"Add {model} OCR results ({len(ds)} samples)"
        + (f" [{config}]" if config else ""),
    )

    # Update dataset card
    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset)

    logger.info("✓ Processing complete!")
    logger.info(f"✓ Dataset: https://huggingface.co/datasets/{output_dataset}")
    logger.info(f"✓ Processing time: {processing_time}")
    logger.info(f"✓ Samples processed: {len(ds):,}")
    if create_pr and getattr(commit_info, "pr_url", None):
        logger.info(f"✓ Pull request created: {commit_info.pr_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert document images to markdown using olmOCR-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Basic OCR on a dataset:
   uv run --with vllm==0.29.0 olmocr2-vllm.py input-dataset output-dataset

2. Test with first 10 samples:
   uv run --with vllm==0.29.0 olmocr2-vllm.py input-dataset output-dataset --max-samples 10

3. Process with custom batch size:
   uv run --with vllm==0.29.0 olmocr2-vllm.py input-dataset output-dataset --batch-size 8

4. Custom image column:
   uv run --with vllm==0.29.0 olmocr2-vllm.py input-dataset output-dataset --image-column page_image

5. Private output dataset:
   uv run --with vllm==0.29.0 olmocr2-vllm.py input-dataset output-dataset --private

6. Random sampling:
   uv run --with vllm==0.29.0 olmocr2-vllm.py input-dataset output-dataset --max-samples 100 --shuffle

7. Running on HuggingFace Jobs (header sets image/flavor/secrets; hf CLI 1.32+):
   hf jobs uv run \\
     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/olmocr2-vllm.py \\
     input-dataset output-dataset

8. Real example with historical documents:
   hf jobs uv run --timeout 2h \\
     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/olmocr2-vllm.py \\
     NationalLibraryOfScotland/Britain-and-UK-Handbooks-Dataset \\
     your-username/handbooks-olmocr \\
     --max-samples 100 \\
     --shuffle
        """,
    )

    parser.add_argument("input_dataset", help="Input HuggingFace dataset ID")
    parser.add_argument("output_dataset", help="Output HuggingFace dataset ID")
    parser.add_argument(
        "--image-column",
        default="image",
        help="Column name containing images (default: image)",
    )
    parser.add_argument(
        "--output-column",
        default="markdown",
        help="Column name for markdown output (default: markdown)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column if it already exists in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing (default: 16)",
    )
    parser.add_argument(
        "--model",
        default="allenai/olmOCR-2-7B-1025-FP8",
        help="Model to use (default: allenai/olmOCR-2-7B-1025-FP8)",
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
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1, matches olmOCR transformers example)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
    )
    parser.add_argument(
        "--guided-decoding",
        action="store_true",
        help="Enable guided decoding with regex for YAML front matter structure",
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make output dataset private",
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
        help="Shuffle dataset before processing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )

    args = parser.parse_args()
    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        output_column=args.output_column,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
        model=args.model,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        guided_decoding=args.guided_decoding,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        config=args.config,
        create_pr=args.create_pr,
    )
