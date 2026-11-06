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
Convert document images to markdown using Qianfan-OCR with vLLM.

Qianfan-OCR is a 4.7B end-to-end document intelligence model from Baidu,
built on InternVL architecture with Qianfan-ViT encoder + Qwen3-4B LLM.

Features:
- #1 end-to-end model on OmniDocBench v1.5 (93.12) and OlmOCR Bench (79.8)
- Layout-as-Thought: optional reasoning phase for complex layouts via --think
- 192 language support (Latin, CJK, Arabic, Cyrillic, and more)
- Multiple task modes: OCR, table (HTML), formula (LaTeX), chart, scene text
- Key information extraction with custom prompts
- 1.024 PPS on A100 with W8A8 quantization

Model: baidu/Qianfan-OCR
License: Apache 2.0
Paper: https://arxiv.org/abs/2603.13398

Run on HF Jobs (the [tool.hf-jobs] header sets image, flavor and secrets;
needs `hf` CLI 1.32+):

    hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/qianfan-ocr.py \
        input-dataset output-dataset --max-samples 10

Run on your own GPU:

    uv run --with vllm==0.29.0 qianfan-ocr.py input-dataset output-dataset
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

MODEL = "baidu/Qianfan-OCR"

PROMPT_TEMPLATES = {
    "ocr": "Parse this document to Markdown.",
    "table": "Extract tables to HTML format.",
    "formula": "Extract formulas to LaTeX.",
    "chart": "What trends are shown in this chart?",
    "scene": "Extract all visible text from the image.",
    "kie": None,  # requires --custom-prompt
}


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
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


def extract_content_from_thinking(text: str, include_thinking: bool = False) -> str:
    """
    Extract final content from Qianfan-OCR's Layout-as-Thought output.

    When --think is enabled, the model generates layout analysis inside
    <think>...</think> tags before the final markdown output.
    """
    if include_thinking:
        return text.strip()

    # If no thinking tags, return as-is
    if "<think>" not in text:
        return text.strip()

    # Extract everything after </think>
    think_end = text.find("</think>")
    if think_end != -1:
        return text[think_end + 8 :].strip()

    # Thinking started but never closed — return full text
    logger.warning("Found <think> but no </think>, returning full text")
    return text.strip()


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str,
) -> List[Dict]:
    """Create vLLM chat message with image and prompt."""
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    pil_img = pil_img.convert("RGB")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

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
    prompt_mode: str,
    think: bool,
    include_thinking: bool,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    return f"""---
tags:
- ocr
- document-processing
- qianfan-ocr
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains OCR results from [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using Qianfan-OCR, Baidu's 4.7B end-to-end document intelligence model.

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
- **Layout-as-Thought**: {"Enabled" if think else "Disabled"}
- **Thinking Traces**: {"Included" if include_thinking else "Excluded"}
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

Qianfan-OCR key capabilities:
- #1 end-to-end model on OmniDocBench v1.5 (93.12)
- #1 on OlmOCR Bench (79.8)
- 192 language support
- Layout-as-Thought reasoning for complex documents
- Document parsing, table extraction, formula recognition, chart understanding
- Key information extraction

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted text in markdown format
- `inference_info`: JSON list tracking all OCR models applied

## Reproduction

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/qianfan-ocr.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --prompt-mode {prompt_mode} \\
    --batch-size {batch_size}{"  --think" if think else ""}
```

Generated with [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 8,
    max_model_len: int = 16384,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    top_p: float = 1.0,
    gpu_memory_utilization: float = 0.85,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    prompt_mode: str = "ocr",
    think: bool = False,
    include_thinking: bool = False,
    custom_prompt: str = None,
    output_column: str = "markdown",
    overwrite: bool = False,
    config: str = None,
    create_pr: bool = False,
    verbose: bool = False,
):
    """Process images from HF dataset through Qianfan-OCR model."""

    check_cuda_availability()
    start_time = datetime.now()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Build prompt
    if custom_prompt:
        prompt = custom_prompt
        logger.info(f"Using custom prompt: {prompt[:80]}...")
    else:
        if prompt_mode == "kie":
            logger.error("--prompt-mode kie requires --custom-prompt")
            sys.exit(1)
        prompt = PROMPT_TEMPLATES[prompt_mode]
        logger.info(f"Using prompt mode: {prompt_mode}")

    if think:
        prompt = prompt + "<think>"
        logger.info("Layout-as-Thought enabled (appending <think> to prompt)")

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
    logger.info("Initializing vLLM with Qianfan-OCR")
    logger.info("This may take a few minutes on first run...")
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=False,
    )

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
        desc="Qianfan-OCR processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        try:
            batch_messages = [make_ocr_message(img, prompt) for img in batch_images]
            outputs = llm.chat(batch_messages, sampling_params)

            for output in outputs:
                text = output.outputs[0].text.strip()
                if think:
                    text = extract_content_from_thinking(text, include_thinking)
                all_outputs.append(text)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            all_outputs.extend(["[OCR ERROR]"] * len(batch_images))

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    # Add output column
    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Handle inference_info tracking
    inference_entry = {
        "model_id": MODEL,
        "model_name": "Qianfan-OCR",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "prompt_mode": prompt_mode if not custom_prompt else "custom",
        "think": think,
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

    # Push to hub with retry and XET fallback
    logger.info(f"Pushing to {output_dataset}")
    commit_msg = f"Add Qianfan-OCR results ({len(dataset)} samples)" + (
        f" [{config}]" if config else ""
    )
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
                commit_message=commit_msg,
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

    # Create and push dataset card (skip when creating PR to avoid conflicts)
    if not create_pr:
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
            prompt_mode=prompt_mode if not custom_prompt else "custom",
            think=think,
            include_thinking=include_thinking,
            image_column=image_column,
            split=split,
        )
        card = DatasetCard(card_content)
        card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("Qianfan-OCR processing complete!")
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
        print("=" * 80)
        print("Qianfan-OCR - End-to-End Document Intelligence")
        print("=" * 80)
        print("\n4.7B model from Baidu, #1 on OmniDocBench v1.5 (93.12)")
        print("\nFeatures:")
        print("- #1 end-to-end model on OmniDocBench v1.5 and OlmOCR Bench")
        print("- Layout-as-Thought reasoning for complex documents (--think)")
        print("- 192 language support")
        print("- Multiple modes: OCR, table (HTML), formula (LaTeX), chart, scene text")
        print("- Key information extraction with custom prompts")
        print("\nExample usage:")
        print("\n1. Basic OCR:")
        print("   uv run --with vllm==0.29.0 qianfan-ocr.py input-dataset output-dataset")
        print("\n2. With Layout-as-Thought (complex documents):")
        print("   uv run --with vllm==0.29.0 qianfan-ocr.py docs output --think")
        print("\n3. Table extraction:")
        print("   uv run --with vllm==0.29.0 qianfan-ocr.py docs output --prompt-mode table")
        print("\n4. Formula extraction:")
        print("   uv run --with vllm==0.29.0 qianfan-ocr.py docs output --prompt-mode formula")
        print("\n5. Key information extraction:")
        print(
            '   uv run --with vllm==0.29.0 qianfan-ocr.py invoices output --prompt-mode kie --custom-prompt "Extract: name, date, total. Output JSON."'
        )
        print("\n6. Running on HF Jobs (header sets image/flavor/secrets; hf CLI 1.32+):")
        print("   hf jobs uv run \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/qianfan-ocr.py \\"
        )
        print("       input-dataset output-dataset --max-samples 10")
        print("\n7. Running on your own GPU:")
        print("   uv run --with vllm==0.29.0 qianfan-ocr.py input-dataset output-dataset")
        print("\nFor full help, run: uv run --with vllm==0.29.0 qianfan-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using Qianfan-OCR (4.7B, #1 on OmniDocBench v1.5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prompt modes:
  ocr       Document parsing to Markdown (default)
  table     Table extraction to HTML format
  formula   Formula recognition to LaTeX
  chart     Chart understanding and analysis
  scene     Scene text extraction
  kie       Key information extraction (requires --custom-prompt)

Examples:
  uv run --with vllm==0.29.0 qianfan-ocr.py my-docs analyzed-docs
  uv run --with vllm==0.29.0 qianfan-ocr.py docs output --think --max-samples 50
  uv run --with vllm==0.29.0 qianfan-ocr.py docs output --prompt-mode table
  uv run --with vllm==0.29.0 qianfan-ocr.py invoices data --prompt-mode kie --custom-prompt "Extract: name, date, total."
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
        help="Batch size for processing (default: 8)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Maximum model context length (default: 16384, reduce to 8192 if OOM on L4)",
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
        default=0.0,
        help="Sampling temperature (default: 0.0, deterministic)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter (default: 1.0)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization (default: 0.85)",
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
        help="Prompt mode (default: ocr)",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable Layout-as-Thought reasoning (appends <think> to prompt)",
    )
    parser.add_argument(
        "--include-thinking",
        action="store_true",
        help="Include thinking traces in output (default: only final content)",
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
        help="Config/subset name when pushing to Hub (for benchmarking multiple models)",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request instead of pushing directly",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions after processing",
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
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        prompt_mode=args.prompt_mode,
        think=args.think,
        include_thinking=args.include_thinking,
        custom_prompt=args.custom_prompt,
        output_column=args.output_column,
        overwrite=args.overwrite,
        config=args.config,
        create_pr=args.create_pr,
        verbose=args.verbose,
    )
