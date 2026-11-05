# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=4.0.0",
#     "httpx",
#     "huggingface-hub",
#     "setuptools",
#     "toolz",
# ]
#
# [tool.hf-jobs]
# # vllm, torch and transformers come from the image (PYTHONPATH). Listing them above makes uv
# # install newer PyPI copies plus their deps. PYTHONPATH shadows the copies, but the current
# # flashinfer-python wheel (0.7.2.post1) also ships a top-level b12x module that the image lacks.
# # The image's vllm imports it and crashes (missing flashinfer/experimental/b12x/_api.py).
# # Seen on v0.32.0 (2026-11-05); re-test before moving the tag. Local GPU: uv run --with vllm.
# image = "vllm/vllm-openai:v0.32.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "l4x1"
# secrets = ["HF_TOKEN"]
# timeout = "1h"
# ///
"""
Batch text classification using vLLM for efficient GPU inference.

This script loads a dataset from Hugging Face Hub, performs classification using
a BERT-style model via vLLM, and saves the results back to the Hub with predicted
labels and confidence scores.

Example usage:
    # Local execution (Linux + CUDA). The header leaves vllm, torch and
    # transformers to the HF Jobs image, so add vllm with --with.
    uv run --with vllm classify-dataset.py \\
        davanstrien/ModernBERT-base-is-new-arxiv-dataset \\
        username/input-dataset \\
        username/output-dataset \\
        --inference-column text \\
        --batch-size 10000

    # HF Jobs execution (the header sets the vLLM image, flavor and HF_TOKEN secret)
    hf jobs uv run classify-dataset.py model/name input-dataset output-dataset
"""

import argparse
import logging
import os
import sys
from typing import Optional

import httpx
from datasets import load_dataset
from huggingface_hub import hf_hub_url, login
from toolz import concat, keymap, partition_all
from tqdm.auto import tqdm

# torch and vllm are imported inside the functions that use them, so
# `uv run classify-dataset.py --help` works without them. On HF Jobs they come from
# the vLLM image (see the header).

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_gpu_availability():
    """Check if CUDA is available and log GPU information."""
    import torch
    import vllm

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error(
            "Please run on a machine with NVIDIA GPU or use HF Jobs with GPU flavor."
        )
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU detected: {gpu_name} with {gpu_memory:.1f} GB memory")
    logger.info(f"vLLM version: {vllm.__version__}")


def get_model_id2label(hub_model_id: str) -> Optional[dict[int, str]]:
    """Extract label mapping from model's config.json on Hugging Face Hub."""
    try:
        response = httpx.get(
            hf_hub_url(hub_model_id, filename="config.json"), follow_redirects=True
        )
        if response.status_code != 200:
            logger.warning(f"Could not fetch config.json for {hub_model_id}")
            return None

        data = response.json()
        id2label = data.get("id2label")

        if id2label is None:
            logger.info("No id2label mapping found in config.json")
            return None

        # Convert string keys to integers
        label_map = keymap(int, id2label)
        logger.info(f"Found label mapping: {label_map}")
        return label_map

    except Exception as e:
        logger.warning(f"Failed to parse config.json: {e}")
        return None


def get_top_label(output, label_map: Optional[dict[int, str]] = None):
    """
    Extract the top predicted label and confidence score from vLLM output.

    Args:
        output: vLLM ClassificationRequestOutput
        label_map: Optional mapping from label indices to label names

    Returns:
        Tuple of (label, confidence_score)
    """
    import torch
    import torch.nn.functional as F

    logits = torch.tensor(output.outputs.probs)
    probs = F.softmax(logits, dim=0)
    top_idx = torch.argmax(probs).item()
    top_prob = probs[top_idx].item()

    # Use label name if mapping available, otherwise use index
    label = label_map.get(top_idx, str(top_idx)) if label_map else str(top_idx)
    return label, top_prob


def main(
    hub_model_id: str,
    src_dataset_hub_id: str,
    output_dataset_hub_id: str,
    inference_column: str = "text",
    batch_size: int = 10_000,
    hf_token: Optional[str] = None,
):
    """
    Main classification pipeline.

    Args:
        hub_model_id: Hugging Face model ID for classification
        src_dataset_hub_id: Input dataset on Hugging Face Hub
        output_dataset_hub_id: Where to save results on Hugging Face Hub
        inference_column: Column name containing text to classify
        batch_size: Number of examples to process at once
        hf_token: Hugging Face authentication token
    """
    from vllm import LLM

    # GPU check
    check_gpu_availability()

    # Authentication
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        logger.error(
            "HF_TOKEN is required. Set via --hf-token or HF_TOKEN environment variable."
        )
        sys.exit(1)

    # vLLM auto-detects the sequence-classification runner from the model
    # architecture (*ForSequenceClassification); the task= arg was removed in 0.12.0.
    logger.info(f"Loading model: {hub_model_id}")
    llm = LLM(model=hub_model_id)

    # Get label mapping if available
    id2label = get_model_id2label(hub_model_id)

    # Load dataset
    logger.info(f"Loading dataset: {src_dataset_hub_id}")
    dataset = load_dataset(src_dataset_hub_id, split="train")
    total_examples = len(dataset)
    logger.info(f"Dataset loaded with {total_examples:,} examples")

    # Extract text column
    if inference_column not in dataset.column_names:
        logger.error(
            f"Column '{inference_column}' not found. Available columns: {dataset.column_names}"
        )
        sys.exit(1)

    prompts = dataset[inference_column]

    # Process in batches
    logger.info(f"Starting classification with batch size {batch_size:,}")
    all_results = []

    for batch in tqdm(
        list(partition_all(batch_size, prompts)),
        desc="Processing batches",
        unit="batch",
    ):
        batch_results = llm.classify(batch)
        all_results.append(batch_results)

    # Flatten results
    outputs = list(concat(all_results))

    # Extract labels and probabilities
    logger.info("Extracting predictions...")
    labels_and_probs = [get_top_label(output, id2label) for output in outputs]

    # Add results to dataset
    dataset = dataset.add_column("label", [label for label, _ in labels_and_probs])
    dataset = dataset.add_column("prob", [prob for _, prob in labels_and_probs])

    # Push to hub
    logger.info(f"Pushing results to: {output_dataset_hub_id}")
    dataset.push_to_hub(output_dataset_hub_id, token=HF_TOKEN)
    logger.info("✅ Classification complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Classify text data using vLLM and save results to Hugging Face Hub",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Local GPU (Linux + CUDA): the header leaves vllm/torch/transformers to
  # the HF Jobs image, so add vllm with --with.
  # Basic usage
  uv run --with vllm classify-dataset.py model/name input-dataset output-dataset
  
  # With custom column and batch size
  uv run --with vllm classify-dataset.py model/name input-dataset output-dataset \\
    --inference-column prompt \\
    --batch-size 50000
  
  # Using environment variable for token
  HF_TOKEN=hf_xxx uv run --with vllm classify-dataset.py model/name input-dataset output-dataset
            """,
        )

        parser.add_argument(
            "hub_model_id",
            help="Hugging Face model ID for classification (e.g., bert-base-uncased)",
        )
        parser.add_argument(
            "src_dataset_hub_id",
            help="Input dataset on Hugging Face Hub (e.g., username/dataset-name)",
        )
        parser.add_argument(
            "output_dataset_hub_id", help="Output dataset name on Hugging Face Hub"
        )
        parser.add_argument(
            "--inference-column",
            type=str,
            default="text",
            help="Column containing text to classify (default: text)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=10_000,
            help="Batch size for inference (default: 10,000)",
        )
        parser.add_argument(
            "--hf-token",
            type=str,
            help="Hugging Face token (can also use HF_TOKEN env var)",
        )

        args = parser.parse_args()

        main(
            hub_model_id=args.hub_model_id,
            src_dataset_hub_id=args.src_dataset_hub_id,
            output_dataset_hub_id=args.output_dataset_hub_id,
            inference_column=args.inference_column,
            batch_size=args.batch_size,
            hf_token=args.hf_token,
        )
    else:
        # Show HF Jobs example when run without arguments
        print("""
vLLM Classification Script
=========================

This script requires arguments. For usage information:
    uv run classify-dataset.py --help

Example HF Jobs command (the header sets the vLLM image, l4x1 and the HF_TOKEN secret):
    hf jobs uv run \\
        https://huggingface.co/datasets/uv-scripts/vllm/resolve/main/classify-dataset.py \\
        davanstrien/ModernBERT-base-is-new-arxiv-dataset \\
        username/input-dataset \\
        username/output-dataset \\
        --inference-column text \\
        --batch-size 100000
        """)
