#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sglang[all]",
#     "flashinfer-python",
#     "transformers",
#     "torch",
#     "datasets",
#     "huggingface-hub[hf_transfer]",
# ]
#
# [tool.hf-jobs]
# image = "lmsysorg/sglang:v0.5.20-cu130"
# flavor = "l4x1"
# secrets = ["HF_TOKEN"]
# timeout = "1h"
# ///

"""
Classify text columns in Hugging Face datasets using SGLang with reasoning-aware models.

This script provides efficient GPU-based classification with optional reasoning support,
optimized for models like SmolLM3-3B that use <think> tokens for chain-of-thought.

Example:
    # Fast classification without reasoning
    uv run classify-dataset-sglang.py \\
        --input-dataset imdb \\
        --column text \\
        --labels "positive,negative" \\
        --output-dataset user/imdb-classified

    # Complex classification with reasoning
    uv run classify-dataset-sglang.py \\
        --input-dataset arxiv-papers \\
        --column abstract \\
        --labels "reasoning_systems,agents,multimodal,robotics,other" \\
        --output-dataset user/arxiv-classified \\
        --reasoning

HF Jobs example:
    hf jobs uv run --flavor l4x1 \\
        https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-dataset-sglang.py \\
        --input-dataset user/emails \\
        --column content \\
        --labels "spam,ham" \\
        --output-dataset user/emails-classified \\
        --reasoning
"""

import argparse
import logging
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import json
import re

import torch
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, get_token
import sglang as sgl

# Default model - SmolLM3 with reasoning capabilities
DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B"

# Minimum text length for valid classification
MIN_TEXT_LENGTH = 3

# Maximum text length (in characters) to avoid context overflow
MAX_TEXT_LENGTH = 4000

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify text in HuggingFace datasets using SGLang with reasoning support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--input-dataset",
        type=str,
        required=True,
        help="Input dataset ID on Hugging Face Hub",
    )
    parser.add_argument(
        "--column", type=str, required=True, help="Name of the text column to classify"
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Comma-separated list of classification labels (e.g., 'positive,negative')",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        required=True,
        help="Output dataset ID on Hugging Face Hub",
    )

    # Optional arguments
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use for classification (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable reasoning mode (allows model to think through complex cases)",
    )
    parser.add_argument(
        "--save-reasoning",
        action="store_true",
        help="Save reasoning traces to a separate column (requires --reasoning)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token (default: auto-detect from HF_TOKEN env var or huggingface-cli login)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation (default: 0.1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens to generate (default: 500 for reasoning, 50 for non-reasoning)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )
    parser.add_argument(
        "--grammar-backend",
        type=str,
        default="xgrammar",
        choices=["outlines", "xgrammar", "llguidance"],
        help="Grammar backend for structured outputs (default: xgrammar)",
    )

    return parser.parse_args()


def preprocess_text(text: str) -> str:
    """Preprocess text for classification."""
    if not text or not isinstance(text, str):
        return ""

    # Strip whitespace
    text = text.strip()

    # Truncate if too long
    if len(text) > MAX_TEXT_LENGTH:
        text = f"{text[:MAX_TEXT_LENGTH]}..."

    return text


def validate_text(text: str) -> bool:
    """Check if text is valid for classification."""
    return bool(text and len(text) >= MIN_TEXT_LENGTH)


def create_classification_prompt(text: str, labels: List[str], reasoning: bool) -> str:
    """Create a prompt for classification with optional reasoning mode."""
    if reasoning:
        system_prompt = "You are a helpful assistant that thinks step-by-step before answering."
    else:
        system_prompt = "You are a helpful assistant. /no_think"
    
    user_prompt = f"""Classify this text as one of: {', '.join(labels)}

Text: {text}

Classification:"""
    
    # Format as a conversation
    return f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"


def create_ebnf_grammar(labels: List[str]) -> str:
    """Create an EBNF grammar that constrains output to one of the given labels."""
    # Escape any special characters in labels
    escaped_labels = [f'"{label}"' for label in labels]
    choices = ' | '.join(escaped_labels)
    return f"root ::= {choices}"


def parse_reasoning_output(output: str, label: str) -> Optional[str]:
    """Extract reasoning from output if present."""
    # Look for thinking tags
    if "<think>" in output and "</think>" in output:
        start = output.find("<think>")
        end = output.find("</think>") + len("</think>")
        reasoning = output[start:end]
        return reasoning
    return None


def classify_batch_with_sglang(
    engine: sgl.Engine,
    texts: List[str],
    labels: List[str],
    args: argparse.Namespace
) -> List[Dict[str, Any]]:
    """Classify texts using SGLang with optional reasoning."""
    
    # Prepare prompts
    prompts = []
    valid_indices = []
    
    for i, text in enumerate(texts):
        processed_text = preprocess_text(text)
        if validate_text(processed_text):
            prompt = create_classification_prompt(processed_text, labels, args.reasoning)
            prompts.append(prompt)
            valid_indices.append(i)
    
    if not prompts:
        return [{"label": None, "reasoning": None} for _ in texts]
    
    # Set max tokens based on reasoning mode
    max_tokens = args.max_tokens if args.reasoning else 50
    
    # Create EBNF grammar for label constraints
    ebnf_grammar = create_ebnf_grammar(labels)
    
    # Set up sampling parameters with EBNF constraint
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": max_tokens,
        "ebnf": ebnf_grammar,  # This ensures output is one of the valid labels
    }
    
    try:
        # Generate with structured output constraint
        outputs = engine.generate(prompts, sampling_params)
        
        # Process outputs
        results = [{"label": None, "reasoning": None} for _ in texts]
        
        for idx, output in enumerate(outputs):
            original_idx = valid_indices[idx]
            
            # The output text should be just the label due to EBNF constraint
            classification = output["text"].strip().strip('"')  # Remove quotes if present
            
            # Extract reasoning if present and requested
            reasoning = None
            if args.reasoning and args.save_reasoning:
                # Get the full output including reasoning
                # Note: We need to check if SGLang provides access to full output with reasoning
                reasoning = parse_reasoning_output(output["text"], classification)
            
            results[original_idx] = {
                "label": classification,
                "reasoning": reasoning
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error during batch classification: {e}")
        # Return None labels for all texts in case of error
        return [{"label": None, "reasoning": None} for _ in texts]


def main():
    args = parse_args()

    # Validate reasoning arguments
    if args.save_reasoning and not args.reasoning:
        logger.error("--save-reasoning requires --reasoning to be enabled")
        sys.exit(1)

    # Check authentication early
    logger.info("Checking authentication...")
    token = args.hf_token or (os.environ.get("HF_TOKEN") or get_token())

    if not token:
        logger.error("No authentication token found. Please either:")
        logger.error("1. Run 'huggingface-cli login'")
        logger.error("2. Set HF_TOKEN environment variable")
        logger.error("3. Pass --hf-token argument")
        sys.exit(1)

    # Validate token by checking who we are
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        logger.info(f"Authenticated as: {user_info['name']}")
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        logger.error("Please check your token is valid")
        sys.exit(1)

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with GPU support or use HF Jobs.")
        sys.exit(1)

    logger.info(f"CUDA available. Using device: {torch.cuda.get_device_name(0)}")

    # Parse and validate labels
    labels = [label.strip() for label in args.labels.split(",")]
    if len(labels) < 2:
        logger.error("At least two labels are required for classification.")
        sys.exit(1)
    logger.info(f"Classification labels: {labels}")

    # Load dataset
    logger.info(f"Loading dataset: {args.input_dataset}")
    try:
        dataset = load_dataset(args.input_dataset, split=args.split)

        # Limit samples if specified
        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            logger.info(f"Limited dataset to {len(dataset)} samples")

        logger.info(f"Loaded {len(dataset)} samples from split '{args.split}'")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Verify column exists
    if args.column not in dataset.column_names:
        logger.error(f"Column '{args.column}' not found in dataset.")
        logger.error(f"Available columns: {dataset.column_names}")
        sys.exit(1)

    # Extract texts
    texts = dataset[args.column]

    # Initialize SGLang Engine
    logger.info(f"Initializing SGLang Engine with model: {args.model}")
    logger.info(f"Reasoning mode: {'enabled' if args.reasoning else 'disabled (fast mode)'}")
    logger.info(f"Grammar backend: {args.grammar_backend}")
    
    try:
        # Determine reasoning parser based on model
        reasoning_parser = None
        if "smollm3" in args.model.lower() or "qwen" in args.model.lower():
            reasoning_parser = "qwen"  # Uses <think> tokens
        elif "deepseek-r1" in args.model.lower():
            reasoning_parser = "deepseek-r1"
        
        engine_kwargs = {
            "model_path": args.model,
            "trust_remote_code": True,
            "dtype": "auto",
            "grammar_backend": args.grammar_backend,
        }
        
        if reasoning_parser and args.reasoning:
            engine_kwargs["reasoning_parser"] = reasoning_parser
            logger.info(f"Using reasoning parser: {reasoning_parser}")
        
        engine = sgl.Engine(**engine_kwargs)
        logger.info("SGLang engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SGLang: {e}")
        sys.exit(1)

    # Process in batches
    logger.info(f"Starting classification with batch size {args.batch_size}...")
    all_results = []
    
    for i in range(0, len(texts), args.batch_size):
        batch_end = min(i + args.batch_size, len(texts))
        batch_texts = texts[i:batch_end]
        
        logger.info(f"Processing batch {i//args.batch_size + 1}/{(len(texts) + args.batch_size - 1)//args.batch_size}")
        
        batch_results = classify_batch_with_sglang(
            engine, batch_texts, labels, args
        )
        all_results.extend(batch_results)

    # Extract labels and reasoning
    all_labels = [r["label"] for r in all_results]
    all_reasoning = [r["reasoning"] for r in all_results] if args.save_reasoning else None

    # Add classifications to dataset
    dataset = dataset.add_column("classification", all_labels)
    
    # Add reasoning column if requested
    if args.save_reasoning and args.reasoning:
        dataset = dataset.add_column("reasoning", all_reasoning)
        logger.info("Added reasoning traces to dataset")

    # Calculate statistics
    valid_count = sum(1 for label in all_labels if label is not None)
    invalid_count = len(all_labels) - valid_count
    
    if invalid_count > 0:
        logger.warning(
            f"{invalid_count} texts were too short or invalid for classification"
        )

    # Show classification distribution
    label_counts = {label: all_labels.count(label) for label in labels}
    logger.info("Classification distribution:")
    for label, count in label_counts.items():
        percentage = count / len(all_labels) * 100 if all_labels else 0
        logger.info(f"  {label}: {count} ({percentage:.1f}%)")
    if invalid_count > 0:
        none_percentage = invalid_count / len(all_labels) * 100
        logger.info(f"  Invalid/Skipped: {invalid_count} ({none_percentage:.1f}%)")

    # Log success rate
    success_rate = (valid_count / len(all_labels) * 100) if all_labels else 0
    logger.info(f"Classification success rate: {success_rate:.1f}%")

    # Save to Hub
    logger.info(f"Pushing dataset to Hub: {args.output_dataset}")
    try:
        commit_msg = f"Add classifications using {args.model} with SGLang"
        if args.reasoning:
            commit_msg += " (reasoning mode)"
        
        dataset.push_to_hub(
            args.output_dataset,
            token=token,
            commit_message=commit_msg,
        )
        logger.info(
            f"Successfully pushed to: https://huggingface.co/datasets/{args.output_dataset}"
        )
    except Exception as e:
        logger.error(f"Failed to push to Hub: {e}")
        sys.exit(1)

    # Clean up
    engine.shutdown()
    logger.info("SGLang engine shutdown complete")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Example HF Jobs commands:")
        print("\n# Fast classification (no reasoning):")
        print("hf jobs uv run \\")
        print("  --flavor l4x1 \\")
        print("  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-dataset-sglang.py \\")
        print("  --input-dataset stanfordnlp/imdb \\")
        print("  --column text \\")
        print("  --labels 'positive,negative' \\")
        print("  --output-dataset user/imdb-classified")
        print("\n# Complex classification with reasoning:")
        print("hf jobs uv run \\")
        print("  --flavor l4x1 \\")
        print("  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-dataset-sglang.py \\")
        print("  --input-dataset arxiv-papers \\")
        print("  --column abstract \\")
        print("  --labels 'reasoning_systems,agents,multimodal,robotics,other' \\")
        print("  --output-dataset user/arxiv-classified \\")
        print("  --reasoning --save-reasoning")
        sys.exit(0)

    main()