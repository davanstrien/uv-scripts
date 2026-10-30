# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "huggingface-hub",
#     "pillow",
#     "toolz",
#     "torch",
#     "tqdm",
#     "transformers",
#     "vllm>=0.11.0",  # StructuredOutputsParams API (guided_decoding removed in 0.12.0)
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.32.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "l40sx1"
# secrets = ["HF_TOKEN"]
# timeout = "1h"
# ///

"""
Classify images using Vision Language Models with vLLM.

This script processes images through VLMs to classify them into user-defined categories,
using vLLM's GuidedDecodingParams for structured output.

Examples:
    # Basic classification
    uv run vlm-classify.py \\
        username/input-dataset \\
        username/output-dataset \\
        --classes "document,photo,diagram,other"
    
    # With custom prompt and model
    uv run vlm-classify.py \\
        username/input-dataset \\
        username/output-dataset \\
        --classes "index-card,manuscript,title-page,other" \\
        --prompt "What type of historical document is this?" \\
        --model Qwen/Qwen2-VL-7B-Instruct
    
    # Quick test with sample limit
    uv run vlm-classify.py \\
        davanstrien/sloane-index-cards \\
        username/test-output \\
        --classes "index,content,other" \\
        --max-samples 10
"""

import argparse
import base64
import io
import logging
import os
import sys
from collections import Counter
from typing import List, Optional, Union, Dict, Any

import torch
from PIL import Image
from datasets import load_dataset
from huggingface_hub import login
from toolz import partition_all
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def image_to_data_uri(
    image: Union[Image.Image, Dict[str, Any]], max_size: Optional[int] = None
) -> str:
    """Convert image to base64 data URI for VLM processing.

    Args:
        image: PIL Image or dict with image bytes
        max_size: Optional maximum dimension (width or height) to resize to.
                 Preserves aspect ratio using thumbnail method.
    """
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Resize if max_size is specified and image exceeds it
    if max_size and (pil_img.width > max_size or pil_img.height > max_size):
        # Use thumbnail to preserve aspect ratio
        pil_img = pil_img.copy()  # Don't modify original
        pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Convert to RGB if necessary (handle RGBA, grayscale, etc.)
    if pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")

    # Convert to base64
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    base64_str = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{base64_str}"


def create_classification_messages(
    image: Union[Image.Image, Dict[str, Any]],
    prompt: str,
    max_size: Optional[int] = None,
) -> List[Dict]:
    """Create chat messages for VLM classification."""
    image_uri = image_to_data_uri(image, max_size=max_size)

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def main(
    input_dataset: str,
    output_dataset: str,
    classes: str,
    prompt: Optional[str] = None,
    image_column: str = "image",
    model: str = "Qwen/Qwen2-VL-7B-Instruct",
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    max_size: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: Optional[int] = None,
    split: str = "train",
    hf_token: Optional[str] = None,
    private: bool = False,
):
    """Classify images from a dataset using a Vision Language Model."""

    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("If running locally, ensure you have a CUDA-capable GPU.")
        logger.error("For cloud execution, use: hf jobs uv run --flavor l40sx1 ...")
        sys.exit(1)

    # Parse classes
    class_list = [c.strip() for c in classes.split(",")]
    logger.info(f"Classes: {class_list}")

    # Create default prompt if not provided
    if prompt is None:
        prompt = f"Classify this image into one of the following categories: {', '.join(class_list)}"
    logger.info(f"Prompt template: {prompt}")

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

    # Limit samples if requested
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Log resizing configuration
    if max_size:
        logger.info(f"Image resizing enabled: max dimension = {max_size}px")

    # Auto-detect tensor parallel size if not specified
    if tensor_parallel_size is None:
        tensor_parallel_size = torch.cuda.device_count()
        logger.info(f"Auto-detected {tensor_parallel_size} GPUs for tensor parallelism")

    # Initialize vLLM
    logger.info(f"Loading model: {model}")
    llm_kwargs = {
        "model": model,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,  # Required for some VLMs
    }

    if max_model_len:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)

    # Constrain output to one of the class labels (vLLM structured outputs).
    # StructuredOutputsParams + structured_outputs= is the API from vLLM 0.11.0+;
    # the old GuidedDecodingParams/guided_decoding= was removed in 0.12.0.
    structured = StructuredOutputsParams(choice=class_list)
    sampling_params = SamplingParams(
        temperature=0.1,  # Low temperature for consistent classification
        max_tokens=50,  # Classifications are short
        structured_outputs=structured,
    )

    # Process images in batches to avoid memory issues
    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")

    all_classifications = []

    # Process in batches using lazy loading
    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="Classifying images",
    ):
        batch_indices = list(batch_indices)

        # Load only this batch's images
        batch_images = []
        valid_batch_indices = []

        for idx in batch_indices:
            try:
                image = dataset[idx][image_column]
                batch_images.append(image)
                valid_batch_indices.append(idx)
            except Exception as e:
                logger.warning(f"Skipping image at index {idx}: {e}")
                all_classifications.append(None)

        if not batch_images:
            continue

        try:
            # Create messages for just this batch
            batch_messages = [
                create_classification_messages(img, prompt, max_size=max_size)
                for img in batch_images
            ]

            # Process with vLLM
            outputs = llm.chat(
                messages=batch_messages,
                sampling_params=sampling_params,
                use_tqdm=False,  # Already have outer progress bar
            )

            # Extract classifications
            for output in outputs:
                if output.outputs:
                    label = output.outputs[0].text.strip()
                    all_classifications.append(label)
                else:
                    all_classifications.append(None)
                    logger.warning("Empty output for an image")

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add None for failed batch
            all_classifications.extend([None] * len(batch_images))

    # Ensure we have the right number of classifications
    while len(all_classifications) < len(dataset):
        all_classifications.append(None)

    # Add classifications to dataset
    logger.info("Adding classifications to dataset...")
    dataset = dataset.add_column("label", all_classifications[: len(dataset)])

    # Push to hub
    logger.info(f"Pushing to {output_dataset}...")
    dataset.push_to_hub(output_dataset, private=private, token=HF_TOKEN)

    # Print summary
    logger.info("Classification complete!")
    logger.info(f"Processed {len(all_classifications)} images")
    logger.info(f"Output dataset: {output_dataset}")

    # Show distribution of classifications
    label_counts = Counter(all_classifications)
    logger.info("Classification distribution:")
    for label, count in sorted(label_counts.items()):
        if label is not None:  # Skip None values in summary
            percentage = (
                (count / len(all_classifications)) * 100 if all_classifications else 0
            )
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify images using Vision Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic classification
    uv run vlm-classify.py \\
        username/input-dataset \\
        username/output-dataset \\
        --classes "document,photo,diagram,other"
    
    # With custom prompt
    uv run vlm-classify.py \\
        username/input-dataset \\
        username/output-dataset \\
        --classes "index-card,manuscript,other" \\
        --prompt "What type of historical document is this?"
    
    # HF Jobs execution
    hf jobs uv run \\
        --flavor l40sx1 \\
        https://huggingface.co/datasets/uv-scripts/vllm/raw/main/vlm-classify.py \\
        username/input-dataset \\
        username/output-dataset \\
        --classes "title-page,content,index,other"
        """,
    )

    parser.add_argument(
        "input_dataset",
        help="Input dataset ID on Hugging Face Hub",
    )
    parser.add_argument(
        "output_dataset",
        help="Output dataset ID on Hugging Face Hub",
    )
    parser.add_argument(
        "--classes",
        required=True,
        help='Comma-separated list of classes (e.g., "cat,dog,other")',
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom classification prompt (default: auto-generated)",
    )
    parser.add_argument(
        "--image-column",
        default="image",
        help="Column name containing images (default: image)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Vision Language Model to use (default: Qwen/Qwen2-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum image dimension in pixels. Images larger than this will be resized while preserving aspect ratio (e.g., 768, 1024)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism (default: auto-detect)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make output dataset private",
    )

    args = parser.parse_args()

    # Show example command if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n" + "=" * 60)
        print("Example HF Jobs command:")
        print("=" * 60)
        print("""
hf jobs uv run \\
    --flavor l40sx1 \\
    -e HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; print(get_token())") \\
    https://huggingface.co/datasets/uv-scripts/vllm/raw/main/vlm-classify.py \\
    davanstrien/sloane-index-cards \\
    username/classified-cards \\
    --classes "index-card,manuscript,title-page,other" \\
    --max-size 768 \\
    --max-samples 100
        """)
        sys.exit(0)

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        classes=args.classes,
        prompt=args.prompt,
        image_column=args.image_column,
        model=args.model,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_size=args.max_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        split=args.split,
        hf_token=args.hf_token,
        private=args.private,
    )
