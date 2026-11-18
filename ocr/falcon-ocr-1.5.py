# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "torch>=2.5",
#     "torchvision",
#     "falcon-perception",
# ]
#
# [tool.uv-scripts]
# maintenance = "active"
#
# [tool.hf-jobs]
# flavor = "a10g-small"
# timeout = "1h"
# secrets = ["HF_TOKEN"]
# ///

"""
Convert document images to text using Falcon OCR 1.5 with the falcon-perception engine.

Uses the optimized OCRInferenceEngine with CUDA graphs and paged inference
for much faster throughput than the raw transformers API.

Features:
- Compact: Only 0.3B parameters
- Fast: Optimized inference with CUDA graphs
- Multi-format: Plain text, LaTeX formulas, HTML tables
- Layout-aware: Optional 2-stage pipeline (layout detection + per-region OCR)

Model: tiiuae/Falcon-OCR (pinned to the v1.5 release head)
Backend: falcon-perception (OCRInferenceEngine)
License: Apache 2.0

Versions: TII released Falcon OCR v1.5 on 2026-09-11 by overwriting the weights on
`main` of the same repo (no tag, no branch). This script is the **v1.5** recipe: it pins
the v1.5 release head (fe757d59…, the third of three same-day commits — the first one,
d259a7fb, has older code files) so the next in-place swap cannot change it silently.
v1 is its own recipe, `falcon-ocr.py`. `--revision` overrides the pin (`--revision main`
tracks whatever the repo root holds); the resolved commit is always recorded in
`inference_info` and on the dataset card.

Launch config: the `[tool.hf-jobs]` header sets the flavor (a10g-small), a 1h timeout and
the HF_TOKEN secret. The hf CLI reads it from version 1.32; older CLIs and plain `uv run`
ignore it. Explicit flags win.

Examples:
    # Basic text OCR
    uv run falcon-ocr-1.5.py input-dataset output-dataset

    # Test with small sample
    uv run falcon-ocr-1.5.py dataset test --max-samples 5 --shuffle

    # Track the repo root instead of the v1.5 pin
    uv run falcon-ocr-1.5.py dataset out --revision main

    # Run on HF Jobs (flavor + HF_TOKEN come from the [tool.hf-jobs] header on hf >= 1.32)
    hf jobs uv run falcon-ocr-1.5.py input-dataset output-dataset --max-samples 10

    # Older hf CLI: pass the launch flags yourself
    hf jobs uv run --flavor a10g-small -s HF_TOKEN falcon-ocr-1.5.py input-dataset output-dataset
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Union

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, HfApi, login
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "tiiuae/Falcon-OCR"
# Upstream swapped the weights in place (v1 -> v1.5) on 2026-09-11 with no tag or
# branch, so `main` is a moving target. This pin IS this script's v1.5 identity — the
# release head, not the first v1.5 commit (d259a7fb) whose code files were superseded
# the same day. Never loosen it to "main" here; v1 lives in falcon-ocr.py. Override
# with --revision only to reproduce old runs or to follow the repo root deliberately.
DEFAULT_REVISION = "fe757d59ecd79d4d68760162306a70a015761ad9"  # v1.5 release head, 2026-09-11
MODEL_LABEL = "Falcon-OCR-1.5"

TASK_MODES = {
    "plain": "Full-page text extraction",
}


def check_cuda_availability():
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("For cloud execution, use HF Jobs with --flavor a10g-small or similar.")
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


def prepare_image(image: Union[Image.Image, Dict[str, Any], str]) -> Image.Image:
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    return pil_img.convert("RGB")


def create_dataset_card(
    source_dataset: str,
    task_mode: str,
    num_samples: int,
    processing_time: str,
    image_column: str = "image",
    split: str = "train",
    revision: str = "main",
    model_commit: str | None = None,
) -> str:
    task_description = TASK_MODES[task_mode]
    commit_note = f" (resolved to `{model_commit}`)" if model_commit else ""
    return f"""---
tags:
- ocr
- document-processing
- falcon-ocr
- falcon-ocr-1.5
- {task_mode}
- uv-script
- generated
---

# Document Processing using Falcon OCR 1.5 ({task_mode} mode)

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using [Falcon OCR 1.5](https://huggingface.co/tiiuae/Falcon-OCR), a 0.3B early-fusion vision-language model (v1.5 weights released 2026-09-11).

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{MODEL_ID}](https://huggingface.co/{MODEL_ID})
- **Model Revision**: `{revision}`{commit_note}
- **Task Mode**: `{task_mode}` - {task_description}
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}
- **Backend**: falcon-perception (OCRInferenceEngine)

## Reproduction

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/falcon-ocr-1.5.py \\
    {source_dataset} \\
    <output-dataset> \\
    --task-mode {task_mode} \\
    --image-column {image_column} \\
    --revision {model_commit or revision}
```

Generated with [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    task_mode: str = "plain",
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
    compile: bool = True,
    cudagraph: bool = True,
    progress: bool = False,
    verbose: bool = False,
    revision: str | None = None,
):
    check_cuda_availability()
    if revision is None:
        revision = DEFAULT_REVISION
        logger.info(f"Pinning {MODEL_ID} to {MODEL_LABEL} revision {revision[:8]}")
    start_time = datetime.now()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    if task_mode not in TASK_MODES:
        raise ValueError(
            f"Invalid task_mode '{task_mode}'. Choose from: {list(TASK_MODES.keys())}"
        )

    logger.info(f"Task mode: {task_mode} - {TASK_MODES[task_mode]}")
    logger.info(f"Output column: {output_column}")

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

    # Resolve the revision to a commit sha up front so the run is reproducible even
    # when `main` moves again (it did on 2026-09-11: v1 -> v1.5, same repo).
    try:
        model_commit = HfApi().model_info(MODEL_ID, revision=revision).sha
    except Exception as e:  # offline / transient API error: run anyway, less provenance
        logger.warning(f"Could not resolve revision {revision!r} to a commit: {e}")
        model_commit = None
    logger.info(f"Model revision: {revision} -> {model_commit or 'unresolved'}")

    # Load model using falcon-perception
    logger.info(f"Loading model: {MODEL_ID} via falcon-perception engine")
    from falcon_perception import load_and_prepare_model
    from falcon_perception.data import ImageProcessor
    from falcon_perception.paged_ocr_inference import OCRInferenceEngine

    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=MODEL_ID,
        hf_revision=model_commit or revision,  # load the commit that inference_info records
        device="cuda",
        dtype="bfloat16",
        compile=compile,
    )

    image_processor = ImageProcessor(patch_size=16, merge_size=1)
    engine = OCRInferenceEngine(
        model, tokenizer, image_processor, capture_cudagraph=cudagraph
    )
    logger.info(f"Engine loaded. compile={compile}, cudagraph={cudagraph}")

    # Prepare all images
    logger.info(f"Processing {len(dataset)} images...")
    all_outputs = []

    # Batch plain OCR for better throughput
    batch_size = 8
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    for batch_idx, batch_start in enumerate(range(0, len(dataset), batch_size), 1):
        batch_end = min(batch_start + batch_size, len(dataset))
        logger.info(f"Batch {batch_idx}/{total_batches} ({batch_start}/{len(dataset)} done)")
        batch_images = []
        for i in range(batch_start, batch_end):
            try:
                batch_images.append(prepare_image(dataset[i][image_column]))
            except Exception as e:
                logger.error(f"Error preparing image {i}: {e}")
                batch_images.append(Image.new("RGB", (100, 100)))

        try:
            texts = engine.generate_plain(
                images=batch_images, use_tqdm=progress
            )
            all_outputs.extend(texts)
        except Exception as e:
            logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
            all_outputs.extend(
                [f"[OCR ERROR: {str(e)[:200]}]"] * len(batch_images)
            )

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    # Add output column
    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Track inference info
    inference_entry = {
        "model_id": MODEL_ID,
        "model_name": MODEL_LABEL,
        "model_size": "0.3B",
        "revision": revision,
        "model_commit": model_commit,
        "task_mode": task_mode,
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "backend": "falcon-perception",
    }

    if "inference_info" in dataset.column_names:
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
        inference_list = [json.dumps([inference_entry])] * len(dataset)
        dataset = dataset.add_column("inference_info", inference_list)

    # Push to hub
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
                commit_message=f"Add {MODEL_ID} OCR results ({len(dataset)} samples)"
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
    logger.info("Creating dataset card")
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        task_mode=task_mode,
        num_samples=len(dataset),
        processing_time=processing_time_str,
        image_column=image_column,
        split=split,
        revision=revision,
        model_commit=model_commit,
    )
    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("Falcon OCR processing complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")
    logger.info(
        f"Speed: {len(dataset) / processing_duration.total_seconds():.2f} images/sec"
    )

    if verbose:
        import importlib.metadata

        logger.info("--- Resolved package versions ---")
        for pkg in [
            "falcon-perception", "transformers", "torch", "datasets", "pillow"
        ]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("Falcon OCR - 0.3B Document OCR (falcon-perception engine)")
        print("=" * 70)
        print(f"\nModel: {MODEL_ID} @ {DEFAULT_REVISION[:10]} ({MODEL_LABEL}; --revision to override)")
        print("License: Apache 2.0")
        print("\nTask Modes:")
        for mode, description in TASK_MODES.items():
            print(f"  {mode:10} - {description}")
        print("\nExamples:")
        print("  uv run falcon-ocr.py input-dataset output-dataset")
        print("  uv run falcon-ocr.py dense-docs output --task-mode layout")
        print("\nFor full help: uv run falcon-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using Falcon OCR (0.3B, falcon-perception engine)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("input_dataset", help="Input dataset ID from Hugging Face Hub")
    parser.add_argument("output_dataset", help="Output dataset ID for Hugging Face Hub")
    parser.add_argument(
        "--image-column", default="image",
        help="Column containing images (default: image)",
    )
    parser.add_argument(
        "--task-mode", choices=list(TASK_MODES.keys()), default="plain",
        help="Task type: plain (default), layout",
    )
    parser.add_argument("--hf-token", help="Hugging Face API token")
    parser.add_argument(
        "--split", default="train", help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--max-samples", type=int,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make output dataset private",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle dataset before processing",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--output-column", default="markdown",
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
        help="Config/subset name for Hub (for benchmarking multiple models)",
    )
    parser.add_argument(
        "--create-pr", action="store_true",
        help="Create a pull request instead of pushing directly",
    )
    parser.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--no-cudagraph", action="store_true",
        help="Disable CUDA graph capture",
    )
    parser.add_argument(
        "--progress", action="store_true",
        help="Show per-image progress bar from the inference engine",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Log resolved package versions",
    )
    parser.add_argument(
        "--revision", default=None,
        help=f"Model repo revision (branch, tag or commit sha). Default: the {MODEL_LABEL} "
        f"pin {DEFAULT_REVISION[:10]}. Pass 'main' to follow the repo root (v1.5 since "
        "2026-09-11; same as this pin today) or any commit to reproduce a run; the resolved commit is recorded "
        "in inference_info.",
    )

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        task_mode=args.task_mode,
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
        compile=not args.no_compile,
        cudagraph=not args.no_cudagraph,
        progress=args.progress,
        verbose=args.verbose,
        revision=args.revision,
    )
