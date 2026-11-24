# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     # 2.21.0 is the first release that ships the OCRia-nano backend (`liteparse.models`).
#     "liteparse>=2.21.0",
#     "datasets>=3.1.0",
#     "huggingface-hub[hf_transfer]",
#     "pillow",
#     "tqdm",
# ]
#
# [tool.hf-jobs]
# flavor = "cpu-upgrade"
# secrets = ["HF_TOKEN"]
# ///
"""
OCR page images to Markdown on CPU with **OCRia-nano-1**, run through LiteParse.

[OCRia-nano-1](https://huggingface.co/llamaindex/OCRia-nano-1) is an 8.1M-parameter
pipeline from LlamaIndex: a text-line detector, a script router, four small
recognizers (Latin, CJK, right-to-left, and a world-script model) and a table and
layout model. It has no standalone inference script. The model files run inside
[LiteParse](https://github.com/run-llama/liteparse), which this recipe calls once
per page.

CPU-ONLY: no GPU is used and LiteParse has no GPU path for this model. Run it on a
`cpu-upgrade` flavor. `--num-workers` runs pages in parallel across cores.

Where it sits: far cheaper per page than the VLM recipes in this folder, with more
character errors and weaker structure on hard layouts. Use it as a cheap first pass
or a CPU baseline, and compare a few outputs with their pages before you scale up.

LICENCE: the weights are under the OCRia-nano-1 Model License (a modified
OpenRAIL-M), not an open-source licence. Research, evaluation and personal use are
free, and so is commercial use by organisations under $2M in revenue and under $2M
in funding. Above those thresholds you need a commercial licence from LlamaIndex,
and so does any use in a document parsing or OCR service that competes with
LlamaParse or LlamaCloud, whatever your size. Read
https://huggingface.co/llamaindex/OCRia-nano-1/blob/main/LICENSE before you use it.
LiteParse itself is Apache-2.0.

HF Jobs examples (the header sets the flavor and the HF_TOKEN secret, hf CLI 1.32+):

    # Try it on seven pages
    hf jobs uv run \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ocria-nano.py \\
        uv-scripts/ocr-demo your-username/ocr-demo-ocria-nano

    # A sample of a larger dataset
    hf jobs uv run \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ocria-nano.py \\
        INPUT_DATASET OUTPUT_DATASET --max-samples 100 --shuffle
"""

import argparse
import importlib.metadata
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "llamaindex/OCRia-nano-1"
MODEL_NAME = "OCRia-nano-1"
MODEL_PACKAGE = "ocria-nano"  # the name LiteParse uses for this model
LICENSE_URL = "https://huggingface.co/llamaindex/OCRia-nano-1/blob/main/LICENSE"
SCRIPT_URL = "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ocria-nano.py"

# Error sentinel written to the output column when a single image fails, so one
# bad page never sinks the whole run (matches the other recipes in this repo).
OCR_ERROR = "[OCR ERROR]"


def ensure_output_columns_free(dataset, columns, overwrite=False):
    """Fail fast if an output column would collide with an existing input column.

    Adding a column that already exists silently overwrites it (e.g. a
    ground-truth `text`/`markdown` column) or crashes on push with a
    duplicate-column error only *after* OCR has run. Catch it up front. With
    overwrite=True, drop the clashing column(s) here instead (logged).
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
    logger.error(
        "Choose a different --output-column, or pass --overwrite to replace them."
    )
    sys.exit(1)


def to_pil(image: Image.Image | dict[str, Any] | str) -> Image.Image:
    """Coerce a datasets image cell to an RGB PIL image.

    Handles the three shapes a HF image column yields: a decoded PIL image, a
    `{"bytes": ...}` dict, or a file path string.
    """
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and image.get("bytes"):
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, dict) and image.get("path"):
        pil_img = Image.open(image["path"])
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    return pil_img.convert("RGB")


def load_image(dataset, index: int, image_column: str) -> Any:
    """Read one image cell, or None if it cannot be decoded.

    `datasets` decodes the image when the row is read, so a corrupt file raises
    here. Returning None lets that page get the error sentinel like any other
    failed page, without losing the pages already done.
    """
    try:
        return dataset[index][image_column]
    except Exception as e:  # noqa: BLE001 — one bad page shouldn't sink the run
        logger.error(f"Error reading image at row {index}: {e}")
        return None


def ocr_image(parser, image: Any, tmp_dir: Path) -> str:
    """Run LiteParse on one image, returning Markdown (or the error sentinel).

    LiteParse takes a file path, so the page is written to a PNG first. Each
    thread writes its own file name, so parallel pages do not overwrite each other.
    """
    if image is None:
        return OCR_ERROR
    page_path = tmp_dir / f"page-{threading.get_ident()}.png"
    try:
        # Light PNG compression: the file is read once and deleted, so speed matters more than size.
        to_pil(image).save(page_path, compress_level=1)
        result = parser.parse(page_path)
        # Markdown only: `page.text` still holds the page headers and footers that
        # LiteParse removes from the Markdown, so it is not a fallback.
        return "\n\n".join(page.markdown for page in result.pages).strip()
    except Exception as e:  # noqa: BLE001 — one bad page shouldn't sink the run
        logger.error(f"Error OCR'ing image: {e}")
        return OCR_ERROR


def run_ocr(parser, dataset, image_column: str, num_workers: int) -> list[str]:
    """OCR every row's image, preserving dataset order.

    LiteParse releases the GIL while it parses, so threads give real parallelism
    across pages. We process in chunks so only a chunk's worth of decoded images
    is held in memory at once — the full dataset stays on disk.
    """
    n = len(dataset)
    results: list[str] = []
    chunk = max(num_workers * 4, 16)

    with (
        tempfile.TemporaryDirectory() as tmp,
        tqdm(total=n, desc="OCR", unit="img") as pbar,
    ):
        tmp_dir = Path(tmp)
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for start in range(0, n, chunk):
                stop = min(start + chunk, n)
                images = [
                    load_image(dataset, i, image_column) for i in range(start, stop)
                ]
                for text in pool.map(
                    lambda img: ocr_image(parser, img, tmp_dir), images
                ):
                    results.append(text)
                    pbar.update(1)
    return results


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
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                ).stdout
                names = [line.strip() for line in listed.splitlines() if line.strip()]
                if names:
                    gpu_name, gpu_count = names[0], len(names)
            except (OSError, subprocess.SubprocessError):
                pass
        if gpu_count == 0 and hardware == "gpu":
            return ""  # a GPU run whose GPU cannot be identified: do not guess a CPU flavor

        # Whole-word match, so "L4" does not match "NVIDIA L40S" ("NVIDIA A100-SXM4-80GB" -> A100).
        gpu_words = gpu_name.upper().replace("-", " ").split()
        candidates = []
        for flavor in HfApi(token=False).list_jobs_hardware():
            accelerator = flavor.accelerator
            if gpu_count == 0 and accelerator is None:
                candidates.append(flavor)
            elif (
                gpu_count > 0
                and accelerator is not None
                and accelerator.model.upper() in gpu_words
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
            return (
                abs(listed_cores - cores) / listed_cores
                + abs(listed_memory - memory_gb) / listed_memory
            )

        return min(candidates, key=distance).name
    except Exception:
        return ""


def create_dataset_card(
    source_dataset: str,
    num_samples: int,
    processing_time: str,
    liteparse_version: str,
    num_workers: int,
    output_column: str,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCRia-nano-1 run."""
    on_jobs = os.environ.get("JOB_ID") is not None  # set by HF Jobs in-container
    hw = jobs_flavor()  # flavor name, e.g. "cpu-upgrade"; "" when it cannot be told
    origin = (
        (
            "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
            + (f" (`{hw}`)" if hw else "")
        )
        if on_jobs
        else "Generated"
    )
    jobs_tag = "\n- hf-jobs" if on_jobs else ""
    return f"""---
tags:
- ocr
- document-processing
- ocria-nano
- liteparse
- markdown
- uv-script{jobs_tag}
- generated
---

# Document OCR using {MODEL_NAME}

This dataset contains Markdown OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using [{MODEL_NAME}](https://huggingface.co/{MODEL_ID}), an 8.1M-parameter CPU pipeline from LlamaIndex, run through [LiteParse](https://github.com/run-llama/liteparse).

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{MODEL_ID}](https://huggingface.co/{MODEL_ID})
- **Runtime**: LiteParse `{liteparse_version}` (CPU)
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `{output_column}`
- **Dataset Split**: `{split}`
- **Workers**: {num_workers}

## Model Information

{MODEL_NAME} is a detector and recognizer pipeline, not a vision-language model:
- Runs on CPU, with no GPU
- A router sends each text line to a Latin, CJK, right-to-left or world-script recognizer
- A layout model adds headings, lists and tables to the Markdown
- Page headers and footers are removed from the Markdown by default

The weights are under the [OCRia-nano-1 Model License]({LICENSE_URL}), a modified OpenRAIL-M licence with limits on commercial use. The licence makes no claim on the outputs. Read it before you use the model.

## Dataset Structure

The dataset contains all original columns plus:
- `{output_column}`: The recognised text as Markdown (`{OCR_ERROR}` where a page failed)
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Reproduction

{origin} with the [`ocria-nano.py`]({SCRIPT_URL}) recipe from [uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:

```bash
hf jobs uv run {SCRIPT_URL} \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --split {split}
```
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    num_workers: int = 0,
    hf_token: str | None = None,
    split: str = "train",
    max_samples: int | None = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    output_column: str = "markdown",
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    config: str | None = None,
    create_pr: bool = False,
):
    """Process images from an HF dataset through OCRia-nano-1."""
    from liteparse import LiteParse, models

    start_time = datetime.now(timezone.utc)

    if num_workers <= 0:
        # On CPU Jobs os.cpu_count() can report the host's cores (64 on a
        # cpu-upgrade node with an 8-core quota). Jobs sets CPU_CORES to the quota.
        num_workers = int(float(os.environ.get("CPU_CORES") or os.cpu_count() or 1))
    liteparse_version = importlib.metadata.version("liteparse")
    logger.info(f"Using {MODEL_NAME} via LiteParse {liteparse_version}")
    logger.info(f"CPU-only run with {num_workers} worker(s)")
    logger.info(f"The weights are under the OCRia-nano-1 Model License: {LICENSE_URL}")

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN and not dry_run:
        login(token=HF_TOKEN)

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

    # Download the model files (about 70 MB) and build one parser shared by all
    # threads. Each input is one page, so LiteParse's own per-document OCR pool
    # is set to 1 and the parallelism comes from running pages side by side.
    logger.info(f"Pulling the {MODEL_PACKAGE} model package")
    models.pull(MODEL_PACKAGE, quiet=True)
    parser = LiteParse(
        ocr_backend=MODEL_PACKAGE,
        output_format="markdown",
        num_workers=1,
        quiet=True,
    )

    logger.info(f"Processing {len(dataset)} images -> column '{output_column}'")
    ocr_start = time.perf_counter()
    all_outputs = run_ocr(parser, dataset, image_column, num_workers)
    ocr_seconds = time.perf_counter() - ocr_start
    if all_outputs:
        logger.info(
            f"OCR took {ocr_seconds:.1f}s for {len(all_outputs)} pages "
            f"({len(all_outputs) / ocr_seconds:.2f} pages/sec)"
        )

    n_errors = sum(1 for t in all_outputs if t == OCR_ERROR)
    n_empty = sum(1 for t in all_outputs if not t)
    if n_errors:
        logger.warning(f"{n_errors}/{len(all_outputs)} images failed OCR ({OCR_ERROR})")
    if n_empty:
        logger.info(f"{n_empty}/{len(all_outputs)} pages returned no text")

    processing_duration = datetime.now(timezone.utc) - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Inference info tracking. `model_id` + `column_name` are the fields consumers
    # (e.g. ocr-bench) read to find the text column and the leaderboard label.
    inference_entry = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "column_name": output_column,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": "liteparse",
        "liteparse_version": liteparse_version,
        "num_workers": num_workers,
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

    if dry_run:
        logger.info("--dry-run: skipping push to Hub.")
        preview = next((t for t in all_outputs if t and t != OCR_ERROR), "")
        logger.info(f"Sample OCR output (first non-empty, truncated):\n{preview[:500]}")
        logger.info(
            f"Done (dry run). {len(dataset)} rows OCR'd in {processing_time_str}."
        )
        return

    # Push to hub with retry and XET fallback
    logger.info(f"Pushing to {output_dataset}")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.warning("Disabling XET (fallback to HTTP upload)")
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            commit_info = dataset.push_to_hub(
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
        except Exception as e:  # noqa: BLE001 — any upload failure gets a retry
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
        num_samples=len(dataset),
        processing_time=processing_time_str,
        liteparse_version=liteparse_version,
        num_workers=num_workers,
        output_column=output_column,
        image_column=image_column,
        split=split,
    )
    card = DatasetCard(card_content)
    # With --create-pr the data went to a pull request, so the card goes to the same
    # one. Pushing it to main would change the repo before the PR is merged.
    card_revision = getattr(commit_info, "pr_revision", None) if create_pr else None
    card.push_to_hub(output_dataset, token=HF_TOKEN, revision=card_revision)

    logger.info(f"Done! {MODEL_NAME} OCR processing complete.")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")

    if verbose:
        logger.info("--- Resolved package versions ---")
        for pkg in ["liteparse", "datasets", "pyarrow", "pillow"]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")
        logger.info("--- End versions ---")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("OCRia-nano-1 — CPU OCR to Markdown, via LiteParse")
        print("=" * 70)
        print("\n8.1M-parameter detector + recognizer + layout pipeline. No GPU.")
        print(f"Weights licence (not open source): {LICENSE_URL}")
        print("\nExamples:")
        print("\n1. Basic OCR:")
        print("   uv run ocria-nano.py input-dataset output-dataset")
        print("\n2. Test with a small sample, no push:")
        print("   uv run ocria-nano.py large-dataset out --max-samples 5 --dry-run")
        print("\n3. Running on HF Jobs (CPU flavor):")
        print("   # header sets flavor + HF_TOKEN secret (hf CLI 1.32+)")
        print("   hf jobs uv run \\")
        print(f"     {SCRIPT_URL} \\")
        print("       input-dataset output-dataset --max-samples 100 --shuffle")
        print("\nFor full help: uv run ocria-nano.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="CPU OCR to Markdown with OCRia-nano-1, run through LiteParse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run ocria-nano.py my-docs analyzed-docs
  uv run ocria-nano.py large-dataset test --max-samples 50 --shuffle --dry-run
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
        "--num-workers",
        type=int,
        default=0,
        help="Pages to OCR in parallel (default: 0 = all CPU cores)",
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
        help="Column name for output text (default: markdown, for cross-model consistency)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column if it already exists in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run OCR but do NOT push to the Hub (local smoke testing).",
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
        num_workers=args.num_workers,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        output_column=args.output_column,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        verbose=args.verbose,
        config=args.config,
        create_pr=args.create_pr,
    )
