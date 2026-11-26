# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=3.1.0",
#     "pyarrow>=17.0.0,<18.0.0",
#     "huggingface-hub",
#     "pillow",
#     "vllm",
#     "toolz",
#     "torch",
# ]
#
# [[tool.uv.index]]
# url = "https://wheels.vllm.ai/nightly/cu129"
#
# [tool.uv]
# prerelease = "allow"
# override-dependencies = ["transformers>=5.1.0"]
# ///

"""
Convert document images to markdown using GLM-OCR with vLLM.

v2: Incremental uploads via CommitScheduler + checkpoint/resume support.
Results are saved as parquet shards per batch and uploaded in the background,
so a crash or upload failure never loses completed OCR work. Use --resume to
pick up from the last completed batch after an interruption.

GLM-OCR is a compact 0.9B parameter OCR model achieving 94.62% on OmniDocBench V1.5.
Uses CogViT visual encoder with GLM-0.5B language decoder and Multi-Token Prediction
(MTP) loss for fast, accurate document parsing.

NOTE: Requires vLLM nightly wheels from cu129 variant (GLM-OCR added in v0.16.0,
PR #33005) and transformers>=5.1.0 (GLM-OCR support landed in stable release).
Uses https://wheels.vllm.ai/nightly/cu129 which has x86_64 wheels.
First run may take a few minutes to download and install dependencies.

Features:
- 0.9B parameters (ultra-compact)
- 94.62% on OmniDocBench V1.5 (SOTA for sub-1B models)
- Text recognition with markdown output
- LaTeX formula recognition
- Table extraction (HTML format)
- Multilingual: zh, en, fr, es, ru, de, ja, ko
- MIT licensed
- Incremental parquet uploads (v2) — never lose results
- Checkpoint/resume (v2) — pick up where you left off

Model: zai-org/GLM-OCR
vLLM: Requires vLLM nightly build + transformers>=5.1.0
Performance: 94.62% on OmniDocBench V1.5
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from datasets import load_dataset
from huggingface_hub import CommitScheduler, DatasetCard, HfApi, login
from PIL import Image
from toolz import partition_all
# Disable vLLM's FlashInfer sampler: it JIT-compiles a CUDA kernel needing nvcc, which the
# default uv-script image lacks (engine init then crashes). Greedy OCR doesn't use it; this
# lets the plain default-image command work. On the vllm/vllm-openai image it's a harmless no-op.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = "zai-org/GLM-OCR"

# Task prompts as specified by the model
TASK_PROMPTS = {
    "ocr": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
}

# Metadata keys that must match between runs for --resume
_RESUMABLE_KEYS = [
    "input_dataset",
    "split",
    "shuffle",
    "seed",
    "max_samples",
    "batch_size",
    "source_dataset_sha",
    "temperature",
    "top_p",
    "repetition_penalty",
    "max_tokens",
    "task",
    "gpu_memory_utilization",
    "max_model_len",
]

METADATA_FILENAME = "_run_metadata.json"


class CleanupScheduler(CommitScheduler):
    """CommitScheduler that deletes local parquet files after successful upload.

    Prevents disk from filling up on long-running HF Jobs.
    """

    def push_to_hub(self):
        parquet_files = list(self.folder_path.glob("train-*.parquet"))
        if not parquet_files:
            return None
        result = super().push_to_hub()
        if result is not None:
            for f in parquet_files:
                f.unlink(missing_ok=True)
                logger.info(f"Cleaned up uploaded shard: {f.name}")
        return result


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
    task: str = "ocr",
) -> List[Dict]:
    """
    Create chat message for OCR processing.

    GLM-OCR uses a chat format with an image and a task prompt prefix.
    Supported tasks: ocr, formula, table.
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

    # Convert to RGB
    pil_img = pil_img.convert("RGB")

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    prompt_text = TASK_PROMPTS.get(task, TASK_PROMPTS["ocr"])

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def build_run_metadata(
    *,
    input_dataset: str,
    split: str,
    shuffle: bool,
    seed: int,
    max_samples: int | None,
    batch_size: int,
    source_dataset_sha: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_tokens: int,
    task: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    total_batches: int,
    total_samples: int,
    model: str = MODEL,
) -> dict:
    """Build the run metadata dict for persistence."""
    return {
        "input_dataset": input_dataset,
        "split": split,
        "shuffle": shuffle,
        "seed": seed,
        "max_samples": max_samples,
        "batch_size": batch_size,
        "source_dataset_sha": source_dataset_sha,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens,
        "task": task,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "total_batches": total_batches,
        "total_samples": total_samples,
        "model": model,
        "script": "glm-ocr-v2.py",
        "created_at": datetime.now().isoformat(),
    }


def save_run_metadata(metadata: dict, folder: Path) -> Path:
    """Save run metadata to the staging folder."""
    path = folder / METADATA_FILENAME
    path.write_text(json.dumps(metadata, indent=2))
    return path


def fetch_remote_metadata(api: HfApi, repo_id: str, token: str | None) -> dict | None:
    """Download _run_metadata.json from the Hub dataset repo. Returns None if missing."""
    try:
        from huggingface_hub.utils import EntryNotFoundError

        local_path = api.hf_hub_download(
            repo_id=repo_id,
            filename=f"data/{METADATA_FILENAME}",
            repo_type="dataset",
            token=token,
        )
        return json.loads(Path(local_path).read_text())
    except (EntryNotFoundError, Exception) as e:
        logger.debug(f"Could not fetch remote metadata: {e}")
        return None


def find_completed_batches(api: HfApi, repo_id: str, token: str | None) -> set[int]:
    """List completed batch numbers from existing parquet files on the Hub."""
    completed = set()
    try:
        files = api.list_repo_tree(
            repo_id=repo_id, path_in_repo="data", repo_type="dataset", token=token
        )
        for item in files:
            name = item.rfilename if hasattr(item, "rfilename") else str(item)
            # Extract batch number from e.g. "data/train-00003-of-00043.parquet"
            basename = name.split("/")[-1] if "/" in name else name
            if basename.startswith("train-") and basename.endswith(".parquet"):
                try:
                    batch_num = int(basename.split("-")[1])
                    completed.add(batch_num)
                except (IndexError, ValueError):
                    continue
    except Exception as e:
        logger.warning(f"Could not list remote files: {e}")
    return completed


def verify_run_metadata(current: dict, remote: dict) -> list[str]:
    """Compare current run params against saved metadata. Returns list of mismatches."""
    mismatches = []
    for key in _RESUMABLE_KEYS:
        current_val = current.get(key)
        remote_val = remote.get(key)
        if current_val != remote_val:
            mismatches.append(f"  {key}: current={current_val!r}, saved={remote_val!r}")
    return mismatches


def create_dataset_card(
    source_dataset: str,
    model: str,
    num_samples: int,
    processing_time: str,
    batch_size: int,
    max_model_len: int,
    max_tokens: int,
    gpu_memory_utilization: float,
    temperature: float,
    top_p: float,
    task: str,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]
    task_desc = {
        "ocr": "text recognition",
        "formula": "formula recognition",
        "table": "table recognition",
    }

    return f"""---
tags:
- ocr
- document-processing
- glm-ocr
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using GLM-OCR, a compact 0.9B OCR model achieving SOTA performance.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Task**: {task_desc.get(task, task)}
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
- **Temperature**: {temperature}
- **Top P**: {top_p}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

GLM-OCR is a compact, high-performance OCR model:
- 0.9B parameters
- 94.62% on OmniDocBench V1.5
- CogViT visual encoder + GLM-0.5B language decoder
- Multi-Token Prediction (MTP) loss for efficiency
- Multilingual: zh, en, fr, es, ru, de, ja, ko
- MIT licensed

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted text in markdown format
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Reproduction

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr-v2.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --batch-size {batch_size} \\
    --task {task}
```

Generated with [UV Scripts](https://huggingface.co/uv-scripts) (glm-ocr-v2.py)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    max_model_len: int = 8192,
    max_tokens: int = 8192,
    temperature: float = 0.01,
    top_p: float = 0.00001,
    repetition_penalty: float = 1.1,
    gpu_memory_utilization: float = 0.8,
    task: str = "ocr",
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    output_column: str = "markdown",
    overwrite: bool = False,
    verbose: bool = False,
    config: str = None,
    create_pr: bool = False,
    resume: bool = False,
    force: bool = False,
    upload_every: int = 5,
):
    """Process images from HF dataset through GLM-OCR model with incremental uploads."""

    check_cuda_availability()

    start_time = datetime.now()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    api = HfApi(token=HF_TOKEN)

    # Validate task
    if task not in TASK_PROMPTS:
        logger.error(f"Unknown task '{task}'. Supported: {list(TASK_PROMPTS.keys())}")
        sys.exit(1)

    # Warn about --create-pr fallback
    if create_pr:
        logger.warning(
            "CommitScheduler does not support PRs. "
            "Falling back to v1 behavior (single push_to_hub at end)."
        )
        if resume:
            logger.error("--resume is not compatible with --create-pr (v1 fallback).")
            sys.exit(1)

    logger.info(f"Using model: {MODEL}")
    logger.info(f"Task: {task} (prompt: '{TASK_PROMPTS[task]}')")

    # Get source dataset SHA for resume verification
    logger.info(f"Fetching source dataset info: {input_dataset}")
    source_info = api.dataset_info(input_dataset, token=HF_TOKEN)
    source_sha = source_info.sha
    logger.info(f"Source dataset SHA: {source_sha}")

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)

    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    # Fail fast if the output column would collide with an existing input column.
    # The incremental path derives batch datasets via dataset.select(...), so dropping
    # it here (on --overwrite) also keeps those per-batch add_columns clean.
    dataset = ensure_output_columns_free(dataset, [output_column], overwrite=overwrite)

    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    total_samples = len(dataset)
    total_batches = (total_samples + batch_size - 1) // batch_size

    # Build metadata for this run
    run_metadata = build_run_metadata(
        input_dataset=input_dataset,
        split=split,
        shuffle=shuffle,
        seed=seed,
        max_samples=max_samples,
        batch_size=batch_size,
        source_dataset_sha=source_sha,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        task=task,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        total_batches=total_batches,
        total_samples=total_samples,
    )

    # Resume logic
    completed_batches: set[int] = set()
    if resume and not force:
        logger.info("Checking for existing run to resume...")
        remote_meta = fetch_remote_metadata(api, output_dataset, HF_TOKEN)
        if remote_meta is None:
            logger.error(
                f"No existing metadata found at {output_dataset}. "
                "Cannot resume. Run without --resume to start fresh."
            )
            sys.exit(1)

        mismatches = verify_run_metadata(run_metadata, remote_meta)
        if mismatches:
            logger.error("Run parameters do not match saved metadata:")
            for m in mismatches:
                logger.error(m)
            logger.error("Use --force to ignore and start fresh.")
            sys.exit(1)

        completed_batches = find_completed_batches(api, output_dataset, HF_TOKEN)
        if completed_batches:
            logger.info(
                f"Found {len(completed_batches)} completed batches: "
                f"{sorted(completed_batches)}"
            )
        else:
            logger.info("No completed batches found. Starting from beginning.")
    elif force:
        logger.info("--force: ignoring any existing data, starting fresh.")

    # Initialize vLLM
    logger.info("Initializing vLLM with GLM-OCR")
    logger.info("This may take a few minutes on first run...")
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )

    # Inference info entry for this run
    inference_entry = {
        "model_id": MODEL,
        "model_name": "GLM-OCR",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens,
    }

    logger.info(f"Processing {total_samples} images in batches of {batch_size}")
    logger.info(f"Output will be written to column: {output_column}")

    # --- create-pr fallback: v1 behavior (collect all, push once) ---
    if create_pr:
        all_outputs = []
        processed = 0

        for batch_num, batch_indices in enumerate(
            partition_all(batch_size, range(total_samples))
        ):
            batch_indices = list(batch_indices)
            batch_images = [dataset[i][image_column] for i in batch_indices]

            logger.info(
                f"Batch {batch_num + 1}/{total_batches} "
                f"({processed}/{total_samples} images done)"
            )

            try:
                batch_messages = [
                    make_ocr_message(img, task=task) for img in batch_images
                ]
                outputs = llm.chat(batch_messages, sampling_params)
                for output in outputs:
                    text = output.outputs[0].text.strip()
                    all_outputs.append(text)
                processed += len(batch_images)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                all_outputs.extend(["[OCR ERROR]"] * len(batch_images))
                processed += len(batch_images)

        processing_duration = datetime.now() - start_time
        processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

        dataset = dataset.add_column(output_column, all_outputs)

        # Inference info
        if "inference_info" in dataset.column_names:

            def update_inference_info(example):
                try:
                    existing = (
                        json.loads(example["inference_info"])
                        if example["inference_info"]
                        else []
                    )
                except (json.JSONDecodeError, TypeError):
                    existing = []
                existing.append(inference_entry)
                return {"inference_info": json.dumps(existing)}

            dataset = dataset.map(update_inference_info)
        else:
            inference_list = [json.dumps([inference_entry])] * len(dataset)
            dataset = dataset.add_column("inference_info", inference_list)

        logger.info(f"Pushing to {output_dataset} (create-pr mode)")
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
                    create_pr=True,
                    commit_message=f"Add {MODEL} OCR results ({len(dataset)} samples)"
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
                    logger.error("All upload attempts failed.")
                    sys.exit(1)

        _push_dataset_card(
            output_dataset=output_dataset,
            input_dataset=input_dataset,
            num_samples=total_samples,
            processing_time=processing_time_str,
            batch_size=batch_size,
            max_model_len=max_model_len,
            max_tokens=max_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            temperature=temperature,
            top_p=top_p,
            task=task,
            image_column=image_column,
            split=split,
            token=HF_TOKEN,
        )
        _log_completion(
            total_samples, processing_duration, processing_time_str, output_dataset
        )
        return

    # --- v2 behavior: incremental parquet uploads via CommitScheduler ---
    staging_dir = Path(tempfile.mkdtemp(prefix="glm-ocr-v2-"))
    logger.info(f"Staging directory: {staging_dir}")

    # Save metadata to staging dir so it gets uploaded with the first commit
    save_run_metadata(run_metadata, staging_dir)

    processed = 0
    skipped = 0

    with CleanupScheduler(
        repo_id=output_dataset,
        repo_type="dataset",
        folder_path=staging_dir,
        path_in_repo="data",
        every=upload_every,
        private=private,
        token=HF_TOKEN,
    ) as _scheduler:
        for batch_num, batch_indices in enumerate(
            partition_all(batch_size, range(total_samples))
        ):
            batch_indices = list(batch_indices)

            # Skip already-completed batches on resume
            if batch_num in completed_batches:
                skipped += len(batch_indices)
                logger.info(
                    f"Batch {batch_num + 1}/{total_batches} — skipped (already uploaded)"
                )
                continue

            batch_images = [dataset[i][image_column] for i in batch_indices]

            logger.info(
                f"Batch {batch_num + 1}/{total_batches} "
                f"({processed + skipped}/{total_samples} images done, "
                f"{skipped} skipped)"
            )

            try:
                batch_messages = [
                    make_ocr_message(img, task=task) for img in batch_images
                ]
                outputs = llm.chat(batch_messages, sampling_params)
                batch_texts = [o.outputs[0].text.strip() for o in outputs]
            except Exception as e:
                logger.error(f"Error processing batch {batch_num + 1}: {e}")
                batch_texts = ["[OCR ERROR]"] * len(batch_images)

            # Build batch dataset from the source subset
            batch_ds = dataset.select(batch_indices)
            batch_ds = batch_ds.add_column(output_column, batch_texts)

            # Handle inference_info per row
            if "inference_info" in batch_ds.column_names:
                info_values = []
                for i in range(len(batch_ds)):
                    raw = batch_ds[i]["inference_info"]
                    try:
                        existing = json.loads(raw) if raw else []
                    except (json.JSONDecodeError, TypeError):
                        existing = []
                    existing.append(inference_entry)
                    info_values.append(json.dumps(existing))
                batch_ds = batch_ds.remove_columns("inference_info")
                batch_ds = batch_ds.add_column("inference_info", info_values)
            else:
                info_values = [json.dumps([inference_entry])] * len(batch_ds)
                batch_ds = batch_ds.add_column("inference_info", info_values)

            # Save shard to staging dir
            shard_name = f"train-{batch_num:05d}-of-{total_batches:05d}.parquet"
            shard_path = staging_dir / shard_name
            batch_ds.to_parquet(shard_path)
            logger.info(f"Saved shard: {shard_name} ({len(batch_ds)} rows)")

            processed += len(batch_indices)

    # Context manager exit triggers final flush — blocks until upload completes
    logger.info("All batches processed. Final upload flush complete.")

    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    # Push dataset card as separate commit
    _push_dataset_card(
        output_dataset=output_dataset,
        input_dataset=input_dataset,
        num_samples=total_samples,
        processing_time=processing_time_str,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        temperature=temperature,
        top_p=top_p,
        task=task,
        image_column=image_column,
        split=split,
        token=HF_TOKEN,
    )

    _log_completion(
        total_samples, processing_duration, processing_time_str, output_dataset
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


def _push_dataset_card(
    *,
    output_dataset: str,
    input_dataset: str,
    num_samples: int,
    processing_time: str,
    batch_size: int,
    max_model_len: int,
    max_tokens: int,
    gpu_memory_utilization: float,
    temperature: float,
    top_p: float,
    task: str,
    image_column: str,
    split: str,
    token: str | None,
):
    """Create and push the dataset card."""
    logger.info("Creating dataset card")
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        model=MODEL,
        num_samples=num_samples,
        processing_time=processing_time,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        temperature=temperature,
        top_p=top_p,
        task=task,
        image_column=image_column,
        split=split,
    )
    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=token)


def _log_completion(
    total_samples: int,
    processing_duration,
    processing_time_str: str,
    output_dataset: str,
):
    """Log final completion stats."""
    logger.info("Done! GLM-OCR processing complete.")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")
    if processing_duration.total_seconds() > 0:
        logger.info(
            f"Processing speed: "
            f"{total_samples / processing_duration.total_seconds():.2f} images/sec"
        )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("GLM-OCR Document Processing (v2 — incremental uploads)")
        print("=" * 70)
        print("\n0.9B OCR model - 94.62% on OmniDocBench V1.5")
        print("\nv2 improvements:")
        print("  - Incremental parquet uploads (never lose results)")
        print("  - Checkpoint/resume (--resume)")
        print("  - Background upload every N minutes (--upload-every)")
        print("\nTask modes:")
        print("  ocr      - Text recognition (default)")
        print("  formula  - LaTeX formula recognition")
        print("  table    - Table extraction")
        print("\nExamples:")
        print("\n1. Basic OCR:")
        print("   uv run glm-ocr-v2.py input-dataset output-dataset")
        print("\n2. Resume after interruption:")
        print("   uv run glm-ocr-v2.py input-dataset output-dataset --resume")
        print("\n3. Running on HF Jobs:")
        print("   hf jobs uv run --flavor l4x1 \\")
        print("     -s HF_TOKEN \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr-v2.py \\"
        )
        print("       input-dataset output-dataset --batch-size 16")
        print("\nFor full help: uv run glm-ocr-v2.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using GLM-OCR v2 (incremental uploads + resume)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Task modes:
  ocr      Text recognition to markdown (default)
  formula  LaTeX formula recognition
  table    Table extraction

v2 features:
  Parquet shards uploaded incrementally via CommitScheduler.
  Use --resume to pick up from the last completed batch.
  Use --force to ignore existing data and start fresh.

Examples:
  uv run glm-ocr-v2.py my-docs analyzed-docs
  uv run glm-ocr-v2.py docs results --task formula
  uv run glm-ocr-v2.py large-dataset test --max-samples 50 --shuffle
  uv run glm-ocr-v2.py large-dataset test --resume
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
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length (default: 8192)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192, capped by max-model-len)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Sampling temperature (default: 0.01, near-greedy for OCR accuracy)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.00001,
        help="Top-p sampling parameter (default: 0.00001, near-greedy)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty to prevent loops (default: 1.1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
    )
    parser.add_argument(
        "--task",
        choices=["ocr", "formula", "table"],
        default="ocr",
        help="OCR task mode (default: ocr)",
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
        help="Create a pull request instead of pushing directly (falls back to v1 single-push behavior)",
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
        help="Column name for output text (default: markdown)",
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
    # v2-specific args
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed batch (requires matching run metadata on Hub)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing data on Hub and start fresh (skips metadata check)",
    )
    parser.add_argument(
        "--upload-every",
        type=int,
        default=5,
        help="CommitScheduler upload interval in minutes (default: 5)",
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
        repetition_penalty=args.repetition_penalty,
        gpu_memory_utilization=args.gpu_memory_utilization,
        task=args.task,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        output_column=args.output_column,
        overwrite=args.overwrite,
        verbose=args.verbose,
        config=args.config,
        create_pr=args.create_pr,
        resume=args.resume,
        force=args.force,
        upload_every=args.upload_every,
    )
