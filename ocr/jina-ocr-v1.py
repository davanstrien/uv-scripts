# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "vllm>=0.21",
#     "tqdm",
#     "toolz",
#     "torch",
# ]
#
# [tool.hf-jobs]
# flavor = "a10g-small"
# secrets = ["HF_TOKEN"]
# env = { UV_TORCH_BACKEND = "auto" }
# ///

"""
UNSUPPORTED: Broken (2026-09-23): vLLM 0.30 DeepEncoder kernel error; pinning vllm/vllm-openai:v0.29.0 likely fixes it (as for deepseek-ocr-vllm.py). CC-BY-NC-4.0 model. See models.json (`support`).

Convert document images to markdown using jina-ocr-v1 with vLLM (offline batch).

jina-ocr-v1 (Jina AI, 2026-09) is a DeepSeek-OCR fine-tune: the DeepEncoder vision
tower plus a 3B mixture-of-experts decoder with ~570M active parameters, and a
FastMTP speculative-decoding head (one draft block reused for K=3 steps). The
card reports 91.14 on OmniDocBench v1.6 and 83.4 on olmOCR-Bench. Licence:
CC-BY-NC-4.0 (non-commercial).

The model ships its own vLLM glue in the checkpoint (`deepseek_ocr_mtp.py`):
architecture registration, the speculative config, and an n-gram repetition
stop built on vLLM's own `repetition_detection`. This recipe follows the card's
offline pattern exactly: put the snapshot on the import path, `register()`
once, then `LLM(**vllm_llm_kwargs(...))` and `llm.chat(...)`.

Run on HF Jobs (hardware and the HF_TOKEN secret come from the script's
[tool.hf-jobs] header; `hf` CLI 1.32+):

  hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/jina-ocr-v1.py \\
      <input-dataset> <output-dataset>

Notes:
- `--spec 0` disables speculative decoding (A/B it: FastMTP is a latency trick
  that helps most at low concurrency; at batch 16 the gain may be small).
- `--no-repetition-stop` disables the card's n-gram stop.
- Uses vLLM from PyPI on the default uv image; the first run spends a few
  minutes installing it. `--image vllm/vllm-openai:<tag>` skips that.

Model: jinaai/jina-ocr-v1
"""

import argparse
import io
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Union

from datasets import load_dataset
from huggingface_hub import DatasetCard, login, snapshot_download
from PIL import Image
from toolz import partition_all
from tqdm.auto import tqdm

# The default uv-script image has no nvcc; vLLM's FlashInfer sampler would JIT a
# kernel and crash engine init. Greedy OCR does not need it. No-op on vllm images.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# register() adds architectures to this process's ModelRegistry; the engine worker only
# sees them if it is forked, not spawned. Nothing below initialises CUDA before LLM().
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "jinaai/jina-ocr-v1"
SCRIPT_NAME = "jina-ocr-v1.py"
SCRIPT_URL = f"https://huggingface.co/datasets/uv-scripts/ocr/raw/main/{SCRIPT_NAME}"


def check_cuda_availability():
    """GPU check WITHOUT initialising CUDA in this process.

    torch.cuda.is_available() would initialise CUDA here, and vLLM then forces the
    `spawn` start method for its engine worker. A spawned worker does not inherit the
    ModelRegistry entries that register() adds in this process, so the custom
    architecture is "not supported" in the worker (seen on the first smoke run).
    nvidia-smi answers the same question with no CUDA context.
    """
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        logger.error("nvidia-smi not found. This script requires a GPU.")
        sys.exit(1)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"nvidia-smi failed: {e}. This script requires a GPU.")
        sys.exit(1)
    logger.info(f"GPU: {out.splitlines()[0] if out else 'unknown'}")


def ensure_output_columns_free(dataset, columns, overwrite=False):
    """Fail fast if an output column would overwrite an existing input column."""
    clash = [c for c in columns if c in dataset.column_names]
    if not clash:
        return dataset
    if overwrite:
        logger.warning(f"--overwrite: replacing existing column(s) {clash}")
        return dataset.remove_columns(clash)
    logger.error(f"Output column(s) {clash} already exist in the input dataset (columns: {dataset.column_names}).")
    logger.error("Choose a different --output-column, or pass --overwrite to replace them.")
    sys.exit(1)


def to_pil(image: Union[Image.Image, Dict[str, Any], str]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict) and "bytes" in image:
        return Image.open(io.BytesIO(image["bytes"]))
    if isinstance(image, str):
        return Image.open(image)
    raise ValueError(f"Unsupported image type: {type(image)}")


def load_model_glue(revision: str | None):
    """Download the checkpoint and import its vLLM helpers.

    vLLM v1 starts its engine core as a *spawned* process, which does not inherit
    sys.path edits, so the snapshot directory also goes into PYTHONPATH before LLM()
    is created. The architecture override points at `deepseek_ocr_mtp`, which the
    worker must be able to import.
    """
    snapshot = snapshot_download(MODEL_ID, revision=revision)
    os.environ["PYTHONPATH"] = snapshot + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, snapshot)
    import deepseek_ocr_mtp  # noqa: E402  (lives in the checkpoint)

    return snapshot, deepseek_ocr_mtp


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
                    capture_output=True, text=True, timeout=10, check=True,
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
            return abs(listed_cores - cores) / listed_cores + abs(listed_memory - memory_gb) / listed_memory

        return min(candidates, key=distance).name
    except Exception:
        return ""


def create_dataset_card(
    source_dataset: str,
    num_samples: int,
    processing_time: str,
    batch_size: int,
    max_model_len: int,
    max_tokens: int,
    spec: int,
    repetition_stop: bool,
    image_column: str,
    output_column: str,
    split: str,
    image_count_per_sec: float,
) -> str:
    on_jobs = os.environ.get("JOB_ID") is not None
    hw = jobs_flavor()  # flavor name, e.g. "a10g-small"; "" on CPU
    origin = (
        "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
        + (f" (`{hw}`)" if hw else "")
    ) if on_jobs else "Generated"
    tags = ["ocr", "document-processing", "jina-ocr", "markdown", "uv-script", "generated"]
    if on_jobs:
        tags.append("hf-jobs")
    tag_lines = "\n".join(f"- {t}" for t in tags)

    return f"""---
tags:
{tag_lines}
---

# Document OCR using jina-ocr-v1

Markdown OCR of the images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
using [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) (DeepSeek-OCR fine-tune, ~570M active parameters,
FastMTP speculative decoding). The model is released under CC-BY-NC-4.0; check that licence before
reusing these outputs commercially.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{MODEL_ID}](https://huggingface.co/{MODEL_ID})
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time} (~{image_count_per_sec:.2f} images/second including model load)
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `{output_column}`
- **Dataset Split**: `{split}`
- **Batch Size**: {batch_size}
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **Speculative decoding (FastMTP draft depth)**: {spec}
- **N-gram repetition stop**: {"on" if repetition_stop else "off"}

## Dataset Structure

All original columns plus:
- `{output_column}`: the extracted text as markdown (HTML tables, LaTeX math)
- `inference_info`: JSON list tracking the OCR models applied to this dataset

## Reproduction

{origin} with the [`{SCRIPT_NAME}`]({SCRIPT_URL}) recipe from [uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:

```bash
hf jobs uv run {SCRIPT_URL} \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --batch-size {batch_size} \\
    --spec {spec}
```
"""


def main():
    ap = argparse.ArgumentParser(description="jina-ocr-v1 batch OCR with vLLM (offline)")
    ap.add_argument("input_dataset")
    ap.add_argument("output_dataset")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--output-column", default="markdown")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing output column")
    ap.add_argument("--split", default="train")
    ap.add_argument("--input-config", default=None, help="Config (subset) name of the input dataset")
    ap.add_argument("--config", default=None, help="Config name for the output dataset")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--spec", type=int, default=3, help="FastMTP draft depth K; 0 disables speculative decoding")
    ap.add_argument("--no-repetition-stop", action="store_true", help="Disable the card's n-gram repetition stop")
    ap.add_argument("--prompt", default=None, help="Override the model's default OCR prompt")
    ap.add_argument("--revision", default=None, help="Model revision to pin")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--create-pr", action="store_true")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--verbose", action="store_true", help="Log resolved package versions after the run")
    args = ap.parse_args()

    check_cuda_availability()
    start_time = datetime.now()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    logger.info(f"Loading dataset: {args.input_dataset}")
    dataset = load_dataset(args.input_dataset, name=args.input_config, split=args.split)
    if args.image_column not in dataset.column_names:
        raise ValueError(f"Column '{args.image_column}' not found. Available: {dataset.column_names}")
    dataset = ensure_output_columns_free(dataset, [args.output_column], overwrite=args.overwrite)
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # --- model: the checkpoint's own vLLM glue, in the card's order ---------------
    snapshot, glue = load_model_glue(args.revision)
    from vllm import LLM

    glue.register()
    llm_kwargs = glue.vllm_llm_kwargs(
        snapshot, num_speculative_tokens=args.spec, mtp_heads=1, mtp_recursive=True,
    )
    llm_kwargs.setdefault("max_model_len", args.max_model_len)
    llm_kwargs.setdefault("gpu_memory_utilization", args.gpu_memory_utilization)
    llm_kwargs.setdefault("enable_prefix_caching", False)
    llm_kwargs.setdefault("mm_processor_cache_gb", 0)
    llm_kwargs.setdefault("limit_mm_per_prompt", {"image": 1})
    logger.info(f"Initializing vLLM with {MODEL_ID} (spec={args.spec}); this can take a few minutes")
    llm = LLM(**llm_kwargs)

    sampling_kwargs = {"max_tokens": args.max_tokens}
    if args.no_repetition_stop:
        sampling_kwargs["repetition_detection"] = None
    sampling_params = glue.vllm_sampling_params(**sampling_kwargs)
    prompt = args.prompt or glue.DEFAULT_OCR_PROMPT
    logger.info(f"Prompt: {prompt}")

    # --- inference ------------------------------------------------------------------
    all_markdown = []
    n_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    for batch_indices in tqdm(partition_all(args.batch_size, range(len(dataset))), total=n_batches, desc="jina-ocr-v1"):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][args.image_column] for i in batch_indices]
        try:
            messages = []
            for img in batch_images:
                pil_img = to_pil(img).convert("RGB")
                messages.append([{"role": "user", "content": [
                    {"type": "image_pil", "image_pil": pil_img},
                    {"type": "text", "text": prompt},
                ]}])
            outputs = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
            for output in outputs:
                all_markdown.append(output.outputs[0].text.strip())
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            all_markdown.extend(["[OCR FAILED]"] * len(batch_images))

    if args.spec > 0:
        glue.log_spec_stats(llm)

    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"
    images_per_sec = len(dataset) / max(1.0, processing_duration.total_seconds())

    # --- output dataset ---------------------------------------------------------------
    dataset = dataset.add_column(args.output_column, all_markdown)
    inference_entry = {
        "model_id": MODEL_ID,
        "model_name": "jina-ocr-v1",
        "column_name": args.output_column,
        "timestamp": datetime.now().isoformat(),
        "prompt": "custom" if args.prompt else "default",
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "speculative_tokens": args.spec,
        "repetition_stop": not args.no_repetition_stop,
        "script": SCRIPT_NAME,
        "script_url": SCRIPT_URL,
    }
    if "inference_info" in dataset.column_names:
        def update_inference_info(example):
            try:
                existing = json.loads(example["inference_info"]) if example["inference_info"] else []
            except (json.JSONDecodeError, TypeError):
                existing = []
            existing.append(inference_entry)
            return {"inference_info": json.dumps(existing)}
        dataset = dataset.map(update_inference_info)
    else:
        dataset = dataset.add_column("inference_info", [json.dumps([inference_entry])] * len(dataset))

    logger.info(f"Pushing to {args.output_dataset}")
    dataset.push_to_hub(
        args.output_dataset,
        private=args.private,
        token=hf_token,
        **({"config_name": args.config} if args.config else {}),
        create_pr=args.create_pr,
        commit_message=f"Add {MODEL_ID} OCR results ({len(dataset)} samples)" + (f" [{args.config}]" if args.config else ""),
    )
    card = DatasetCard(create_dataset_card(
        source_dataset=args.input_dataset, num_samples=len(dataset), processing_time=processing_time_str,
        batch_size=args.batch_size, max_model_len=args.max_model_len, max_tokens=args.max_tokens,
        spec=args.spec, repetition_stop=not args.no_repetition_stop, image_column=args.image_column,
        output_column=args.output_column, split=args.split, image_count_per_sec=images_per_sec,
    ))
    card.push_to_hub(args.output_dataset, token=hf_token)
    logger.info(f"Dataset available at: https://huggingface.co/datasets/{args.output_dataset}")
    logger.info(f"Processing time: {processing_time_str}")

    failed = sum(1 for m in all_markdown if m == "[OCR FAILED]")
    if failed:
        logger.warning(f"{failed} of {len(all_markdown)} rows are [OCR FAILED] (a whole batch fails together in offline mode)")

    if args.verbose:
        import importlib.metadata
        for pkg in ["vllm", "transformers", "torch", "datasets", "pillow"]:
            try:
                logger.info(f"{pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"{pkg}: not installed")


if __name__ == "__main__":
    main()
