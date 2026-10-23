# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "requests",
# ]
# ///

"""
Convert document images to markdown using LightOnOCR-2 via an in-job vLLM server.

Same model, message shape, and sampling as lighton-ocr2.py, but serves the
model behind `vllm serve` inside the job (the model card's own documented
path) and posts images concurrently — continuous batching stays fed instead
of draining at each offline batch barrier, and a bad image fails one request
instead of a whole batch. Measured ~1.8x the offline recipe's inference
throughput on a 100-page historical-scan smoke test (l4x1, concurrency 32).

This script is the *driver* half: it expects the server on localhost
(started by the job command below), loads the input dataset, posts images
concurrently, and pushes the result dataset. The driver has no torch/vllm
deps, so `uv run` starts in seconds while the server warms up in parallel.

Run on HF Jobs (standard uv-run shape — the script starts `vllm serve` itself
as a subprocess when no server is already reachable; the only thing to get
right is the --image flag, which provides the `vllm` binary):

  hf jobs uv run --detach --flavor l4x1 -s HF_TOKEN --timeout 4h \\
      --image vllm/vllm-openai:v0.22.1 \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr2-server.py \\
      <input-dataset> <output-dataset>

To use an already-running or remote endpoint instead, pass --server URL — the
script only spawns a server when the (default localhost) URL is unreachable.
The serve flags live in SERVE_ARGS below (the model card's own recommended
command: `--limit-mm-per-prompt`, `--mm-processor-cache-gb 0`,
`--no-enable-prefix-caching` — OCR never reuses images, so the caches only
cost memory).

Model: lightonai/LightOnOCR-2-1B (1B, Apache-2.0)
- Message is the image ONLY (no text prompt) — LightOnOCR-2's trained format.
- Images resized client-side so the longest dimension is 1540px (training
  resolution at 200 DPI), same as the offline recipe.
- Sampling per the card: temperature 0.2, top_p 0.9, max_tokens 4096.
"""

import argparse
import atexit
import base64
import concurrent.futures
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, Union
from urllib.parse import urlparse

import requests
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "lightonai/LightOnOCR-2-1B"

DEFAULT_TARGET_SIZE = 1540  # longest dimension; LightOnOCR-2 training resolution

# The serve command this script spawns when no server is reachable — the model
# card's own recommended flags; single source of truth for the serving config.
SERVE_ARGS = [
    "vllm", "serve", MODEL,
    "--limit-mm-per-prompt", '{"image": 1}',
    "--mm-processor-cache-gb", "0",
    "--no-enable-prefix-caching",
    "--max-model-len", "8192",
    "--gpu-memory-utilization", "0.8",
    "--port", "8000",
]

RUN_COMMAND = (
    "hf jobs uv run --detach --flavor l4x1 -s HF_TOKEN --timeout 4h \\\n"
    "    --image vllm/vllm-openai:v0.22.1 \\\n"
    "    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr2-server.py \\\n"
    "    <input-dataset> <output-dataset>"
)


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


def to_pil_image(image: Union[Image.Image, Dict[str, Any], str]) -> Image.Image:
    """Convert a dataset image cell (PIL image, bytes dict, or path) to RGB PIL."""
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    return pil_img.convert("RGB")


def encode_image(image, target_size: int) -> str:
    """RGB-convert, resize longest dimension to target_size, return base64 PNG."""
    img = to_pil_image(image)
    if target_size:
        w, h = img.size
        if max(w, h) != target_size:
            scale = target_size / max(w, h)
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def server_alive(server: str) -> bool:
    try:
        return requests.get(f"{server}/health", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def wait_for_server(server: str, timeout_s: int, proc: "subprocess.Popen | None" = None):
    logger.info(f"Waiting for server at {server}...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server_alive(server):
            logger.info("Server is ready")
            return
        if proc is not None and proc.poll() is not None:
            logger.error(f"Spawned vllm serve exited with code {proc.returncode} before becoming ready")
            sys.exit(1)
        time.sleep(10)
    logger.error(f"Server did not become ready within {timeout_s}s")
    sys.exit(1)


def ensure_server(server: str, timeout_s: int = 1800):
    """Use a reachable server; otherwise spawn `vllm serve` ourselves; else fail fast.

    Spawning is only attempted for a localhost URL — a remote --server that is
    down is the user's to fix, not ours to shadow with a local model.
    """
    if server_alive(server):
        logger.info(f"Using already-running server at {server}")
        return

    host = urlparse(server).hostname or ""
    if host not in ("127.0.0.1", "localhost", "0.0.0.0"):
        logger.info(f"Remote server {server} not up yet — waiting for it")
        wait_for_server(server, timeout_s)
        return

    if shutil.which("vllm") is None:
        logger.error("No server is running and the `vllm` binary is not on PATH.")
        logger.error("Run this script on a vLLM image so it can start the server itself:\n")
        logger.error(RUN_COMMAND)
        logger.error("\n(or start `vllm serve` yourself / pass --server URL of a running endpoint)")
        sys.exit(1)

    logger.info(f"Starting server: {' '.join(SERVE_ARGS)}")
    proc = subprocess.Popen(SERVE_ARGS)  # logs interleave with ours on stdout/stderr
    atexit.register(proc.terminate)  # don't leave a GPU server behind on local runs
    wait_for_server(server, timeout_s, proc=proc)


def ocr_one(
    server: str,
    image,
    target_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: int,
    retries: int = 2,
) -> str:
    """OCR a single image via the chat completions API. Returns raw model text."""
    b64 = encode_image(image, target_size)
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    # Image ONLY — LightOnOCR-2 uses no text prompt.
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{server}/v1/chat/completions", json=payload, timeout=timeout_s
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"request failed after {retries + 1} attempts: {last_err}")


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

        candidates = []
        for flavor in HfApi(token=False).list_jobs_hardware():
            accelerator = flavor.accelerator
            if gpu_count == 0 and accelerator is None:
                candidates.append(flavor)
            elif (
                gpu_count > 0
                and accelerator is not None
                and accelerator.model in gpu_name
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
    model: str,
    num_samples: int,
    num_errors: int,
    processing_time: str,
    images_per_sec: float,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    target_size: int,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    # Canonical provenance stamp (see AGENTS.md): Jobs claim gated on JOB_ID, set by HF Jobs in-container.
    on_jobs = os.environ.get("JOB_ID") is not None
    hw = jobs_flavor()  # flavor name, e.g. "a10g-small"; "" on CPU
    origin = (
        "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
        + (f" (`{hw}`)" if hw else "")
    ) if on_jobs else "Generated"
    jobs_tag = "\n- hf-jobs" if on_jobs else ""

    return f"""---
tags:
- ocr
- document-processing
- lighton-ocr
- markdown
- uv-script
- generated{jobs_tag}
---

# Document OCR using {model_name} (server mode)

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using LightOnOCR-2 (1B), served behind an in-job vLLM server with concurrent requests (continuous batching).

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Number of Samples**: {num_samples:,}
- **Failed Requests**: {num_errors:,} (marked `[OCR ERROR]`)
- **Processing Time**: {processing_time}
- **Throughput**: {images_per_sec:.2f} images/sec
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Mode**: vLLM server (`vllm serve`) + concurrent driver, {concurrency} concurrent requests
- **Image Column**: `{image_column}`
- **Dataset Split**: `{split}`
- **Target Image Size**: {target_size}px (longest dimension)
- **Max Output Tokens**: {max_tokens:,}
- **Temperature**: {temperature}

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted text in markdown format
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Reproduction

{origin} with the [`lighton-ocr2-server.py`](https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr2-server.py) recipe from [uv-scripts](https://huggingface.co/uv-scripts) — see the script docstring for the single `hf jobs run` command that starts the server and driver together. The offline-vLLM sibling recipe is [`lighton-ocr2.py`](https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr2.py).
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    server: str = "http://127.0.0.1:8000",
    concurrency: int = 32,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    top_p: float = 0.9,
    target_size: int = DEFAULT_TARGET_SIZE,
    request_timeout: int = 1800,
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
):
    """Process images from HF dataset through a LightOnOCR-2 vLLM server."""

    start_time = datetime.now()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    logger.info(f"Using model: {MODEL} via server {server}")

    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)

    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    dataset = ensure_output_columns_free(dataset, [output_column], overwrite=overwrite)

    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Reuse a reachable server, else spawn `vllm serve` (needs the vllm binary,
    # i.e. a vllm/vllm-openai image), else fail fast with the correct command.
    ensure_server(server)

    n = len(dataset)
    logger.info(f"Processing {n} images, concurrency {concurrency}")
    all_outputs = [None] * n
    errors = 0
    done = 0
    inference_start = time.time()
    lock = threading.Lock()

    def worker(i: int) -> None:
        nonlocal errors, done
        try:
            text = ocr_one(
                server,
                dataset[i][image_column],
                target_size,
                max_tokens,
                temperature,
                top_p,
                request_timeout,
            )
            all_outputs[i] = text.strip()
        except Exception as e:
            logger.error(f"Image {i} failed: {e}")
            all_outputs[i] = "[OCR ERROR]"
            with lock:
                errors += 1
        with lock:
            done += 1
            if done % 25 == 0 or done == n:
                rate = done / max(time.time() - inference_start, 1e-9)
                logger.info(f"{done}/{n} done ({rate:.2f} img/s, {errors} errors)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(worker, range(n)))

    inference_secs = time.time() - inference_start
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"
    images_per_sec = n / inference_secs if inference_secs else 0.0

    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Inference info tracking
    inference_entry = {
        "model_id": MODEL,
        "model_name": "LightOnOCR-2-1B",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "target_size": target_size,
        "mode": "vllm-server",
        "concurrency": concurrency,
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
                commit_message=f"Add {MODEL} OCR results ({len(dataset)} samples, server mode)"
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

    logger.info("Creating dataset card")
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        model=MODEL,
        num_samples=len(dataset),
        num_errors=errors,
        processing_time=processing_time_str,
        images_per_sec=images_per_sec,
        concurrency=concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        target_size=target_size,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("Done! LightOnOCR-2 server-mode processing complete.")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")
    logger.info(
        f"Throughput: {images_per_sec:.2f} images/sec "
        f"(inference only, excl. dataset load/push; {errors} errors)"
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("LightOnOCR-2 Document Processing (vLLM server mode)")
        print("=" * 70)
        print("\nSame model + outputs as lighton-ocr2.py, but drives an in-job")
        print("`vllm serve` with concurrent requests — no batch barriers,")
        print("per-image (not per-batch) failure isolation.")
        print("\nThe server must already be running (the job command starts")
        print("both — see the module docstring for the full `hf jobs run`).")
        print("\nExamples:")
        print("\n1. Basic OCR (server on localhost:8000):")
        print("   uv run lighton-ocr2-server.py input-dataset output-dataset")
        print("\n2. Test with a small sample:")
        print("   uv run lighton-ocr2-server.py large-dataset test --max-samples 10 --shuffle")
        print("\nFor full help: uv run lighton-ocr2-server.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using LightOnOCR-2 via an in-job vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run lighton-ocr2-server.py my-docs analyzed-docs
  uv run lighton-ocr2-server.py large-dataset test --max-samples 50 --shuffle
See the module docstring for the full `hf jobs run` command (server + driver in one job).
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
        "--server",
        default="http://127.0.0.1:8000",
        help="vLLM server base URL (default: in-job localhost:8000)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Concurrent OCR requests (default: 32; vLLM queues excess internally, "
        "so this mainly needs to be high enough to keep continuous batching fed)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096, the model card value)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2, the model card value)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9, the model card value)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help=f"Resize images so the longest dimension is this many pixels before upload "
        f"(default: {DEFAULT_TARGET_SIZE}, the model's training resolution); 0 disables",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=1800,
        help="Per-request timeout in seconds (default: 1800)",
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
        help="Column name for output text (default: markdown)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column if it already exists in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        server=args.server,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        target_size=args.target_size,
        request_timeout=args.request_timeout,
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
    )
