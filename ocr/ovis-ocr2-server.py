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
Convert document images to markdown using OvisOCR2 via an in-job vLLM server.

Same model, prompt, and outputs as ovis-ocr2.py, but serves the model behind
`vllm serve` inside the job and posts images concurrently — continuous batching
stays fed instead of draining at each offline `llm.generate()` batch barrier,
and a bad image fails one request instead of a whole batch of 16.

This script is the *driver* half: it expects the server on localhost (started
by the job command below), loads the input dataset, posts images concurrently,
postprocesses (repeat-trim + img-tag filter, as in ovis-ocr2.py), and pushes
the result dataset. The driver has no torch/vllm deps, so `uv run` starts in
seconds while the server warms up in parallel.

NOTE: OvisOCR2's card documents offline vLLM only (checked 2026-07-16) — this
server translation is ours, not the authors'. The serve flags below mirror the
card's offline args; treat A/B output parity with ovis-ocr2.py as part of any
benchmark run (same --max-samples slice, diff the markdown columns).

Run on HF Jobs (standard uv-run shape — the script starts `vllm serve` itself
as a subprocess when no server is already reachable; the only thing to get
right is the --image flag, which provides the `vllm` binary):

  hf jobs uv run --detach --flavor l4x1 -s HF_TOKEN --timeout 4h \\
      --image vllm/vllm-openai:v0.22.1 \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ovis-ocr2-server.py \\
      <input-dataset> <output-dataset>

To use an already-running or remote endpoint instead, pass --server URL — the
script only spawns a server when the (default localhost) URL is unreachable.
The serve flags live in SERVE_ARGS below, the single source of truth.

Serve-flag provenance (the card only shows offline `LLM(...)` args):
- `--no-enable-prefix-caching --mm-processor-cache-gb 0`: the recurring official
  OCR-serving pattern (DeepSeek-OCR vLLM recipe, LightOnOCR, PaddleOCR-VL recipe)
  — OCR never reuses images, the caches only cost memory.
- `--mm-processor-kwargs`: server-side equivalent of the card's per-request
  `images_kwargs` pixel bounds. The driver ALSO downscales oversized images
  client-side to the same max_pixels, so outputs match even if the server flag
  is dropped.
- The card's offline `gdn_prefill_backend="triton"` (JIT/nvcc workaround for the
  bare uv image) is NOT needed here: the vllm-openai image ships the full CUDA
  toolchain. Add `--gdn-prefill-backend triton` to the serve command if the
  default backend misbehaves.
- `enable_thinking=False` (card-mandated, Qwen3.5 templates can inject a
  thinking preamble) is passed per-request via `chat_template_kwargs`.

Model: ATH-MaaS/OvisOCR2 (0.9B, Apache-2.0, 96.58 OmniDocBench v1.6)
vLLM: stock Qwen3_5ForConditionalGeneration arch, needs vllm >= 0.22.1.
"""

import argparse
import atexit
import base64
import concurrent.futures
import io
import json
import logging
import math
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

MODEL = "ATH-MaaS/OvisOCR2"

# Fixed instruction prompt, verbatim from the model card (including the leading newline).
# The card warns outputs are tuned to this exact wording — don't "improve" it.
OCR_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    'represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_'
    '{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box '
    "coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as "
    "HTML: <table>...</table>. Transcribe all other text as standard Markdown. "
    "Preserve the original text without translation or paraphrasing."
)

# Image bounds from the model card's parser.
DEFAULT_MIN_PIXELS = 448 * 448  # 200,704
DEFAULT_MAX_PIXELS = 2880 * 2880  # 8,294,400

# The serve command this script spawns when no server is reachable — single
# source of truth for the serving configuration (see docstring for provenance).
SERVE_ARGS = [
    "vllm", "serve", MODEL,
    "--max-model-len", "32768",
    "--gpu-memory-utilization", "0.85",
    "--limit-mm-per-prompt", '{"image": 1}',
    "--no-enable-prefix-caching",
    "--mm-processor-cache-gb", "0",
    "--mm-processor-kwargs",
    f'{{"images_kwargs": {{"min_pixels": {DEFAULT_MIN_PIXELS}, "max_pixels": {DEFAULT_MAX_PIXELS}}}}}',
    "--port", "8000",
]

RUN_COMMAND = (
    "hf jobs uv run --detach --flavor l4x1 -s HF_TOKEN --timeout 4h \\\n"
    "    --image vllm/vllm-openai:v0.22.1 \\\n"
    "    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ovis-ocr2-server.py \\\n"
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


def encode_image(image, max_pixels: int) -> str:
    """RGB-convert, downscale to max_pixels if oversized, return base64 JPEG.

    The server's processor clamps to the same bound, so this only changes where
    the downscale happens — doing it client-side shrinks the request payload
    (a 30MP scan is ~4x the bytes of its 8.3MP clamp) and keeps outputs
    identical even if the serve command omits --mm-processor-kwargs.
    """
    img = to_pil_image(image)
    w, h = img.size
    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """Trim degenerate trailing repetition (verbatim port of the model card's cleanup).

    Long outputs that hit max_tokens can end in a repeated unit (a char, phrase, or
    table row); this detects the shortest repeating tail unit and keeps one copy.
    """
    n = len(text)
    if n < min_text_len:
        return text

    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue

        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1

        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len

        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len :]

    return text


def filter_image_tags(text: str) -> str:
    """Drop visual-region <img> blocks (upstream parser's default behaviour)."""
    return "\n\n".join(
        block
        for block in text.split("\n\n")
        if not block.strip().startswith('<img src="images/bbox_')
    )


def postprocess_output(text: str, keep_image_tags: bool) -> str:
    text = text.strip()
    if not keep_image_tags:
        text = filter_image_tags(text)
    return clean_truncated_repeats(text)


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
    max_pixels: int,
    max_tokens: int,
    timeout_s: int,
    retries: int = 2,
) -> str:
    """OCR a single image via the chat completions API. Returns raw model text."""
    b64 = encode_image(image, max_pixels)
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    # Image first, then text — same order the card's chat template uses.
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        # Card-mandated: Qwen3.5 templates can inject a thinking preamble otherwise.
        "chat_template_kwargs": {"enable_thinking": False},
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
    keep_image_tags: bool,
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
- ovis-ocr2
- markdown
- uv-script
- generated{jobs_tag}
---

# Document OCR using {model_name} (server mode)

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using OvisOCR2, a compact 0.9B document parsing model (96.58 on OmniDocBench v1.6), served behind an in-job vLLM server with concurrent requests (continuous batching).

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
- **Max Output Tokens**: {max_tokens:,}
- **Temperature**: 0.0 (greedy, per model card)
- **Visual-region image tags**: {"kept" if keep_image_tags else "filtered (default)"}

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted text in markdown format
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Reproduction

{origin} with the [`ovis-ocr2-server.py`](https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ovis-ocr2-server.py) recipe from [uv-scripts](https://huggingface.co/uv-scripts) — see the script docstring for the single `hf jobs run` command that starts the server and driver together. The offline-vLLM sibling recipe is [`ovis-ocr2.py`](https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ovis-ocr2.py).
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    server: str = "http://127.0.0.1:8000",
    concurrency: int = 32,
    max_tokens: int = 16384,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    request_timeout: int = 1800,
    keep_image_tags: bool = False,
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
    """Process images from HF dataset through an OvisOCR2 vLLM server."""

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
                max_pixels,
                max_tokens,
                request_timeout,
            )
            all_outputs[i] = postprocess_output(text, keep_image_tags)
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
        "model_name": "OvisOCR2",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "max_pixels": max_pixels,
        "keep_image_tags": keep_image_tags,
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
        keep_image_tags=keep_image_tags,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("Done! OvisOCR2 server-mode processing complete.")
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
        print("OvisOCR2 Document Processing (vLLM server mode)")
        print("=" * 70)
        print("\nSame model + outputs as ovis-ocr2.py, but drives an in-job")
        print("`vllm serve` with concurrent requests — no batch barriers,")
        print("per-image (not per-batch) failure isolation.")
        print("\nThe server must already be running (the job command starts")
        print("both — see the module docstring for the full `hf jobs run`).")
        print("\nExamples:")
        print("\n1. Basic OCR (server on localhost:8000):")
        print("   uv run ovis-ocr2-server.py input-dataset output-dataset")
        print("\n2. Test with a small sample:")
        print("   uv run ovis-ocr2-server.py large-dataset test --max-samples 10 --shuffle")
        print("\n3. Throughput A/B vs the offline recipe:")
        print("   run both scripts on the same --max-samples slice and compare")
        print("   the images/sec lines + diff the markdown columns")
        print("\nFor full help: uv run ovis-ocr2-server.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using OvisOCR2 via an in-job vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run ovis-ocr2-server.py my-docs analyzed-docs
  uv run ovis-ocr2-server.py large-dataset test --max-samples 50 --shuffle
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
        default=16384,
        help="Maximum tokens to generate (default: 16384, the model card value)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help=f"Maximum image pixels; larger images are downscaled client-side before "
        f"upload (default: {DEFAULT_MAX_PIXELS}, = 2880*2880, the model card value)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=1800,
        help="Per-request timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--keep-image-tags",
        action="store_true",
        help="Keep visual-region <img src=\"images/bbox_...\"> tags in the output "
        "(default: filtered, matching the upstream parser)",
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
        max_pixels=args.max_pixels,
        request_timeout=args.request_timeout,
        keep_image_tags=args.keep_image_tags,
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
