# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=3.1.0",
#     "huggingface-hub>=1.0",
#     "pillow",
# ]
#
# [tool.hf-jobs]
# image = "cr.hf.co/davanstrien/minicpm-ocr:1.0-vllm-5d35ce3"
# flavor = "l4x1"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# ///

"""
Convert document images to markdown with OpenBMB's MiniCPM-OCR 1.0 pipeline.

MiniCPM-OCR 1.0 (1.3B, Apache-2.0) is a region reader: the official SDK finds layout regions
with PP-DocLayoutV3, crops them, sends each crop to the model with the prompt for its type
(`Text Recognition:`, `Table Recognition:`, `Formula Recognition:`, ...) and assembles the page
as Markdown (tables as HTML, formulas as LaTeX). This script runs that pipeline unchanged over a
Hub dataset of page images and adds a `markdown` column.

  hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/minicpm-ocr.py \\
      input-dataset output-dataset --max-samples 20

Runtime image (from the [tool.hf-jobs] header): cr.hf.co/davanstrien/minicpm-ocr, built from
the Dockerfile linked below. It holds:
- vLLM nightly 5d35ce3. MiniCPMOcrForConditionalGeneration landed in vLLM main on 23 Oct 2026
  and is not in a release yet. Drop the custom image for a stock vllm/vllm-openai tag once a
  release has the architecture.
- transformers at OpenBMB's pin (549e5b4).
- The minicpm-ocr SDK 1.0.1 in its own venv (/opt/sdk) with PaddlePaddle GPU, as OpenBMB's docs
  require (SDK and inference service in separate environments), and the PP-DocLayoutV3 layout
  model baked in at /opt/minicpm-ocr-assets.

Server settings follow OpenBMB's SDK tutorial (docs/minicpmvocr_tutorial_en.md): max-model-len
40960, 32 sequences, image limit 36, `downsample_mode` 4x, thinking off, DSpark speculative
decoding with the matching draft model (`--no-dspark` for plain decoding). The SDK's bundled
config sets sampling and per-region prompts; nothing here overrides them.

Model: https://huggingface.co/openbmb/MiniCPM-OCR-1.0
SDK: https://github.com/OpenBMB/MiniCPM-OCR
Image: https://huggingface.co/containers/davanstrien/minicpm-ocr
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import DatasetCard, login, snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = "openbmb/MiniCPM-OCR-1.0"
MODEL_REVISION = "a59a4acf215e34c6d73149e9eafdb88ed2086e39"
DRAFT_MODEL = "openbmb/MiniCPM-OCR-1.0-DSpark"  # must come from the same release as MODEL
DRAFT_REVISION = "31e3feaf80e0061cd6aa73db23480d2d691d6452"
SERVED_NAME = "minicpm-ocr"
PORT = 18499
SDK_PYTHON = "/opt/sdk/bin/python"
LAYOUT_MODEL_DIR = "/opt/minicpm-ocr-assets/models/PP-DocLayoutV3_infer"

# Runs inside the SDK venv: the SDK's documented Python API (docs/minicpmvocr_api_en.md),
# with the bundled config and only the input, server and layout path filled in.
SDK_RUNNER = """
import json, sys
from importlib.resources import files
from minicpmvocr import load_config, pipelined_run
args = json.loads(sys.argv[1])
cfg = load_config(str(files("minicpmvocr").joinpath("configs/minicpmvocr.yaml")), overrides={
    "input": {"image_dir": args["image_dir"]},
    "run": {"run_root": args["run_root"], "run_tag": args["run_tag"]},
    "engine": {"server_url": args["server_url"], "model_name": args["model_name"], "max_model_len": 40960},
    "layout": {"model_dir": args["layout_dir"], "device": "cuda:0"},
    "pipeline": {"overlap_layout": True},
})
pipelined_run(cfg)
print(json.dumps({"run_dir": cfg["_meta"]["run_dir"]}))
"""


def check_image():
    """Fail fast, naming the fix, when not on the MiniCPM-OCR image."""
    if not (shutil.which("vllm") and Path(SDK_PYTHON).exists() and Path(LAYOUT_MODEL_DIR).exists()):
        logger.error("This script needs the cr.hf.co/davanstrien/minicpm-ocr image (vLLM with "
                     "MiniCPM-OCR support + the minicpm-ocr SDK + the layout model).")
        logger.error("Run it with `hf jobs uv run`, which reads the image from the script header, "
                     "or pass --image cr.hf.co/davanstrien/minicpm-ocr:1.0-vllm-5d35ce3.")
        sys.exit(1)
    if subprocess.run(["nvidia-smi"], capture_output=True).returncode != 0:
        logger.error("No GPU found. This script needs a CUDA GPU.")
        sys.exit(1)


def start_server(dspark: bool) -> subprocess.Popen:
    """Start vLLM with OpenBMB's tutorial settings and wait until it serves the model."""
    model_dir = snapshot_download(MODEL, revision=MODEL_REVISION)
    cmd = [
        "vllm", "serve", model_dir, "--served-model-name", SERVED_NAME,
        "--host", "127.0.0.1", "--port", str(PORT),
        "--max-model-len", "40960", "--max-num-batched-tokens", "40960", "--max-num-seqs", "32",
        "--limit-mm-per-prompt", '{"image":36,"video":0}',
        "--mm-processor-kwargs", '{"downsample_mode":"4x"}',
        "--gpu-memory-utilization", "0.85", "--trust-request-mm-kwargs",
        "--default-chat-template-kwargs", '{"enable_thinking":false}',
    ]
    if dspark:
        draft_dir = snapshot_download(DRAFT_MODEL, revision=DRAFT_REVISION)
        cmd += ["--speculative-config", json.dumps({
            "method": "dspark", "model": draft_dir, "num_speculative_tokens": 15,
            "enable_adaptive_verification": False})]
    # The uv venv's VIRTUAL_ENV and PATH entry must not shadow the image's vLLM.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    logger.info("Starting vLLM: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env)
    deadline = time.time() + 1800
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"vLLM exited during startup (code {proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
                if SERVED_NAME in r.read().decode():
                    logger.info("vLLM ready")
                    return proc
        except OSError:
            pass
        time.sleep(5)
    proc.kill()
    sys.exit("vLLM did not become ready within 30 minutes")


def parse_batch(images, work: Path, tag: str) -> list[str]:
    """Write one batch of page images, run the SDK over the folder, return markdown in order."""
    image_dir = work / tag / "images"
    image_dir.mkdir(parents=True)
    for i, img in enumerate(images):
        img.convert("RGB").save(image_dir / f"{i:06d}.png")
    args = {"image_dir": str(image_dir), "run_root": str(work / tag / "runs"), "run_tag": tag,
            "server_url": f"http://127.0.0.1:{PORT}", "model_name": SERVED_NAME,
            "layout_dir": LAYOUT_MODEL_DIR}
    out = subprocess.run([SDK_PYTHON, "-c", SDK_RUNNER, json.dumps(args)],
                         check=True, capture_output=True, text=True)
    run_dir = Path(json.loads(out.stdout.strip().splitlines()[-1])["run_dir"])
    pages = []
    for i in range(len(images)):
        md = run_dir / "output" / f"{i:06d}.md"
        pages.append(md.read_text(encoding="utf-8").strip() if md.exists() else "[OCR ERROR]")
    shutil.rmtree(work / tag)
    return pages


def ensure_output_columns_free(dataset, columns, overwrite=False):
    """Fail fast if an output column would collide with an existing input column."""
    clash = [c for c in columns if c in dataset.column_names]
    if not clash:
        return dataset
    if overwrite:
        logger.warning(f"--overwrite: replacing existing column(s) {clash}")
        return dataset.remove_columns(clash)
    logger.error(f"Output column(s) {clash} already exist in the input dataset.")
    logger.error("Choose a different --output-column, or pass --overwrite to replace them.")
    sys.exit(1)


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


def create_dataset_card(source_dataset, num_samples, processing_time, dspark, image_column, split):
    # Canonical provenance stamp (see AGENTS.md): Jobs claim gated on JOB_ID, set by HF Jobs in-container.
    on_jobs = os.environ.get("JOB_ID") is not None
    hw = jobs_flavor()
    origin = (
        "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
        + (f" (`{hw}`)" if hw else "")
    ) if on_jobs else "Generated"
    jobs_tag = "\n- hf-jobs" if on_jobs else ""
    return f"""---
tags:
- ocr
- document-processing
- minicpm-ocr
- markdown
- uv-script
- generated{jobs_tag}
---

# Document OCR using MiniCPM-OCR 1.0

OCR of [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) with OpenBMB's
[MiniCPM-OCR 1.0](https://huggingface.co/{MODEL}) pipeline: PP-DocLayoutV3 layout regions, each
region read by the model, assembled into Markdown by the official SDK.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) (split `{split}`, column `{image_column}`)
- **Model**: [{MODEL}](https://huggingface.co/{MODEL}) @ `{MODEL_REVISION[:7]}`
- **Decoding**: {"DSpark speculative decoding" if dspark else "standard"}
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

## Columns

- `markdown`: the SDK's Markdown for each page (tables as HTML, formulas as LaTeX)
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Reproduction

{origin} with the [uv-scripts](https://huggingface.co/datasets/uv-scripts/ocr) MiniCPM-OCR recipe:

```bash
hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/minicpm-ocr.py \\
    {source_dataset} <output-dataset>
```
"""


def main(args):
    check_image()
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    dataset = load_dataset(args.input_dataset, split=args.split)
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = ensure_output_columns_free(dataset, [args.output_column], args.overwrite)
    logger.info(f"{len(dataset)} pages from {args.input_dataset}")

    start = datetime.now()
    server = start_server(dspark=not args.no_dspark)
    outputs = []
    try:
        with tempfile.TemporaryDirectory() as work:
            for b in range(0, len(dataset), args.batch_size):
                images = dataset[b: b + args.batch_size][args.image_column]
                outputs += parse_batch(images, Path(work), f"batch{b:08d}")
                logger.info(f"{len(outputs)}/{len(dataset)} pages")
    finally:
        server.terminate()
    elapsed = datetime.now() - start

    dataset = dataset.add_column(args.output_column, outputs)
    entry = {"model_id": MODEL, "model_name": "MiniCPM-OCR-1.0", "column_name": args.output_column,
             "timestamp": datetime.now().isoformat(), "model_revision": MODEL_REVISION,
             "pipeline": "minicpm-ocr SDK 1.0.1 (PP-DocLayoutV3)", "dspark": not args.no_dspark}
    if "inference_info" in dataset.column_names:
        def update(example):
            try:
                info = json.loads(example["inference_info"]) if example["inference_info"] else []
            except (json.JSONDecodeError, TypeError):
                info = []
            return {"inference_info": json.dumps(info + [entry])}
        dataset = dataset.map(update)
    else:
        dataset = dataset.add_column("inference_info", [json.dumps([entry])] * len(dataset))

    dataset.push_to_hub(args.output_dataset, private=args.private, token=token)
    card = create_dataset_card(args.input_dataset, len(dataset), f"{elapsed.total_seconds() / 60:.1f} min",
                               not args.no_dspark, args.image_column, args.split)
    DatasetCard(card).push_to_hub(args.output_dataset, token=token)
    logger.info(f"Done: https://huggingface.co/datasets/{args.output_dataset} "
                f"({len(dataset) / elapsed.total_seconds():.2f} pages/s including startup)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document OCR with the MiniCPM-OCR 1.0 pipeline")
    parser.add_argument("input_dataset", help="Input dataset ID on the Hub")
    parser.add_argument("output_dataset", help="Output dataset ID on the Hub")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--output-column", default="markdown")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, help="Process only the first N pages (for testing)")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Pages handed to the SDK per run (it parallelises regions inside a run)")
    parser.add_argument("--no-dspark", action="store_true", help="Standard decoding, no draft model")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output column")
    parser.add_argument("--hf-token", help="Hugging Face token (default: HF_TOKEN env var)")
    main(parser.parse_args())
