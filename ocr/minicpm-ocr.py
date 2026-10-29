# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=3.1.0",
#     "huggingface-hub>=1.0",
#     "pillow",
#     "bucketbag @ https://github.com/davanstrien/bucketbag/archive/refs/tags/v0.3.1.tar.gz",
# ]
#
# [tool.hf-jobs]
# image = "cr.hf.co/davanstrien/minicpm-ocr:1.0-vllm-5d35ce3"
# flavor = "a10g-large"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# ///

"""
Convert document images to markdown with OpenBMB's MiniCPM-OCR 1.0 pipeline.

MiniCPM-OCR 1.0 (1.3B, Apache-2.0) is a region reader: the official SDK finds layout regions
with PP-DocLayoutV3, crops them, sends each crop to the model with the prompt for its type
(`Text Recognition:`, `Table Recognition:`, `Formula Recognition:`, ...) and assembles the page
as Markdown (tables as HTML, formulas as LaTeX). This script runs that pipeline unchanged.

Two ways in and out:

  # Hub dataset -> Hub dataset (adds `markdown` and `markdown_status` columns)
  hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/minicpm-ocr.py \\
      input-dataset output-dataset --max-samples 20

  # Bucket (or folder) of page images -> one .md per image in a Bucket (or folder)
  hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/minicpm-ocr.py \\
      hf://buckets/<ns>/<bucket>/pages hf://buckets/<ns>/<bucket>/minicpm-ocr

Bucket input is listed and downloaded in batches with bucketbag (no mount); `x/y.jp2` becomes
`x/y.md` under the output prefix. A page whose regions all read cleanly is written; a page with
failed regions is not, and is listed at the end, so re-running the same command retries it (pages
with a `.md` already are skipped). In dataset mode every row gets a `markdown_status`: `ok`,
`partial_failed` (some regions failed; the text is kept), `failed` or `error: ...` (no text).

Runtime image (from the [tool.hf-jobs] header): cr.hf.co/davanstrien/minicpm-ocr, built from
images/minicpm-ocr/Dockerfile. It holds:
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
config sets sampling and per-region prompts; nothing here overrides them. The SDK reads page
numbers, headers, footers, footnotes and side notes but leaves them out of the Markdown;
`--keep-labels footnote aside_text` puts those region types back.

Model: https://huggingface.co/openbmb/MiniCPM-OCR-1.0
SDK: https://github.com/OpenBMB/MiniCPM-OCR
Image: https://huggingface.co/containers/davanstrien/minicpm-ocr
"""

import argparse
import contextlib
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_URL = "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/minicpm-ocr.py"
MODEL = "openbmb/MiniCPM-OCR-1.0"
MODEL_REVISION = "a59a4acf215e34c6d73149e9eafdb88ed2086e39"
DRAFT_MODEL = "openbmb/MiniCPM-OCR-1.0-DSpark"  # must come from the same release as MODEL
DRAFT_REVISION = "31e3feaf80e0061cd6aa73db23480d2d691d6452"
SERVED_NAME = "minicpm-ocr"
PORT = 18499
SDK_PYTHON = "/opt/sdk/bin/python"
LAYOUT_MODEL_DIR = "/opt/minicpm-ocr-assets/models/PP-DocLayoutV3_infer"
# minicpm-ocr SDK 1.0.1 markdown_assembly.ignore_labels: read by the model, left out of the Markdown
SDK_IGNORED_LABELS = ["number", "footnote", "header", "footer", "aside_text"]
IMAGE_TAG = "cr.hf.co/davanstrien/minicpm-ocr:1.0-vllm-5d35ce3"
BUCKET_SCHEME = "hf://buckets/"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".jp2"}

# Runs inside the SDK venv: the SDK's documented Python API (docs/minicpmvocr_api_en.md),
# with the bundled config and only the input, server and layout path filled in.
SDK_RUNNER = """
import json, sys
from importlib.resources import files
from pathlib import Path
from minicpmvocr import load_config, pipelined_run
from minicpmvocr.prompts import DEFAULT_PROMPT_MAP
args = json.loads(sys.argv[1])
prompt = {}
if args["keep_labels"]:
    # A prompt-map file replaces whole top-level sections, so write the full assembly section.
    assembly = dict(DEFAULT_PROMPT_MAP["markdown_assembly"])
    assembly["ignore_labels"] = [l for l in assembly["ignore_labels"] if l not in args["keep_labels"]]
    prompt_map = Path(args["run_root"]) / "prompt_map.json"
    prompt_map.parent.mkdir(parents=True, exist_ok=True)
    prompt_map.write_text(json.dumps({"markdown_assembly": assembly}))
    prompt = {"prompt_map": str(prompt_map)}
cfg = load_config(str(files("minicpmvocr").joinpath("configs/minicpmvocr.yaml")), overrides={
    "input": {"image_dir": args["image_dir"]},
    "run": {"run_root": args["run_root"], "run_tag": args["run_tag"]},
    "engine": {"server_url": args["server_url"], "model_name": args["model_name"], "max_model_len": 40960},
    "layout": {"model_dir": args["layout_dir"], "device": "cuda:0"},
    "pipeline": {"overlap_layout": True},
    "prompt": prompt,
})
pipelined_run(cfg)
print(json.dumps({"run_dir": cfg["_meta"]["run_dir"]}))
"""


def image_python() -> str | None:
    """The image's own python3 (the one with vLLM and torch), looking past this script's venv."""
    own_bin = os.path.realpath(os.path.join(sys.prefix, "bin"))
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                           if os.path.realpath(p) != own_bin)
    return shutil.which("python3", path=path)


def check_runtime():
    """Fail fast, naming the fix, when not on the MiniCPM-OCR image or without a CUDA GPU."""
    py = image_python()
    if not (py and shutil.which("vllm") and Path(SDK_PYTHON).exists() and Path(LAYOUT_MODEL_DIR).exists()):
        logger.error("This script needs the %s image (vLLM with MiniCPM-OCR support, the "
                     "minicpm-ocr SDK and the layout model).", IMAGE_TAG)
        logger.error("Run it with `hf jobs uv run`, which reads the image from the script header, "
                     "or pass --image %s.", IMAGE_TAG)
        sys.exit(1)
    probe = subprocess.run([py, "-c", "import torch; print(torch.cuda.is_available())"],
                           capture_output=True, text=True)
    if probe.stdout.strip() != "True":
        logger.error("CUDA is not available to the image's torch. This script needs a GPU "
                     "(e.g. --flavor a10g-large). %s", probe.stderr.strip()[-300:])
        sys.exit(1)


@contextlib.contextmanager
def vllm_server(dspark: bool):
    """Run vLLM with OpenBMB's tutorial settings for the duration of the block.

    The server gets its own process group, and is terminated, then killed, then reaped, on every
    exit path, including a failed or interrupted startup.
    """
    from huggingface_hub import snapshot_download

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
    # This script's VIRTUAL_ENV must not leak into the image's vLLM.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    logger.info("Starting vLLM: %s", shlex.join(cmd))
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        deadline = time.time() + 1800
        while True:
            if proc.poll() is not None:
                sys.exit(f"vLLM exited during startup (code {proc.returncode})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
                    if SERVED_NAME in r.read().decode():
                        break
            except OSError:
                pass
            if time.time() > deadline:
                sys.exit("vLLM did not become ready within 30 minutes")
            time.sleep(5)
        logger.info("vLLM ready")
        yield
    finally:
        stop_process_group(proc)


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def stop_process_group(proc: subprocess.Popen):
    """SIGTERM the whole group, wait up to 60 s for it to empty, then SIGKILL; always reap the
    parent. Checks the group, not just the parent, since vLLM's workers can outlive it."""
    pgid = proc.pid  # start_new_session=True: the server leads its own group
    for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
        if not group_alive(pgid):
            break
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, sig)
        deadline = time.time() + wait
        while group_alive(pgid) and time.time() < deadline:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=1)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5)


def parse_batch(images: dict, work: Path, tag: str, keep_labels: list[str]) -> dict:
    """OCR one batch of pages with the SDK.

    `images` maps a key to a PIL image, a path, or an undecoded datasets image dict. Returns
    key -> (status, markdown or None). A page that cannot be read gets an `error: ...` status and
    does not stop the batch; if the SDK run itself fails, every page in the batch gets one.
    """
    from datasets import Image as ImageFeature
    from PIL import Image, ImageOps

    decoder = ImageFeature()
    image_dir = work / tag / "images"
    image_dir.mkdir(parents=True)
    results, ids = {}, {}
    for i, (key, source) in enumerate(images.items()):
        try:
            if isinstance(source, dict):
                # datasets Image(decode=False): {"bytes", "path"}. The datasets decoder handles
                # remote paths and auth, and turns EXIF-rotated photos upright.
                img = decoder.decode_example(source)
            else:
                img = source if isinstance(source, Image.Image) else Image.open(source)
                img = ImageOps.exif_transpose(img)
            img.convert("RGB").save(image_dir / f"{i:06d}.png")
            ids[f"{i:06d}"] = key
        except Exception as e:  # one unreadable page must not lose the batch
            logger.warning("Cannot read %s: %s", key, e)
            results[key] = (f"error: unreadable image ({type(e).__name__})", None)
    if ids:
        args = {"image_dir": str(image_dir), "run_root": str(work / tag / "runs"), "run_tag": tag,
                "server_url": f"http://127.0.0.1:{PORT}", "model_name": SERVED_NAME,
                "layout_dir": LAYOUT_MODEL_DIR, "keep_labels": keep_labels}
        run = subprocess.run([SDK_PYTHON, "-c", SDK_RUNNER, json.dumps(args)],
                             capture_output=True, text=True)
        if run.returncode != 0:
            logger.error("SDK run failed for %s (exit %d):\n%s", tag, run.returncode, run.stderr[-3000:])
            results.update({key: (f"error: SDK exit {run.returncode}", None) for key in ids.values()})
        else:
            run_dir = Path(json.loads(run.stdout.strip().splitlines()[-1])["run_dir"])
            status = {}
            metrics = run_dir / "per_page_metrics.jsonl"
            if metrics.exists():
                for line in metrics.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    status[row["image_id"]] = row.get("status") or "unknown"
            for image_id, key in ids.items():
                md = run_dir / "output" / f"{image_id}.md"
                page_status = status.get(image_id, "failed" if not md.exists() else "unknown")
                text = md.read_text(encoding="utf-8").strip() if md.exists() else None
                results[key] = (page_status, text if page_status in ("ok", "partial_failed") else None)
    shutil.rmtree(work / tag, ignore_errors=True)
    return results


# --- Bucket / folder mode ------------------------------------------------------------------


def is_bucket(location: str) -> bool:
    return location.startswith(BUCKET_SCHEME)


def as_bucket_dir(url: str) -> tuple[str, str]:
    """Split a bucket URL into (bucket root URL, directory key prefix ending in "/").

    The prefix goes to bucketbag explicitly: it matches prefixes as strings, so `.../vol`
    would also list `.../vol2/...`.
    """
    parts = url[len(BUCKET_SCHEME):].strip("/").split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"Not a bucket URL: {url} (expected hf://buckets/<ns>/<bucket>[/<prefix>])")
    prefix = parts[2] + "/" if len(parts) == 3 else ""
    return BUCKET_SCHEME + "/".join(parts[:2]), prefix


def md_rel(rel: str) -> str:
    return str(Path(rel).with_suffix(".md"))


def list_rels(location: str) -> list[str]:
    """Relative paths of every file under a bucket prefix or folder, sorted."""
    if is_bucket(location):
        from bucketbag import iter_keys

        url, prefix = as_bucket_dir(location)
        return sorted(k[len(prefix):] for k in iter_keys(url, prefix=prefix))
    root = Path(location)
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.is_dir() else []


def iter_local_batches(source: str, rels: list[str], n: int):
    """Yield batches of (rel, local path): from disk, or downloaded by bucketbag to scratch."""
    if not is_bucket(source):
        for i in range(0, len(rels), n):
            yield [(rel, Path(source) / rel) for rel in rels[i: i + n]]
        return
    from bucketbag import batched_files

    url, prefix = as_bucket_dir(source)
    for batch in batched_files(url, keys=[prefix + rel for rel in rels], n=n, prefetch=2):
        yield [(item.key[len(prefix):], item.path) for item in batch]


def write_outputs(dest: str, pairs: list[tuple[str, str]]):
    if not pairs:
        return
    if is_bucket(dest):
        from bucketbag import put_files

        url, prefix = as_bucket_dir(dest)
        put_files([(prefix + rel, content) for rel, content in pairs], url)
        return
    for rel, content in pairs:
        # Temp file + rename: a stopped job never leaves a partial .md that a re-run counts as done.
        path = Path(dest) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)


def run_files(args):
    inputs = [r for r in list_rels(args.input) if Path(r).suffix.lower() in IMAGE_EXTENSIONS]
    owners: dict[str, list[str]] = {}
    for rel in inputs:
        owners.setdefault(md_rel(rel), []).append(rel)
    clashes = [" + ".join(group) for group in owners.values() if len(group) > 1]
    if clashes:
        sys.exit("These inputs would write the same .md file; rename or move them first:\n  "
                 + "\n  ".join(clashes))
    if args.max_samples:
        inputs = inputs[: args.max_samples]  # the same pages on every run, so a re-run resumes them
    done = {r for r in list_rels(args.output) if r.endswith(".md")}
    todo = [r for r in inputs if args.overwrite or md_rel(r) not in done]
    already = sum(md_rel(r) in done for r in inputs)
    logger.info("%d images under %s, %d already have a .md, %d to OCR", len(inputs), args.input,
                already, len(todo))
    if not todo:
        return
    failed = []
    with vllm_server(dspark=not args.no_dspark), tempfile.TemporaryDirectory() as work:
        for b, batch in enumerate(iter_local_batches(args.input, todo, args.batch_size)):
            results = parse_batch(dict(batch), Path(work), f"batch{b:06d}", args.keep_labels)
            write_outputs(args.output, [(md_rel(rel), text) for rel, (status, text) in results.items()
                                        if status == "ok"])
            failed += [f"{rel} ({status})" for rel, (status, _) in results.items() if status != "ok"]
            logger.info("%d/%d images", min((b + 1) * args.batch_size, len(todo)), len(todo))
    if failed:
        logger.warning("%d pages not written (re-run the same command to retry them):\n  %s",
                       len(failed), "\n  ".join(failed))
        stale = [f for f in failed if md_rel(f.rsplit(" (", 1)[0]) in done]
        if args.overwrite and stale:
            logger.warning("%d of them still have a .md from an earlier run, which a re-run without "
                           "--overwrite will skip; delete those files to retry them:\n  %s",
                           len(stale), "\n  ".join(stale))


# --- Dataset mode ----------------------------------------------------------------------------


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


def create_dataset_card(args, num_samples: int, counts: dict, processing_time: str) -> str:
    # Canonical provenance stamp (see AGENTS.md): Jobs claim gated on JOB_ID, set by HF Jobs in-container.
    on_jobs = os.environ.get("JOB_ID") is not None
    hw = jobs_flavor()
    origin = (
        "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
        + (f" (`{hw}`)" if hw else "")
    ) if on_jobs else "Generated"
    jobs_tag = "\n- hf-jobs" if on_jobs else ""
    status_col = f"{args.output_column}_status"
    flags = [f"--flavor {hw or 'a10g-large'}"]
    for flag, value, default in (("--split", args.split, "train"), ("--image-column", args.image_column, "image"),
                                 ("--output-column", args.output_column, "markdown"),
                                 ("--batch-size", args.batch_size, 64)):
        if value != default:
            flags.append(f"{flag} {shlex.quote(str(value))}")
    if args.max_samples:
        flags.append(f"--max-samples {args.max_samples}")
    if args.shuffle:
        flags.append(f"--shuffle --seed {args.seed}")
    if args.no_dspark:
        flags.append("--no-dspark")
    if args.keep_labels:
        flags.append("--keep-labels " + " ".join(args.keep_labels))
    if args.overwrite:
        flags.append("--overwrite")
    command = f"hf jobs uv run {' '.join(flags[:1])} \\\n    {SCRIPT_URL} \\\n    {args.input} <output-dataset>"
    if flags[1:]:
        command += " \\\n    " + " ".join(flags[1:])
    counts_line = ", ".join(f"`{k}` {v:,}" for k, v in sorted(counts.items()))
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

OCR of [{args.input}](https://huggingface.co/datasets/{args.input}) with OpenBMB's
[MiniCPM-OCR 1.0](https://huggingface.co/{MODEL}) pipeline: PP-DocLayoutV3 layout regions, each
region read by the model, assembled into Markdown by the official SDK.

## Processing Details

- **Source Dataset**: [{args.input}](https://huggingface.co/datasets/{args.input}) (split `{args.split}`, column `{args.image_column}`)
- **Model**: [{MODEL}](https://huggingface.co/{MODEL}) @ `{MODEL_REVISION[:7]}`
- **Decoding**: {"standard" if args.no_dspark else "DSpark speculative decoding"}
- **Region types left out of the Markdown**: {", ".join(f"`{l}`" for l in SDK_IGNORED_LABELS if l not in args.keep_labels) or "none"}
- **Number of Samples**: {num_samples:,}
- **Page status**: {counts_line}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

## Columns

- `{args.output_column}`: the SDK's Markdown for each page (tables as HTML, formulas as LaTeX); empty when
  the page failed
- `{status_col}`: `ok`, `partial_failed` (some regions failed; text kept), `failed` or `error: ...`
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Reproduction

{origin} with the [`minicpm-ocr.py`]({SCRIPT_URL}) recipe from [uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:

```bash
{command}
```
"""


def run_dataset(args):
    from datasets import load_dataset
    from huggingface_hub import DatasetCard

    status_col = f"{args.output_column}_status"
    reserved = {"inference_info", args.image_column}
    if args.output_column in reserved or status_col in reserved:
        sys.exit(f"--output-column {args.output_column!r} clashes with {sorted(reserved)}; pick another name")

    dataset = load_dataset(args.input, split=args.split)
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    # Undecoded: a corrupt image then fails inside parse_batch's per-page guard, not in the slice.
    from datasets import Image as ImageFeature

    dataset = dataset.cast_column(args.image_column, ImageFeature(decode=False))
    clash = [c for c in (args.output_column, status_col) if c in dataset.column_names]
    if clash and not args.overwrite:
        sys.exit(f"Output column(s) {clash} already exist; choose another --output-column or pass --overwrite")
    if clash:
        logger.warning("--overwrite: replacing existing column(s) %s", clash)
        dataset = dataset.remove_columns(clash)
    logger.info("%d pages from %s", len(dataset), args.input)

    # Slice only the image column: slicing the full dataset would decode every other image
    # column too, outside the per-page guard.
    image_only = dataset.select_columns([args.image_column])
    start = datetime.now()
    texts, statuses = [], []
    with vllm_server(dspark=not args.no_dspark), tempfile.TemporaryDirectory() as work:
        for b in range(0, len(dataset), args.batch_size):
            images = image_only[b: b + args.batch_size][args.image_column]
            results = parse_batch(dict(enumerate(images)), Path(work), f"batch{b:08d}", args.keep_labels)
            for i in range(len(images)):
                status, text = results[i]
                statuses.append(status)
                texts.append(text or "")
            logger.info("%d/%d pages", len(texts), len(dataset))
    elapsed = datetime.now() - start

    counts = {}
    for s in statuses:
        key = "error" if s.startswith("error") else s
        counts[key] = counts.get(key, 0) + 1
    logger.info("Page status: %s", counts)

    dataset = dataset.cast_column(args.image_column, ImageFeature())
    dataset = dataset.add_column(args.output_column, texts).add_column(status_col, statuses)
    entry = {"model_id": MODEL, "model_name": "MiniCPM-OCR-1.0", "column_name": args.output_column,
             "timestamp": datetime.now().isoformat(), "model_revision": MODEL_REVISION,
             "pipeline": "minicpm-ocr SDK 1.0.1 (PP-DocLayoutV3)", "dspark": not args.no_dspark,
             "keep_labels": args.keep_labels}
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

    dataset.push_to_hub(args.output, private=args.private)
    card = create_dataset_card(args, len(dataset), counts, f"{elapsed.total_seconds() / 60:.1f} min")
    DatasetCard(card).push_to_hub(args.output)
    logger.info("Done: https://huggingface.co/datasets/%s (%.2f pages/s including startup)",
                args.output, len(dataset) / elapsed.total_seconds())


def main():
    parser = argparse.ArgumentParser(
        description="Document OCR with the MiniCPM-OCR 1.0 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Input/output: Hub dataset IDs (dataset mode), or hf://buckets/<ns>/<bucket>[/<prefix>] "
               "URLs / folders (one .md per image).")
    parser.add_argument("input", help="Hub dataset ID, or hf://buckets/... URL or folder of page images")
    parser.add_argument("output", help="Hub dataset ID (dataset input), or hf://buckets/... URL or folder")
    parser.add_argument("--image-column", default="image", help="Dataset mode: image column")
    parser.add_argument("--output-column", default="markdown", help="Dataset mode: output column")
    parser.add_argument("--split", default="train", help="Dataset mode: split")
    parser.add_argument("--max-samples", type=int, help="Process only the first N pages (for testing)")
    parser.add_argument("--shuffle", action="store_true", help="Dataset mode: shuffle before --max-samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Pages handed to the SDK per run (it parallelises regions inside a run)")
    parser.add_argument("--no-dspark", action="store_true", help="Standard decoding, no draft model")
    parser.add_argument("--keep-labels", nargs="+", default=[], choices=SDK_IGNORED_LABELS,
                        help="Region types to keep in the Markdown that the SDK leaves out by default")
    parser.add_argument("--private", action="store_true", help="Dataset mode: private output dataset")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing output column / re-OCR images that already have a .md")
    parser.add_argument("--hf-token", help="Hugging Face token (default: HF_TOKEN env var)")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    if args.max_samples is not None and args.max_samples < 1:
        sys.exit("--max-samples must be at least 1")
    if args.batch_size < 1:
        sys.exit("--batch-size must be at least 1")
    # The input decides the mode: a Bucket or an existing folder means files, anything else is a
    # Hub dataset ID. In file mode the output is a Bucket URL or a folder (created if new).
    file_in = is_bucket(args.input) or Path(args.input).is_dir()
    if not file_in and (is_bucket(args.output) or Path(args.output).is_dir()):
        sys.exit("Dataset input writes a Hub dataset; for Bucket output, give a Bucket or folder as input.")
    check_runtime()
    if file_in:
        run_files(args)
    else:
        run_dataset(args)


if __name__ == "__main__":
    main()
