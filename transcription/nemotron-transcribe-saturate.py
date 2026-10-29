# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf] @ git+https://github.com/davanstrien/saturate@main",
#     "soundfile",
# ]
#
# [tool.hf-jobs]
# image = "hongccc/sglang-omni@sha256:06ed306707bcf3d9e9beb5ba7ce66f09a666cac11c23455ece50c34199c0e985"
# flavor = "rtx-pro-6000"
# timeout = "4h"
# secrets = ["HF_TOKEN"]
# ///
"""
Transcribe audio with NVIDIA Nemotron 3.5 Transcribe on SGLang-Omni, via saturate.

Batch companion to `nemotron-transcribe.py`: same model, but served by
SGLang-Omni (no NeMo) on `/v1/audio/transcriptions`, with the driver half
(concurrency, output, resume) handled by the `saturate` package. What that buys
over the vLLM recipe:

- **A fixed window of files in flight** (`--concurrency`, default 32). It is
  fixed because transcription responses carry no token counts, which saturate's
  adaptive window needs to grow. Each in-flight file is held in memory, about
  two windows' worth, so lower it for large WAV inputs.
- **Crash-safe, resumable output**: results stream to the output as parquet
  parts; re-running the same command skips files already done (by path).
- **Durable error rows**: each file is sent once (saturate does not retry
  multipart uploads); a failed file is recorded as `{id, error}`, and
  `--retry-errors` re-admits only those on a later run.
- **No NeMo install**: SGLang-Omni is one pip install on the image below.

What it does NOT do (SGLang-Omni's integration, per the model card): no word
timestamps, no speaker tags, no translation, no speculative decoding. Use
`nemotron-transcribe.py --timestamps` for those. Each file is one request of up to
one hour (the model's limit); there is no splitting here.

Run on HF Jobs. The script installs SGLang-Omni into its own venv on the image and
starts `sgl-omni serve` itself; the [tool.hf-jobs] header declares the image (the
SGLang-Omni install guide's image, pinned by digest), the hardware and the HF_TOKEN
secret (`hf` CLI 1.32+):

  hf jobs uv run --detach \\
      https://huggingface.co/datasets/uv-scripts/transcription/raw/main/nemotron-transcribe-saturate.py \\
      "hf://buckets/user/audio-files/**/*.mp3" user/transcripts

The output repo holds `data/part-*.parquet` with rows
`{id, text, finish_reason, duration_s, model, prompt, error}`, where `id` is the
input path relative to the glob's folder. Read it with
`datasets.load_dataset(<output>, data_dir="data")`. A row with
`finish_reason == "length"` hit the token budget: SGLang-Omni's cookbook warns that
long recordings can repeat sentences until it runs out. The model card's
generation-loop guardrail is applied to every transcript.

With `--endpoint https://<job_id>--8000.hf.jobs/v1` the script uses a running
SGLang-Omni server instead of starting one, and sends HF_TOKEN as Bearer (only to
https `*.hf.jobs` hosts). Run it locally with `uv run`, or as a job with
`--flavor cpu-basic --image python:3.12`: the header's GPU is not needed.

Model: nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B (30B MoE, ~4B active,
OpenMDW-1.1). 40 transcription languages; the prompt presets are the card's.

The SERVING dict below is the per-model tuning prior (serve flags + install +
request shape). Agents can `ast.literal_eval` it without running the script; the
script itself consumes it, so it cannot drift from reality.
"""

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Serving starting values for Nemotron 3.5 Transcribe on SGLang-Omni. Provenance:
# - package / serve_args / served_name: the model card's SGLang-Omni section,
#   verbatim (`sgl-omni serve ... --model-name nemotron-transcribe
#   --mem-fraction-static 0.85`).
# - package: a pinned main commit, not the card's "SGLang-Omni 0.2.0". The 0.2.0
#   wheel (2026-10-27 17:10 UTC) predates the Nemotron Transcribe integration
#   (sgl-project/sglang-omni#2845, merged 2026-10-28 04:11 UTC). Source tarball,
#   not git+: the image needs no git.
# - arch_rename: the integration registers NeMoSpeechLMForConditionalGeneration
#   (the NeMo export's name); the Hub checkpoint's config.json says
#   Nemotron3_5TranscribeForConditionalGeneration with the same NeMo-shaped
#   config (llm_config + perception), so the local copy is renamed before serving.
# - env SGLANG_OMNI_STARTUP_TIMEOUT=1800: the cookbook's setting for a first
#   launch, which builds FlashInfer kernels.
# - venv python 3.12 + --prerelease=allow: the SGLang-Omni install guide.
SERVING = {
    "model": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
    "served_name": "nemotron-transcribe",
    "package": "sglang-omni @ https://github.com/sgl-project/sglang-omni/archive/61a23967f010de309e65e6f1b1c3819be4b51e86.tar.gz",
    "arch_rename": {
        "Nemotron3_5TranscribeForConditionalGeneration": "NeMoSpeechLMForConditionalGeneration"
    },
    "model_dir": "/opt/nemotron-transcribe-model",
    "venv": "/opt/sglang-omni-venv",
    "venv_python": "3.12",
    "serve_args": ["--mem-fraction-static", "0.85"],
    "env": {"SGLANG_OMNI_STARTUP_TIMEOUT": "1800"},
    "route": "/audio/transcriptions",
    "response_format": "json",
}

# User prompts from the model card's prompt guide.
PROMPTS = {
    "verbatim": (
        "Audio transcription. Language: as spoken. Wording: verbatim, including "
        "fillers and repetitions. Formatting: punctuation and capitalization. "
        "Output: transcript only."
    ),
    "clean": (
        "Audio transcription. Language: as spoken. Wording: omit fillers and "
        "accidental repetitions; keep the remaining spoken wording unchanged. "
        "Formatting: punctuation and capitalization. Output: transcript only."
    ),
}

MIME = {
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".opus": "audio/ogg", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".wma": "audio/x-ms-wma",
}  # fmt: skip

# Generation-loop guardrail from the model card (adapted there from the
# Qwen3-ASR repetition fix, Apache 2.0).
CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)


def cleanup_generation_loops(text: str) -> str:
    text = CHAR_RUN.sub(lambda m: m.group(1), text)
    return BLOCK_RUN.sub(lambda m: m.group(1), text)


def audio_seconds(data: bytes) -> float | None:
    import soundfile

    try:
        return round(soundfile.info(io.BytesIO(data)).duration, 1)
    except Exception:  # duration is informational; never fail a row over it
        return None


def prepare_checkpoint() -> Path:
    """Download the checkpoint and rename its architecture for SGLang-Omni."""
    import json

    from huggingface_hub import snapshot_download

    model_dir = Path(
        snapshot_download(SERVING["model"], local_dir=SERVING["model_dir"])
    )
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    renamed = [
        SERVING["arch_rename"].get(a, a) for a in config.get("architectures", [])
    ]
    if renamed != config.get("architectures"):
        print(
            f"config.json architectures {config['architectures']} -> {renamed}",
            file=sys.stderr,
        )
        config["architectures"] = renamed
        config_path.write_text(json.dumps(config, indent=2))
    return model_dir


def install_sglang_omni() -> Path:
    """Install SGLang-Omni into its own venv, as the install guide does."""
    venv = Path(SERVING["venv"])
    sgl_omni = venv / "bin" / "sgl-omni"
    if sgl_omni.exists():
        return sgl_omni
    uv = shutil.which("uv")
    if uv is None:
        sys.exit("uv is not on PATH: it is needed to install SGLang-Omni")
    env = {**os.environ}
    env.pop("VIRTUAL_ENV", None)  # never install into this script's venv
    print(f"installing {SERVING['package']} into {venv}", file=sys.stderr)
    subprocess.run([uv, "venv", "-q", "-p", SERVING["venv_python"], str(venv)],
                   check=True, env=env)  # fmt: skip
    subprocess.run([uv, "pip", "install", "-q", "--python", str(venv / "bin" / "python"),
                    "--prerelease=allow", SERVING["package"]], check=True, env=env)  # fmt: skip
    return sgl_omni


def main():
    ap = argparse.ArgumentParser(
        description="Nemotron 3.5 Transcribe batch transcription on SGLang-Omni via saturate"
    )
    ap.add_argument("input", help='Glob of audio files: "hf://buckets/org/name/**/*.mp3", '
                    '"/input/**/*.wav" (a mounted dir), ...')  # fmt: skip
    ap.add_argument(
        "output",
        help="Output dataset id (org/name) or fsspec URI (hf://buckets/..., a dir)",
    )
    ap.add_argument("--style", choices=sorted(PROMPTS), default="verbatim",
                    help="Prompt preset from the model card (default: verbatim)")  # fmt: skip
    ap.add_argument("--prompt", default=None,
                    help="Custom instruction (overrides --style), e.g. with 'Context terms: <names>'")  # fmt: skip
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Only the first N files (smoke tests)",
    )
    ap.add_argument(
        "--retry-errors", action="store_true", help="Re-admit files recorded as errors"
    )
    ap.add_argument("--concurrency", type=int, default=32,
                    help="Files in flight (default: 32; the server runs up to 64)")  # fmt: skip
    ap.add_argument("--endpoint", default=None,
                    help="Use a running SGLang-Omni server (e.g. https://<job_id>--8000.hf.jobs/v1) "
                    "instead of starting one; sends HF_TOKEN as Bearer")  # fmt: skip
    args = ap.parse_args()

    import pyarrow as pa
    from saturate import (
        Engine,
        Fixed,
        bucket_rows,
        existing_ids,
        make_multipart_request,
        pump,
    )

    is_repo_id = re.fullmatch(
        r"[\w-][\w.-]*/[\w.-]*[\w-]", args.output
    ) and not os.path.exists(args.output)
    output = f"hf://datasets/{args.output}/data" if is_repo_id else args.output
    prompt = args.prompt or PROMPTS[args.style]

    # Resume by path before any bytes are read: a re-run pays listing only.
    skip = existing_ids(output, retry_errors=args.retry_errors)
    rows = bucket_rows(args.input, skip=skip, limit=args.max_files)
    total_audio = [0.0]
    # Declared, so a part holding only error rows has the same columns as the rest.
    schema = pa.schema([
        ("id", pa.string()), ("error", pa.string()), ("text", pa.string()),
        ("finish_reason", pa.string()), ("duration_s", pa.float64()),
        ("model", pa.string()), ("prompt", pa.string()),
    ])  # fmt: skip

    def to_request(row):
        path = row["path"]
        files = {"file": (Path(path).name, row["bytes"],
                          MIME.get(Path(path).suffix.lower(), "application/octet-stream"))}  # fmt: skip
        data = {"model": SERVING["served_name"], "prompt": prompt,
                "response_format": SERVING["response_format"]}  # fmt: skip
        return make_multipart_request(SERVING["route"], data, files)

    def parse(row, body):
        duration = audio_seconds(row["bytes"])
        total_audio[0] += duration or 0.0
        return {
            "text": cleanup_generation_loops(body.get("text") or "").strip(),
            # SGLang-Omni's json response carries finish_reason only on truncation.
            "finish_reason": body.get("finish_reason") or "stop",
            "duration_s": duration,
            "model": SERVING["model"],
            "prompt": prompt,
        }

    def run(endpoint, headers=None):
        start = time.time()
        # One request is a whole recording (up to an hour of decode): flush often.
        stats = pump(rows, to_request, parse, endpoint, output,
                     window=Fixed(args.concurrency), flush_every=4,
                     read_timeout=3600, headers=headers, retry_errors=args.retry_errors,
                     schema=schema)  # fmt: skip
        elapsed = time.time() - start
        rtfx = f", {total_audio[0] / elapsed:.1f}x realtime" if total_audio[0] else ""
        print(f"{output}: {stats.rows_processed} ok, {stats.rows_failed} error rows, "
              f"{total_audio[0] / 3600:.2f} h of audio in {elapsed / 60:.1f} min{rtfx}",
              file=sys.stderr)  # fmt: skip
        print("NEMOTRON_TRANSCRIBE_SATURATE " + stats.to_json(), flush=True)

    if args.endpoint:
        from urllib.parse import urlparse

        url = urlparse(args.endpoint)
        is_jobs = url.scheme == "https" and (url.hostname or "").endswith(".hf.jobs")
        token = os.environ.get("HF_TOKEN") if is_jobs else None
        run(args.endpoint, {"Authorization": f"Bearer {token}"} if token else None)
        return

    sgl_omni = install_sglang_omni()
    model_dir = prepare_checkpoint()
    os.environ.update(SERVING["env"])  # inherited by the server process
    cmd = [str(sgl_omni), "serve", "--model-path", str(model_dir),
           "--model-name", SERVING["served_name"], *SERVING["serve_args"],
           "--host", "127.0.0.1", "--port", "8000"]  # fmt: skip
    # Readiness probe on the transcription route: SGLang-Omni answers saturate's default
    # chat-completions probe (no audio) with a 500, which reads as "not up yet"; a
    # JSON post to the multipart route gets a 4xx, which proves the server is alive.
    with Engine(
        SERVING["model"],
        engine="sglang",
        cmd=cmd,
        boot_timeout=3600,
        ready_route=SERVING["route"],
    ) as endpoint:
        run(endpoint)


if __name__ == "__main__":
    main()
