# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub",
#     "requests",
#     "librosa",
#     "soundfile",
# ]
#
# [tool.hf-jobs]
# image = "cr.hf.co/davanstrien/nemotron-transcribe:v0.32.0-nemo-7b05f096d54d"
# flavor = "rtx-pro-6000"
# secrets = ["HF_TOKEN"]
# ///

"""
Transcribe audio with NVIDIA Nemotron 3.5 Transcribe via an in-job vLLM server.

Serves nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B behind vLLM + the NeMo
Speech plugin inside the job and posts files concurrently, so continuous
batching decodes many recordings at once.

The model only runs on vLLM 0.32.0 + NeMo Speech main (no stable wheel ships
the plugin). The header's image has that serving stack prebuilt (NeMo Speech
pinned to commit 7b05f096d54d, FlashInfer kernels warm for rtx-pro-6000; built
by nemotron-transcribe-image.sh). On any other image the script builds the
stack itself first: it downloads the NeMo Speech source, installs it with the
model card's locked `uv sync` plus FlashInfer's precompiled kernels. Either way
it downloads the checkpoint, starts `vllm serve` with the card's single-GPU
flags as a subprocess, splits audio longer than the model's one-hour limit,
posts files concurrently, and writes .txt (+ JSON with word timestamps and
speakers when --timestamps is set). If a server already answers on --server,
the setup is skipped.

Run on HF Jobs (image, flavor and HF_TOKEN secret come from the
[tool.hf-jobs] header above; needs `hf` CLI 1.32+):

  hf jobs uv run --detach --timeout 4h \\
      -v hf://buckets/user/audio-files:/input:ro \\
      -v hf://buckets/user/transcripts:/output \\
      https://huggingface.co/datasets/uv-scripts/transcription/raw/main/nemotron-transcribe.py \\
      /input /output

Setup is a fixed ~8 min per job on the prebuilt image (2 min image pull,
35 s checkpoint download, ~4 min engine warmup); ~10 min with
`--image vllm/vllm-openai:v0.32.0`, which installs the stack at runtime. Batch
many files into one job: 10 half-hour episodes (4.9 h) transcribed in 2.5 min
(116x realtime) once the server was up, for a $0.48 job.

The server flags are the model card's single-GPU settings (written for an
H100 80 GB; Jobs has no H100, so rtx-pro-6000 96 GB is the default here).
The timestamp middleware finds the checkpoint's bundled CTC head only in a
local model directory, which is why the checkpoint is downloaded first and
served by path.

Input:                              Output:
  /input/tape1.mp3            ->      /output/tape1.txt
  /input/sub/tape2.mp3        ->      /output/sub/tape2.txt
                                      (+ tape2.json with --timestamps)

Model: nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B (30B MoE, ~4B active,
OpenMDW-1.1)
  - 40 transcription languages, speaker tags in 13, up to 8 speakers.
  - Trained on recordings up to 1 hour; longer files are split into parts.
    Speaker tags are consistent WITHIN a part and reset BETWEEN parts
    (the part index is recorded on every word as "part").
  - The card's generation-loop guardrail is applied to every transcript.
"""

import argparse
import base64
import concurrent.futures
import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"

# Where the prebuilt image (nemotron-transcribe-image.sh) installs the serving
# stack and its warm kernel caches.
PREBUILT_WORKDIR = Path("/opt/nemotron-transcribe")

# Written last by install_serving_stack, so a half-finished install is redone.
INSTALL_MARKER = ".install-complete"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus"}

SAMPLE_RATE = 16000

# The model is trained on recordings up to one hour. Stay a little under it.
DEFAULT_PART_SECONDS = 3300  # 55 min

# A dense one-hour conversation can need about 20,000 output tokens (card).
MAX_TOKENS = 24576

# User prompts from the model card's prompt guide. Transcription keeps the
# system prompt empty.
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
    "speakers": (
        "Transcribe this audio with punctuation and capitalization. Include all "
        "spoken words and verbal fillers. Write numbers in their standard written "
        "form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
    ),
}

# Generation-loop guardrail from the model card (adapted there from the
# Qwen3-ASR repetition fix, Apache 2.0).
CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)


def cleanup_generation_loops(text: str) -> str:
    text = CHAR_RUN.sub(lambda m: m.group(1), text)
    return BLOCK_RUN.sub(lambda m: m.group(1), text)


def write_atomic(path: Path, text: str):
    """Write via a temp file, so an interrupted write never looks finished."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def discover_audio_files(input_dir: Path) -> list[Path]:
    return [
        p
        for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]


def encode_flac(audio) -> str:
    """Encode a 16 kHz mono waveform as base64 FLAC (lossless, ~half of WAV)."""
    import soundfile

    buf = io.BytesIO()
    soundfile.write(buf, audio, SAMPLE_RATE, format="FLAC")
    return base64.b64encode(buf.getvalue()).decode()


def transcribe_part(
    server: str, audio_b64: str, prompt: str, timestamps: bool, timeout_s: int
) -> dict:
    import requests

    extra = {
        "chat_template_kwargs": {"enable_thinking": False},
        "skip_special_tokens": False,  # keep speaker and event tags
        "spaces_between_special_tokens": False,
        "seed": 0,
    }
    if timestamps:
        extra["mm_processor_kwargs"] = {"capture_ctc_timestamps": True}
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/flac;base64,{audio_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": MAX_TOKENS,
        **extra,
    }
    resp = requests.post(f"{server}/v1/chat/completions", json=body, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def shift_alignment(alignment: dict, offset_s: float, part_index: int) -> dict:
    """Move a part's word and diarization times onto the whole-file timeline."""
    words = [
        {
            **w,
            "start": round(w["start"] + offset_s, 3),
            "end": round(w["end"] + offset_s, 3),
            "part": part_index,
        }
        for w in alignment.get("words") or []
    ]
    diarization = [
        {
            **seg,
            "start": round(seg["start"] + offset_s, 3),
            "end": round(seg["end"] + offset_s, 3),
            "part": part_index,
        }
        for seg in alignment.get("diarization") or []
    ]
    return {
        "words": words,
        "diarization": diarization,
        "speaker_tag_to_diarization_speaker": alignment.get(
            "speaker_tag_to_diarization_speaker"
        ),
        "error": alignment.get("error"),
    }


def server_is_up(server: str) -> bool:
    import requests

    try:
        return requests.get(f"{server}/health", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def install_serving_stack(workdir: Path, nemo_ref: str) -> Path:
    """Install vLLM 0.32.0 + NeMo Speech as the model card does; return the venv."""
    src = workdir / "nemo-speech"
    venv = src / ".venv"
    marker = src / INSTALL_MARKER
    # The first prebuilt image predates the marker; the builder only publishes
    # complete installs, so a vllm binary in the prebuilt workdir counts too.
    prebuilt = workdir == PREBUILT_WORKDIR and (venv / "bin" / "vllm").exists()
    if marker.exists() or prebuilt:
        logger.info(f"Using the installed serving stack in {src}")
        return venv
    if src.exists():
        logger.info(f"Removing an incomplete install in {src}")
        shutil.rmtree(src)

    # Resolve the ref to a commit so the run records exactly what it served.
    with urllib.request.urlopen(
        f"https://api.github.com/repos/NVIDIA-NeMo/Speech/commits/{nemo_ref}"
    ) as resp:
        sha = json.load(resp)["sha"]
    # The source tarball, not `git clone`: the vLLM image ships without git.
    url = f"https://github.com/NVIDIA-NeMo/Speech/archive/{sha}.tar.gz"
    logger.info(f"Downloading NeMo Speech source ({nemo_ref} = {sha})...")
    archive = workdir / "nemo-speech.tar.gz"
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive) as tar:
        top = tar.getnames()[0].split("/")[0]
        tar.extractall(workdir, filter="data")
    (workdir / top).rename(src)
    (src / ".nemo-ref").write_text(sha)

    uv = shutil.which("uv")
    if uv is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        uv = shutil.which("uv")
    python = str(venv / "bin" / "python")
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv)}
    steps = [
        [uv, "venv", "-q", "-p", "3.13", str(venv)],
        [uv, "sync", "-q", "--extra", "asr", "--extra", "vllm", "--locked", "--no-install-project"],
        [uv, "pip", "install", "-q", "--python", python, "--no-deps", "-e", "."],
        # PyAV decodes compressed audio; vLLM needs OpenCV, which NeMo excludes.
        [uv, "pip", "install", "-q", "--python", python, "--no-deps",
         "av==19.0.2", "opencv-python-headless==5.0.0.93"],
    ]  # fmt: skip
    logger.info("Installing vLLM 0.32.0 + NeMo Speech (locked uv sync)...")
    for step in steps:
        subprocess.run(step, cwd=src, env=env, check=True)

    # The locked env has flashinfer-python but not its precompiled kernels, so
    # FlashInfer JIT-builds them at startup (~10 min on rtx-pro-6000). vLLM's
    # `download-kernels` installs them; FlashInfer's installer needs pip.
    logger.info("Installing FlashInfer precompiled kernels...")
    subprocess.run([uv, "pip", "install", "-q", "--python", python, "pip"], check=True)
    result = subprocess.run([str(venv / "bin" / "vllm"), "download-kernels"])
    if result.returncode != 0:
        logger.warning("Kernel download failed; FlashInfer will JIT-build at startup")
    marker.write_text(sha)
    return venv


def start_server(venv: Path, model_dir: Path, port: int) -> subprocess.Popen:
    """Start `vllm serve` with the model card's single-GPU flags."""
    env = {
        **os.environ,
        "VLLM_PLUGINS": "nemo_speechlm",
        # Allow recordings up to one hour (the default is 600 seconds).
        "VLLM_MAX_AUDIO_DECODE_DURATION_S": "3600",
        "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "180",
        # Word timestamps require the V1 model runner.
        "VLLM_USE_V2_MODEL_RUNNER": "0",
    }
    cmd = [
        str(venv / "bin" / "vllm"), "serve", str(model_dir),
        "--served-model-name", MODEL,
        "--port", str(port),
        "--trust-remote-code", "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.94",
        "--max-model-len", "65536",
        "--max-num-batched-tokens", "4096",
        "--max-num-seqs", "192",
        "--enable-chunked-prefill", "--no-enable-prefix-caching",
        "--use-replayssm", "--mamba-backend", "flashinfer", "--mamba-cache-mode", "none",
        "--mm-processor-cache-gb", "0", "--no-async-scheduling",
        "--limit-mm-per-prompt", '{"audio":{"count":6,"length":120000000}}',
        "--speculative-config", '{"method":"mtp","num_speculative_tokens":8}',
        "--reasoning-parser", "nemotron_v3",
        "--trust-request-mm-kwargs",
        "--middleware",
        "nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware",
    ]  # fmt: skip
    logger.info("Starting vllm serve...")
    # Own process group, so stop_server also reaches vLLM's engine workers.
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def stop_server(process: subprocess.Popen):
    if process.poll() is not None:
        return
    logger.info("Stopping vllm serve...")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def wait_for_server(
    server: str, process: subprocess.Popen | None = None, timeout_s: int = 3600
):
    logger.info(f"Waiting for server at {server} (model load + warmup)...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server_is_up(server):
            logger.info("Server is ready")
            return
        if process is not None and process.poll() is not None:
            logger.error(f"vllm serve exited with code {process.returncode}")
            sys.exit(1)
        time.sleep(15)
    logger.error(f"Server did not become ready within {timeout_s}s")
    sys.exit(1)


def process_file(
    file_path: Path,
    input_dir: Path,
    output_dir: Path,
    server: str,
    prompt: str,
    timestamps: bool,
    part_seconds: int,
    request_timeout: int,
) -> dict:
    import librosa

    rel = file_path.relative_to(input_dir)
    start_time = time.time()

    audio, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / SAMPLE_RATE

    texts = []
    parts = []
    offset = 0.0
    part_index = 0
    while offset < duration:
        lo = int(offset * SAMPLE_RATE)
        hi = int(min(offset + part_seconds, duration) * SAMPLE_RATE)
        payload = transcribe_part(
            server, encode_flac(audio[lo:hi]), prompt, timestamps, request_timeout
        )
        choice = payload["choices"][0]
        text = cleanup_generation_loops(choice["message"]["content"] or "").strip()
        texts.append(text)
        part = {
            "part": part_index,
            "offset_s": round(offset, 1),
            "duration_s": round((hi - lo) / SAMPLE_RATE, 1),
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
            "text": text,
        }
        if timestamps:
            part["alignment"] = shift_alignment(
                payload.get("ctc_timestamps") or {"error": "no ctc_timestamps"},
                offset,
                part_index,
            )
        if part["finish_reason"] == "length":
            logger.warning(f"  {rel}: part {part_index} hit max_tokens — truncated")
        parts.append(part)
        offset += part_seconds
        part_index += 1

    txt_path = output_dir / rel.with_suffix(".txt")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(txt_path, "\n\n".join(texts) + "\n")

    alignment_errors = [
        p["alignment"]["error"]
        for p in parts
        if timestamps and p["alignment"]["error"] is not None
    ]
    if alignment_errors:
        # No JSON, so a rerun retries this file instead of skipping it.
        raise RuntimeError(f"alignment failed: {alignment_errors}")
    if timestamps:
        record = {
            "file": str(rel),
            "model": MODEL,
            "prompt": prompt,
            "duration_s": round(duration, 1),
            "parts": parts,
        }
        write_atomic(
            txt_path.with_suffix(".json"),
            json.dumps(record, ensure_ascii=False, indent=2),
        )

    elapsed = time.time() - start_time
    logger.info(
        f"  {rel}: {duration / 60:.1f} min, {len(parts)} part(s), "
        f"{sum(len(t.split()) for t in texts)} words in {elapsed:.0f}s"
        + (f", alignment errors: {alignment_errors}" if alignment_errors else "")
    )
    return {
        "file": str(rel),
        "duration_s": round(duration, 1),
        "num_parts": len(parts),
        "truncated_parts": sum(p["finish_reason"] == "length" for p in parts),
        "alignment_errors": alignment_errors,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Concurrent transcription via an in-job Nemotron Transcribe vLLM server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for the full `hf jobs run` command.",
    )
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("output_dir", help="Directory to write transcripts")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8000",
        help="vLLM server base URL (default: in-job localhost:8000). If no "
        "server answers there, the script installs and starts one.",
    )
    parser.add_argument(
        "--nemo-ref",
        default="main",
        help="NeMo Speech branch, tag or commit to install (default: main, "
        "as the model card requires)",
    )
    parser.add_argument(
        "--workdir",
        default=PREBUILT_WORKDIR
        if PREBUILT_WORKDIR.exists()
        else "/tmp/nemotron-transcribe",
        help="Where the serving stack and checkpoint are installed (default: "
        f"{PREBUILT_WORKDIR} in the prebuilt image, else /tmp/nemotron-transcribe)",
    )
    parser.add_argument(
        "--style",
        choices=sorted(PROMPTS),
        default="verbatim",
        help="Prompt preset from the model card (default: verbatim). "
        "'speakers' adds <spk:N> tags; --timestamps implies it.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom user prompt, e.g. with 'Context terms: <names>' "
        "(overrides --style)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Also write JSON with word timestamps, speakers and diarization "
        "segments (needs the server's --middleware flag)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Concurrent files in flight (default: 16)",
    )
    parser.add_argument(
        "--part-minutes",
        type=float,
        default=DEFAULT_PART_SECONDS / 60,
        help="Split audio into parts of this length (default: 55 min — the "
        "model is trained on recordings up to one hour). Speaker tags reset "
        "between parts.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=3600,
        help="Per-request timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Limit files (for testing)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    part_seconds = int(args.part_minutes * 60)
    if part_seconds < 1:
        logger.error("--part-minutes must be at least 1/60 (one second)")
        sys.exit(1)

    if args.prompt:
        prompt = args.prompt
    elif args.timestamps:
        prompt = PROMPTS["speakers"]
    else:
        prompt = PROMPTS[args.style]

    files = discover_audio_files(input_dir)
    if not files:
        logger.error(f"No audio files found in {input_dir}")
        sys.exit(1)
    if args.max_files:
        files = files[: args.max_files]

    # Resume: outputs are written per file, so a rerun after a timeout or
    # crash only processes what's missing.
    done_suffix = ".json" if args.timestamps else ".txt"
    done = {
        f
        for f in files
        if (output_dir / f.relative_to(input_dir).with_suffix(done_suffix)).exists()
    }
    if done:
        logger.info(f"Skipping {len(done)} file(s) with existing output")
        files = [f for f in files if f not in done]
        if not files:
            logger.info("Nothing to do — all outputs exist")
            return
    logger.info(f"Found {len(files)} audio file(s) to process")

    server_process = None
    if server_is_up(args.server):
        logger.info(f"Using the running server at {args.server}")
    else:
        from urllib.parse import urlparse

        from huggingface_hub import snapshot_download

        # Only start a server where the requests will go: this machine, on
        # the port named in --server.
        url = urlparse(args.server)
        if url.hostname not in {"127.0.0.1", "localhost"} or url.port is None:
            logger.error(
                f"No server answers at {args.server}. To start one here, pass "
                "--server http://127.0.0.1:<port> (the default is port 8000)."
            )
            sys.exit(1)
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        venv = install_serving_stack(workdir, args.nemo_ref)
        logger.info(f"Downloading {MODEL}...")
        model_dir = Path(snapshot_download(MODEL, local_dir=workdir / "model"))
        server_process = start_server(venv, model_dir, url.port)

    # Results are appended as files finish, so a crash keeps what completed
    # and a resumed run adds to the earlier records instead of replacing them.
    summary_path = output_dir / "summary.jsonl"
    results = []
    try:
        wait_for_server(args.server, server_process)
        start_time = time.time()
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool,
            open(summary_path, "a", encoding="utf-8") as summary,
        ):
            futures = {
                pool.submit(
                    process_file,
                    f,
                    input_dir,
                    output_dir,
                    args.server,
                    prompt,
                    args.timestamps,
                    part_seconds,
                    args.request_timeout,
                ): f
                for f in files
            }
            for fut in concurrent.futures.as_completed(futures):
                f = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    logger.error(f"  {f.name} FAILED: {exc}")
                    result = {"file": str(f.relative_to(input_dir)), "error": str(exc)}
                results.append(result)
                summary.write(json.dumps(result, ensure_ascii=False) + "\n")
                summary.flush()
        elapsed = time.time() - start_time
    finally:
        if server_process is not None:
            stop_server(server_process)

    ok = [r for r in results if "error" not in r]
    total_audio = sum(r["duration_s"] for r in ok)
    elapsed_str = f"{elapsed / 60:.1f} min" if elapsed > 60 else f"{elapsed:.1f}s"
    logger.info("=" * 50)
    logger.info(f"Done! {len(ok)}/{len(files)} file(s) in {elapsed_str}")
    if total_audio:
        logger.info(f"  Audio: {total_audio / 3600:.1f} h total")
        logger.info(f"  RTFx: {total_audio / elapsed:.1f}x realtime (aggregate)")
    logger.info(f"  Summary: {summary_path}")
    if len(ok) < len(files):
        # Non-zero, so the job (and the image builder's warm-up) shows as failed.
        logger.error(f"  Failed: {len(files) - len(ok)} file(s) — see {summary_path}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("Nemotron 3.5 Transcribe (in-job vLLM server)")
        print("=" * 60)
        print("\nStarts an in-job vLLM server (vLLM 0.32.0 + NeMo Speech), splits")
        print("audio over one hour, posts files concurrently, writes .txt")
        print("transcripts (+ JSON word timestamps and speakers with --timestamps).")
        print("\nSee module docstring for the `hf jobs uv run` command")
        print("(image + flavor come from the [tool.hf-jobs] header).")
        print("\nFor full help: uv run nemotron-transcribe.py --help")
        sys.exit(0)

    main()
