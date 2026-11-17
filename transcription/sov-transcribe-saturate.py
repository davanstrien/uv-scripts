# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]==0.3.1",
#     "av",
#     "numpy",
#     "soundfile",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.32.0"
# flavor = "rtx-pro-6000"
# timeout = "4h"
# secrets = ["HF_TOKEN"]
# ///
"""
Transcribe audio with Salesforce SOV-Transcribe-1 on vLLM, via saturate.

SOV-Transcribe-1 is a Whisper large-v3 encoder on a Gemma 4 26B-A4B MoE decoder
(26.6B total, ~4.6B active, Apache 2.0). It transcribes English, French, German,
Spanish, Italian, Portuguese and Dutch, and takes a natural-language instruction:
verbatim or cleaned-up text, a language hint, or a list of names and terms to
expect. The card tests recordings up to 30 minutes long.

The script decodes each file to 16 kHz mono and cuts it into equal parts of at most
`--part-minutes` (default 10). Each part is one request and one output row. On
~30-minute radio episodes sent whole, the model often stopped early or skipped
long stretches, with `finish_reason == "stop"`; vLLM also rejects audio over
1,800 s. Parts are cut at fixed times, so a word at a boundary can be split or
lost.

The script starts `vllm serve` in the Job with the model's vLLM plugin and, by
default, its DFlash speculative-decoding draft (the model card's serving setup).
saturate sends the files and writes the results:

- **Adaptive concurrency.** Each part is a chat request with the audio inline
  (FLAC, base64), so responses carry token counts and saturate's window grows and
  shrinks with the server's load.
- **Crash-safe, resumable output.** Results stream to the output as parquet files.
  Re-running the same command skips files whose parts are all done, without
  downloading them, and sends only the missing parts of the others.
- **Retries, then durable error rows.** A part that still fails, or a file that
  cannot be decoded, is recorded as `{id, error}`; `--retry-errors` re-admits only
  those on a later run.

Run on HF Jobs. The [tool.hf-jobs] header sets the image, the GPU and the HF_TOKEN
secret (`hf` CLI 1.32+):

  hf jobs uv run --detach \\
      https://huggingface.co/datasets/uv-scripts/transcription/raw/main/sov-transcribe-saturate.py \\
      "hf://buckets/user/audio-files/**/*.mp3" user/transcripts

The output repo holds `data/part-*.parquet` with one row per audio part:
`{id, file, part, n_parts, start_s, duration_s, skipped_packets, text, finish_reason,
prompt_tokens, completion_tokens, model, prompt, error}`. `file` is the input path
without the scheme (`buckets/user/audio-files/a.mp3`) and `id` is
`<file>#p02of03`. To get one transcript per file, sort by `file` and `part` and
join `text`:

  df = load_dataset(<output>, data_dir="data", split="train").to_pandas()
  full = df.sort_values(["file", "part"]).groupby("file")["text"].agg(" ".join)

A row with `finish_reason == "length"` hit `--max-tokens`; check it for a
repetition loop. The text is stored as the model returned it: no repetition
cleanup is applied.

`--no-draft` turns speculative decoding off. The model card reports a 2-3x speed-up
for one request at a time; under a full batch the gain is smaller, so compare both
on your audio.

With `--endpoint https://<job_id>--8000.hf.jobs/v1` the script uses a running server
instead of starting one, and sends HF_TOKEN as Bearer (only to https `*.hf.jobs`
hosts).

The SERVING dict below is the per-model tuning prior (serve flags + install +
request shape). Agents can `ast.literal_eval` it without running the script; the
script itself consumes it, so it cannot drift from reality.
"""

import argparse
import base64
import collections
import io
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Serving starting values for SOV-Transcribe-1 on vLLM 0.32.0. Provenance: the model
# card's "vLLM" and "Speculative decoding" sections.
# - plugin: the repo's vllm_plugin/ folder registers the architecture with vLLM (a
#   vllm.general_plugins entry point). It pins vllm==0.32.0, which the image already
#   has, so it is installed with --no-deps into the image's Python.
# - serve_args: --max-model-len 32768 as on the card (30 min of audio is 60 windows
#   x 375 tokens = 22,500 prompt tokens, so --part-minutes up to 30 fits). --max-num-seqs 64 with the draft: the card
#   says vLLM's default of 1,024 runs out of memory at start-up with it.
# - env VLLM_MAX_AUDIO_DECODE_DURATION_S=1800: the card's setting for 30-minute audio.
# - chat_template_kwargs enable_thinking=False and temperature 0: the card's request.
SERVING = {
    "model": "salesforce-sovereign-intelligence/SOV-Transcribe-1",
    "plugin_pattern": "vllm_plugin/*",
    "serve_args": ["--max-model-len", "32768"],
    "draft_args": [
        "--max-num-seqs", "64",
        "--speculative-config",
        (
            '{"method": "dflash", "model": "salesforce-sovereign-intelligence/SOV-Transcribe-1-DFlash", '
            '"num_speculative_tokens": 15}'
        ),
    ],
    "env": {"VLLM_MAX_AUDIO_DECODE_DURATION_S": "1800"},
    "route": "/chat/completions",
    "chat_template_kwargs": {"enable_thinking": False},
}  # fmt: skip

# User prompts from the model card's prompt guide and evaluation protocol.
PROMPTS = {
    # Prompt guide, "Verbatim transcript" (the server's default instruction).
    "verbatim": "Transcribe the audio verbatim, outputting only spoken words in sequence.",
    # Evaluation protocol, Open ASR English sets: fillers kept, punctuation and case.
    "verbatim-punct": (
        "Transcribe the audio verbatim, including fillers and repetitions. "
        "Use punctuation and capitalization. Output only the transcript."
    ),
    # Evaluation protocol, SPGISpeech: fillers and accidental repetitions dropped.
    "clean": (
        "Transcribe the audio, omitting fillers and accidental repetitions while "
        "preserving the remaining words. Use punctuation and capitalization. "
        "Output only the transcript."
    ),
    # Prompt guide, "Automatic language".
    "auto-language": (
        "Transcribe the audio verbatim in the language's standard writing system, "
        "using punctuation and capitalization. Output only the transcript."
    ),
}
LANGUAGE_PROMPT = "Transcribe the {language} audio into text, ensuring all punctuation marks are included."
KEYWORDS_PROMPT = "This audio may contain the following words or phrases: {keywords}. Transcribe the audio into text."

SAMPLE_RATE = 16000  # the model's Whisper encoder input rate
PART_ID = re.compile(r"^(?P<file>.+)#p(?P<part>\d+)of(?P<n_parts>\d+)$")


def build_prompt(args) -> str:
    if args.prompt:
        return args.prompt
    if args.keywords:
        return KEYWORDS_PROMPT.format(keywords=args.keywords)
    if args.language:
        return LANGUAGE_PROMPT.format(language=args.language)
    return PROMPTS[args.style]


def decode_16k_mono(data: bytes, suffix: str = ""):
    """Decode any audio container ffmpeg reads (via PyAV) to 16 kHz mono int16.

    Returns (samples, skipped). Tolerates three faults seen in old MP3 rips, each of
    which makes a strict decode (and vLLM's own) reject the whole file:
    - damaged packets mid-file: skipped and counted, not fatal;
    - a container that probing misreads (e.g. an ID3 tag on a RIFF/WAVE file that
      holds MP3 data): retried with the format the file's suffix names;
    - a sample rate that changes mid-stream: the resampler is rebuilt for it."""
    import av
    import numpy as np

    try:
        container = av.open(io.BytesIO(data))
    except av.error.InvalidDataError:
        fmt = suffix.lower().lstrip(".")
        if not fmt:
            raise
        container = av.open(io.BytesIO(data), format=fmt)

    chunks, skipped = [], 0
    resampler, layout_key = None, None

    def drain(frame):
        chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))

    with container:
        for packet in container.demux(container.streams.audio[0]):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                skipped += 1
                continue
            for frame in frames:
                key = (frame.sample_rate, frame.layout.name, frame.format.name)
                if key != layout_key:  # first frame, or the stream changed mid-file
                    if resampler is not None:
                        drain(None)
                    resampler = av.AudioResampler(
                        format="s16", layout="mono", rate=SAMPLE_RATE
                    )
                    layout_key = key
                drain(frame)
        if resampler is not None:
            drain(None)
    samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    return samples, skipped


def split_file(file_id: str, data: bytes, part_s: float) -> list[tuple[str, dict]]:
    """One file -> (part id, row) pairs: equal parts of at most part_s seconds, as FLAC.

    Runs in a worker process. A file that cannot be decoded becomes one row that
    carries the error, so it lands in the output as an error row."""
    import soundfile

    try:
        samples, skipped = decode_16k_mono(data, Path(file_id).suffix)
    except Exception as e:  # noqa: BLE001 -- any decode failure is this file's error row
        return [
            (
                f"{file_id}#decode",
                {"file": file_id, "decode_error": f"{type(e).__name__}: {e}"},
            )
        ]
    if len(samples) == 0:
        return [
            (f"{file_id}#decode", {"file": file_id, "decode_error": "no audio samples"})
        ]
    n_parts = max(1, math.ceil(len(samples) / (part_s * SAMPLE_RATE)))
    size = math.ceil(len(samples) / n_parts)
    parts = []
    for i in range(n_parts):
        chunk = samples[i * size : (i + 1) * size]
        buf = io.BytesIO()
        soundfile.write(buf, chunk, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        row = {
            "file": file_id, "part": i + 1, "n_parts": n_parts,
            "start_s": round(i * size / SAMPLE_RATE, 2),
            "duration_s": round(len(chunk) / SAMPLE_RATE, 2),
            "skipped_packets": skipped,
            "flac": buf.getvalue(),
        }  # fmt: skip
        parts.append((f"{file_id}#p{i + 1:02d}of{n_parts:02d}", row))
    return parts


def done_files(ids: set[str]) -> set[str]:
    """Files whose parts are all in the output: a re-run skips them before download."""
    seen = collections.defaultdict(set)
    totals = {}
    for row_id in ids:
        m = PART_ID.match(row_id)
        if m:
            seen[m["file"]].add(int(m["part"]))
            totals[m["file"]] = int(m["n_parts"])
    return {f for f, parts in seen.items() if len(parts) == totals[f]}


def part_rows(files, part_s: float, skip: set[str], workers: int):
    """Decode and split files in worker processes, a bounded number ahead, in order."""
    pending = collections.deque()
    with ProcessPoolExecutor(workers) as pool:
        for file_id, row in files:
            pending.append(pool.submit(split_file, file_id, row["bytes"], part_s))
            if len(pending) >= 2 * workers:
                yield from (p for p in pending.popleft().result() if p[0] not in skip)
        while pending:
            yield from (p for p in pending.popleft().result() if p[0] not in skip)


def vllm_python() -> str:
    """The image's Python, where vLLM is installed (not this script's uv venv)."""
    from saturate.engine import python_with

    python = python_with("vllm")
    if python == sys.executable:
        sys.exit(
            "vLLM is not installed on this machine: run on the vllm/vllm-openai:v0.32.0 image"
        )
    return python


def install_plugin(python: str) -> None:
    """Install the model's vLLM plugin into the image's Python."""
    from huggingface_hub import snapshot_download

    repo = Path(
        snapshot_download(SERVING["model"], allow_patterns=[SERVING["plugin_pattern"]])
    )
    plugin = repo / "vllm_plugin"
    print(f"installing the SOV-Transcribe vLLM plugin from {plugin}", file=sys.stderr)
    pip = [python, "-m", "pip", "install", "-q", "--no-deps"]
    subprocess.run([*pip, str(plugin)], check=True)
    has_librosa = subprocess.run(
        [python, "-c", "import librosa"], check=False, capture_output=True
    )
    if has_librosa.returncode != 0:  # the plugin's only dependency besides vLLM
        subprocess.run([python, "-m", "pip", "install", "-q", "librosa"], check=True)


def main():
    ap = argparse.ArgumentParser(
        description="SOV-Transcribe-1 batch transcription on vLLM via saturate"
    )
    ap.add_argument("input", help='Glob of audio files: "hf://buckets/org/name/**/*.mp3", '
                    '"/input/**/*.wav" (a mounted dir), ...')  # fmt: skip
    ap.add_argument(
        "output",
        help="Output dataset id (org/name) or fsspec URI (hf://buckets/..., a dir)",
    )
    ap.add_argument("--style", choices=sorted(PROMPTS), default="verbatim",
                    help="Prompt preset from the model card (default: verbatim)")  # fmt: skip
    ap.add_argument("--language", default=None,
                    help="Language hint, e.g. French (the card's language-hint prompt; overrides --style)")  # fmt: skip
    ap.add_argument("--keywords", default=None,
                    help="Comma-separated names and terms to expect (the card's keyword prompt; "
                    "overrides --style and --language)")  # fmt: skip
    ap.add_argument(
        "--prompt", default=None, help="Custom instruction (overrides all of the above)"
    )
    ap.add_argument(
        "--no-draft",
        action="store_true",
        help="Serve without the DFlash speculative-decoding draft",
    )
    ap.add_argument(
        "--part-minutes",
        type=float,
        default=10,
        help="Split audio into equal parts of at most this length, up to 30 (default: 10)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Output token budget per part (default: 8192)",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Only the first N files (smoke tests)",
    )
    ap.add_argument(
        "--retry-errors", action="store_true", help="Re-admit parts recorded as errors"
    )
    ap.add_argument("--endpoint", default=None,
                    help="Use a running vLLM server (e.g. https://<job_id>--8000.hf.jobs/v1) "
                    "instead of starting one; sends HF_TOKEN as Bearer")  # fmt: skip
    args = ap.parse_args()
    if not 0 < args.part_minutes <= 30:
        ap.error(
            "--part-minutes must be more than 0 and at most 30 (the model card's tested limit)"
        )

    import pyarrow as pa
    from saturate import Engine, bucket_rows, existing_ids, make_json_request, pump

    is_repo_id = re.fullmatch(
        r"[\w-][\w.-]*/[\w.-]*[\w-]", args.output
    ) and not os.path.exists(args.output)
    output = f"hf://datasets/{args.output}/data" if is_repo_id else args.output
    prompt = build_prompt(args)

    # Resume: files with every part done are skipped before any bytes are read; the
    # others are decoded again and only their missing parts are sent.
    skip = existing_ids(output, retry_errors=args.retry_errors)
    files = bucket_rows(args.input, skip=done_files(skip), limit=args.max_files)
    workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    rows = part_rows(files, args.part_minutes * 60, skip, workers)
    total_audio = [0.0]
    # Declared, so a parquet file holding only error rows has the same columns as the rest.
    schema = pa.schema([
        ("id", pa.string()), ("error", pa.string()), ("file", pa.string()),
        ("part", pa.int64()), ("n_parts", pa.int64()), ("start_s", pa.float64()),
        ("duration_s", pa.float64()), ("skipped_packets", pa.int64()), ("text", pa.string()),
        ("finish_reason", pa.string()), ("prompt_tokens", pa.int64()),
        ("completion_tokens", pa.int64()), ("model", pa.string()), ("prompt", pa.string()),
    ])  # fmt: skip

    def to_request(row):
        if "decode_error" in row:  # saturate records this as the row's error
            raise ValueError(f"could not decode audio: {row['decode_error']}")
        audio_url = f"data:audio/flac;base64,{base64.b64encode(row['flac']).decode()}"
        body = {
            "model": SERVING["model"],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "audio_url", "audio_url": {"url": audio_url}},
            ]}],
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": SERVING["chat_template_kwargs"],
        }  # fmt: skip
        return make_json_request(SERVING["route"], body)

    def parse(row, body):
        total_audio[0] += row["duration_s"]
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        return {
            "file": row["file"],
            "part": row["part"],
            "n_parts": row["n_parts"],
            "start_s": row["start_s"],
            "duration_s": row["duration_s"],
            "skipped_packets": row["skipped_packets"],
            "text": (choice["message"].get("content") or "").strip(),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "model": SERVING["model"],
            "prompt": prompt,
        }

    def run(endpoint, headers=None):
        start = time.time()
        stats = pump(rows, to_request, parse, endpoint, output,
                     read_timeout=1800, headers=headers,
                     retry_errors=args.retry_errors, schema=schema,
                     prepare_workers=4)  # fmt: skip
        elapsed = time.time() - start
        rtfx = f", {total_audio[0] / elapsed:.1f}x realtime" if total_audio[0] else ""
        print(f"{output}: {stats.rows_processed} ok, {stats.rows_failed} error rows, "
              f"{total_audio[0] / 3600:.2f} h of audio in {elapsed / 60:.1f} min{rtfx}",
              file=sys.stderr)  # fmt: skip
        print("SOV_TRANSCRIBE_SATURATE " + stats.to_json(), flush=True)

    if args.endpoint:
        from urllib.parse import urlparse

        url = urlparse(args.endpoint)
        is_jobs = url.scheme == "https" and (url.hostname or "").endswith(".hf.jobs")
        token = os.environ.get("HF_TOKEN") if is_jobs else None
        run(args.endpoint, {"Authorization": f"Bearer {token}"} if token else None)
        return

    python = vllm_python()
    install_plugin(python)
    os.environ.update(SERVING["env"])  # inherited by the server process
    extra = [*SERVING["serve_args"], *([] if args.no_draft else SERVING["draft_args"])]
    cmd = [python, "-m", "vllm.entrypoints.cli.main", "serve", SERVING["model"],
           "--port", "8000", "--host", "127.0.0.1", *extra]  # fmt: skip
    with Engine(
        SERVING["model"], engine="vllm", cmd=cmd, boot_timeout=3600
    ) as endpoint:
        run(endpoint)


if __name__ == "__main__":
    main()
