# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]==0.3.0",
#     "vllm",
#     "qwen-vl-utils",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.32.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "a10g-small"
# secrets = ["HF_TOKEN"]
# timeout = "2h"
# ///

"""
Caption videos with timestamped events using Marlin-2B, writing a resumable dataset.

Marlin-2B (NemoStation/Marlin-2B, gated — accept the license on its model page first)
is a 2B video VLM producing dense scene captions with second-precise <start - end>
events, plus a temporal-grounding mode ("when does X happen?"). This recipe serves it
on vLLM and pumps every video in INPUT_DIR through it with saturate: crash-safe
parquet out, exact resume (re-running skips completed videos), congestion-aware
concurrency.

Videos longer than ~60s are split into chunks before captioning and event timestamps
are offset back to global film time. This is not an optimisation: Marlin compresses
any input onto a ~60s timeline (it was trained on short clips), so captioning a long
film in one request produces plausible-looking but wrong-scale timestamps.

Input:                            Output (one parquet dataset):
  /input/film.mp4  (11 min)  →      11 rows (one per 60s chunk), each with
  /input/clip.mp4  (45 s)    →      1 row  — columns: video, chunk_start,
                                    chunk_end, scene, events (JSON), caption,
                                    prompt_tokens, completion_tokens

Examples:

  # Caption a bucket of videos on HF Jobs (a10g-small handles ~24 concurrent 60s clips)
  hf jobs uv run --image vllm/vllm-openai:v0.32.0 --flavor a10g-small \\
      -s HF_TOKEN \\
      -v hf://buckets/user/my-videos:/input:ro \\
      marlin-caption.py /input hf://buckets/user/my-videos/captions

  # Temporal grounding instead of captioning
  ... marlin-caption.py /input hf://buckets/user/out --find "a person enters the room"

Find mode returns CANDIDATES, not detections: Marlin always emits a span, even in
chunks where the event never occurs (the model has no "not present" answer). Treat
spans as a shortlist to rank or verify downstream — when the event is really there,
they are precise (matches caption-mode events to the half-second in testing).

  # Local machine with a CUDA GPU
  uv run marlin-caption.py ./videos ./captions-out

Memory safety (learned the hard way — defaults encode a measured config):
  * --mm-processor-cache-gb 0 is passed to vLLM: its multimodal cache grows without
    bound on distinct videos and OOM-kills 15 GB nodes. Do not re-enable for batch work.
  * In-flight window is capped (default 24 ≈ the measured a10g-small ceiling for 60s
    clips at 640px; each in-flight request holds ~290 MB of decoded frames). Use
    --window-max 64 on RAM-rich flavors (a10g-large and up).

Model: NemoStation/Marlin-2B (Apache-2.0, Qwen3.5-2B fine-tune; served via vLLM's
native qwen3_5 implementation through an architecture override — no custom code).
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "NemoStation/Marlin-2B"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
WORK_DIR = Path("/tmp/marlin_work")

# Canonical training-time prompts from the model's modeling_marlin.py — must match
# exactly; the model card warns that diverging silently degrades quality.
CAPTION_PROMPT = (
    "Provide a spatial description of this clip followed by time-ranged events.\n"
    "For each event, give the time range as <start - end> and a short description."
)
GROUNDING_PROMPT_TEMPLATE = (
    'Identify the timestamps during which "{event}" takes place. '
    'Output the time range as "From <start> to <end>." (numbers in seconds).'
)

THINK = re.compile(r"<think>.*?</think>\s*|^\s*<think>\s*\n*|</think>\s*", re.DOTALL)
EVENT_LINE = re.compile(r"<(\d+\.?\d*)\s*-\s*(\d+\.?\d*)>\s*(.*)")
SPAN = re.compile(r"From\s+(\d+\.?\d*)\s+to\s+(\d+\.?\d*)", re.IGNORECASE)


def probe_duration(path: Path) -> float | None:
    """Video duration in seconds via ffprobe, or None if unreadable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120)
        return float(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return None


def stage_chunks(videos: list[Path], input_root: Path, chunk_seconds: int) -> list[tuple[str, dict]]:
    """Copy short videos / split long ones into WORK_DIR; return pump rows.

    Chunk boundaries are re-encoded (libx264) rather than stream-copied: stream copy
    snaps to keyframes and skews the timestamps we ground against — same choice
    Marlin's own multi_find makes.
    """
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, video in enumerate(videos):
        rel = video.relative_to(input_root).as_posix()
        duration = probe_duration(video)
        if duration is None:
            logger.warning("skipping unreadable video: %s", rel)
            continue
        if duration <= chunk_seconds * 1.25:  # tolerate slightly-long clips unsplit
            staged = WORK_DIR / f"v{n:05d}.mp4"
            if not staged.exists():
                shutil.copyfile(video, staged)
            rows.append((f"{rel}#0", {"video": rel, "path": str(staged),
                                      "start": 0.0, "end": round(duration, 2)}))
            continue
        start = 0.0
        c = 0
        while start < duration - 5:  # drop tails shorter than 5s
            end = min(start + chunk_seconds, duration)
            staged = WORK_DIR / f"v{n:05d}_c{c:04d}.mp4"
            if not staged.exists():
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error",
                     "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video),
                     "-c:v", "libx264", "-preset", "fast", "-an", "-y", str(staged)],
                    check=True, stdin=subprocess.DEVNULL)
            rows.append((f"{rel}#{int(start)}", {"video": rel, "path": str(staged),
                                                 "start": round(start, 2), "end": round(end, 2)}))
            start, c = end, c + 1
        logger.info("split %s (%.0fs) into %d chunks", rel, duration, c)
    return rows


def parse_caption_text(text: str, offset: float) -> tuple[str, list[dict]]:
    """Split a Mode-1 caption into (scene, events); event times offset to global."""
    scene, events = "", []
    body = text.split("Events:", 1)
    scene = body[0].replace("Scene:", "", 1).strip()
    for line in (body[1] if len(body) > 1 else "").splitlines():
        m = EVENT_LINE.match(line.strip())
        if m:
            events.append({"start": round(float(m.group(1)) + offset, 2),
                           "end": round(float(m.group(2)) + offset, 2),
                           "text": m.group(3).strip()})
    return scene, events


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", help="Directory of videos (e.g. a mounted bucket)")
    parser.add_argument("output", help="Dataset output: hf://datasets/..., hf://buckets/..., or local path")
    parser.add_argument("--find", metavar="EVENT",
                        help="Temporal grounding mode: locate EVENT instead of captioning. "
                             "Every chunk returns a candidate span — filter downstream; "
                             "the model cannot say 'not present'.")
    parser.add_argument("--chunk-seconds", type=int, default=60,
                        help="Chunk length for long videos (default 60 — Marlin's training scale)")
    parser.add_argument("--max-videos", type=int, help="Only process the first N videos (testing)")
    parser.add_argument("--window-max", type=int, default=24,
                        help="Max in-flight requests (default 24 for 15 GB nodes; 64 on a10g-large+)")
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--retry-errors", action="store_true",
                        help="Re-attempt rows that errored in a previous run")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.exit("ffmpeg/ffprobe not found — use the vllm/vllm-openai image (has both) "
                 "or install ffmpeg")

    input_root = Path(args.input_dir)
    videos = sorted(p for p in input_root.rglob("*")
                    if p.suffix.lower() in VIDEO_EXTENSIONS)
    if args.max_videos:
        videos = videos[: args.max_videos]
    if not videos:
        sys.exit(f"no videos found under {input_root}")
    logger.info("found %d videos; staging chunks...", len(videos))
    rows = stage_chunks(videos, input_root, args.chunk_seconds)
    logger.info("%d chunk rows to process", len(rows))

    prompt = (GROUNDING_PROMPT_TEMPLATE.format(event=args.find.strip())
              if args.find else CAPTION_PROMPT)
    max_tokens = 64 if args.find else 1024

    def to_request(row: dict) -> dict:
        return {
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": f"file://{row['path']}"}},
                {"type": "text", "text": prompt},
            ]}],
            "temperature": 0,  # greedy, matching Marlin's own wrappers
            "max_tokens": max_tokens,
        }

    def parse(row: dict, resp: dict) -> dict:
        text = THINK.sub("", resp["choices"][0]["message"]["content"]).strip()
        usage = resp.get("usage") or {}
        out = {"video": row["video"], "chunk_start": row["start"], "chunk_end": row["end"],
               "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens")}
        if args.find:
            m = SPAN.search(text)
            out.update({
                "span_start": round(float(m.group(1)) + row["start"], 2) if m else None,
                "span_end": round(float(m.group(2)) + row["start"], 2) if m else None,
                "format_ok": m is not None,
                "raw": text,
            })
        else:
            scene, events = parse_caption_text(text, offset=row["start"])
            out.update({"scene": scene, "events": json.dumps(events), "caption": text})
        return out

    from saturate import Auto, Engine, pump

    with Engine(
        MODEL,
        engine="vllm",
        extra_args=[
            # Marlin is a stock Qwen3.5-2B fine-tune; its custom code is only
            # convenience wrappers, so route onto vLLM's native implementation.
            "--hf-overrides", '{"architectures": ["Qwen3_5ForConditionalGeneration"]}',
            "--allowed-local-media-path", str(WORK_DIR),
            "--max-model-len", str(args.max_model_len),
            # Unbounded growth on distinct videos — OOM-kills the node if left on.
            "--mm-processor-cache-gb", "0",
            "--enforce-eager",
        ],
    ) as endpoint:
        stats = pump(
            rows,
            to_request=to_request,
            parse=parse,
            endpoint=endpoint,
            output=args.output,
            window=Auto(initial=8, max_limit=args.window_max),
            retry_errors=args.retry_errors,
        )

    logger.info("done: %d ok, %d failed, %.1f tok/s (window settled at %d)",
                stats.rows_processed, stats.rows_failed,
                stats.tokens_per_sec, stats.final_limit)
    if stats.rows_failed:
        logger.warning("failed rows are recorded in the output; re-run with "
                       "--retry-errors to attempt them again")


if __name__ == "__main__":
    main()
