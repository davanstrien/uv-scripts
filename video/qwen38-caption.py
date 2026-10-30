# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]==0.3.0",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.32.0"
# flavor = "a100-large"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# ///

"""
Caption whole videos — no chunking — with Qwen3.8-27B, writing a resumable dataset.

Qwen3.8-27B (Apache 2.0, not gated) ingests videos natively at hour scale: an
11-minute film is one request, and in --timestamps mode the model returns a
timeline of <start - end> events whose times are read off per-frame time tags
(verified accurate to the ~2s sampling interval, with on-screen text quoted
verbatim). This is the structural difference from marlin-caption.py, which must
split anything over ~60s into chunks.

Serves the model on vLLM inside the job and pumps every video in INPUT_DIR
through it with saturate: crash-safe parquet out, exact resume (re-running
skips completed videos), congestion-aware concurrency.

Input:                              Output (one parquet dataset):
  /input/film.mp4   (11 min)  ->      1 row per video — columns: video,
  /input/clip.mp4   (45 s)    ->      duration_s, caption, events (JSON,
                                      --timestamps mode), token counts

Examples:

  # Caption a bucket of videos on HF Jobs (the [tool.hf-jobs] header sets the
  # image, a100-large and HF_TOKEN; hf CLI >= 1.32)
  hf jobs uv run --timeout 2h \\
      -v hf://buckets/user/my-videos:/input:ro \\
      https://huggingface.co/datasets/uv-scripts/video/raw/main/qwen38-caption.py \\
      /input hf://buckets/user/my-videos/captions

  # Timestamped event timeline per film instead of a prose caption
  ... qwen38-caption.py /input hf://buckets/user/out --timestamps

  # Local machine with a CUDA GPU (needs ~60 GB VRAM); vLLM comes from the
  # Jobs image, so add it yourself
  uv run --with vllm==0.32.0 qwen38-caption.py ./videos ./captions-out

Thinking mode defaults off for captions (little gain, big latency) but ON for
--timestamps: measured on the 11-min film, the non-thinking model may answer a
timeline request with a one-line summary, while thinking mode produced a
frame-verified 96-event timeline. --thinking / --no-thinking overrides.

Frame budget: the model's default video processor caps a video at 768 frames
and a fixed pixel budget (~12k video tokens). At the default 2 fps anything longer than
~6.4 min is capped, so a long film is seen at lower frame rate and resolution:
an 11-min film at ~1 frame per 0.9 s, an hour at ~1 frame per 4.7 s. Small
on-screen text gets harder to read as videos get longer. The model card's
"Long Video Understanding" section explains how to raise the budget.

Model: Qwen/Qwen3.8-27B (27B dense, hybrid GDN linear attention, vision tower).
bf16 weights are ~52 GB — a100-large is the smallest sensible Jobs flavor;
h200 generates ~2.3x faster if latency matters.
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3.8-27B"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}

CAPTION_PROMPT = (
    "Describe this video: the setting, the people, what happens over its course, "
    "and any text that appears on screen."
)
TIMESTAMPS_PROMPT = (
    "Watch this video and produce a timeline of events covering the WHOLE video "
    "from start to finish. Output ONLY the timeline, one event per line. For "
    "each event give the time range as <start - end> in decimal seconds "
    "(e.g. <95.0 - 112.5>) followed by a one-sentence description. "
    "Include title cards, on-screen text (quoted), and scene changes. "
    "Do not summarize; cover the full duration."
)

THINK = re.compile(r"<think>.*?</think>\s*|^.*?</think>\s*", re.DOTALL)
# accepts decimal seconds (<95.0 - 112.5>) and clock forms (<01:35 - 01:52>)
EVENT_LINE = re.compile(
    r"<(\d+(?::\d{2}){0,2}(?:\.\d+)?)\s*-\s*(\d+(?::\d{2}){0,2}(?:\.\d+)?)>\s*[:–\-]?\s*(.+)")


def to_seconds(ts: str) -> float:
    parts = [float(x) for x in ts.split(":")]
    out = 0.0
    for part in parts:
        out = out * 60 + part
    return round(out, 2)


def probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120)
        return round(float(out.stdout.strip()), 2)
    except (ValueError, subprocess.SubprocessError):
        return None


def config_tag(**config) -> str:
    """Short hash of everything that changes the output, so resume never reuses
    a row produced under a different prompt, mode or frame rate."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def parse_events(text: str) -> list[dict]:
    return [{"start": to_seconds(a), "end": to_seconds(b), "text": d.strip()}
            for a, b, d in EVENT_LINE.findall(text)]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", help="Directory of videos (e.g. a mounted bucket)")
    parser.add_argument("output", help="Dataset output: hf://datasets/..., hf://buckets/..., or local path")
    parser.add_argument("--timestamps", action="store_true",
                        help="Return a <start - end> event timeline per video instead of a prose caption")
    parser.add_argument("--prompt", help="Override the built-in prompt")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                        help="Thinking mode. Default: on for --timestamps (measured: without it the "
                             "model may summarize instead of covering the video), off otherwise")
    parser.add_argument("--max-videos", type=int, help="Only process the first N videos (testing)")
    parser.add_argument("--window-max", type=int, default=8,
                        help="Max in-flight requests (default 8 — whole-video requests are large)")
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--retry-errors", action="store_true",
                        help="Re-attempt rows that errored in a previous run")
    args = parser.parse_args()
    if args.window_max < 1:
        sys.exit("--window-max must be at least 1")

    if shutil.which("ffprobe") is None:
        sys.exit("ffprobe not found — use the vllm/vllm-openai image (has it) or install ffmpeg")

    input_root = Path(args.input_dir).resolve()
    videos = sorted(p for p in input_root.rglob("*")
                    if p.suffix.lower() in VIDEO_EXTENSIONS)
    if args.max_videos:
        videos = videos[: args.max_videos]
    if not videos:
        sys.exit(f"no videos found under {input_root}")
    logger.info("found %d videos", len(videos))

    prompt = args.prompt or (TIMESTAMPS_PROMPT if args.timestamps else CAPTION_PROMPT)
    thinking = args.timestamps if args.thinking is None else args.thinking
    tag = config_tag(model=MODEL, prompt=prompt, timestamps=args.timestamps,
                     thinking=thinking)
    logger.info("config tag %s (part of every row id; changing the prompt or mode "
                "re-processes videos instead of reusing old rows)", tag)

    # Row id = relative path + file size + config tag: a replaced video or a
    # changed setting gets a new id, so resume cannot return a stale result.
    rows = [(f"{p.relative_to(input_root)}|{p.stat().st_size}|{tag}",
             {"video": str(p.relative_to(input_root)), "uri": p.as_uri(),
              "duration_s": probe_duration(p)})
            for p in videos]
    # Timeline output scales with video length; measured 9.6K tokens for an
    # 11-min film — leave generous headroom. Thinking needs its own budget on top.
    max_tokens = 16384 if args.timestamps else 4096
    if thinking:
        max_tokens += 32768

    def to_request(row: dict) -> dict:
        req = {
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": row["uri"]}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": max_tokens,
        }
        if thinking:  # model-card sampling params per mode
            req.update({"temperature": 1.0, "top_p": 0.95, "top_k": 20})
        else:
            req.update({"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                        "presence_penalty": 1.5,
                        "chat_template_kwargs": {"enable_thinking": False}})
        return req

    def parse(row: dict, resp: dict) -> dict:
        # Raising here makes saturate store an error row, which --retry-errors
        # re-attempts; a truncated or empty answer must not count as done.
        choice = resp["choices"][0]
        raw = choice["message"]["content"] or ""
        if choice.get("finish_reason") == "length":
            raise ValueError(f"truncated at max_tokens={max_tokens} (finish_reason=length)")
        if "<think>" in raw and "</think>" not in raw:
            raise ValueError("reasoning never finished (no </think>)")
        text = THINK.sub("", raw).strip()
        if not text:
            raise ValueError("empty response")
        usage = resp.get("usage") or {}
        out = {"video": row["video"], "duration_s": row["duration_s"],
               "caption": text,
               "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens")}
        if args.timestamps:
            events = parse_events(text)
            if not events:
                raise ValueError("no <start - end> events in the response")
            duration = row["duration_s"]
            if duration and any(e["end"] > duration + 2.0 for e in events):
                raise ValueError(f"event ends past the video's {duration}s duration")
            out["events"] = json.dumps(events)
        return out

    from saturate import Auto, Engine, pump

    with Engine(
        MODEL,
        engine="vllm",
        extra_args=[
            "--allowed-local-media-path", str(input_root),
            "--max-model-len", str(args.max_model_len),
            # Model card's video flag. vLLM 0.32 then samples at the model's
            # default 2 fps (a 60 s clip -> 119 frames). Do NOT also pass
            # --mm-processor-kwargs '{"fps": ..., "do_sample_frames": true}':
            # on 0.32 the frames are sampled twice and every request 400s.
            "--media-io-kwargs", json.dumps({"video": {"num_frames": -1}}),
            # Unbounded growth on distinct videos — OOM-kills the node if left on.
            "--mm-processor-cache-gb", "0",
        ],
    ) as endpoint:
        stats = pump(
            rows,
            to_request=to_request,
            parse=parse,
            endpoint=endpoint,
            output=args.output,
            window=Auto(initial=min(4, args.window_max), min_limit=1,
                        max_limit=args.window_max),
            retry_errors=args.retry_errors,
            # A thinking timeline can run to ~49k tokens; at 8 in flight one
            # request can take longer than saturate's 1800 s default.
            read_timeout=7200,
        )

    logger.info("done: %d ok, %d failed, %.1f tok/s (window settled at %d)",
                stats.rows_processed, stats.rows_failed,
                stats.tokens_per_sec, stats.final_limit)
    if stats.rows_failed:
        logger.warning("failed rows are recorded in the output; re-run with "
                       "--retry-errors to attempt them again")


if __name__ == "__main__":
    main()
