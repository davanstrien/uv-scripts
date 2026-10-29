---
viewer: false
tags:
  - uv-script
  - video
  - video-text-to-text
  - video-captioning
  - temporal-grounding
---

# Video

Scripts for captioning and temporally grounding video files using HF Buckets and Jobs.

What the output looks like — a frame from *Joan Avoids a Cold* (1947, Prelinger Archives) with the event Marlin-2B produced for that moment:

![Example: film frame with its timestamped event caption](example-card.jpg)

## Quick Start

Scripts run directly from their Hub URL — no clone or local checkout needed:

```bash
# Caption every video in a bucket: dense scene captions + timestamped events
hf jobs uv run --image vllm/vllm-openai:latest --flavor a10g-small \
    -s HF_TOKEN \
    -v hf://buckets/user/my-videos:/input:ro \
    https://huggingface.co/datasets/uv-scripts/video/raw/main/marlin-caption.py \
    /input hf://buckets/user/my-videos/captions

# Temporal grounding: when does an event happen?
hf jobs uv run --image vllm/vllm-openai:latest --flavor a10g-small \
    -s HF_TOKEN \
    -v hf://buckets/user/my-videos:/input:ro \
    https://huggingface.co/datasets/uv-scripts/video/raw/main/marlin-caption.py \
    /input hf://buckets/user/out --find "a person enters the room"
```

## Scripts

### qwen38-caption.py

Runs [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (27B dense VLM,
Apache 2.0, not gated) over a directory of videos via vLLM — **whole videos, no
chunking**. The model ingests video natively at hour scale: an 11-minute film is
one request. The default video processor caps each video at 768 frames and a
fixed pixel budget, so videos longer than ~6.4 min are seen at a lower frame rate
and resolution (an hour at ~1 frame per 4.7 s). Output is a resumable parquet dataset, one row per video with
`caption` (and, with `--timestamps`, an `events` column of `<start - end>`
descriptions in global seconds).

`--timestamps` asks for a full event timeline instead of a prose caption. The
model reads times off per-frame time tags rather than estimating. On one
11-minute film (96 events), spot-checked event times matched extracted frames to
within ~2 s, and on-screen text was quoted verbatim; that is one measured
example, not a guarantee.

Thinking mode is on for `--timestamps` (without it, the model sometimes answered
a timeline request with a one-line summary) and off for captions.
`--thinking` / `--no-thinking` override this. A truncated, empty or
event-less answer is stored as an error row; `--retry-errors` tries it again.
Row ids include the prompt and mode, so changing them re-processes videos
instead of reusing old rows.

**Hardware**: bf16 weights are ~52 GB. The script's `[tool.hf-jobs]` header
sets `a100-large` and a pinned vLLM image, so `hf jobs uv run` needs no flavor
or image flags (hf CLI 1.32+). `--flavor h200` generated ~2.3x faster in our
test.

**vs marlin-caption.py**: Marlin is ~13x smaller and cheaper per minute of
footage, but must chunk at 60s (with per-chunk timestamp offsetting) and returns
grounding candidates rather than reliable timelines. Qwen3.8 handles the whole
film in one pass with verbatim on-screen-text reading. Rough guide: bulk
captioning of large collections → Marlin; rich timelines, text-heavy or
long-form material → Qwen3.8.

### marlin-caption.py

Runs [NemoStation/Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B) (2B video
VLM, gated — accept the license on the model page first) over a directory of videos
via vLLM. Output is a resumable parquet dataset: one row per ~60s chunk with `scene`,
`caption`, and an `events` column of `<start - end>` descriptions in seconds.
Re-running skips completed rows; failed rows are recorded, not dropped
(`--retry-errors` re-attempts them).

Videos longer than ~60s are split into chunks and event timestamps offset back to
global film time. This is required for correct timestamps, not an optimisation:
Marlin was trained on short clips and compresses any input onto a ~60s timeline.

`--find "event"` switches to grounding mode: each chunk returns a candidate
`(span_start, span_end)`. Spans are candidates, not detections — the model cannot
say "not present", so filter or verify downstream. When the event is real, spans
are precise to fractions of a second.

**Cost**: ~3s of GPU per minute of film on `a10g-small` at batch scale — about
$0.05 per hour of footage.

**Memory**: defaults encode a measured config (`--mm-processor-cache-gb 0`, in-flight
window capped at 24). vLLM's multimodal cache grows without bound on distinct videos
and will OOM a 15 GB node if re-enabled. On `a10g-large` and up, `--window-max 64`
is safe.

Run `--help` on the script for all options.
