# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]>=0.1.1",
#     "pillow>=10",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.22.1"
# flavor = "a10g-small"
# timeout = "4h"
# secrets = ["HF_TOKEN"]
# ///
"""
Convert document images to markdown using OvisOCR2 via saturate.

Companion to `ovis-ocr2-server.py`: same model, prompt, message shape, sampling,
and post-processing, same in-job `vllm serve` — but the driver half (concurrency,
retries, output, resume) is the `saturate` library instead of hand-rolled code.
What that buys over the -server recipe:

- **Adaptive concurrency** — the window sizes itself from live engine signals
  (no `--concurrency` flag to tune).
- **Crash-safe, resumable output** — results stream to the output repo as
  parquet parts while the run is hot; re-running the same command skips
  everything already done (exact anti-join on id).
- **Durable error rows** — a failed page is recorded as `{id, error}` instead of
  an `[OCR ERROR]` string in the text column; `--retry-errors` re-admits only
  those rows on a later run.

Run on HF Jobs. The script starts `vllm serve` itself; its [tool.hf-jobs] header
declares the image that provides the `vllm` binary, the hardware and the HF_TOKEN
secret (`hf` CLI 1.32+). Pass --timeout for a long run; flags override the header:

  hf jobs uv run --detach --timeout 4h \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ovis-ocr2-saturate.py \\
      <input-dataset> <output-dataset>

Smoke test first with --max-samples 3. On your own GPU (vllm on PATH):

  uv run --with vllm==0.22.1 ovis-ocr2-saturate.py <input-dataset> <output-dataset>

Output layout (differs from the -server recipe, which pushes input+markdown):
the output repo holds `data/part-*.parquet` with rows
`{id, markdown, model, prompt_tokens, completion_tokens, error}` keyed by the
input row id (`--id-column`, or `<split>-<index>` by default). Read it with
`datasets.load_dataset(<output>, data_dir="data")` or `saturate.read_output`;
join back to the input on id. Run metadata lands in `data/completions/`.

Model: ATH-MaaS/OvisOCR2 (0.9B, Apache-2.0, 96.58 OmniDocBench)
- The card's exact OCR prompt (leading newline included — outputs are tuned to
  this wording), image before text, `enable_thinking=False` via
  chat_template_kwargs (the Qwen3.5 template otherwise injects a thinking
  preamble).
- Images downscaled client-side to the processor's max_pixels bound (8.3MP) and
  sent as JPEG q95 — the same clamp the server would apply, moved client-side to
  shrink the payload; min/max pixel bounds ride on the engine boot flag.
- Post-processing per the card: bbox `<img>` placeholder blocks dropped (keep
  with --keep-image-tags) and degenerate trailing repeats trimmed.

The SERVING dict below is the per-model tuning prior (serve flags + client
sampling + context math). Agents can `ast.literal_eval` it without running
the script; the script itself consumes it, so it cannot drift from reality.
"""

import argparse
import base64
import io
import math
import sys

# Serving starting values for ATH-MaaS/OvisOCR2. Per-value provenance:
# - The card documents OFFLINE inference only — no `vllm serve` command exists
#   upstream. The whole server arrangement here (incl. serve_args) is the
#   uv-scripts construction inherited from ovis-ocr2-server.py.
# - max_model_len 32768: house choice (card sets none; native ctx is 262144 —
#   NEVER boot without a cap on 24GB, the full-context KV profile kills boot).
# - cache/mm flags: house OCR defaults (OCR never reuses images, so prefix/
#   processor caches only cost memory).
# - mm-processor-kwargs pixel bounds: card's offline example, verbatim
#   (min 448*448=200704, max 2880*2880=8294400), moved to the engine flag.
# - max_tokens 16384 / temperature 0.0: card's sampling, verbatim.
# Throughput receipt (a10g-small, 20 pages): 4,057 tok/s, window ramped to 32.
SERVING = {
    "model": "ATH-MaaS/OvisOCR2",
    "image": "vllm/vllm-openai:v0.22.1",
    "max_model_len": 32768,
    "serve_args": [
        "--limit-mm-per-prompt", '{"image": 1}',
        "--mm-processor-cache-gb", "0",
        "--no-enable-prefix-caching",
        "--mm-processor-kwargs",
        '{"images_kwargs": {"min_pixels": 200704, "max_pixels": 8294400}}',
    ],
    "max_tokens": 16384,
    "temperature": 0.0,
    "max_pixels": 8294400,
}
assert SERVING["max_tokens"] < SERVING["max_model_len"], (
    "context math: max_tokens must leave room for the image tokens "
    "(input + output <= max_model_len, or every request 400s)"
)

OCR_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    'represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, '
    "where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. Preserve the original text "
    "without translation or paraphrasing."
)


def to_pil(value):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value))
    raise ValueError(f"unsupported image value: {type(value)}")


def encode_image(value, max_pixels: int) -> str:
    """RGB-convert, downscale to max_pixels if needed, return base64 JPEG q95."""
    from PIL import Image

    img = to_pil(value).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
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
    """Trim degenerate trailing repetition (verbatim port of the model card's cleanup)."""
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
            return text[: n - total_len + unit_len] + text[n - tail_len:]

    return text


def filter_image_tags(text: str) -> str:
    blocks = text.split("\n\n")
    return "\n\n".join(
        b for b in blocks if not b.strip().startswith('<img src="images/bbox_')
    )


def main():
    ap = argparse.ArgumentParser(description="OvisOCR2 batch OCR via saturate")
    ap.add_argument("input_dataset", help="Input dataset repo id (rows with an image column)")
    ap.add_argument("output_dataset", help="Output dataset repo id (created if missing)")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--config", default=None, help="Dataset config name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--id-column", default=None,
                    help="Column to use as row id (default: split-index ids)")
    ap.add_argument("--max-samples", "--limit", dest="limit", type=int, default=None,
                    help="Process only the first N rows (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=SERVING["max_tokens"])
    ap.add_argument("--keep-image-tags", action="store_true",
                    help="Keep the bbox <img> placeholder blocks in the output")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-admit rows whose only record is an error row")
    args = ap.parse_args()

    from saturate import Auto, Engine, dataset_rows, pump

    rows = dataset_rows(
        args.input_dataset, config=args.config, split=args.split,
        columns=[args.image_column], ids=args.id_column or "index", limit=args.limit,
    )

    def to_request(row):
        b64 = encode_image(row[args.image_column], SERVING["max_pixels"])
        return {
            "model": SERVING["model"],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": OCR_PROMPT},
            ]}],
            "temperature": SERVING["temperature"],
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def parse(row, body):
        text = body["choices"][0]["message"]["content"].strip()
        if not args.keep_image_tags:
            text = filter_image_tags(text)
        text = clean_truncated_repeats(text)
        usage = body.get("usage") or {}
        return {
            "markdown": text,
            "model": SERVING["model"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    extra = ["--max-model-len", str(SERVING["max_model_len"]), *SERVING["serve_args"]]
    output = f"hf://datasets/{args.output_dataset}/data"
    with Engine(SERVING["model"], engine="vllm", extra_args=extra) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, output,
                     window=Auto(initial=8, max_limit=48),
                     retry_errors=args.retry_errors)

    print(f"https://huggingface.co/datasets/{args.output_dataset} "
          f"({stats.rows_processed} ok, {stats.rows_failed} error rows)", file=sys.stderr)
    print("OVIS_OCR2_SATURATE " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
