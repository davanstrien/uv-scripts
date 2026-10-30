# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]==0.3.0",
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
Convert document images to markdown using LightOnOCR-2 via saturate.

Companion to `lighton-ocr2-server.py`: same model, same message shape, same
sampling, same in-job `vllm serve` — but the driver half (concurrency, retries,
output, resume) is the `saturate` library instead of hand-rolled code. What that
buys over the -server recipe:

- **Adaptive concurrency** — the window sizes itself from live engine signals
  (no `--concurrency` flag to tune).
- **Crash-safe, resumable output** — results stream to the output repo as
  parquet parts while the run is hot; re-running the same command skips
  everything already done (exact anti-join on id). A 10k-page job that dies at
  9k resumes at 9k.
- **Durable error rows** — a failed page is recorded as `{id, error}` instead of
  an `[OCR ERROR]` string in the text column; `--retry-errors` re-admits only
  those rows on a later run.

Run on HF Jobs. The script starts `vllm serve` itself; its [tool.hf-jobs] header
declares the image that provides the `vllm` binary, the hardware and the HF_TOKEN
secret (`hf` CLI 1.32+). Pass --timeout for a long run; flags override the header:

  hf jobs uv run --detach --timeout 4h \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr2-saturate.py \\
      <input-dataset> <output-dataset>

Output layout (differs from the -server recipe, which pushes input+markdown):
the output repo holds `data/part-*.parquet` with rows
`{id, markdown, model, prompt_tokens, completion_tokens, error}` keyed by the
input row id (`--id-column`, or `<split>-<index>` by default). Read it with
`datasets.load_dataset(<output>, data_dir="data")` or `saturate.read_output`;
join back to the input on id. Run metadata lands in `data/completions/`.

Model: lightonai/LightOnOCR-2-1B (1B, Apache-2.0)
- Message is the image ONLY (no text prompt) — LightOnOCR-2's trained format.
- Images resized client-side so the longest dimension is 1540px (training
  resolution at 200 DPI), same as the offline recipe.
- Sampling per the card: temperature 0.2, top_p 0.9, max_tokens 4096.

The SERVING dict below is the per-model tuning prior (serve flags + client
sampling + context math). Agents can `ast.literal_eval` it without running
the script; the script itself consumes it, so it cannot drift from reality.
"""

import argparse
import base64
import io
import sys

# Serving starting values for lightonai/LightOnOCR-2-1B. Per-value provenance:
# - serve_args: the model card's own `vllm serve` command, verbatim (the three
#   cache/mm flags; OCR never reuses images, so those caches only cost memory).
# - max_model_len 8192: NOT in the card's serve command (card default = native
#   16384). House choice inherited from lighton-ocr2-server.py: halves the KV
#   allocation on 24GB and still fits a 1540px page (~2.5k image tokens) + 4096
#   output with headroom.
# - max_tokens/temperature/top_p: card's sampling example, verbatim.
# - target_size 1540: card's stated training resolution (200 DPI longest side).
# Throughput receipt (a10g-small): 0.955 img/s at 1k pages incl. streaming.
SERVING = {
    "model": "lightonai/LightOnOCR-2-1B",
    "image": "vllm/vllm-openai:v0.22.1",
    "max_model_len": 8192,
    "serve_args": [
        "--limit-mm-per-prompt", '{"image": 1}',
        "--mm-processor-cache-gb", "0",
        "--no-enable-prefix-caching",
    ],
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.9,
    "target_size": 1540,
}
assert SERVING["max_tokens"] < SERVING["max_model_len"], (
    "context math: max_tokens must leave room for the image tokens "
    "(input + output <= max_model_len, or every request 400s)"
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


def encode_image(value, target_size: int) -> str:
    """RGB-convert, resize longest dimension to target_size, return base64 PNG."""
    from PIL import Image

    img = to_pil(value).convert("RGB")
    if target_size:
        w, h = img.size
        if max(w, h) != target_size:
            scale = target_size / max(w, h)
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser(description="LightOnOCR-2 batch OCR via saturate")
    ap.add_argument("input_dataset", help="Input dataset repo id (rows with an image column)")
    ap.add_argument("output_dataset", help="Output dataset repo id (created if missing)")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--config", default=None, help="Dataset config name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--id-column", default=None,
                    help="Column to use as row id (default: split-index ids)")
    ap.add_argument("--max-samples", "--limit", dest="limit", type=int, default=None,
                    help="Process at most N rows (same name as the other OCR recipes; --limit still works)")
    ap.add_argument("--max-tokens", type=int, default=SERVING["max_tokens"])
    ap.add_argument("--temperature", type=float, default=SERVING["temperature"])
    ap.add_argument("--target-size", type=int, default=SERVING["target_size"])
    ap.add_argument("--no-resize", action="store_true")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-admit rows whose only record is an error row")
    args = ap.parse_args()

    import pyarrow as pa
    from saturate import Auto, Engine, dataset_rows, pump

    target_size = 0 if args.no_resize else args.target_size
    rows = dataset_rows(
        args.input_dataset, config=args.config, split=args.split,
        columns=[args.image_column], ids=args.id_column or "index", limit=args.limit,
    )

    def to_request(row):
        b64 = encode_image(row[args.image_column], target_size)
        return {
            "model": SERVING["model"],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            "temperature": args.temperature,
            "top_p": SERVING["top_p"],
            "max_tokens": args.max_tokens,
        }

    # Declared, so a part holding only error rows has the same columns as the rest
    # (datasets.load_dataset reads every part with one schema).
    schema = pa.schema([
        ("id", pa.string()), ("error", pa.string()), ("markdown", pa.string()),
        ("model", pa.string()), ("prompt_tokens", pa.int64()), ("completion_tokens", pa.int64()),
    ])  # fmt: skip

    def parse(row, body):
        usage = body.get("usage") or {}
        return {
            "markdown": body["choices"][0]["message"]["content"].strip(),
            "model": SERVING["model"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    extra = ["--max-model-len", str(SERVING["max_model_len"]), *SERVING["serve_args"]]
    output = f"hf://datasets/{args.output_dataset}/data"
    with Engine(SERVING["model"], engine="vllm", extra_args=extra) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, output,
                     window=Auto(initial=8, max_limit=48),
                     retry_errors=args.retry_errors, schema=schema)

    print(f"https://huggingface.co/datasets/{args.output_dataset} "
          f"({stats.rows_processed} ok, {stats.rows_failed} error rows)", file=sys.stderr)
    print("LIGHTON_OCR2_SATURATE " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
