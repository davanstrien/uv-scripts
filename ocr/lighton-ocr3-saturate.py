# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]==0.3.0",
#     "pillow>=10",
#     # LightOn's official client, pinned to a commit (not on PyPI). Used for its page
#     # resize, grounding parser and dollar-delimiter repair so outputs match upstream.
#     "lightonocr @ https://github.com/lightonai/LightOnOCR/archive/36755d461be079737860a5f03ae0c803501269e9.tar.gz",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.30.0"
# flavor = "a10g-small"
# timeout = "4h"
# secrets = ["HF_TOKEN"]
# ///
"""
Convert document images to markdown (or layout blocks) using LightOnOCR-3 via saturate.

Same model, prompt modes, sampling and page preparation as `lighton-ocr3.py`, but the
script starts `vllm serve` in the Job and the `saturate` library sends the pages to it:
many pages in flight, resumable parquet output and durable error rows. Use this recipe
for large runs.

Run on HF Jobs. The script starts the server itself; its [tool.hf-jobs] header declares
the image that provides `vllm`, the hardware and the HF_TOKEN secret (`hf` CLI 1.32+).
Flags on the command line override the header:

  hf jobs uv run --detach \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr3-saturate.py \\
      <input-dataset> <output-dataset>

Smoke test first with --max-samples 10. Layout blocks with labels and boxes:

  hf jobs uv run --detach \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr3-saturate.py \\
      <input-dataset> <output-dataset> --mode grounding

Output: the output repo holds `data/part-*.parquet` with rows
`{id, markdown, raw, finish_reason, layout_blocks, model, prompt_tokens, completion_tokens,
error}` keyed by the input row id (`--id-column`, or `<repo>:<config>:<split>:<000000000>`
by default). Read it with `saturate.read_output`, which keeps one row per id (a success
over an earlier error). `datasets.load_dataset(<output>, data_dir="data")` returns every
record, so after a --retry-errors run it also holds the old error rows: drop rows with an
`error` before you join back to the input on id. Re-running the same command skips
finished rows; --retry-errors re-admits failed ones.

`finish_reason` = "length" marks a page that hit --max-tokens (long tables, or the model
repeating a line). --repair-max-tokens N sends those pages once more, at the end of the
run, with a cap of N tokens and temperature 0.1, and writes the results to `repair/`
(`load_dataset(<output>, data_dir="repair")`); join on id and prefer the repair row where
its finish_reason is "stop". Only pages truncated in this run are repaired.

Models (Apache-2.0):
  lightonai/LightOnOCR-3-0.8B  (default)  85.5 olmOCR-Bench, 74.6 ParseBench
  lightonai/LightOnOCR-3-4B               86.3 olmOCR-Bench, 75.1 ParseBench
LightOnOCR-3-1B is not supported: it keeps the LightOnOCR-2 (Pixtral) architecture, and
LightOn reports that transformers 5.17 (the version in the vLLM 0.30.0 image) breaks it.

Upstream settings kept:
- Serve flags: the LightOnOCR README's flags for the 0.8B and 4B (`--max-model-len 18000`,
  thinking off, `--mm-processor-cache-gb 0`, `--no-enable-prefix-caching`,
  `--limit-mm-per-prompt '{"image": 1}'`). vLLM 0.30.0, the version LightOn tests.
- Prompt modes: plain sends the image alone; grounding adds the text "grounding". The card
  says other instructions are out of distribution.
- Sampling: the model card's vLLM example (temperature 0.2, top_p 0.9, max_tokens 4096).
- Pages: downscaled to 2048 px on the longest edge with LightOn's own resize (never
  upscaled), sent as PNG, as the official client does. The PNG uses a fast compression
  level; PNG is lossless, so the server decodes the same pixels.
- Escaped dollar delimiters are repaired (`\\$x\\$` -> `$x$`) as the client does; `raw`
  keeps the model's text.

Speed settings: pages are prepared in `--prepare-workers` threads (default: the CPU_CORES
that HF Jobs sets), and saturate's adaptive window sets how many pages are in flight, up to
a cap that fits the Job's memory (window_cap below). `--window N` pins it instead. The
server also gets `--max-num-batched-tokens 16384` (see SERVING).

Measured speed, LightOnOCR-3-0.8B (2026-11-05, vLLM 0.30.0). 600 pages: 350 book pages from
finebooks/bhl-impact-gt and 250 pages of typed industry documents, all fitted to 2048 px.
Steady state, excluding the server start (about 4 min, also paid by lighton-ocr3.py):

  flavor          pages/s   $ per 1,000 pages   lighton-ocr3.py (offline batches)
  a10g-small        2.2         0.12                0.90 pages/s, $0.31
  l4x1              1.6         0.14
  a100-large        6.2         0.11                1.27 pages/s, $0.55
  rtx-pro-6000      9.2         0.08

The 4B on a100-large: 2.7 pages/s, $0.26 per 1,000 pages; on these book pages its
character error rate was higher than the 0.8B's (0.10 mean, 0.023 median, one run), so
check a sample before paying for it. The header's a10g-small is a
good default for small and medium runs; for large runs add --flavor rtx-pro-6000 (or
a100-large). On a100-large, --serve-args "--api-server-count 4" gave 7.3 pages/s instead
of 6.2 (it needs spare CPU cores, so it is not the default). In every run the outputs
differed from lighton-ocr3.py's no more than two runs of this recipe differ from each
other (sampling at temperature 0.2), and the character error rate on the book pages stayed
at 0.08 mean, 0.02 median.

Truncation and repair: 9 to 12 of the 600 pages stopped at max_tokens 4096 in each run,
most of them typed tables. With --repair-max-tokens 12000, 8 of 12 such pages finished:
3 were loops that ended on the second try and 5 were long pages (4,900 to 6,100 tokens).
The other 4 looped again up to the cap. The pass added 22% to the run's output tokens.
With --repair-max-tokens 8192 (suggested), 5 of 9 such pages finished, the same 4 looped
again, and the pass added 16%. Pages that loop on both tries keep their first output in
data/.

The SERVING dict below is the per-model tuning prior. Agents can `ast.literal_eval` it
without running the script; the script itself consumes it.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import shlex
import sys
import tempfile

# Serving starting values for LightOnOCR-3 (0.8B and 4B). Per-value provenance:
# - image: vLLM 0.30.0, the version LightOn's client pins and tests (0.27.1 makes the
#   models loop, per the LightOnOCR README).
# - context 18000 and serve_args: the LightOnOCR README "Models" table for the 0.8B and 4B,
#   plus its `--limit-mm-per-prompt '{"image": 1}'`.
# - max_tokens/temperature/top_p: the model card's vLLM example, verbatim.
# - longest_edge 2048: the card's rendering tip and the client's page size for these models.
# - patch 32: 16 px patches with a 2x2 merge, so a 2048x2048 page is at most 64x64 = 4,096
#   image tokens (used for the context check).
# - --max-num-batched-tokens 16384: speed only. vLLM 0.30 serves with 2048 on A10G, L4 and
#   A100 (8192 on other GPUs of 70 GB or more), less than one page's image tokens. Measured on a100-large:
#   6.20 and 6.30 pages/s against 5.83 without it, outputs within the run-to-run noise.
# - pages_per_gib: see window_cap().
SERVING = {
    "models": ["lightonai/LightOnOCR-3-0.8B", "lightonai/LightOnOCR-3-4B"],
    "image": "vllm/vllm-openai:v0.30.0",
    "context": 18000,
    "serve_args": [
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--limit-mm-per-prompt", '{"image": 1}',
        "--mm-processor-cache-gb", "0",
        "--no-enable-prefix-caching",
        "--max-num-batched-tokens", "16384",
    ],
    "pages_per_gib": 6,
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.9,
    "repair_temperature": 0.1,
    "longest_edge": 2048,
    "patch": 32,
}


def window_cap() -> int:
    """Most pages in flight that this machine's memory allows.

    Every page in flight holds about 0.1 GiB of image data on the host (the client's copy and
    the server's preprocessed image), on top of 7 to 9 GiB for the engine itself. On
    a10g-small (14.9 GiB limit), 64 pages in flight reached the limit and the Job stopped
    making progress; 32 peaked at 12.0 GiB, and up to 39 peaked at 13.0 and 14.8 GiB on two
    runs. So the window may grow by SERVING "pages_per_gib" pages per GiB above 10 GiB: 29 on
    a10g-small (which is GPU-bound from about 20 pages in flight), about 100 on l4x1, the full
    512 on a100-large. The limit is read from the container (cgroup v2), then from the MEMORY
    variable that HF Jobs sets.
    """
    limit_gib = None
    try:
        with open("/sys/fs/cgroup/memory.max") as handle:
            value = handle.read().strip()
        if value != "max":
            limit_gib = int(value) / 2**30
    except (OSError, ValueError):
        pass
    if limit_gib is None and os.environ.get("MEMORY"):
        try:  # e.g. "16.0G" (decimal gigabytes)
            limit_gib = float(os.environ["MEMORY"].upper().rstrip("GIB ")) * 1e9 / 2**30
        except ValueError:
            pass
    if limit_gib is None:
        return 512
    return max(8, min(512, int((limit_gib - 10) * SERVING["pages_per_gib"])))


def max_image_tokens(longest_edge: int) -> int:
    """Upper bound on the image tokens of a page fitted to longest_edge (square page)."""
    side = -(-longest_edge // SERVING["patch"])
    return side * side


def to_pil(value):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, dict) and value.get("path"):
        return Image.open(value["path"])
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value))
    raise ValueError(f"unsupported image value: {type(value)}")


def page_png(value, longest_edge: int) -> bytes:
    """RGB page fitted to longest_edge with LightOn's resize, as PNG bytes.

    compress_level=1 instead of Pillow's default 6: PNG is lossless at every level, so the
    pixels are identical, and level 1 encodes several times faster (CPU is the limit on
    small flavors).
    """
    from lightonocr import fit

    image = fit(to_pil(value).convert("RGB"), longest_edge)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def chat_request(model: str, png: bytes, mode: str, max_tokens: int, temperature: float, top_p: float) -> dict:
    """One page as the client sends it: the image, plus the word "grounding" in grounding mode."""
    from lightonocr import GROUNDING_PROMPT

    url = "data:image/png;base64," + base64.b64encode(png).decode()
    content = [{"type": "image_url", "image_url": {"url": url}}]
    if mode == "grounding":
        content.append({"type": "text", "text": GROUNDING_PROMPT})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def postprocess(raw: str, mode: str, fix_dollars: bool):
    """Return (markdown, layout_blocks JSON or None) for one model output, as lighton-ocr3.py."""
    from lightonocr import fix_escaped_dollars, parse_blocks, to_markdown

    text = fix_escaped_dollars(raw) if fix_dollars else raw
    if mode != "grounding":
        return text, None
    blocks = parse_blocks(text)
    if not blocks:
        return text, json.dumps([])  # no markers parsed: keep the text so nothing is lost
    return to_markdown(blocks), json.dumps([b.to_dict() for b in blocks], ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description="LightOnOCR-3 batch OCR via saturate")
    ap.add_argument("input_dataset", help="Input dataset repo id (rows with an image column)")
    ap.add_argument("output_dataset", help="Output dataset repo id (created private if missing)")
    ap.add_argument("--model", default=SERVING["models"][0], choices=SERVING["models"])
    ap.add_argument("--mode", default="plain", choices=["plain", "grounding"],
                    help="plain = markdown; grounding = labelled layout blocks with 0-1000 boxes")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--config", default=None, help="Dataset config name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--id-column", default=None,
                    help="Column to use as row id (default: split-index ids)")
    ap.add_argument("--max-samples", "--limit", dest="limit", type=int, default=None,
                    help="Process only the first N rows (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=SERVING["max_tokens"],
                    help="Output cap per page (default 4096, the model card's)")
    ap.add_argument("--temperature", type=float, default=SERVING["temperature"])
    ap.add_argument("--top-p", type=float, default=SERVING["top_p"],
                    help="Default 0.9, the model card's; LightOn's client uses 1.0")
    ap.add_argument("--longest-edge", type=int, default=SERVING["longest_edge"],
                    help="Downscale pages to at most this many pixels on the longest edge")
    ap.add_argument("--no-fix-dollars", action="store_true",
                    help="Keep the model's text in `markdown` without repairing escaped dollars")
    ap.add_argument("--repair-max-tokens", type=int, default=0,
                    help="Re-send pages that hit --max-tokens once, with this cap and "
                         "temperature 0.1, into repair/ (default 0: off; 8192 suggested)")
    ap.add_argument("--window", default="auto",
                    help="Pages in flight: 'auto' (saturate adapts it) or a fixed number")
    ap.add_argument("--prepare-workers", type=int, default=None,
                    help="Threads that decode, resize and encode pages (default: CPU_CORES)")
    ap.add_argument("--serve-args", default="",
                    help="Extra `vllm serve` flags, one quoted string, e.g. "
                         "\"--max-num-batched-tokens 16384\"")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-admit rows whose only record is an error row")
    args = ap.parse_args()

    room = SERVING["context"] - max_image_tokens(args.longest_edge) - 64
    for flag, value in (("--max-tokens", args.max_tokens), ("--repair-max-tokens", args.repair_max_tokens)):
        if value > room:
            ap.error(f"{flag} {value} can overflow the {SERVING['context']}-token context with a "
                     f"{args.longest_edge} px page; use at most {room}")
    if args.max_tokens < 1:
        ap.error("--max-tokens must be at least 1")
    if args.repair_max_tokens and args.repair_max_tokens <= args.max_tokens:
        ap.error("--repair-max-tokens must be larger than --max-tokens")
    if args.window != "auto" and not args.window.isdigit():
        ap.error("--window must be 'auto' or a number")

    import pyarrow as pa
    from saturate import Auto, Engine, Fixed, dataset_rows, pump

    if args.id_column is None:
        ids = "index"
    elif args.id_column in ("index", "content"):
        # saturate reads these two names as id strategies, so a column with one of them
        # needs a callable. It returns the raw value, so saturate still refuses a null id.
        ids = lambda example: example[args.id_column]
    else:
        ids = args.id_column

    rows = dataset_rows(
        args.input_dataset, config=args.config, split=args.split,
        columns=[args.image_column], ids=ids, limit=args.limit,
    )

    # Declared, so a part holding only error rows has the same columns as the rest
    # (datasets.load_dataset reads every part with one schema).
    schema = pa.schema([
        ("id", pa.string()), ("error", pa.string()), ("markdown", pa.string()), ("raw", pa.string()),
        ("finish_reason", pa.string()), ("layout_blocks", pa.string()), ("model", pa.string()),
        ("prompt_tokens", pa.int64()), ("completion_tokens", pa.int64()),
    ])  # fmt: skip

    # Pages that hit --max-tokens, kept on local disk for the repair pass.
    repair_dir = tempfile.mkdtemp(prefix="lighton-ocr3-repair-") if args.repair_max_tokens else None
    truncated = []

    def to_request(row):
        png = page_png(row[args.image_column], args.longest_edge)
        return chat_request(args.model, png, args.mode, args.max_tokens, args.temperature, args.top_p)

    def make_parse(stash: bool):
        def parse(row, body):
            choice = body["choices"][0]
            raw = (choice["message"].get("content") or "").strip()
            markdown, blocks = postprocess(raw, args.mode, fix_dollars=not args.no_fix_dollars)
            usage = body.get("usage") or {}
            if stash and choice.get("finish_reason") == "length":
                path = os.path.join(repair_dir, hashlib.sha1(row["__id"].encode()).hexdigest() + ".png")
                with open(path, "wb") as handle:
                    handle.write(page_png(row[args.image_column], args.longest_edge))
                truncated.append((row["__id"], path))
            return {
                "markdown": markdown,
                "raw": raw,
                "finish_reason": choice.get("finish_reason"),
                "layout_blocks": blocks,
                "model": args.model,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        return parse

    def with_ids(source):
        # Keep each row's id on the row, so parse can name the page it stashes for repair.
        for row_id, row in source:
            yield row_id, {**row, "__id": row_id}

    cap = window_cap()
    if args.window == "auto":
        window = Auto(initial=min(64, cap), max_limit=cap)
        print(f"window: auto, at most {cap} pages in flight (host memory)", file=sys.stderr)
    else:
        window = Fixed(int(args.window))
        if int(args.window) > cap:
            print(f"warning: --window {args.window} is above the {cap} pages this machine's "
                  "memory allows; the Job may run out of memory", file=sys.stderr)
    workers = args.prepare_workers
    if workers is None:
        workers = max(1, int(float(os.environ.get("CPU_CORES") or "4")))
    extra = [
        "--max-model-len", str(SERVING["context"]),
        *SERVING["serve_args"],
        *shlex.split(args.serve_args),
    ]

    output = f"hf://datasets/{args.output_dataset}/data"
    with Engine(args.model, engine="vllm", extra_args=extra) as endpoint:
        stats = pump(with_ids(rows) if repair_dir else rows, to_request, make_parse(bool(repair_dir)),
                     endpoint, output, window=window, prepare_workers=workers,
                     retry_errors=args.retry_errors, schema=schema)
        print("LIGHTON_OCR3_SATURATE " + stats.to_json(), flush=True)

        if repair_dir and truncated:
            print(f"repair: re-sending {len(truncated)} pages that hit --max-tokens "
                  f"with a cap of {args.repair_max_tokens}", file=sys.stderr, flush=True)

            def repair_rows():
                for row_id, path in truncated:
                    with open(path, "rb") as handle:
                        yield row_id, {"png": handle.read()}

            def repair_request(row):
                return chat_request(args.model, row["png"], args.mode, args.repair_max_tokens,
                                    SERVING["repair_temperature"], args.top_p)

            repair_stats = pump(repair_rows(), repair_request, make_parse(False), endpoint,
                                f"hf://datasets/{args.output_dataset}/repair", window=window,
                                schema=schema)
            print("LIGHTON_OCR3_SATURATE_REPAIR " + repair_stats.to_json(), flush=True)

    print(f"https://huggingface.co/datasets/{args.output_dataset} "
          f"({stats.rows_processed} ok, {stats.rows_failed} error rows, "
          f"{len(truncated) if repair_dir else 'n/a'} truncated pages repaired)", file=sys.stderr)


if __name__ == "__main__":
    main()
