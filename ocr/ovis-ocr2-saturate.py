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
import os
import shlex
import sys

SCRIPT_URL = "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ovis-ocr2-saturate.py"

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


def jobs_flavor() -> str:
    """Return the HF Jobs hardware flavor this run is on, or "" when it cannot be told.

    HF Jobs sets HF_JOB_FLAVOR to the flavor name. Without it (ACCELERATOR holds only
    "cpu" or "gpu", not the flavor name), the flavor is looked up
    in the public Jobs hardware list: first by GPU model and count, then by the CPU_CORES and
    MEMORY values that Jobs sets in the container (to tell a10g-small from a10g-large, or
    cpu-basic from cpu-upgrade). Any failure (no network, older huggingface_hub) returns "".
    """
    if os.environ.get("HF_JOB_FLAVOR"):
        return os.environ["HF_JOB_FLAVOR"]
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


def reproduce_options(args):
    """Return the options that give this dataset again, as a shell-quoted string."""
    options = []
    if args.image_column != "image":
        options += ["--image-column", args.image_column]
    if args.config:
        options += ["--config", args.config]
    if args.split != "train":
        options += ["--split", args.split]
    if args.id_column:
        options += ["--id-column", args.id_column]
    if args.limit:
        options += ["--max-samples", str(args.limit)]
    if args.max_tokens != SERVING["max_tokens"]:
        options += ["--max-tokens", str(args.max_tokens)]
    if args.keep_image_tags:
        options += ["--keep-image-tags"]
    return "".join(" " + shlex.quote(option) for option in options)


def write_card(args, stats):
    """Write the dataset card, with the canonical uv-scripts reproduction section."""
    from huggingface_hub import HfApi

    on_jobs = os.environ.get("HF_JOB_ID") is not None
    flavor = jobs_flavor()
    origin = (
        "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
        + (f" (`{flavor}`)" if flavor else "")
    ) if on_jobs else "Generated"
    jobs_tag = "\n- hf_jobs" if on_jobs else ""
    model = SERVING["model"]
    # A resumed run processes only the rows that were missing, so the card gives both counts.
    earlier = f", {stats.rows_done_prior} already written by an earlier run" if stats.rows_done_prior else ""
    card = (
        f"---\ntags:\n- ocr\n- document-processing\n- ovis-ocr2\n- markdown\n- uv-script\n- generated{jobs_tag}\n---\n\n"
        f"# Document OCR using {model.split('/')[-1]}\n\n"
        f"OCR results for the `{args.image_column}` column of "
        f"[`{args.input_dataset}`](https://huggingface.co/datasets/{args.input_dataset}) (split `{args.split}`), "
        f"made with [`{model}`](https://huggingface.co/{model}) on a vLLM server driven by `saturate`.\n\n"
        f"- Rows in the last run: {stats.rows_processed} done, {stats.rows_failed} failed{earlier}. "
        f"A failed row has a message in `error`.\n"
        f"- Visual-region image tags: {'kept' if args.keep_image_tags else 'filtered (default)'}\n"
        f"- Files: `data/part-*.parquet`, one record per input row: `id`, `markdown`, `model`, "
        f"`prompt_tokens`, `completion_tokens`, `error`. Join back to the input on `id`.\n\n"
        f"## Reproduction\n\n"
        f"{origin} with the [`ovis-ocr2-saturate.py`]({SCRIPT_URL}) recipe "
        f"from [uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:\n\n"
        f"```bash\nhf jobs uv run --flavor {flavor or 'a10g-small'} {SCRIPT_URL} \\\n"
        f"    {args.input_dataset} <output-dataset>{reproduce_options(args)}\n```\n"
    )
    HfApi().upload_file(
        path_or_fileobj=card.encode(), path_in_repo="README.md",
        repo_id=args.output_dataset, repo_type="dataset",
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

    import pyarrow as pa
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

    # Declared, so a part holding only error rows has the same columns as the rest
    # (datasets.load_dataset reads every part with one schema).
    schema = pa.schema([
        ("id", pa.string()), ("error", pa.string()), ("markdown", pa.string()),
        ("model", pa.string()), ("prompt_tokens", pa.int64()), ("completion_tokens", pa.int64()),
    ])  # fmt: skip

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
                     retry_errors=args.retry_errors, schema=schema)

    write_card(args, stats)
    print(f"https://huggingface.co/datasets/{args.output_dataset} "
          f"({stats.rows_processed} ok, {stats.rows_failed} error rows)", file=sys.stderr)
    print("OVIS_OCR2_SATURATE " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
