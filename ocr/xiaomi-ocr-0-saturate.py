# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]==0.3.0",
#     "pillow>=10",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.29.0"
# flavor = "a10g-small"
# timeout = "4h"
# secrets = ["HF_TOKEN"]
# ///
"""
Convert document images to markdown using Xiaomi-OCR-0 via saturate.

Whole-page parsing on an in-job vLLM or SGLang server, driven by the `saturate`
library: adaptive concurrency, resumable parquet output and durable error rows.

Run on HF Jobs. The script starts the server itself; its [tool.hf-jobs] header
declares the image that provides `vllm`, the hardware and the HF_TOKEN secret
(`hf` CLI 1.32+). Flags on the command line override the header:

  hf jobs uv run --detach \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/xiaomi-ocr-0-saturate.py \\
      <input-dataset> <output-dataset>

SGLang instead of vLLM, on the matching image:

  hf jobs uv run --detach --image lmsysorg/sglang:v0.5.20-cu130 \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/xiaomi-ocr-0-saturate.py \\
      <input-dataset> <output-dataset> --engine sglang

Smoke test first with --max-samples 3. For speed at the same cost per page, add
--flavor rtx-pro-6000.

Output: the output repo holds `data/part-*.parquet` with rows
`{id, markdown, raw, finish_reason, model, prompt_tokens, completion_tokens, error}`
keyed by the input row id (`--id-column`, or `<repo>:<config>:<split>:<000000000>` by
default). Read it with `saturate.read_output`, which keeps one row per id (a success over
an earlier error). `datasets.load_dataset(<output>, data_dir="data")` returns every
record, so after a --retry-errors run it also holds the old error rows: drop rows with an
`error` before you join back to the input on id. Re-running the same command skips
finished rows; --retry-errors re-admits failed ones. A corrupt image stops the run when
it is read (images are decoded as the dataset streams); remove that row and run again.
`finish_reason` = "length" marks a page that hit --max-tokens. Empty `markdown` with no `error` means the model returned no text,
which is the correct result for a blank page; greedy decoding would repeat it on retry.

Model: xiaomi-research/Xiaomi-OCR-0 (0.8B, Apache-2.0)
- Card scores (OmniDocBench v1.6 96.83, Real5 95.24, Wild 87.94) come from Xiaomi's
  TWO-STAGE pipeline (PP-DocLayoutV3 layout, then per-region prompts). This recipe
  runs the model's single-stage whole-page mode, so do not quote those scores for it.
- The card's document prompt, image before text, `enable_thinking=False`, greedy
  decoding, and images fitted to 2048 px on the long edge and sent as PNG
  (skills/xiaomi-ocr/mcp_ocr_server.py `_chat`, evaluation_pipeline/cropping.py).
- Post-processing as upstream's `finalize_document`: repeated-tail cleanup, math
  delimiters to `$`/`$$`, OTSL tables to HTML. The model's text is kept in `raw`.
- Deliberate deviation: the card and MCP server use max_tokens 4096, which truncates
  dense pages. Upstream's whole-page evaluation (evaluation_pipeline/e2e_img2md.py) uses
  16384 on the model's full context. This recipe serves the 16384 context from the repo
  README, to keep KV memory small, so the default is 12000: the most that context leaves
  after a worst-case 2048x2048 page (about 4200 image tokens). Lower it with --max-tokens.
- Deliberate deviation: upstream's evaluation re-requests pages that its repetition
  guard flags (--retry-repetitive, presence penalty 1.5). This recipe does not. It only
  trims a repeated tail, so a page where the model loops keeps the looped text.
- Upstream behaviour kept: a paragraph (text between blank lines) that holds an OTSL
  table tag is replaced by the converted table, so text in the same paragraph is lost.
  `raw` keeps the model's full output.

The SERVING dict below is the per-model tuning prior. Agents can `ast.literal_eval`
it without running the script; the script itself consumes it.
"""

import argparse
import base64
import io
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

# Serving starting values for xiaomi-research/Xiaomi-OCR-0. Per-value provenance:
# - context 16384: the repo README's `vllm serve --max-model-len` and SGLang
#   `--context-length` value, verbatim.
# - vllm_args cache/mm flags: house OCR defaults (see SERVING.md).
# - max_tokens 12000: house choice, see the docstring. Upstream sends 4096 (card, MCP)
#   and 16384 in its whole-page evaluation, on the full model context.
# - temperature, top_p, presence_penalty, repetition_penalty: mcp_ocr_server.py `_chat`.
# - max_edge 2048: evaluation_pipeline/cropping.py MAX_IMAGE_EDGE.
# - window: saturate's default Auto(). Receipts below; a fixed cap only slowed large GPUs.
# - prepare_workers: threads for PNG encoding, sized from CPU_CORES (set by HF Jobs).
# Throughput receipts on UFO scans, vLLM, excluding boot (2026-10-30):
#   a10g-small 3.4 pages/s, rtx-pro-6000 9.7 pages/s, both about $0.08 per 1,000 pages.
# - images: vLLM pin matches ovis-ocr2.py (same Qwen3_5 arch); SGLang pin matches
#   unlimited-ocr2.py. Move them forward once a newer image passes a smoke test.
SERVING = {
    "model": "xiaomi-research/Xiaomi-OCR-0",
    "images": {"vllm": "vllm/vllm-openai:v0.29.0", "sglang": "lmsysorg/sglang:v0.5.20-cu130"},
    "context": 16384,
    "vllm_args": [
        "--limit-mm-per-prompt", '{"image": 1}',
        "--mm-processor-cache-gb", "0",
        "--no-enable-prefix-caching",
    ],
    "max_tokens": 12000,
    "temperature": 0.0,
    "top_p": 0.8,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_edge": 2048,
}

# Card "Task prompts" table, Document row, verbatim.
OCR_PROMPT = (
    "Extract all information from the main body of the document image and represent it "
    "in markdown format, ignoring headers and footers. Tables should be expressed in OTSL "
    "format, formulas in the document should be represented using LATEX format, and the "
    "parsing should be organized according to the reading order."
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


def encode_image(value, max_edge: int) -> str:
    """RGB-convert, fit within max_edge keeping aspect ratio, return base64 PNG."""
    from PIL import Image

    img = to_pil(value).convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# Vendored from github.com/xiaomi-research/Xiaomi-OCR-0 at b1a9bb4 (Apache-2.0,
# Copyright 2026 Xiaomi Corporation): postprocess/otsl.py and postprocess/document.py.


ROW_BREAK = "<nl>"
CELL = "<fcel>"
EMPTY = "<ecel>"
LEFT = "<lcel>"
UP = "<ucel>"
BOTH = "<xcel>"

ALL_TAGS = (CELL, EMPTY, LEFT, UP, BOTH, ROW_BREAK)
TAG_RE = re.compile("(" + "|".join(re.escape(tag) for tag in ALL_TAGS) + ")")

ORIGIN_TAGS = frozenset({CELL, EMPTY})
RIGHT_CONTINUATION = frozenset({LEFT, BOTH})
DOWN_CONTINUATION = frozenset({UP, BOTH})


@dataclass
class Cell:
    """A resolved cell placed on the rectangular grid."""

    text: str
    row: int
    column: int
    rowspan: int = 1
    colspan: int = 1


def _tag_of(chunk: str) -> str:
    match = TAG_RE.match(chunk)
    return match.group(0) if match else ""


def _split_chunks(line: str) -> list[str]:
    """Split a row into chunks, each holding one tag plus the text after it."""
    starts = [match.start() for match in TAG_RE.finditer(line)]
    ends = starts[1:] + [len(line)]
    return [line[start:end] for start, end in zip(starts, ends)]


def _required_width(chunks: list[str]) -> int:
    """Number of leading cells a row must keep to preserve its last content."""
    width = 0
    for position, chunk in enumerate(chunks, start=1):
        if _tag_of(chunk) == CELL:
            width = position
    return width


def _target_width(rows: list[list[str]]) -> int:
    """Pick one cell count every row is padded or truncated to.

    Every row must keep at least the cells it declared content for, and the
    cheapest width balances the total amount of truncation and padding.
    """
    lower = max(_required_width(row) for row in rows)
    upper = max(len(row) for row in rows)
    best_width = upper
    best_cost = None
    for width in range(lower, upper + 1):
        cost = sum(abs(len(row) - width) for row in rows)
        if best_cost is None or cost < best_cost:
            best_width = width
            best_cost = cost
    return best_width


def pad_rows(otsl: str) -> str:
    """Give every row the same cell count so the grid can be indexed."""
    text = otsl.strip()
    if ROW_BREAK not in text:
        return text + ROW_BREAK

    rows = []
    for line in text.split(ROW_BREAK):
        chunks = _split_chunks(line)
        if chunks:
            rows.append(chunks)
    if not rows:
        return ROW_BREAK

    width = _target_width(rows)
    padded = []
    for chunks in rows:
        fixed = chunks[:width]
        fixed += [EMPTY] * (width - len(fixed))
        padded.append("".join(fixed))
    return ROW_BREAK.join(padded) + ROW_BREAK


def parse_grid(otsl: str) -> tuple[list[list[str]], list[list[str]]]:
    """Return the padded table as parallel tag and text grids."""
    tag_rows: list[list[str]] = []
    text_rows: list[list[str]] = []

    for line in otsl.split(ROW_BREAK):
        chunks = _split_chunks(line)
        if not chunks:
            continue
        tags = []
        texts = []
        for chunk in chunks:
            tag = _tag_of(chunk)
            tags.append(tag)
            texts.append(chunk[len(tag):] if tag == CELL else "")
        tag_rows.append(tags)
        text_rows.append(texts)

    width = max((len(row) for row in tag_rows), default=0)
    for tags, texts in zip(tag_rows, text_rows):
        tags += [EMPTY] * (width - len(tags))
        texts += [""] * (width - len(texts))
    return tag_rows, text_rows


def resolve_cells(
    tag_rows: list[list[str]], text_rows: list[list[str]]
) -> list[Cell]:
    """Turn origin tags into cells, measuring spans from the continuation tags."""
    height = len(tag_rows)
    cells: list[Cell] = []

    for row_index, tags in enumerate(tag_rows):
        for column, tag in enumerate(tags):
            if tag not in ORIGIN_TAGS:
                continue

            colspan = 1
            while (
                column + colspan < len(tags)
                and tags[column + colspan] in RIGHT_CONTINUATION
            ):
                colspan += 1

            rowspan = 1
            while (
                row_index + rowspan < height
                and column < len(tag_rows[row_index + rowspan])
                and tag_rows[row_index + rowspan][column] in DOWN_CONTINUATION
            ):
                rowspan += 1

            text = text_rows[row_index][column] if tag == CELL else ""
            cells.append(
                Cell(
                    text=text.strip(),
                    row=row_index,
                    column=column,
                    rowspan=rowspan,
                    colspan=colspan,
                )
            )
    return cells


def render_html(cells: list[Cell], num_rows: int, num_cols: int) -> str:
    """Render resolved cells as an HTML table element.

    Layout is walked column by column, so every row emits exactly one cell per
    grid column and the result stays a well-formed table even when the input
    OTSL contradicts itself. Spans are measured against the ownership grid, so
    a span never claims a position owned by another cell.
    """
    if not cells:
        return ""

    grid: list[list[Cell | None]] = [
        [None] * num_cols for _ in range(num_rows)
    ]
    for cell in cells:
        for row in range(cell.row, min(cell.row + cell.rowspan, num_rows)):
            for column in range(cell.column, min(cell.column + cell.colspan, num_cols)):
                if row >= 0 and column >= 0:
                    grid[row][column] = cell

    covered_until: dict[int, int] = {}
    parts = []
    for row in range(num_rows):
        parts.append("<tr>")
        column = 0
        while column < num_cols:
            if covered_until.get(column, 0) > row:
                column += 1
                continue

            cell = grid[row][column]
            if cell is None or cell.row != row or cell.column != column:
                # A continuation tag with no origin of its own: keep the column
                # so the row still spans the full grid width.
                parts.append("<td></td>")
                column += 1
                continue

            colspan = 1
            while (
                column + colspan < num_cols
                and grid[row][column + colspan] is cell
            ):
                colspan += 1

            rowspan = 1
            while row + rowspan < num_rows and all(
                grid[row + rowspan][offset] is cell
                for offset in range(column, column + colspan)
            ):
                rowspan += 1

            attributes = ""
            if rowspan > 1:
                attributes += f' rowspan="{rowspan}"'
            if colspan > 1:
                attributes += f' colspan="{colspan}"'
            parts.append(f"<td{attributes}>{cell.text}</td>")

            for offset in range(column, column + colspan):
                covered_until[offset] = max(
                    covered_until.get(offset, 0), row + rowspan
                )
            column += colspan
        parts.append("</tr>")
    return "<table>" + "".join(parts) + "</table>"


def convert_otsl_to_html(otsl_content: str) -> str:
    """Convert an OTSL table into an HTML ``<table>`` element."""
    if not isinstance(otsl_content, str):
        raise TypeError("otsl_content must be a string")

    tag_rows, text_rows = parse_grid(pad_rows(otsl_content))
    cells = resolve_cells(tag_rows, text_rows)
    num_rows = len(tag_rows)
    num_cols = max((len(row) for row in tag_rows), default=0)
    return render_html(cells, num_rows, num_cols)


OTSL_TAG_RE = re.compile(r"<(?:fcel|ecel|nl|lcel|ucel|xcel)>")


def postprocess_math_delimiters(markdown: str) -> str:
    """Convert LaTeX delimiters in whole-page markdown to markdown math delimiters."""
    if not markdown:
        return markdown
    markdown = markdown.replace("\\[\\[", "\\[").replace("\\]\\]", "\\]")
    markdown = re.sub(
        r"\$\$?\s*\\\[\s*(.*?)\s*\\\]\s*\$\$?",
        r" $$ \1 $$ ",
        markdown,
        flags=re.DOTALL,
    )
    markdown = re.sub(
        r"\$\s*\\\(\s*(.*?)\s*\\\)\s*\$",
        r" $ \1 $",
        markdown,
        flags=re.DOTALL,
    )
    markdown = re.sub(
        r"\\\[\s*(.*?)\s*\\\]",
        r" $$ \1 $$ ",
        markdown,
        flags=re.DOTALL,
    )
    markdown = re.sub(
        r"\\\(\s*(.*?)\s*\\\)",
        r" $ \1 $",
        markdown,
        flags=re.DOTALL,
    )
    return markdown


def _split_markdown_blocks(markdown: str):
    parts = re.split(r"(\n\s*\n)", markdown)
    blocks = []
    for i in range(0, len(parts), 2):
        text = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        blocks.append([text, sep])
    return blocks


def postprocess_otsl_tables(markdown: str) -> str:
    """Convert OTSL-looking markdown blocks to HTML tables; keep original on failure."""
    if not markdown or not OTSL_TAG_RE.search(markdown):
        return markdown

    blocks = _split_markdown_blocks(markdown)
    in_fence = False
    converted = []
    for text, sep in blocks:
        stripped = text.strip()
        fence_count = stripped.count("```")
        should_skip = in_fence
        if fence_count % 2 == 1:
            in_fence = not in_fence
            should_skip = True

        if not should_skip and OTSL_TAG_RE.search(stripped):
            try:
                html = convert_otsl_to_html(stripped)
                if html:
                    text = html
            except Exception:
                pass
        converted.append(text + sep)
    return "".join(converted)


def _clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """Trim a degenerate repeated tail, keeping one period plus the remainder."""
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


def finalize_document(text: str) -> str:
    """Upstream's whole-page post-processing order (postprocess/document.py)."""
    return postprocess_otsl_tables(postprocess_math_delimiters(_clean_truncated_repeats(text)))


def sglang_command() -> list[str]:
    """Return the SGLang server command, using the image's Python rather than this script's venv."""
    # The SGLang image keeps sglang in its own venv, first on PATH; `uv run` puts
    # this script's venv in front, so look past it.
    own_bin = os.path.realpath(os.path.join(sys.prefix, "bin"))
    path = os.pathsep.join(
        p for p in os.environ.get("PATH", "").split(os.pathsep) if os.path.realpath(p) != own_bin
    )
    python = shutil.which("python3", path=path)
    found = python and subprocess.run([python, "-c", "import sglang"], check=False, capture_output=True).returncode == 0
    if not found:
        sys.exit(f"--engine sglang needs sglang: add --image {SERVING['images']['sglang']} to hf jobs uv run.")
    return [
        python, "-m", "sglang.launch_server", "--model-path", SERVING["model"],
        "--context-length", str(SERVING["context"]), "--enable-metrics",
        "--host", "127.0.0.1", "--port", "8000",
    ]  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description="Xiaomi-OCR-0 batch OCR via saturate")
    ap.add_argument("input_dataset", help="Input dataset repo id (rows with an image column)")
    ap.add_argument("output_dataset", help="Output dataset repo id (created if missing)")
    ap.add_argument("--engine", choices=["vllm", "sglang"], default="vllm",
                    help="Serving engine; sglang also needs the SGLang --image (see docstring)")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--config", default=None, help="Dataset config name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--id-column", default=None,
                    help="Column to use as row id (default: split-index ids)")
    ap.add_argument("--max-samples", "--limit", dest="limit", type=int, default=None,
                    help="Process only the first N rows (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=SERVING["max_tokens"],
                    help="Output cap; must leave room for the image in the 16384 context")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-admit rows whose only record is an error row")
    args = ap.parse_args()

    if args.max_tokens < 1:
        ap.error("--max-tokens must be at least 1")
    if args.max_tokens > SERVING["max_tokens"]:
        ap.error(f"--max-tokens above {SERVING['max_tokens']} can overflow the {SERVING['context']} context")

    import pyarrow as pa
    from saturate import Engine, dataset_rows, pump

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

    def to_request(row):
        b64 = encode_image(row[args.image_column], SERVING["max_edge"])
        return {
            "model": SERVING["model"],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": OCR_PROMPT},
            ]}],
            "max_tokens": args.max_tokens,
            "temperature": SERVING["temperature"],
            "top_p": SERVING["top_p"],
            "presence_penalty": SERVING["presence_penalty"],
            "repetition_penalty": SERVING["repetition_penalty"],
            "chat_template_kwargs": {"enable_thinking": False},
        }

    # Declared, so a part holding only error rows has the same columns as the rest
    # (datasets.load_dataset reads every part with one schema).
    schema = pa.schema([
        ("id", pa.string()), ("error", pa.string()), ("markdown", pa.string()), ("raw", pa.string()),
        ("finish_reason", pa.string()), ("model", pa.string()),
        ("prompt_tokens", pa.int64()), ("completion_tokens", pa.int64()),
    ])  # fmt: skip

    def parse(row, body):
        choice = body["choices"][0]
        raw = choice["message"]["content"].strip()
        usage = body.get("usage") or {}
        return {
            "markdown": finalize_document(raw),
            "raw": raw,
            "finish_reason": choice.get("finish_reason"),
            "model": SERVING["model"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    if args.engine == "sglang":
        engine = Engine(SERVING["model"], engine="sglang", cmd=sglang_command())
    else:
        extra = ["--max-model-len", str(SERVING["context"]), *SERVING["vllm_args"]]
        engine = Engine(SERVING["model"], engine="vllm", extra_args=extra)

    output = f"hf://datasets/{args.output_dataset}/data"
    with engine as endpoint:
        stats = pump(rows, to_request, parse, endpoint, output,
                     prepare_workers=int(float(os.environ.get("CPU_CORES") or "4")),
                     retry_errors=args.retry_errors, schema=schema)

    print(f"https://huggingface.co/datasets/{args.output_dataset} "
          f"({stats.rows_processed} ok, {stats.rows_failed} error rows)", file=sys.stderr)
    print("XIAOMI_OCR_0_SATURATE " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
