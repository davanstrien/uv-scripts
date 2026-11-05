# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = [
#     "saturate[hf]==0.3.0",
#     "pillow>=10",
# ]
#
# [tool.hf-jobs]
# image = "cr.hf.co/davanstrien/hpd-parsing-vllm:87496aa5-uv0.13.6"
# flavor = "l4x1"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# ///
"""
Parse document images with HPD-Parsing via saturate.

HPD-Parsing (1B, InternVL3.5-1B vision + Qwen3-0.6B) parses a page with
*hierarchical parallel decoding*: a main layout branch emits `<BLOCK> <type>
[bbox]` headers and forks a child branch per region (children reuse the parent's
prefix KV), with a P-MTP medusa head drafting tokens inside each branch. Its
output carries a type, a bbox and reading order for every block; this recipe keeps
them in an `hpd_blocks` column next to the markdown.

The server half is the vendor's own `vllm serve` command, started in the job. The
driver half (concurrency, retries, output, resume) is the `saturate` library:
adaptive concurrency, resumable parquet output and durable error rows.

NEEDS PADDLE'S vLLM FORK. Forking is a scheduler feature of Paddle's vLLM fork
(`0.17.1+hpdparsing`), which ships in their registry image. The [tool.hf-jobs]
header above names `cr.hf.co/davanstrien/hpd-parsing-vllm`: Paddle's image
(`latest-nvidia-gpu`, pinned by digest, built 2026-07-16) plus `uv`, which their
image lacks. Build script and digest: images/hpd-parsing/ in the uv-scripts repo.
Run on HF Jobs (`hf` CLI 1.32+ reads image, flavor, timeout and secrets from the
header):

  hf jobs uv run --detach \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/hpd-parsing-saturate.py \\
      <input-dataset> <output-dataset>

Smoke test first with --max-samples 5. Budget for an ~11 GB image pull on a cold
node before anything runs. requires-python includes 3.10 so that uv uses the
image's own Python (3.10) instead of downloading one.

On Paddle's image directly (no uv in it), install uv in the job command:

  hf jobs run --detach --flavor l4x1 -s HF_TOKEN --timeout 2h \\
      ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm@sha256:87496aa5dd702a7df6c70bb00c29d5bf8d1a0f0505d66b613fafa6f71cd72de2 \\
      -- bash -lc 'pip install -q uv==0.13.6 && uv run \\
          https://huggingface.co/datasets/uv-scripts/ocr/raw/main/hpd-parsing-saturate.py \\
          <input-dataset> <output-dataset>'

Output: the output repo (created private if missing) holds `data/part-*.parquet`
with rows `{id, markdown, hpd_blocks, raw, looped_blocks, finish_reason, model,
prompt_tokens, completion_tokens, error}` keyed by the input row id
(`--id-column`, or `<repo>:<config>:<split>:<000000000>` by default). Read it with
`saturate.read_output`, which keeps one row per id (a success over an earlier
error). `datasets.load_dataset(<output>, data_dir="data")` returns every record,
so after a --retry-errors run it also holds the old error rows: drop rows with an
`error` before you join back to the input on id. Re-running the same command
skips finished rows; --retry-errors re-admits failed ones.

- `hpd_blocks`: JSON list, one entry per region in the model's reading order:
  `{"type": "text|title|table|formula|figure|...", "bbox": [x1, y1, x2, y2], "text": "..."}`.
  BBOX SPACE: 0-1000 **normalized**, not pixels. Convert with
  `px = value / 1000 * width`. Container blocks (`list`, the `table` wrapper)
  and `chart`/`seal` regions carry a bbox with empty `text`.
- `raw`: the model's untouched `<BLOCK>/<FORK>/<CHILD>` stream.
- `looped_blocks`: number of blocks whose text is a repetition loop (see Limits).

Model: PaddlePaddle/HPD-Parsing (Apache-2.0), paper arXiv:2607.18839
- Card prompt `document parsing with fork.` (hierarchical parallel decoding);
  --no-fork sends the card's single-trajectory `document parsing.` for an A/B.
- Image before text, greedy decoding, max_tokens 8000, context 16384: the card's
  values.
- Block parsing and formula clean-up adapted from the model repo's Apache-2.0
  `eval/hpd_to_markdown.py`, via the bbox-preserving `parse_blocks` in
  multimodalart's demo Space (hugging-apps/hpd-parsing, `hpd_postprocess.py`).

Limits seen in testing:
- `max_tokens` bounds ONE decoding branch, not the page. A dense page that
  fragments into hundreds of blocks can loop inside many children at once: on 12
  Chronicling America pages (1774-1809), 3 pages looped, one emitting 544k
  characters against 24.9k in LoC's own OCR. Nothing errors. This recipe flags
  such blocks in `looped_blocks` and leaves the text as the model wrote it.
  Filter on `looped_blocks == 0`, or drop the looped blocks from `hpd_blocks`
  yourself. On those 12 pages the 3 looping pages also ended with
  `finish_reason` "length" and the other 9 with "stop".
- Card languages are en and zh. A smoke run on degraded French scans produced
  stray CJK tokens.

The SERVING dict below is the per-model tuning prior. Agents can `ast.literal_eval`
it without running the script; the script itself consumes it.
"""

import argparse
import base64
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zlib

# Serving starting values for PaddlePaddle/HPD-Parsing. Per-value provenance:
# - serve_args: the vendor image's own /home/hpd/entrypoint.sh, verbatim, except
#   for the speculative head path (see serve_command) and an added --revision pin.
#   Two flags look wrong and are not:
#   * --enable-prefix-caching contradicts the house OCR default (SERVING.md: OCR
#     never reuses images). Here the forked children share the parent branch's
#     prefix KV, so the cache is load-bearing (52.9% hit rate measured, 2026-07-27).
#   * --attention-config use_prefill_query_quantization is a fork-only flag.
# - served_model_name / port 8118: the entrypoint's values; requests use the name.
# - max_model_len 16384, max_tokens 8000, temperature 0.0: card values.
# - max_pixels 0 (no client-side downscale): the model picks its 448 px tile grid
#   partly from image area (find_closest_aspect_ratio_optim), so pre-shrinking can
#   land a page on a smaller grid. Set --max-pixels only for huge payloads.
# - image: Paddle's `latest-nvidia-gpu` tag pinned by digest on 2026-11-05
#   (base_image), plus uv (image; built by images/hpd-parsing/build.sh).
SERVING = {
    "model": "PaddlePaddle/HPD-Parsing",
    "revision": "91de80054c23ab4238e3d7073fa2b83c2a7e301e",
    "served_model_name": "HPD-Parsing",
    "image": "cr.hf.co/davanstrien/hpd-parsing-vllm:87496aa5-uv0.13.6",
    "base_image": (
        "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm"
        "@sha256:87496aa5dd702a7df6c70bb00c29d5bf8d1a0f0505d66b613fafa6f71cd72de2"
    ),
    "port": 8118,
    "max_model_len": 16384,
    "serve_args": [
        "--trust-remote-code",
        "--host", "0.0.0.0",
        "--limit-mm-per-prompt", '{"image": 1}',
        "--gpu-memory-utilization", "0.9",
        "--attention-backend", "FLASHINFER",
        "--attention-config", '{"use_prefill_query_quantization":true}',
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
    ],
    "num_speculative_tokens": 6,
    "max_tokens": 8000,
    "temperature": 0.0,
    "max_pixels": 0,
}  # fmt: skip
assert SERVING["max_tokens"] < SERVING["max_model_len"], (
    "context math: max_tokens must leave room for the image tokens "
    "(input + output <= max_model_len, or every request 400s)"
)

# Card-verbatim prompts. Do not "improve" either.
FORK_PROMPT = "document parsing with fork."
PLAIN_PROMPT = "document parsing."

RUN_COMMAND = (
    f"hf jobs uv run --detach --image {SERVING['image']} --flavor l4x1 -s HF_TOKEN \\\n"
    "    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/hpd-parsing-saturate.py \\\n"
    "    <input-dataset> <output-dataset>"
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
    """RGB-convert, downscale only if max_pixels is set, return base64 JPEG q95."""
    from PIL import Image

    img = to_pil(value).convert("RGB")
    w, h = img.size
    if max_pixels and w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


# --- Formula cleaning (verbatim from the model repo's eval/hpd_to_markdown.py) ---

_TALL = re.compile(
    r'\\d?frac|\\tfrac|\\cfrac|\\binom|\\sqrt'
    r'|\\sum|\\prod|\\coprod|\\int|\\iint|\\iiint|\\oint'
    r'|\\bigcup|\\bigcap|\\bigoplus|\\bigotimes|\\bigsqcup'
    r'|\\begin\{'
    r'|\\overbrace|\\underbrace|\\overset|\\underset|\\stackrel'
    r'|\\substack|\\atop|\\\\'
)


def _scan_delims(s):
    out = []
    for m in re.finditer(r'\\(left|right)\s*', s):
        dm = re.match(r'\\[a-zA-Z]+|\\.|.', s[m.end():])
        if not dm:
            continue
        out.append({'kind': m.group(1), 'delim': dm.group(0),
                    'start': m.start(), 'end': m.end() + dm.end()})
    return out


def simplify_left_right(s: str) -> str:
    """Downgrade `\\left( ... \\right)` with no tall inner structure to plain `( )`."""
    if '\\left' not in s:
        return s
    stack, pairs = [], []
    for d in _scan_delims(s):
        if d['kind'] == 'left':
            stack.append(d)
        elif stack:
            pairs.append((stack.pop(), d))
    edits = []
    for L, R in pairs:
        if L['delim'] == '(' and R['delim'] == ')' and not _TALL.search(s[L['end']:R['start']]):
            edits.append((L['start'], L['end'], '('))
            edits.append((R['start'], R['end'], ')'))
    for st, en, rep in sorted(edits, key=lambda x: x[0], reverse=True):
        s = s[:st] + rep + s[en:]
    return s


_ELLIPSIS = r'(?:\\dots|\\cdots|\\ldots|\\dotsb|\\dotsc)'
_CLOSER = r'(?:\\right\s*[.\}\]\)]|\\end\s*\{(?:array|matrix|cases|bmatrix|pmatrix|vmatrix|smallmatrix)\})'
_TAIL_WRAP = re.compile(r'^(?P<core>.*?)(?P<wrap>\s*(?:\\\]|\\\)|\$\$))?\s*$', re.DOTALL)


def clean_formula_tail(s: str) -> str:
    """Strip degenerate formula tails (repeated/dangling ellipses, stray `\\quad`)."""
    if not s:
        return s
    m = _TAIL_WRAP.match(s)
    core, wrap = m.group('core'), m.group('wrap') or ''
    prev = None
    while prev != core:
        prev = core
        core = re.sub(r'(' + _ELLIPSIS + r')(?:\s*' + _ELLIPSIS + r')+', r'\1', core)
        core = re.sub(r'(?P<keep>' + _CLOSER + r')\s*(?:\\q?quad\s*)*' + _ELLIPSIS + r'\s*$',
                      lambda mm: mm.group('keep'), core)
        core = re.sub(r'(?:\s*\\q?quad)+\s*' + _ELLIPSIS + r'\s*$', '', core)
        core = re.sub(r'(?:\s*\\q?quad)+\s*$', '', core)
        core = core.rstrip()
    return core + wrap


_OP_MAP = {
    '≈': r'\approx', '≠': r'\neq', '≤': r'\leq', '≥': r'\geq', '×': r'\times',
    '÷': r'\div', '±': r'\pm', '∓': r'\mp', '·': r'\cdot', '∙': r'\cdot',
    '⋅': r'\cdot', '∗': '*', '−': '-', '≡': r'\equiv', '∝': r'\propto',
    '∞': r'\infty', '√': r'\sqrt', '→': r'\to', '≪': r'\ll', '≫': r'\gg',
}
_ARITH_ALLOWED = re.compile(r'^[0-9A-Za-z\s=+\-*/^_().,:;<>|%!\u4e00-\u9fff' + ''.join(_OP_MAP.keys()) + r']+$')
_ARITH_HASOP = re.compile(r'[=+\-*/' + ''.join(_OP_MAP.keys()) + r']')
_KNOWN_FUNCS = {'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'log', 'ln', 'exp',
                'lim', 'max', 'min', 'det', 'mod', 'arcsin', 'arccos', 'arctan', 'sqrt'}
_CJK_RUN = re.compile(r'[\u4e00-\u9fff]+')
_MATH_SPAN = re.compile(r'(\\\[.*?\\\]|\$\$.*?\$\$|\\\(.*?\\\)|\$.*?\$)', re.DOTALL)

WRAP_CJK_IN_ARITH = True


def _convert_unicode_ops(s: str) -> str:
    for k, v in _OP_MAP.items():
        s = s.replace(k, (v + ' ') if v.startswith('\\') else v)
    if WRAP_CJK_IN_ARITH:
        s = _CJK_RUN.sub(lambda m: r'\text{' + m.group(0) + '}', s)
    return re.sub(r'[ \t]{2,}', ' ', s)


def _is_pure_arith_line(line: str) -> bool:
    t = line.strip()
    if not t or '\\(' in t or '\\[' in t or '$' in t or '<' in t:
        return False
    if not WRAP_CJK_IN_ARITH and re.search(r'[\u4e00-\u9fff]', t):
        return False
    if not _ARITH_ALLOWED.match(t) or not _ARITH_HASOP.search(t):
        return False
    return all(w.lower() in _KNOWN_FUNCS for w in re.findall(r'[A-Za-z]{2,}', t))


def normalize_arith(text: str) -> str:
    """Normalize Unicode operators to LaTeX and wrap pure-arithmetic lines as `\\( .. \\)`."""
    if not text:
        return text
    text = _MATH_SPAN.sub(lambda m: _convert_unicode_ops(m.group(0)), text)
    out = []
    for line in text.split('\n'):
        if _is_pure_arith_line(line):
            out.append('\\( ' + _convert_unicode_ops(line.strip()) + ' \\)')
        else:
            out.append(line)
    return '\n'.join(out)


def clean_text(text: str) -> str:
    """Per-block cleaning, matching the official hpd_to_markdown.py steps."""
    text = text.strip()
    text = text.replace('The image is too blurry to recognize any text content.', '').strip()
    text = text.replace(
        "The image contains no text or characters. It is a graphical element (a horizontal "
        "line with a vertical line) and does not contain any chart, graph, or data points "
        "that can be extracted. Therefore, the correct OCR output is an empty string.", ""
    ).strip()
    if not text or text == '[Non-Text]':
        return ''
    if text.startswith('\\[') and not text.endswith('\n\\]'):
        text += '\n\\]'
    if text.startswith('<table>') and not text.endswith('</table>'):
        text += '</table>'
    if '\\[\n' in text and '\\\\' not in text:
        text = text.replace('\\[\n', '\\(').replace('\n\\]', '\\)')
    text = text.replace('\\) \\(', '\\)\n\n\\(')
    if '÷' in text and '\\(' not in text:
        text = '\\( ' + text + ' \\)'
    text = re.sub(r'\\tag\s*\{[^{}]*\}', '', text)
    text = text.replace('\\supset', '\\sqsupset')
    text = simplify_left_right(text)
    text = clean_formula_tail(text)
    return normalize_arith(text)


# --- Block parsing (bbox-preserving; the official script discards bboxes) ---

# type + [bbox], usually followed by <FORK|CHILD|BLOCK>. Container blocks such as
# `list [x1,y1,x2,y2]` may have no trailing tag at all, so the tag is optional.
_BLOCK_HEADER = re.compile(r'([a-zA-Z_]+)\s*\[\s*([-\d.,\s]+)\]\s*(?:<(?:FORK|CHILD|BLOCK)>)?')
_NO_CONTENT_TYPES = {'chart', 'seal'}


def parse_blocks(raw_text: str) -> list[dict]:
    """Parse the `<BLOCK> <type> [bbox] <CHILD> <content>` stream.

    Returns `[{"type": str, "bbox": [x1,y1,x2,y2] | None, "text": str}, ...]` in the
    model's own reading order. `bbox` is in the model's native 0-1000 NORMALIZED
    coordinate space, not pixels. `text` is empty for container blocks (e.g. `list`,
    the `table` wrapper) and for `chart`/`seal`, matching the official export.
    """
    blocks = []
    for seg in raw_text.split('<BLOCK>')[1:]:
        header_m = _BLOCK_HEADER.match(seg.strip())
        type_m = re.match(r'\s*([a-zA-Z_]+)', seg)
        block_type = type_m.group(1) if type_m else 'unknown'

        bbox = None
        if header_m:
            block_type = header_m.group(1)
            nums = [float(x) for x in re.split(r'[,\s]+', header_m.group(2).strip()) if x]
            if len(nums) == 4:
                bbox = nums

        text = ''
        if block_type.lower() not in _NO_CONTENT_TYPES:
            content_m = re.search(r'<CHILD>(.*)', seg, re.DOTALL)
            if content_m:
                # stop at the next control tag if any leaked through
                raw_content = re.split(r'<(?:FORK|CHILD|BLOCK)>', content_m.group(1))[0]
                text = clean_text(raw_content)

        blocks.append({'type': block_type, 'bbox': bbox, 'text': text})
    return blocks


def blocks_to_markdown(blocks: list[dict]) -> str:
    """Join the non-empty block texts in reading order into one markdown string."""
    return '\n\n'.join(b['text'] for b in blocks if b['text']).strip()


# --- Loop flag ---

# Calibrated on the 2026-07-27 test outputs (ChronAm, OmniDocBench, UFO; 345 blocks of
# 200+ bytes). Ordinary text blocks of 2000+ bytes compress about 2x with zlib; looping
# blocks compressed 6x (a counting loop: "of the 860th of the 861th ...") to 330x. The
# most compressible real block was a 1.5 KB table at 5.3x, so tables get a higher bar.
LOOP_MIN_BYTES = 2000
LOOP_MIN_RATIO = 4.0
LOOP_MIN_RATIO_TABLE = 10.0


def is_looped(block: dict) -> bool:
    """True when a block's text is long and highly repetitive (a decoding loop)."""
    data = block["text"].encode()
    if len(data) < LOOP_MIN_BYTES:
        return False
    ratio = len(data) / len(zlib.compress(data))
    threshold = LOOP_MIN_RATIO_TABLE if block["type"] == "table" else LOOP_MIN_RATIO
    return ratio >= threshold


# --- Server ---


def vendor_vllm() -> str:
    """Return the vendor fork's `vllm` binary, or exit with the command that provides it."""
    vllm = shutil.which("vllm")
    version = ""
    if vllm:
        out = subprocess.run([vllm, "--version"], capture_output=True, text=True, check=False)
        lines = out.stdout.strip().splitlines()
        version = lines[-1] if lines else ""
    if "hpdparsing" not in version:
        found = f"found vllm {version!r} at {vllm}" if vllm else "no `vllm` on PATH"
        sys.exit(
            f"HPD-Parsing needs Paddle's vLLM fork (0.17.1+hpdparsing); {found}. "
            f"Run the job on an image that has it:\n\n{RUN_COMMAND}"
        )
    print(f"vendor vllm {version} at {vllm}", file=sys.stderr)
    return vllm


def resolve_spec_model() -> str:
    """Download the P-MTP speculative head and return its local directory.

    The vendor entrypoint passes `PaddlePaddle/HPD-Parsing/P-MTP`, a local directory in
    the image's offline variant (weights baked in). This image has no weights, so pull
    that subfolder (~644 MB) into the normal HF cache and pass its path.
    """
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        SERVING["model"], revision=SERVING["revision"], allow_patterns=["P-MTP/*"]
    )
    spec_dir = os.path.join(snapshot, "P-MTP")
    if not os.path.isfile(os.path.join(spec_dir, "config.json")):
        sys.exit(f"P-MTP head has no config.json under {spec_dir}")
    return spec_dir


def serve_command(vllm: str, max_model_len: int) -> list[str]:
    """The vendor entrypoint's `vllm serve` command, with a local speculative head."""
    speculative = {
        "method": "medusa",
        "model": resolve_spec_model(),
        "num_speculative_tokens": SERVING["num_speculative_tokens"],
    }
    return [
        vllm, "serve", SERVING["model"],
        "--revision", SERVING["revision"],
        "--served-model-name", SERVING["served_model_name"],
        "--port", str(SERVING["port"]),
        "--max-model-len", str(max_model_len),
        *SERVING["serve_args"],
        "--speculative-config", json.dumps(speculative),
    ]  # fmt: skip


def main():
    ap = argparse.ArgumentParser(
        description="HPD-Parsing document parsing via saturate (vendor vLLM fork image only)"
    )
    ap.add_argument("input_dataset", help="Input dataset repo id (rows with an image column)")
    ap.add_argument("output_dataset", help="Output dataset repo id (created private if missing)")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--config", default=None, help="Dataset config name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--id-column", default=None,
                    help="Column to use as row id (default: split-index ids)")
    ap.add_argument("--max-samples", "--limit", dest="limit", type=int, default=None,
                    help="Process only the first N rows (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=SERVING["max_tokens"],
                    help="Output cap PER DECODING BRANCH, not per page (card: 8000)")
    ap.add_argument("--max-pixels", type=int, default=SERVING["max_pixels"],
                    help="Downscale images above this pixel count before upload (default 0 = off)")
    ap.add_argument("--no-fork", action="store_true",
                    help="Send the single-trajectory prompt ('document parsing.') instead of "
                         "hierarchical parallel decoding; for an A/B only")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-admit rows whose only record is an error row")
    args = ap.parse_args()

    if not 1 <= args.max_tokens < SERVING["max_model_len"]:
        ap.error(f"--max-tokens must be between 1 and {SERVING['max_model_len'] - 1}")

    vllm = vendor_vllm()  # fail before the dataset is touched if this is the wrong image

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
    prompt = PLAIN_PROMPT if args.no_fork else FORK_PROMPT

    def to_request(row):
        b64 = encode_image(row[args.image_column], args.max_pixels)
        return {
            "model": SERVING["served_model_name"],
            "messages": [{"role": "user", "content": [
                # Image first, then text: the order of the card's example.
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            "temperature": SERVING["temperature"],
            "max_tokens": args.max_tokens,
        }

    # Declared, so a part holding only error rows has the same columns as the rest
    # (datasets.load_dataset reads every part with one schema).
    schema = pa.schema([
        ("id", pa.string()), ("error", pa.string()), ("markdown", pa.string()),
        ("hpd_blocks", pa.string()), ("raw", pa.string()), ("looped_blocks", pa.int64()),
        ("finish_reason", pa.string()), ("model", pa.string()),
        ("prompt_tokens", pa.int64()), ("completion_tokens", pa.int64()),
    ])  # fmt: skip

    def parse(row, body):
        choice = body["choices"][0]
        raw = choice["message"]["content"] or ""
        blocks = parse_blocks(raw)
        usage = body.get("usage") or {}
        return {
            "markdown": blocks_to_markdown(blocks),
            "hpd_blocks": json.dumps(blocks, ensure_ascii=False),
            "raw": raw,
            "looped_blocks": sum(is_looped(b) for b in blocks),
            "finish_reason": choice.get("finish_reason"),
            "model": SERVING["model"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    engine = Engine(
        SERVING["model"],
        port=SERVING["port"],
        cmd=serve_command(vllm, SERVING["max_model_len"]),
    )
    output = f"hf://datasets/{args.output_dataset}/data"
    with engine as endpoint:
        stats = pump(rows, to_request, parse, endpoint, output,
                     prepare_workers=int(float(os.environ.get("CPU_CORES") or "4")),
                     retry_errors=args.retry_errors, schema=schema)

    print(f"https://huggingface.co/datasets/{args.output_dataset} "
          f"({stats.rows_processed} ok, {stats.rows_failed} error rows)", file=sys.stderr)
    print("HPD_PARSING_SATURATE " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
