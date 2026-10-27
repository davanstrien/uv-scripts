# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]>=0.2.0",
#     "pillow>=10",
#     "pymupdf>=1.24",
# ]
#
# [tool.hf-jobs]
# image = "lmsysorg/sglang:v0.5.20-cu130"
# flavor = "rtx-pro-6000"
# timeout = "4h"
# secrets = ["HF_TOKEN"]
# ///
"""
OCR whole documents (book volumes, PDFs) with Baidu Unlimited-OCR-2 via saturate.

Unlimited-OCR-2 reads many pages in ONE request and keeps context across page
breaks: a paragraph that runs onto the next page comes back tagged as a
continuation, not as two fragments. So the unit of work here is a document, not
a page. Input is a folder (bucket, mounted dir or local) of page images and/or
PDFs; pages are grouped into documents, each document is cut into windows of
--pages-per-request consecutive pages, and each window is one request.

Run on HF Jobs (the [tool.hf-jobs] table above supplies image, flavor, timeout
and HF_TOKEN; flags you pass on the command line win):

  hf jobs uv run --detach \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/unlimited-ocr2-saturate.py \\
      "hf://buckets/biglam/britannica/source/pages/1771EncyclopediaBritannicaNLS/*.jp2" \\
      <user>/britannica-1771-uocr2

Grouping pages into documents:
- Page files: the document is the parent folder plus the file name with its
  trailing page number removed (`Volume 1, A-B_0007.jp2` -> `Volume 1, A-B`),
  so an Internet Archive item holding several volumes splits into volumes.
  Pages are ordered by that number (natural sort).
- PDFs: one document per file, pages rendered at --pdf-dpi.

Output: `data/part-*.parquet` in the output dataset (or any fsspec URI, e.g.
hf://buckets/...), one row per window:
`{id, doc_id, first_page, page_files, num_pages, status, finish_reason,
page_markdown, markdown, raw, prompt_tokens, completion_tokens, error}`. `page_markdown` is aligned with
`page_files`; `raw` keeps the model's grounded output (block types + boxes).
`markdown` is the whole window through Baidu's own post-processor (vendored
below at a pinned commit): cross-page paragraph/table chains merged, figures as
image links, page furniture kept. `page_markdown` is a lighter per-page cleanup
that stays aligned with page files (page numbers and running heads dropped
unless --keep-furniture). A window whose output is truncated or has the wrong
number of <PAGE> blocks keeps its text with `status` = truncated / page_mismatch
(greedy decoding repeats it on retry); `page_markdown` is set only when status = ok.
Re-running the same command resumes; --retry-errors re-admits failed windows.

Repair pass (on by default; --no-repair skips it, --repair-only runs just it):
about 2% of 1771 Britannica pages, mostly dense mathematics, sent the model
into a loop inside a window; the forced page break then shifted the next
page's text by one. After the main pass, looping pages, their empty neighbours
and every page of a failed window are re-OCR'd one at a time in gundam mode
into `repairs/` (one row per page: `{page_file, page, doc_id, window, status,
markdown, raw}`). That fixed 28 of 33 flagged pages in testing; the rest keep
their text with status = degenerate. To assemble final per-page text, take the
`repairs/` row for a page when it exists, else `page_markdown` from its window.

Why this is a saturate recipe, not an offline one: the multi-page engine lives
in Baidu's own SGLang build (no vLLM support for v2). The script installs that
wheel into the image's Python, boots `sglang.launch_server`, and lets saturate
drive it (adaptive concurrency from SGLang's /metrics, resumable parquet output).

Model: baidu/Unlimited-OCR-2 (3.3B, MIT). https://huggingface.co/baidu/Unlimited-OCR-2
"""

import argparse
import base64
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
import types
import zlib

# Per-value provenance:
# - image lmsysorg/sglang:v0.5.20-cu130: the SGLang release whose pins match
#   Baidu's wheel exactly (torch 2.13.0, transformers 5.12.1, flashinfer 0.6.18
#   cu13). The wheel is installed --no-deps on top. Move the pin only together
#   with the wheel; drop both once upstream SGLang ships the architecture.
# - env + serve_args: the model card's launch command, verbatim (tuned for an
#   80 GB A800; validated 2026-10-27 on a100-large and rtx-pro-6000, boot ~3.5 min).
# - flavor rtx-pro-6000: on the same 512 1771 Britannica pages it finished 32/32
#   windows in 805 s (1,571 tok/s end to end) vs a100-large 1,267 s (866 tok/s):
#   ~30% cheaper per page. The n-gram guard makes decode scheduler-CPU-bound, so
#   CPU matters as much as the GPU. 24 GB cards (a10g, l4) thrash the KV cache on
#   16-page windows (~31k tokens each); use >= 48 GB.
# - tokens_per_page 8000, ngram 35, window 1024 multi / 128 single, page guard
#   20000, temperature 0: the card's client example.
# - crop_mode False (base, one global view per page): the card's multi-page
#   default (infer_multi). On a 32-page 1861 scan gundam (crop) mode read the
#   same text with 6.7x the image tokens (58k vs 8.7k).
SERVING = {
    "model": "baidu/Unlimited-OCR-2",
    "served_name": "Unlimited-OCR-2",
    "image": "lmsysorg/sglang:v0.5.20-cu130",
    "wheel": "https://huggingface.co/baidu/Unlimited-OCR-2/resolve/main/wheel/"
             "sglang-0.5.20.dev10%2Bg8d62f5fb9c-py3-none-any.whl",
    "extra_pip": ["fastokens==0.3.2"],
    "env": {
        "SGLANG_UOCR_EANABLE_DOC_LIMIT": "1",  # (sic) upstream spelling
        "SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW": "10",
        "SGLANG_UOCR_DOC_BASE_PAGE_WINDOW": "10",
        "SGLANG_UOCR_MAX_PAGE_TOKENS": "20000",
        "SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB": "64",
        "SGLANG_UOCR_SHM_WRAP_MODEL_DATA": "1",
        "SGLANG_UOCR_SHM_WRAP_MIN_MB": "1",
        "SGLANG_VLM_CACHE_SIZE_MB": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    "serve_args": [
        "--chat-template", "unlimited-ocr",
        "--tokenizer-backend", "fastokens", "--tokenizer-worker-num", "8",
        "--mem-fraction-static", "0.7",
        "--max-running-requests", "64",
        "--max-prefill-tokens", "2048",
        "--enable-custom-logit-processor", "--enable-streaming-session",
        "--allow-auto-truncate", "--disable-radix-cache",
        "--mm-preprocess-cache-size-mb", "0",
        "--cuda-graph-backend-decode", "full", "--cuda-graph-max-bs-decode", "64",
        "--cuda-graph-backend-prefill", "full",
        "--cuda-graph-max-bs-prefill", "16384",
        "--cuda-graph-bs-prefill", "1024", "2048",
        "--cuda-graph-config", '{"prefill": {"backend": "full", "full_prefill_max_req": 64}}',
    ],
    "tokens_per_page": 8000,
    "temperature": 0.0,
    "crop_mode": False,
    "ngram_size": 35,
    "page_token_id": 128827,  # <PAGE>, = sparse_attention_config.page_token_id
    "eos_token_id": 1,
    "max_page_tokens": 20000,
}

PDF_LOCK = threading.Lock()
PAGE_EXTS = (".jp2", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
FURNITURE = {"page_number", "header", "footer"}
PAGE_BLOCK_RE = re.compile(r"<BLOCK>(?P<type>[a-z_]+)(?:<LEVEL(?P<level>\d)>)?\[[\d, ]*\]<FORK>")


def ngram_processor() -> str:
    """The card's DeepseekOCRNoRepeatNGramLogitProcessor.to_str(), built without sglang.

    to_str() is json({"callable": dill.dumps(cls).hex()}) and dill pickles an
    importable class by reference, so the payload is just a pickled global
    (module + name). Building it here keeps sglang/torch/dill out of the driver.
    """
    mod, name = "sglang.srt.sampling.custom_logit_processor", "DeepseekOCRNoRepeatNGramLogitProcessor"
    saved = {k: sys.modules.get(k) for k in ("sglang", "sglang.srt", "sglang.srt.sampling", mod)}
    try:
        parts = mod.split(".")
        for i in range(len(parts)):
            sys.modules[".".join(parts[: i + 1])] = types.ModuleType(parts[i])
        cls = type(name, (), {"__module__": mod})
        setattr(sys.modules[mod], name, cls)
        return json.dumps({"callable": pickle.dumps(cls).hex()})
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def doc_key(rel_path: str) -> str:
    """`item/Volume 1, A-B_0007.jp2` -> `item/Volume 1, A-B`; a PDF is its own doc."""
    if rel_path.lower().endswith(".pdf"):
        return rel_path
    stem = rel_path.rsplit(".", 1)[0]
    doc = re.sub(r"[_\-. ]?\d+$", "", stem)
    return doc if doc and not doc.endswith("/") else (doc or ".")


def list_windows(pattern: str, pages_per_request: int, limit_docs, max_pages, pdf_dpi):
    """List page files, group into documents, cut each into windows -> (id, row)."""
    import fsspec
    from saturate import bucket_rows

    if not re.search(r"[*?\[]", pattern) and not pattern.lower().endswith(".pdf"):
        pattern = pattern.rstrip("/") + "/**"
    files = [(rid, row["path"]) for rid, row in bucket_rows(pattern, read=False)
             if rid.lower().endswith(PAGE_EXTS + (".pdf",))]
    if not files:
        sys.exit(f"no page images or PDFs matched {pattern!r}")

    docs: dict[str, list[tuple[str, str]]] = {}
    for rid, path in files:
        docs.setdefault(doc_key(rid), []).append((rid, path))
    doc_ids = sorted(docs, key=natural_key)[:limit_docs]

    fs = fsspec.core.url_to_fs(pattern)[0]
    n_pages = 0
    for d in doc_ids:
        pages = sorted(docs[d], key=lambda p: natural_key(p[0]))
        if d.lower().endswith(".pdf"):  # explode a PDF into (page-index) entries
            import fitz

            with fs.open(pages[0][1], "rb") as f:
                count = fitz.open(stream=f.read(), filetype="pdf").page_count
            pages = [(f"{d}#{i + 1}", f"{pages[0][1]}#{i}") for i in range(count)]
        for start in range(0, len(pages), pages_per_request):
            if max_pages is not None and n_pages >= max_pages:
                return
            window = pages[start: start + pages_per_request]
            if max_pages is not None:
                window = window[: max_pages - n_pages]
            n_pages += len(window)
            # The id encodes the page range: a --max-pages-truncated window must
            # not mark the full window done on a later resume.
            yield f"{d}@{start:06d}-{start + len(window):06d}", {
                "doc_id": d, "first_page": start,
                "page_files": [rid for rid, _ in window],
                "paths": [p for _, p in window],
            }


def load_page(fs, path: str, max_side: int, pdf_dpi: int) -> str:
    """Read one page (image file or `file.pdf#i`), downscale, return base64 JPEG."""
    from PIL import Image

    if "#" in path and path.split("#")[0].lower().endswith(".pdf"):
        import fitz

        pdf_path, idx = path.rsplit("#", 1)
        with fs.open(pdf_path, "rb") as f:
            data = f.read()
        with PDF_LOCK:  # PyMuPDF is not thread-safe; pages are prepared in threads
            png = fitz.open(stream=data, filetype="pdf")[int(idx)].get_pixmap(dpi=pdf_dpi).tobytes("png")
        img = Image.open(io.BytesIO(png))
    else:
        with fs.open(path, "rb") as f:
            data = f.read()
        img = Image.open(io.BytesIO(data))
        if img.format == "JPEG2000":
            # Decode straight at a lower wavelet level when the page stays >= max_side:
            # same pixels the model would see after downscaling, ~2.5x less CPU
            # (measured: 406 -> 160 ms/page at 8 threads on 1771 Britannica scans).
            # Some codestreams fail at a reduced level ("broken data stream"), so step
            # back towards a full decode until one works.
            reduce = 0
            while max(img.size) >> (reduce + 1) >= max_side:
                reduce += 1
            while True:
                try:
                    img = Image.open(io.BytesIO(data))
                    img.reduce = reduce
                    img.load()
                    break
                except OSError:
                    if reduce == 0:
                        raise
                    reduce -= 1
        else:
            img.load()
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def clean_page(raw_page: str, keep_furniture: bool) -> tuple[str, bool]:
    """Grounded page -> (markdown, continues_previous_page).

    Tags and boxes dropped, titles -> #, furniture dropped. A block preceded by
    <MERGE> belongs to the block before it (joined with a newline, e.g. verse
    lines); an `append_*` first block continues the previous page's last block.
    """
    blocks: list[str] = []
    continues = False
    merge_next = False
    matches = list(PAGE_BLOCK_RE.finditer(raw_page))
    for i, m in enumerate(matches):
        body = raw_page[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(raw_page)].strip()
        merge_prev, merge_next = merge_next, body.endswith("<MERGE>")
        body = body.removesuffix("<MERGE>").strip()
        kind = m.group("type")
        if not body or (kind in FURNITURE and not keep_furniture):
            continue
        if kind == "title" and m.group("level"):
            body = "#" * int(m.group("level")) + " " + body
        if not blocks and kind.startswith("append_"):
            continues = True
        if merge_prev and blocks:
            blocks[-1] += "\n" + body
        else:
            blocks.append(body)
    return "\n\n".join(blocks), continues


def degenerate(raw_page: str) -> bool:
    """A looping page: far longer than any real page, or highly repetitive.

    Thresholds from 1771 Britannica (median raw page ~6.5k chars, densest real
    page ~10k; loops ran 14k-109k chars at zlib ratio 0.005-0.19, real text ~0.35).
    """
    if len(raw_page) > 15000:
        return True
    return len(raw_page) > 2000 and len(zlib.compress(raw_page.encode())) / len(raw_page) < 0.2


def repair_rows(windows, output: str):
    """Pages to re-OCR one at a time, from the windows and what the main pass wrote.

    A window with no usable record (error row, or truncated / page_mismatch) has all
    its pages repaired. In an ok window, a looping page is repaired together with
    an empty neighbour: the loop's forced page break shifts the next page's text.
    """
    from saturate import read_output

    done = dict(read_output(output))
    for wid, row in windows:
        n = len(row["paths"])
        rec = done.get(wid)
        if rec is None or rec.get("status", "ok") != "ok" or not rec.get("raw"):
            bad = set(range(n))
        else:
            raw_pages = rec["raw"].split("<PAGE>")[1:]
            bad = {i for i, p in enumerate(raw_pages) if degenerate(p)}
            texts = rec.get("page_markdown") or [""] * n
            for i in list(bad):
                bad |= {j for j in (i - 1, i + 1) if 0 <= j < n and not texts[j].strip()}
        for i in sorted(bad):
            yield row["page_files"][i], {
                "doc_id": row["doc_id"], "page": row["first_page"] + i, "window": wid,
                "page_file": row["page_files"][i], "paths": [row["paths"][i]],
                # alone and tiled: fixed 8 of 10 looping 1771 pages in testing
                "crop": True, "max_side": 2048,
            }


def find_server_python(candidates) -> str:
    """The image's Python (the one that has sglang), not this script's uv venv.

    The SGLang image keeps sglang in its own venv (/opt/sglang/bin), first on PATH;
    `uv run` puts this script's venv in front, so search PATH without it.
    """
    own_bin = os.path.join(sys.prefix, "bin")
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                           if os.path.realpath(p) != os.path.realpath(own_bin))
    on_path = [shutil.which(name, path=path) for name in ("python3", "python")]
    for py in [p for p in on_path if p] + list(candidates):
        if os.path.exists(py) and subprocess.run(
            [py, "-c", "import sglang"], check=False, capture_output=True
        ).returncode == 0:
            return py
    sys.exit(
        "No Python with sglang found. Run on the SGLang image, e.g.\n"
        f"  hf jobs uv run --image {SERVING['image']} --flavor a100-large -s HF_TOKEN ...\n"
        "(the script's [tool.hf-jobs] table sets this when launched with `hf jobs uv run`)."
    )


def main():
    ap = argparse.ArgumentParser(description="Unlimited-OCR-2 document OCR via saturate")
    ap.add_argument("input", help="Folder/glob of page images or PDFs: hf://buckets/..., a mounted dir, ...")
    ap.add_argument("output", help="Output dataset id (org/name) or fsspec URI (hf://buckets/...)")
    ap.add_argument("--pages-per-request", type=int, default=16,
                    help="Window size: consecutive pages sent in one request (default 16). "
                         "Each request decodes as one serial stream, so smaller windows finish "
                         "sooner and leave throughput to concurrency; the server itself works "
                         "in 10-page chunks with KV reuse across them.")
    ap.add_argument("--limit-docs", type=int, default=None, help="Only the first N documents")
    ap.add_argument("--max-pages", type=int, default=None, help="Stop after N pages in total (smoke tests)")
    ap.add_argument("--crop-mode", action="store_true",
                    help="Gundam mode: tile each page (more image tokens; for small print)")
    ap.add_argument("--max-side", type=int, default=None,
                    help="Long side (px) pages are sent at. Default 1280 in base mode (the model "
                         "resizes every page to one 1024 px view, so more pixels are discarded) and "
                         "2048 with --crop-mode (tiles use the detail)")
    ap.add_argument("--pdf-dpi", type=int, default=150)
    ap.add_argument("--keep-furniture", action="store_true",
                    help="Keep page numbers, running heads and footers in `markdown`")
    ap.add_argument("--ngram-window", type=int, default=1024,
                    help="Look-back (tokens) of the card's no-repeat n-gram guard for multi-page "
                         "windows (single pages use 128). It runs in Python on every decode step "
                         "for every request, so it bounds throughput; 0 disables the guard "
                         "(and its per-page length cap)")
    ap.add_argument("--retry-errors", action="store_true", help="Re-admit windows recorded as errors")
    ap.add_argument("--no-repair", action="store_true",
                    help="Skip the repair pass (re-OCR of looping pages one at a time, in gundam mode)")
    ap.add_argument("--repair-only", action="store_true",
                    help="Run only the repair pass over an existing output (same input/output args)")
    ap.add_argument("--prepare-workers", type=int, default=8,
                    help="Threads decoding/encoding pages ahead of the engine")
    ap.add_argument("--dp-size", type=int, default=1,
                    help="SGLang data-parallel replicas (one per GPU on multi-GPU flavors). The n-gram "
                         "guard makes each scheduler CPU-bound, so replicas multiply throughput")
    ap.add_argument("--endpoint", default=None,
                    help="Use a running server (e.g. a Jobs --expose URL + /v1) instead of booting one")
    ap.add_argument("--server-python", nargs="+", default=["/opt/sglang/bin/python3", "/usr/bin/python3"],
                    help="Fallback Pythons to try after PATH (the one with sglang wins)")
    args = ap.parse_args()
    if args.max_side is None:
        # Payload matters as much as pixels: 32 concurrent 16-page bodies at 1600 px
        # (~14 MB of base64 each) stalled SGLang's HTTP front end past saturate's fixed
        # 30 s connect/write timeouts.
        args.max_side = 2048 if args.crop_mode else 1280

    import fsspec
    from saturate import Auto, Engine, pump

    rows = list(list_windows(args.input, args.pages_per_request, args.limit_docs,
                             args.max_pages, args.pdf_dpi))
    n_pages = sum(len(r["paths"]) for _, r in rows)
    print(f"{len({r['doc_id'] for _, r in rows})} documents, {n_pages} pages, "
          f"{len(rows)} windows of <= {args.pages_per_request} pages", file=sys.stderr)
    fs = fsspec.core.url_to_fs(args.input.split("*")[0] or ".")[0]
    processor = ngram_processor()

    def to_request(row):
        max_side = row.get("max_side", args.max_side)
        images = [load_page(fs, p, max_side, args.pdf_dpi) for p in row["paths"]]
        n = len(images)
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
                   for b in images]
        content.append({"type": "text", "text": "<PAGE><image>" * n + "Multi page parsing."})
        body = {
            "model": SERVING["served_name"],
            "messages": [{"role": "user", "content": content}],
            "max_tokens": SERVING["tokens_per_page"] * n,  # global budget for the window
            "temperature": SERVING["temperature"],
            "skip_special_tokens": False,  # <PAGE>/<BLOCK> markers are the structure
            "images_config": {"crop_mode": row.get("crop", args.crop_mode or SERVING["crop_mode"])},
        }
        if args.ngram_window > 0:
            body["custom_logit_processor"] = processor
            body["custom_params"] = {
                "ngram_size": SERVING["ngram_size"],
                "window_size": args.ngram_window if n > 1 else 128,
                "page_token_id": SERVING["page_token_id"],
                "eos_token_id": SERVING["eos_token_id"],
                "max_page_tokens": SERVING["max_page_tokens"],
            }
        return body

    def parse(row, body):
        raw = body["choices"][0]["message"]["content"]
        raw_pages = raw.split("<PAGE>")[1:] if "<PAGE>" in raw else [raw]
        finish = body["choices"][0].get("finish_reason")
        # Decoding is greedy, so a truncated or page-misaligned window comes back the same
        # on retry: keep its text and flag it (`status`) instead of an error row.
        # Only transport failures become error rows (--retry-errors).
        if finish == "length":
            status = "truncated"
        elif len(raw_pages) != len(row["paths"]):
            status = "page_mismatch"
        else:
            status = "ok"
        cleaned = [clean_page(p, args.keep_furniture) for p in raw_pages]
        pages = [text for text, _ in cleaned]
        usage = body.get("usage") or {}
        return {
            "doc_id": row["doc_id"],
            "first_page": row["first_page"],
            "page_files": row["page_files"],
            "num_pages": len(row["paths"]),
            "status": status,
            "finish_reason": finish,
            # aligned with page_files only when status == "ok"
            "page_markdown": pages if status == "ok" else None,
            "markdown": convert(raw),
            "raw": raw,
            "model": SERVING["model"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    def parse_repair(row, body):
        raw = body["choices"][0]["message"]["content"]
        finish = body["choices"][0].get("finish_reason")
        page_raw = raw.split("<PAGE>", 1)[-1]
        usage = body.get("usage") or {}
        return {
            "doc_id": row["doc_id"],
            "page": row["page"],
            "page_file": row["page_file"],
            "window": row["window"],
            # still looping alone: keep the text, flagged, for a human or another model
            "status": "degenerate" if finish == "length" or degenerate(page_raw) else "ok",
            "finish_reason": finish,
            "markdown": clean_page(page_raw, args.keep_furniture)[0],
            "raw": raw,
            "mode": "gundam",
            "model": SERVING["model"],
            "completion_tokens": usage.get("completion_tokens"),
        }

    is_repo_id = re.fullmatch(r"[\w-][\w.-]*/[\w.-]*[\w-]", args.output) and not os.path.exists(args.output)
    output = f"hf://datasets/{args.output}/data" if is_repo_id else args.output
    repairs_out = re.sub(r"/data/?$", "/repairs", output) if re.search(r"/data/?$", output) \
        else output.rstrip("/") + "-repairs"
    # One window is minutes of decode, and Auto widens only on completed-request
    # evidence, so it ramps slowly from a small start. Start near the measured
    # headroom (a100-large: 8 x 32-page windows used ~25% of KV) and flush often,
    # since each row is expensive to redo.
    window = Auto(initial=32, max_limit=64)
    # saturate caps a row's retries at RETRY_BUDGET_S (300 s, counted from the first
    # attempt), which is shorter than one legitimate multi-page generation here, so a
    # transient first-attempt failure always ended in a TimeoutError row. Match the
    # read timeout until saturate exposes this as a pump() argument.
    import saturate.transport

    saturate.transport.RETRY_BUDGET_S = 3600.0

    def run_passes(endpoint, headers=None):
        common = {"retry_errors": args.retry_errors, "headers": headers,
                  "prepare_workers": args.prepare_workers, "read_timeout": 3600}
        if not args.repair_only:
            stats = pump(rows, to_request, parse, endpoint, output, window=window, flush_every=4, **common)
            print(f"{output} ({stats.rows_processed} ok, {stats.rows_failed} error rows)", file=sys.stderr)
            print("UNLIMITED_OCR2_SATURATE " + stats.to_json(), flush=True)
        if args.no_repair:
            return
        targets = list(repair_rows(rows, output))
        print(f"repair pass: {len(targets)} pages to re-OCR alone (gundam) -> {repairs_out}", file=sys.stderr)
        if targets:
            rstats = pump(targets, to_request, parse_repair, endpoint, repairs_out,
                          window=Auto(initial=32, max_limit=64), flush_every=16, **common)
            print("UNLIMITED_OCR2_REPAIR " + rstats.to_json(), flush=True)

    if args.endpoint:
        run_passes(args.endpoint, {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
                   if os.environ.get("HF_TOKEN") else None)
    else:
        py = find_server_python(args.server_python)
        print(f"server python: {py}", file=sys.stderr)
        # The image's venv may have no pip (uv-built); uv is always present under `uv run`.
        has_pip = subprocess.run([py, "-m", "pip", "--version"], check=False,
                                 capture_output=True).returncode == 0
        pip = [py, "-m", "pip", "install", "-q"] if has_pip else ["uv", "pip", "install", "-q", "--python", py]
        pip_env = {**os.environ, "PIP_BREAK_SYSTEM_PACKAGES": "1"}
        pip_env.pop("VIRTUAL_ENV", None)  # never install into this script's venv
        subprocess.run([*pip, "--no-deps", SERVING["wheel"]], check=True, env=pip_env)
        subprocess.run([*pip, *SERVING["extra_pip"]], check=True, env=pip_env)
        os.environ.update(SERVING["env"])  # inherited by the server process
        cmd = [py, "-m", "sglang.launch_server", "--model-path", SERVING["model"],
               "--served-model-name", SERVING["served_name"], *SERVING["serve_args"],
               "--enable-metrics", "--host", "127.0.0.1", "--port", "8000"]
        if args.dp_size > 1:
            cmd += ["--dp-size", str(args.dp_size)]
        with Engine(SERVING["model"], engine="sglang", cmd=cmd) as endpoint:
            run_passes(endpoint)


# ---------------------------------------------------------------------------
# Vendored verbatim from Baidu's Unlimited-OCR repo (MIT License, Copyright (c) 2026 Baidu):
# https://github.com/baidu/Unlimited-OCR/blob/0cdd7877868bd0e17715d742911b9ce937c53d3f/utils/process_unlimitedOCR2.py
# Lines 30-353 (constants, helpers, convert()); only the file's CLI main() is left out.
# To update: re-copy those lines from a newer commit and bump the hash above.
# Kept byte-for-byte, so its two style-rule hits are waived file-wide:
# ruff: noqa: FURB167, SIM114
# ---------------------------------------------------------------------------

# ── block / token patterns ───────────────────────────────────────────────────
PAGE_SPLIT_RE = re.compile(r'<\s*/?\s*PAGE\s*\d*\s*>', re.I)
MERGE_RE = re.compile(r'<\s*/?\s*MERGE\s*>', re.I)
STRAY_TOKEN_RE = re.compile(
    r'<[|｜]/?[a-zA-Z_]+[|｜]>|<image>'
    r'|<\s*/?\s*PAGE\s*\d*\s*>|<\s*/?\s*BLOCK\s*>|<\s*/?\s*FORK\s*>'
    r'|<\s*/?\s*LEVEL\s*\d*\s*[>}]?|<\s*/?\s*MERGE\s*>', re.I)
# <BLOCK>label [<LEVELn>|[bbox]] [<LEVELn>|[bbox]] [<FORK>] ; LEVEL-first or bbox-first.
BLOCK_RE = re.compile(
    r'<\s*BLOCK\s*>\s*([a-zA-Z_]+)'
    r'(?:'
    r'\s*<\s*/?\s*LEVEL\s*(\d*)\s*[>}]?(?:\s*\[\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)'
    r'(?:\s*[,，]\s*-?\d+)*\s*\])?'
    r'|'
    r'\s*\[\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)'
    r'(?:\s*[,，]\s*-?\d+)*\s*\](?:\s*<\s*/?\s*LEVEL\s*(\d*)\s*[>}]?)?'
    r')?'
    r'\s*(?:<\s*FORK\s*>)?', re.I)
TR_RE = re.compile(r'<tr\b[^>]*>.*?</tr>', re.I | re.S)
TABLE_OPEN_RE = re.compile(r'<table\b[^>]*>', re.I)
SPAN_ATTR_RE = re.compile(r'\b(colspan|rowspan)\s*=\s*"([^"]*)"', re.I)

DROP_LABELS = set()                       # header/footer/page_number kept (canonical default)
IMAGE_LABELS = {'image', 'figure', 'picture', 'chart', 'img', 'seal'}
NOISE_CONTENT = {'[non-text]', '[unreadable]', '[image]', '[figure]'}
CHAIN_TYPES = {'text', 'table'}
MAX_HEADER_ROWS = 5

_BASE_LABELS = {
    'text', 'title', 'header', 'footer', 'page_number', 'image', 'figure',
    'chart', 'seal', 'table', 'image_caption', 'image_footnote',
    'page_footnote', 'equation', 'ref_text', 'aside_text', 'algorithm',
}
_BARE_LABELS = _BASE_LABELS | {
    f'{p}{l}' for p in ('cross_', 'append_') for l in _BASE_LABELS}
BARE_BLOCK_RE = re.compile(
    r'(?:(?<=<FORK>)|(?<=<PAGE>)|(?<=\n)|^)'
    r'(?:<[|｜]\s*det\s*[|｜]>)?'
    r'((?:[a-zA-Z_][a-zA-Z_0-9]*)(?:\s*<\s*/?\s*LEVEL\s*\d*\s*[>}]?)?\s*'
    r'\[\s*-?\d+(?:\s*[,，]\s*-?\d+){3,}[^\]]*\])', re.I)


def _scale(v):
    return max(0, min(1000, int(v)))


def _fix_span_attrs(text):
    def repl(m):
        name, val = m.group(1), m.group(2)
        try:
            int(val)
            return m.group(0)
        except ValueError:
            num = re.search(r'\d+', val)
            return f'{name}="{num.group() if num else 1}"'
    return SPAN_ATTR_RE.sub(repl, text)


def _fix_bare_blocks(text):
    """Add the missing <BLOCK> prefix to bare block declarations (known labels only)."""
    def _repl(m):
        label = re.match(r'[a-zA-Z_][a-zA-Z_0-9]*', m.group(1)).group(0)
        if label.lower() not in _BARE_LABELS:
            return m.group(0)
        return f'<BLOCK>{m.group(1)}'
    text = BARE_BLOCK_RE.sub(_repl, text)
    return re.sub(r'\n\s*<FORK>\s*(?=<BLOCK>)', '\n', text, flags=re.I)


def _base_type(label):
    """cross_text/append_text -> text; append_table/cross_table -> table."""
    for p in ('append_', 'cross_'):
        if label.startswith(p):
            return label[len(p):]
    return label


def _parse_blocks(text, page_base=1):
    """Split on <PAGE>; return blocks [{label,bbox,level,content,page,merge_prev}]."""
    text = _fix_bare_blocks(text)
    segments = PAGE_SPLIT_RE.split(text)
    if segments and not segments[0].strip():
        segments = segments[1:]
    blocks = []
    for i, seg in enumerate(segments):
        page_no = page_base + i
        if not seg.strip():
            continue
        matches = list(BLOCK_RE.finditer(seg))
        if not matches:
            blocks.append({'label': '__raw__', 'bbox': None, 'level': None,
                           'content': MERGE_RE.sub('', seg).strip(),
                           'page': page_no, 'merge_prev': False})
            continue
        for j, m in enumerate(matches):
            gap = seg[:m.start()] if j == 0 else seg[matches[j - 1].end():m.start()]
            merge_prev = bool(MERGE_RE.search(gap))
            if j == 0:
                pre = STRAY_TOKEN_RE.sub('', MERGE_RE.sub('', gap)).strip()
                if pre:
                    blocks.append({'label': '__raw__', 'bbox': None, 'level': None,
                                   'content': pre, 'page': page_no, 'merge_prev': False})
            end = matches[j + 1].start() if j + 1 < len(matches) else len(seg)
            content = MERGE_RE.sub('', seg[m.end():end]).strip()
            if m.group(2) is not None:          # LEVEL-first
                level = int(m.group(2)) if m.group(2) else None
                bbox = (tuple(int(m.group(k)) for k in range(3, 7))
                        if m.group(3) is not None else None)
            else:                                # bbox-first (or none)
                bbox = (tuple(int(m.group(k)) for k in range(7, 11))
                        if m.group(7) is not None else None)
                level = int(m.group(11)) if m.group(11) else None
            blocks.append({'label': m.group(1).lower(), 'bbox': bbox, 'level': level,
                           'content': content, 'page': page_no, 'merge_prev': merge_prev})
    return blocks

def _merge_adjacent_titles(blocks):
    """Merge strictly-adjacent, same-page, same-<LEVELn> title blocks into one."""
    if not blocks:
        return blocks
    out = [blocks[0]]
    for blk in blocks[1:]:
        prev = out[-1]
        if (blk['label'] == 'title' and prev['label'] == 'title'
                and blk['page'] == prev['page']
                and blk['level'] is not None and prev['level'] is not None
                and blk['level'] == prev['level']):
            prev['content'] = '\n'.join(x for x in (prev['content'], blk['content']) if x)
            continue
        out.append(blk)
    return out


def _build_groups(blocks):
    """Union-find grouping of append/cross chains and same-page <MERGE> columns."""
    n = len(blocks)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pages_with = {}
    for blk in blocks:
        b = _base_type(blk['label'])
        if blk['label'] != '__raw__' and b in CHAIN_TYPES:
            pages_with.setdefault(b, set()).add(blk['page'])

    def _table_adjacent(tail_page, cur_page):
        if cur_page - tail_page <= 1:
            return True
        return all(p not in pages_with.get('table', ())
                   for p in range(tail_page + 1, cur_page))

    open_chain = {}                       # base type -> current chain-tail index
    for i, blk in enumerate(blocks):
        label, base = blk['label'], _base_type(blk['label'])
        if label == '__raw__' or base not in CHAIN_TYPES:
            continue
        if label.startswith('append_'):
            tail = open_chain.get(base)
            if tail is None:
                continue
            if base == 'table' and not _table_adjacent(blocks[tail]['page'], blk['page']):
                open_chain.pop(base, None)
                continue
            union(tail, i)
            open_chain[base] = i
        elif label.startswith('cross_'):
            open_chain[base] = i
        elif base == 'table':
            open_chain[base] = i

    for i, blk in enumerate(blocks):       # same-page <MERGE> column edges
        if not blk.get('merge_prev') or i == 0:
            continue
        pblk = blocks[i - 1]
        if pblk['page'] != blk['page']:
            continue
        if (blk['label'] == 'title' or _base_type(blk['label']) == 'text') \
                and (pblk['label'] == 'title' or _base_type(pblk['label']) == 'text'):
            union(i - 1, i)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [groups[r] for r in sorted(groups, key=lambda r: groups[r][0])]


def _row_key(tr_html):
    txt = re.sub(r'<[^>]+>', ' ', tr_html)
    txt = ''.join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in txt)
    ncells = len(re.findall(r'<t[hd]\b', tr_html, re.I))
    return (ncells, ''.join(txt.split()).lower())


def _merge_table_fragments(frags):
    """Merge cross-page table fragments, skipping repeated leading header rows."""
    frags = [f for f in frags if f and f.strip()]
    if len(frags) <= 1:
        return frags[0] if frags else ''
    base_rows = TR_RE.findall(frags[0])
    if not base_rows:
        return '\n\n'.join(frags)
    out_rows, base_keys, extra = list(base_rows), [_row_key(r) for r in base_rows], []
    for frag in frags[1:]:
        rows = TR_RE.findall(frag)
        if not rows:
            extra.append(frag.strip())
            continue
        skip = 0
        for r in rows[:MAX_HEADER_ROWS]:
            if skip < len(base_keys) and _row_key(r) == base_keys[skip]:
                skip += 1
            else:
                break
        out_rows.extend(rows[skip:])
    m = TABLE_OPEN_RE.search(frags[0])
    merged = (m.group(0) if m else '<table>') + ''.join(out_rows) + '</table>'
    return merged + ('\n\n' + '\n\n'.join(extra) if extra else '')

def _render_blocks(blocks, groups):
    """Render each group to a Markdown item (figure/table/heading/text)."""
    items = []
    for members in groups:
        head = blocks[members[0]]
        base = head['label'] if head['label'] == '__raw__' else _base_type(head['label'])
        contents = [blocks[k]['content'] for k in members
                    if blocks[k]['content'] and blocks[k]['content'].lower() not in NOISE_CONTENT]
        if base == '__raw__':
            if contents:
                items.append(contents[0])
            continue
        if base in IMAGE_LABELS:
            if head['bbox'] is not None:
                x1, y1, x2, y2 = (_scale(v) for v in head['bbox'])
                items.append(f'![](page{head["page"]}_{x1}_{y1}_{x2}_{y2}.jpg)')
            if contents:
                items.append(contents[0])
            continue
        if not contents or base in DROP_LABELS:
            continue
        if base == 'title':
            lv = max(1, head['level'] or 1)
            joined = ''.join(contents)
            if not STRAY_TOKEN_RE.sub('', joined).strip():
                continue
            items.append(f'{"#" * lv} {joined}')
            continue
        if base == 'table':
            items.append(_merge_table_fragments(contents) if len(contents) > 1 else contents[0])
            continue
        items.append(''.join(contents))       # text family: concatenate, no separator
    return items


def _head_level(item):
    s = item.lstrip()
    return len(s) - len(s.lstrip('#'))


def _single_root_titles(items):
    """Collapse multiple top-level headings under the first one (single root)."""
    idxs = [i for i, it in enumerate(items) if it.lstrip().startswith('#')]
    if len(idxs) <= 1:
        return items
    levels = [_head_level(items[i]) for i in idxs]
    min_lv = min(levels)
    if sum(1 for lv in levels if lv == min_lv) <= 1:
        return items
    first_lv, out = levels[0], list(items)
    for k, i in enumerate(idxs):
        body = items[i].lstrip()[levels[k]:].lstrip(' ')
        new_lv = 1 if k == 0 else max(2, levels[k] - first_lv + 2)
        out[i] = '#' * new_lv + ' ' + body
    return out


def _fold_head_ws(items):
    out = []
    for it in items:
        s = it.lstrip()
        if s.startswith('#'):
            n = len(s) - len(s.lstrip('#'))
            it = '#' * n + ' ' + re.sub(r'\s+', ' ', s[n:]).strip()
        out.append(it)
    return out


def _drop_empty_heads(items):
    return [it for it in items
            if not (it.lstrip().startswith('#') and not it.lstrip().lstrip('#').strip())]


def convert(result):
    """Unlimited-OCR 2 raw document string -> MarathonDocBench evaluation Markdown."""
    text = (result or '').strip()
    if text.startswith('```markdown'):
        text = text[len('```markdown'):].strip()
    if text.endswith('```'):
        text = text[:-3].strip()

    blocks = _parse_blocks(text, page_base=1)
    blocks = _merge_adjacent_titles(blocks)
    groups = _build_groups(blocks)
    items = _drop_empty_heads(_render_blocks(blocks, groups))
    items = _fold_head_ws(items)
    text = '\n\n'.join(items)

    text = text.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    text = STRAY_TOKEN_RE.sub('', text)
    text = _fix_span_attrs(text)
    text = re.sub(r'(?m)^\s*\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]\s*$', '', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return '\n\n'.join(_single_root_titles(text.split('\n\n')))


if __name__ == "__main__":
    main()
