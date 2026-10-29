# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf] @ git+https://github.com/davanstrien/saturate@main",
#     "pillow>=10",
# ]
#
# [tool.hf-jobs]
# image = "lmsysorg/sglang:v0.5.20-cu130"
# flavor = "rtx-pro-6000"
# timeout = "4h"
# secrets = ["HF_TOKEN"]
# ///
"""
OCR whole documents (book volumes, folders of scans) with Baidu Unlimited-OCR-2.

Unlimited-OCR-2 reads many pages in ONE request and keeps context across page
breaks: a paragraph that runs onto the next page comes back tagged as a
continuation, not as two fragments. This script groups page images into
documents and sends 16 consecutive pages per request.

  hf jobs uv run --detach \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/unlimited-ocr2.py \\
      "hf://buckets/biglam/britannica/source/pages/1771EncyclopediaBritannicaNLS/*.jp2" \\
      <user>/britannica-1771

The [tool.hf-jobs] table above sets the image, GPU, timeout and HF_TOKEN.
The GPU needs >= 48 GB: on 24 GB cards a 16-page request (~31k tokens of KV)
thrashes the cache. rtx-pro-6000 was the fastest per dollar in our tests.

Input: a glob or folder of page images (a Bucket, a mounted dir, a local dir).
A document is the parent folder plus the file name without its trailing page
number (`Volume 1, A-B_0007.jp2` -> `Volume 1, A-B`); pages sort by that number.

Output: `data/part-*.parquet`, one row per 16-page window:
`{id, doc_id, first_page, page_files, status, page_markdown, raw, ...}`.
`page_markdown` lines up with `page_files`; `raw` is the model's output with its
layout blocks and boxes. `status` is `truncated` or `page_mismatch` when the model
ran out of budget or lost a page boundary (then `page_markdown` is empty and only
`raw` is kept). Re-running the same command resumes where it stopped.

About 2% of dense 18th-century pages (mostly mathematics) send the model into a
loop; a page stops at `--max-page-tokens` (8000). `unlimited-ocr2-saturate.py` is the
tuned version of this script: it re-reads those pages one at a time, reads PDFs and adds
Baidu's own Markdown post-processing. It may change more often.

How it runs: Unlimited-OCR-2 needs Baidu's own SGLang build (no vLLM support).
The script installs that wheel into the image's Python, starts the server in the
Job, and sends requests with saturate (adaptive concurrency, resumable output).

Model: https://huggingface.co/baidu/Unlimited-OCR-2 (3.3B, MIT)
"""

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys

# The model card's launch settings. The image is the SGLang release whose pins
# match Baidu's wheel (torch 2.13.0, transformers 5.12.1); move both together. The wheel URL
# names a model-repo commit: Baidu replaces the wheel on main (dev10 was removed 2026-10-29).
SERVING = {
    "model": "baidu/Unlimited-OCR-2",
    "served_name": "Unlimited-OCR-2",
    "wheel": "https://huggingface.co/baidu/Unlimited-OCR-2/resolve/91da6546d8ed91e02525f59c012a1d243266d7dd/wheel/"
             "sglang-0.5.20.dev10%2Bg8d62f5fb9c-py3-none-any.whl",
    "extra_pip": ["fastokens==0.3.2"],
    "env": {
        "SGLANG_UOCR_EANABLE_DOC_LIMIT": "1",  # (sic) upstream spelling
        "SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW": "10",
        "SGLANG_UOCR_DOC_BASE_PAGE_WINDOW": "10",
        # Per-page output cap; --max-page-tokens sets it (default 8000).
        "SGLANG_UOCR_MAX_PAGE_TOKENS": "8000",
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
}

# The card's no-repeat n-gram guard: DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),
# i.e. json({"callable": dill.dumps(cls).hex()}). dill pickles an importable class
# by reference (module + name), so this is a fixed string and needs no sglang here.
NGRAM_PROCESSOR = json.dumps({"callable": (
    "80059559000000000000008c2a73676c616e672e7372742e73616d706c696e672e637573746f6d5f6c6f6769"
    "745f70726f636573736f72948c26446565707365656b4f43524e6f5265706561744e4772616d4c6f67697450"
    "726f636573736f729493942e")})

PAGE_EXTS = (".jp2", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
FURNITURE = {"page_number", "header", "footer"}
BLOCK_RE = re.compile(r"<BLOCK>(?P<type>[a-z_]+)(?:<LEVEL(?P<level>\d)>)?\[[\d, ]*\]<FORK>")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def doc_key(rel_path: str) -> str:
    """`item/Volume 1, A-B_0007.jp2` -> `item/Volume 1, A-B`; bare `0007.jp2` -> `.`."""
    stem = rel_path.rsplit(".", 1)[0]
    doc = re.sub(r"[_\-. ]?\d+$", "", stem)
    return doc if doc and not doc.endswith("/") else (doc or ".")


def list_windows(pattern: str, pages_per_request: int, max_pages):
    """List page files, group them into documents, cut each into windows -> (id, row)."""
    from saturate import bucket_rows

    if not re.search(r"[*?\[]", pattern):
        pattern = pattern.rstrip("/") + "/**"
    docs: dict[str, list[tuple[str, str]]] = {}
    for rid, row in bucket_rows(pattern, read=False):
        if rid.lower().endswith(PAGE_EXTS):
            docs.setdefault(doc_key(rid), []).append((rid, row["path"]))
    if not docs:
        sys.exit(f"no page images matched {pattern!r}")

    n_pages = 0
    for doc in sorted(docs, key=natural_key):
        pages = sorted(docs[doc], key=lambda p: natural_key(p[0]))
        for start in range(0, len(pages), pages_per_request):
            if max_pages is not None and n_pages >= max_pages:
                return
            window = pages[start: start + pages_per_request]
            if max_pages is not None:
                window = window[: max_pages - n_pages]
            n_pages += len(window)
            # The id encodes the page range, so a --max-pages-truncated window is not
            # mistaken for the full window when a later run resumes.
            yield f"{doc}@{start:06d}-{start + len(window):06d}", {
                "doc_id": doc, "first_page": start,
                "page_files": [rid for rid, _ in window],
                "paths": [p for _, p in window],
            }


def load_page(fs, path: str, max_side: int) -> str:
    """Read one page image, downscale it, return base64 JPEG."""
    from PIL import Image

    with fs.open(path, "rb") as f:
        data = f.read()
    img = Image.open(io.BytesIO(data))
    if img.format == "JPEG2000":
        # Decode straight at a lower wavelet level while the page stays >= max_side: close
        # to the full decode after downscaling, for a fraction of the CPU. Some codestreams fail
        # at a reduced level, so step back towards a full decode until one works.
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


def clean_page(raw_page: str, keep_furniture: bool) -> str:
    """One page of model output -> plain markdown.

    Drops block tags and boxes, turns titles into `#` headings, drops page
    numbers and running heads, and joins `<MERGE>`d blocks (e.g. verse lines).
    """
    blocks: list[str] = []
    merge_next = False
    matches = list(BLOCK_RE.finditer(raw_page))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_page)
        body = raw_page[m.end(): end].strip()
        merge_prev, merge_next = merge_next, body.endswith("<MERGE>")
        body = body.removesuffix("<MERGE>").strip()
        kind = m.group("type")
        if not body or (kind in FURNITURE and not keep_furniture):
            continue
        if kind == "title" and m.group("level"):
            body = "#" * int(m.group("level")) + " " + body
        if merge_prev and blocks:
            blocks[-1] += "\n" + body
        else:
            blocks.append(body)
    return "\n\n".join(blocks)


def prepare_workers() -> int:
    """Threads that decode and encode pages: enough to keep ~40 requests fed, within the Job's CPU quota.

    os.cpu_count() reports the host's cores (190 on a Jobs node), not the quota; that many
    threads starve the loop that sends requests. Jobs sets CPU_CORES to the quota.
    """
    try:
        cores = int(os.environ.get("CPU_CORES", ""))
    except ValueError:
        cores = 0
    if cores <= 0:
        try:
            cores = len(os.sched_getaffinity(0))
        except AttributeError:  # not Linux
            cores = os.cpu_count() or 1
    return max(4, min(16, cores - 6))  # leave cores for SGLang's tokenizer and scheduler


def start_server_cmd() -> list[str]:
    """Install Baidu's wheel into the image's sglang Python and return the server command."""
    # The SGLang image keeps sglang in its own venv (/opt/sglang/bin), first on PATH;
    # `uv run` puts this script's venv in front, so look past it.
    own_bin = os.path.realpath(os.path.join(sys.prefix, "bin"))
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                           if os.path.realpath(p) != own_bin)
    py = shutil.which("python3", path=path)
    if not py or subprocess.run([py, "-c", "import sglang"], check=False,
                                capture_output=True).returncode != 0:
        sys.exit("No Python with sglang found: run this on the lmsysorg/sglang image "
                 "(`hf jobs uv run` picks it up from the script's [tool.hf-jobs] table).")
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    pip = ["uv", "pip", "install", "-q", "--python", py]
    subprocess.run([*pip, "--no-deps", SERVING["wheel"]], check=True, env=env)
    subprocess.run([*pip, *SERVING["extra_pip"]], check=True, env=env)
    os.environ.update(SERVING["env"])  # inherited by the server process
    return [py, "-m", "sglang.launch_server", "--model-path", SERVING["model"],
            "--served-model-name", SERVING["served_name"], *SERVING["serve_args"],
            "--enable-metrics", "--host", "127.0.0.1", "--port", "8000"]


def main():
    ap = argparse.ArgumentParser(description="Unlimited-OCR-2 document OCR on HF Jobs")
    ap.add_argument("input", help="Glob or folder of page images: hf://buckets/..., a mounted dir, ...")
    ap.add_argument("output", help="Output dataset id (org/name) or fsspec URI (hf://buckets/...)")
    ap.add_argument("--pages-per-request", type=int, default=16,
                    help="Consecutive pages per request (default 16)")
    ap.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (for a test run)")
    ap.add_argument("--max-side", type=int, default=1280,
                    help="Long side (px) pages are sent at. The model resizes each page to a "
                         "1024 px view, so more pixels only make the requests bigger")
    ap.add_argument("--keep-furniture", action="store_true",
                    help="Keep page numbers and running heads in page_markdown")
    ap.add_argument("--max-page-tokens", type=int, default=8000,
                    help="Output cap per page (default 8000). A book page is ~2,000 characters; "
                         "past 8000 tokens it is almost always a repetition loop. Raise it (up to "
                         "20000) for very dense pages: large tables, mathematics")
    ap.add_argument("--retry-errors", action="store_true", help="Retry windows recorded as errors")
    args = ap.parse_args()
    if args.max_side <= 0:
        ap.error("--max-side must be a positive number of pixels")
    if not 0 < args.max_page_tokens <= 20000:
        ap.error("--max-page-tokens must be between 1 and 20000")
    SERVING["env"]["SGLANG_UOCR_MAX_PAGE_TOKENS"] = str(args.max_page_tokens)

    import fsspec
    from saturate import Auto, Engine, pump

    rows = list(list_windows(args.input, args.pages_per_request, args.max_pages))
    print(f"{len({r['doc_id'] for _, r in rows})} documents, "
          f"{sum(len(r['paths']) for _, r in rows)} pages, {len(rows)} requests", file=sys.stderr)
    fs = fsspec.core.url_to_fs(args.input.split("*")[0] or ".")[0]

    def to_request(row):
        n = len(row["paths"])
        content = [{"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{load_page(fs, p, args.max_side)}"}}
                   for p in row["paths"]]
        content.append({"type": "text", "text": "<PAGE><image>" * n + "Multi page parsing."})
        # Sampling and guard settings from the model card's client example.
        return {
            "model": SERVING["served_name"],
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max(8000, args.max_page_tokens) * n,  # output budget for the whole window
            "temperature": 0,
            "skip_special_tokens": False,  # <PAGE>/<BLOCK> markers are the structure
            "images_config": {"crop_mode": False},  # base mode: one 1024 px view per page
            "custom_logit_processor": NGRAM_PROCESSOR,
            "custom_params": {
                "ngram_size": 35, "window_size": 1024 if n > 1 else 128,
                "page_token_id": 128827, "eos_token_id": 1, "max_page_tokens": args.max_page_tokens,
            },
        }

    def parse(row, body):
        choice = body["choices"][0]
        raw = choice["message"]["content"]
        raw_pages = raw.split("<PAGE>")[1:]
        if choice.get("finish_reason") == "length":
            status = "truncated"
        elif len(raw_pages) != len(row["paths"]):
            status = "page_mismatch"
        else:
            status = "ok"
        usage = body.get("usage") or {}
        return {
            "doc_id": row["doc_id"],
            "first_page": row["first_page"],
            "page_files": row["page_files"],
            "status": status,
            "page_markdown": [clean_page(p, args.keep_furniture) for p in raw_pages]
            if status == "ok" else None,
            "raw": raw,
            "model": SERVING["model"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    is_repo_id = re.fullmatch(r"[\w-][\w.-]*/[\w.-]*[\w-]", args.output) and not os.path.exists(args.output)
    output = f"hf://datasets/{args.output}/data" if is_repo_id else args.output

    # One request is minutes of decoding: read_timeout=3600 below, and a row's retries get
    # 2 x read_timeout in total (saturate's default retry budget).
    with Engine(SERVING["model"], engine="sglang", cmd=start_server_cmd()) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, output,
                     window=Auto(initial=32, max_limit=64),  # long requests: start wide
                     flush_every=4,  # each row is minutes of GPU time; write often
                     read_timeout=3600, prepare_workers=prepare_workers(), retry_errors=args.retry_errors)
    print(f"{output}: {stats.rows_processed} windows written, {stats.rows_failed} errors", file=sys.stderr)
    print("UNLIMITED_OCR2 " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
