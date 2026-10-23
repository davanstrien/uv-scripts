# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pillow",
#     "imagecodecs",
#     "toolz",
#     "bucketbag @ https://github.com/davanstrien/bucketbag/archive/refs/tags/v0.3.1.tar.gz",
#     # LightOn's official client, pinned to a commit (not on PyPI). Used for its
#     # parser, page resize, PDF rendering and dollar-delimiter repair.
#     "lightonocr @ https://github.com/lightonai/LightOnOCR/archive/36755d461be079737860a5f03ae0c803501269e9.tar.gz",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.30.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "a10g-small"
# secrets = ["HF_TOKEN"]
# ///

"""
OCR images and PDFs from an HF Bucket (or a directory) using LightOnOCR-3,
writing markdown files.

Bucket sibling of lighton-ocr3.py: same model, prompt, sampling and post-processing.
Writes one .md per image (or per PDF page), preserving the key/directory structure.
In grounding mode each .md gets a .json sidecar with the parsed layout blocks.

INPUT and OUTPUT are each either
  - an `hf://buckets/<ns>/<bucket>[/<prefix>]` URL: read with bucketbag (list, then
    download one batch at a time to scratch, no mount) / written with one upload call
    per batch. Use this for large buckets: mounting a very large bucket can be slow.
  - a directory: a local folder, or a bucket mounted with `-v`.

Input:                          Output:
  /input/page1.jp2        →      /output/page1.md   (+ page1.json with --mode grounding)
  /input/report.pdf       →      /output/report/page_001.md
  (3 pages)                       /output/report/page_002.md
                                  /output/report/page_003.md
                                  /output/report/.complete.plain
  /input/sub/photo.jpg    →      /output/sub/photo.md

Pages that already have their output are skipped, so re-running the same command
resumes a stopped run. A page that fails writes nothing and is retried on the next
run. A PDF gets a `.complete.<mode>` marker in its output folder once every page is
written; until then, a re-run OCRs only its missing pages. --overwrite redoes
everything, replacing outputs as it goes: a page that fails keeps its previous output
(the run lists those inputs), so re-run them with --overwrite or use a fresh output prefix.

Examples:

  # Bucket to bucket on HF Jobs (image, flavor and HF_TOKEN come from the
  # [tool.hf-jobs] header; needs hf CLI 1.32+). One folder of a large bucket:
  hf jobs uv run \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr3-bucket.py \\
      hf://buckets/biglam/britannica/source/pages/00070476bsb \\
      hf://buckets/user/ocr-output/britannica --mode grounding --max-samples 20

  # Mounted volumes work too (simplest for small buckets)
  hf jobs uv run \\
      -v hf://buckets/user/ocr-input:/input:ro \\
      -v hf://buckets/user/ocr-output:/output \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr3-bucket.py \\
      /input /output

  # Own GPU (vLLM and torch come from the image on Jobs, so add them here)
  uv run --with vllm==0.30.0 lighton-ocr3-bucket.py ./images ./output

Models: lightonai/LightOnOCR-3-4B (default) or lightonai/LightOnOCR-3-0.8B (Apache-2.0).
See lighton-ocr3.py for the settings and why the vLLM image is pinned.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import torch
from bucketbag import batched_files, iter_keys, put_files
from lightonocr import (
    DEFAULT_LONGEST_EDGE,
    GROUNDING_PROMPT,
    fit,
    fix_escaped_dollars,
    parse_blocks,
    to_markdown,
)
from PIL import Image
from toolz import partition_all

# Disable vLLM's FlashInfer sampler: it JIT-compiles a CUDA kernel needing nvcc, which the
# default uv-script image lacks (engine init then crashes). On the vllm/vllm-openai image it's
# a harmless no-op.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# DeepGEMM's init asserts on a missing nvcc (non-fatal traceback); OCR doesn't need it.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = ["lightonai/LightOnOCR-3-4B", "lightonai/LightOnOCR-3-0.8B"]
DEFAULT_MODEL = MODELS[0]

# Sampling from the model card's vLLM example (temperature 0.2, top_p 0.9, max_tokens 4096).
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 4096

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".jp2", ".j2k"}
JP2_EXTENSIONS = {".jp2", ".j2k"}
BUCKET_SCHEME = "hf://buckets/"


def check_cuda_availability():
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")


def is_bucket(location: str) -> bool:
    return location.startswith(BUCKET_SCHEME)


def as_bucket_dir(url: str) -> tuple[str, str]:
    """Normalise a bucket URL to a directory and return (url, key prefix).

    The trailing slash matters: bucketbag matches prefixes as strings, so
    `.../pages/00070476bsb` would also match `.../pages/00070476bsb2/...`.
    """
    parts = url[len(BUCKET_SCHEME):].strip("/").split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"Not a bucket URL: {url} (expected hf://buckets/<ns>/<bucket>[/<prefix>])")
    prefix = parts[2] + "/" if len(parts) == 3 else ""
    return BUCKET_SCHEME + "/".join(parts) + "/", prefix


def is_input_file(rel: str) -> bool:
    ext = Path(rel).suffix.lower()
    return ext in IMAGE_EXTENSIONS or ext == ".pdf"


def list_inputs(source: str, include: str | None, limit: int | None) -> list[str]:
    """Relative paths of image/PDF inputs, sorted. With `limit`, stops early."""
    rels = []
    if is_bucket(source):
        url, prefix = as_bucket_dir(source)
        keys = iter_keys(url, include=f"{prefix}{include}" if include else None)
        candidates = (k[len(prefix):] for k in keys)
    else:
        root = Path(source)
        paths = root.rglob(include or "*")
        candidates = (str(p.relative_to(root)) for p in (paths if limit else sorted(paths)) if p.is_file())
    for rel in candidates:
        if is_input_file(rel):
            rels.append(rel)
            if limit is not None and len(rels) >= limit:
                break
    return rels


def list_outputs(dest: str) -> set[str]:
    """Relative paths of outputs already written (for resume)."""
    if is_bucket(dest):
        url, prefix = as_bucket_dir(dest)
        return {k[len(prefix):] for k in iter_keys(url)}
    root = Path(dest)
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} if root.is_dir() else set()


def md_rel(rel: str, page: int | None = None) -> str:
    """Output path for an input: x.jp2 -> x.md; a PDF page -> x/page_001.md."""
    stem = str(Path(rel).with_suffix(""))
    return f"{stem}/page_{page:03d}.md" if page else f"{stem}.md"


def marker_rel(rel: str, mode: str) -> str:
    """Written into a PDF's output folder once all its pages are written in this mode."""
    return f"{Path(rel).with_suffix('')}/.complete.{mode}"


def page_done(md: str, existing: set[str], mode: str) -> bool:
    if md not in existing:
        return False
    return mode != "grounding" or md[: -len(".md")] + ".json" in existing


def is_done(rel: str, existing: set[str], mode: str) -> bool:
    if rel.lower().endswith(".pdf"):
        # Grounding needs the .json sidecars, so only a grounding marker will do;
        # plain is satisfied by either (as for images, whose .md exists in both modes).
        modes = ["grounding"] if mode == "grounding" else ["plain", "grounding"]
        return any(marker_rel(rel, m) in existing for m in modes)
    return page_done(md_rel(rel), existing, mode)


def iter_local_batches(source: str, rels: list[str], n: int):
    """Yield batches of (rel, local_path): straight from disk, or downloaded by bucketbag
    (one batch at a time to scratch, deleted as the loop advances)."""
    if not is_bucket(source):
        for chunk in partition_all(n, rels):
            yield [(rel, Path(source) / rel) for rel in chunk]
        return
    url, prefix = as_bucket_dir(source)
    keys = [prefix + rel for rel in rels]
    for batch in batched_files(url, keys=keys, n=n, prefetch=2):
        yield [(it.key[len(prefix):], it.path) for it in batch]


def open_image(path: Path) -> Image.Image:
    """Open one image as RGB. JPEG-2000 falls back to imagecodecs when Pillow
    was built without OpenJPEG."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        if path.suffix.lower() not in JP2_EXTENSIONS:
            raise
        import imagecodecs

        return Image.fromarray(imagecodecs.imread(str(path))).convert("RGB")


class PageItem(NamedTuple):
    rel: str  # input file
    md: str | None  # output .md path; None for a marker-only item
    image: Image.Image | None
    last: bool  # last page still to do for a PDF: write its .complete.<mode> marker after it


def iter_pages(rel: str, path: Path, existing: set[str], mode: str, longest_edge: int):
    """PageItems for the pages of one input that still need OCR. PDF pages render one at
    a time (as lightonocr.render.render_pdf does), so a long PDF never sits in memory."""
    if not rel.lower().endswith(".pdf"):
        yield PageItem(rel, md_rel(rel), fit(open_image(path), longest_edge), False)
        return
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    todo = [n for n in range(1, len(pdf) + 1) if not page_done(md_rel(rel, n), existing, mode)]
    if not todo:  # every page written by an earlier run: only the marker is missing
        yield PageItem(rel, None, None, True)
        return
    for n in todo:
        page = pdf[n - 1]
        w, h = page.get_size()  # points
        image = page.render(scale=longest_edge / max(w, h)).to_pil().convert("RGB")
        yield PageItem(rel, md_rel(rel, n), image, n == todo[-1])


def build_prompt(llm: LLM, mode: str) -> str:
    """Chat-template prompt with thinking disabled; grounding adds the "grounding" text."""
    content = [{"type": "image"}]
    if mode == "grounding":
        content.append({"type": "text", "text": GROUNDING_PROMPT})
    return llm.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_outputs(md: str, raw: str, image_size, mode: str, fix_dollars: bool, model: str):
    """(relative path, content) pairs for one page: the .md, plus a .json in grounding mode."""
    text = raw.strip()
    if fix_dollars:
        text = fix_escaped_dollars(text)
    if mode != "grounding":
        return [(md, text)]
    blocks = parse_blocks(text)
    sidecar = {
        "model": model,
        "image_size": list(image_size),  # size fed to the model; bbox is 0-1000 of this frame
        "blocks": [b.to_dict() for b in blocks],
    }
    # Sidecar before .md: if a directory write stops between the two, the page is redone.
    return [
        (md[: -len(".md")] + ".json", json.dumps(sidecar, ensure_ascii=False)),
        (md, to_markdown(blocks) if blocks else text),
    ]


def write_outputs(dest: str, pairs: list[tuple[str, str]]):
    if not pairs:
        return
    if is_bucket(dest):
        put_files(pairs, as_bucket_dir(dest)[0].rstrip("/"))  # one upload call per batch
        return
    for rel, content in pairs:
        path = Path(dest) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="OCR images/PDFs from an HF Bucket or directory using LightOnOCR-3, output markdown files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  plain      markdown transcription (default)
  grounding  markdown + a .json sidecar of labelled layout blocks (0-1000 boxes)

Examples:
  lighton-ocr3-bucket.py hf://buckets/ns/scans/vol1 hf://buckets/ns/ocr/vol1 --mode grounding
  lighton-ocr3-bucket.py /input /output        (directories, e.g. -v mounted buckets)
  lighton-ocr3-bucket.py hf://buckets/ns/scans /output --include "**/*.jp2"
        """,
    )
    parser.add_argument("input", help="hf://buckets/<ns>/<bucket>[/<prefix>] or a directory of images/PDFs")
    parser.add_argument("output", help="hf://buckets/<ns>/<bucket>[/<prefix>] or a directory for the .md files")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS,
                        help=f"Model checkpoint (default: {DEFAULT_MODEL})")
    parser.add_argument("--mode", default="plain", choices=["plain", "grounding"],
                        help="plain = markdown; grounding = markdown + .json layout blocks (default: plain)")
    parser.add_argument("--include", default=None,
                        help='Glob relative to the input, e.g. "**/*.jp2" (default: all images and PDFs)')
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Files downloaded, and pages sent to vLLM, per batch (default: 16)")
    parser.add_argument("--max-model-len", type=int, default=16384,
                        help="Max model context length (default: 16384)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max output tokens (default: {DEFAULT_MAX_TOKENS}, the card's vLLM example)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}, model card)")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P,
                        help=f"Top-p (default: {DEFAULT_TOP_P}, model card; LightOn's client uses 1.0)")
    parser.add_argument("--longest-edge", type=int, default=DEFAULT_LONGEST_EDGE,
                        help=f"Downscale pages so the longest edge is at most this many pixels; "
                        f"PDFs render at this size (default: {DEFAULT_LONGEST_EDGE})")
    parser.add_argument("--no-fix-dollars", action="store_true",
                        help="Keep the raw output instead of repairing escaped dollar delimiters")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        help="GPU memory utilization (default: 0.8)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit the number of input files (files, not pages; a PDF still yields "
                        "all its pages). Counted before skipping files that are already done.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redo files that already have output (default: skip them, which resumes a run)")
    parser.add_argument("--verbose", action="store_true", help="Print resolved package versions")

    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be 1 or more")
    if args.max_tokens > args.max_model_len:
        parser.error(f"--max-tokens ({args.max_tokens}) must be <= --max-model-len ({args.max_model_len})")
    for loc in (args.input, args.output):
        if is_bucket(loc):
            try:
                as_bucket_dir(loc)
            except ValueError as e:
                parser.error(str(e))
    if not is_bucket(args.input) and not Path(args.input).is_dir():
        parser.error(f"Input directory does not exist: {args.input}")

    check_cuda_availability()

    start_time = time.time()
    logger.info(f"Listing inputs in {args.input} ...")
    rels = list_inputs(args.input, args.include, args.max_samples)
    if not rels:
        logger.error(f"No image or PDF files found in {args.input}")
        sys.exit(1)
    existing = set() if args.overwrite else list_outputs(args.output)
    todo = [rel for rel in rels if not is_done(rel, existing, args.mode)]
    skipped = len(rels) - len(todo)
    logger.info(f"Found {len(rels)} file(s); {skipped} already done; {len(todo)} to OCR")
    if not todo:
        logger.info("Nothing to do (--overwrite to redo).")
        return

    logger.info(f"Initializing vLLM with {args.model} ({args.mode} mode)...")
    # Stock Qwen3_5ForConditionalGeneration arch; gdn_prefill_backend="triton" avoids the
    # gated-delta-net prefill path that needs a CUDA JIT toolchain (as in lighton-ocr3.py).
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        gdn_prefill_backend="triton",
    )
    prompt = build_prompt(llm, args.mode)
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )

    pages_done = 0
    errors = 0
    files_seen = 0
    failed: set[str] = set()  # inputs with a page that failed: no .complete.<mode> marker this run
    open_errors: list[str] = []

    def file_pages(batch):
        for rel, path in batch:
            try:
                yield from iter_pages(rel, path, existing, args.mode, args.longest_edge)
            except Exception as e:
                logger.error(f"Failed to open {rel}: {e}")
                failed.add(rel)
                open_errors.append(rel)

    total_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    for batch_num, batch in enumerate(iter_local_batches(args.input, todo, args.batch_size), 1):
        logger.info(f"Batch {batch_num}/{total_batches} ({files_seen}/{len(todo)} files done)")
        files_seen += len(batch)

        # Windows of at most --batch-size pages, rendered lazily.
        for window in partition_all(args.batch_size, file_pages(batch)):
            pages = [it for it in window if it.image is not None]
            outputs = []
            if pages:
                try:
                    outputs = llm.generate(
                        [{"prompt": prompt, "multi_modal_data": {"image": it.image}} for it in pages],
                        sampling_params,
                    )
                except Exception as e:
                    # Write nothing: these pages are retried on the next run.
                    logger.error(f"Batch {batch_num} failed: {e}")
                    errors += len(pages)
                    failed.update(it.rel for it in pages)
                    continue

            pairs = []
            for it, output in zip(pages, outputs, strict=True):
                pairs.extend(render_outputs(it.md, output.outputs[0].text, it.image.size, args.mode,
                                            not args.no_fix_dollars, args.model))
            # Markers last, so a stopped directory write never marks a PDF complete early.
            pairs.extend(
                (marker_rel(it.rel, args.mode), "ok\n") for it in window if it.last and it.rel not in failed
            )
            try:
                write_outputs(args.output, pairs)
                pages_done += len(pages)
            except Exception as e:
                logger.error(f"Writing batch {batch_num} failed: {e}")
                errors += len(pages)
                failed.update(it.rel for it in window)

    errors += len(open_errors)
    elapsed = time.time() - start_time
    elapsed_str = f"{elapsed / 60:.1f} min" if elapsed > 60 else f"{elapsed:.1f}s"
    logger.info("=" * 50)
    logger.info(f"Done! Wrote {pages_done} page(s) from {files_seen} file(s) in {elapsed_str}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Skipped (already done): {skipped}")
    logger.info(f"  Errors (no output, retried next run): {errors}")
    if failed:
        listed = sorted(failed)
        logger.warning(f"  Inputs with failures ({len(listed)}): {', '.join(listed[:20])}"
                       + (" ..." if len(listed) > 20 else ""))
        if args.overwrite:
            logger.warning("  With --overwrite these keep their previous output: re-run them with --overwrite.")
    if pages_done:
        logger.info(f"  Speed: {pages_done / elapsed:.2f} pages/sec")

    if args.verbose:
        import importlib.metadata

        logger.info("--- Package versions ---")
        for pkg in ["vllm", "transformers", "torch", "pillow", "imagecodecs", "bucketbag", "lightonocr"]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("LightOnOCR-3 Bucket Script")
        print("=" * 60)
        print("\nOCR images/PDFs from an HF Bucket or directory → markdown files")
        print("(+ .json layout blocks in grounding mode). Re-running resumes.")
        print()
        print("Usage:")
        print("  lighton-ocr3-bucket.py INPUT OUTPUT   (hf://buckets/... URLs or directories)")
        print()
        print("HF Jobs (image, flavor + secrets from the [tool.hf-jobs] header; hf CLI 1.32+):")
        print("  hf jobs uv run \\")
        print("      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr3-bucket.py \\")
        print("      hf://buckets/ns/scans/vol1 hf://buckets/ns/ocr/vol1 --mode grounding")
        print()
        print("For full help: uv run --with vllm==0.30.0 lighton-ocr3-bucket.py --help")
        sys.exit(0)

    main()
