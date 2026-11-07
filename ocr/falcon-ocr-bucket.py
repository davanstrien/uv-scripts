# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pillow",
#     "pymupdf",
#     "torch>=2.5",
#     "torchvision",
#     "falcon-perception[ocr]",
# ]
# ///

"""
OCR images and PDFs from a directory using Falcon OCR, writing markdown files.

Designed to work with HF Buckets mounted as volumes via `hf jobs uv run -v ...`.
Reads images/PDFs from INPUT_DIR, runs Falcon OCR via the optimized falcon-perception
engine (CUDA graphs + paged inference), and writes one .md file per image (or per
PDF page) to OUTPUT_DIR, preserving directory structure.

Input:                          Output:
  /input/page1.png        ->      /output/page1.md
  /input/report.pdf       ->      /output/report/page_001.md
  (3 pages)                       /output/report/page_002.md
                                  /output/report/page_003.md
  /input/sub/photo.jpg    ->      /output/sub/photo.md

Examples:

  # Local test
  uv run falcon-ocr-bucket.py ./test-images ./test-output

  # HF Jobs with bucket volumes
  hf jobs uv run --flavor l4x1 \\
      -s HF_TOKEN \\
      -v hf://buckets/user/ocr-input:/input:ro \\
      -v hf://buckets/user/ocr-output:/output \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/falcon-ocr-bucket.py \\
      /input /output

Model: tiiuae/Falcon-OCR (0.3B, 80.3% olmOCR, Apache 2.0)
Backend: falcon-perception (OCRInferenceEngine with CUDA graphs)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "tiiuae/Falcon-OCR"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def check_cuda_availability():
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")


def discover_files(input_dir: Path, limit: int | None = None) -> list[Path]:
    """Discover image and PDF files under input_dir.

    Without `limit`, returns the full sorted list (deterministic order).
    With `limit`, stops scanning once `limit` matching files are found
    and returns them in filesystem order (much faster on huge mounted
    buckets, but ordering is not deterministic).
    """
    files = []
    iterator = (
        input_dir.rglob("*") if limit is not None else sorted(input_dir.rglob("*"))
    )
    for path in iterator:
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in IMAGE_EXTENSIONS or ext == ".pdf":
            files.append(path)
            if limit is not None and len(files) >= limit:
                break
    return files


def prepare_images(
    files: list[Path], input_dir: Path, output_dir: Path, pdf_dpi: int
) -> list[tuple[Image.Image, Path]]:
    import fitz  # pymupdf

    items: list[tuple[Image.Image, Path]] = []

    for file_path in files:
        rel = file_path.relative_to(input_dir)
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            pdf_output_dir = output_dir / rel.with_suffix("")
            try:
                doc = fitz.open(file_path)
                num_pages = len(doc)
                logger.info(f"PDF: {rel} ({num_pages} pages)")
                for page_num in range(num_pages):
                    page = doc[page_num]
                    zoom = pdf_dpi / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    md_path = pdf_output_dir / f"page_{page_num + 1:03d}.md"
                    items.append((img, md_path))
                doc.close()
            except Exception as e:
                logger.error(f"Failed to open PDF {rel}: {e}")
        else:
            try:
                img = Image.open(file_path).convert("RGB")
                md_path = output_dir / rel.with_suffix(".md")
                items.append((img, md_path))
            except Exception as e:
                logger.error(f"Failed to open image {rel}: {e}")

    return items


def main():
    parser = argparse.ArgumentParser(
        description="OCR images/PDFs from a directory using Falcon OCR, output markdown files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", help="Directory containing images and/or PDFs")
    parser.add_argument("output_dir", help="Directory to write markdown output files")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Images per batch (default: 8)",
    )
    parser.add_argument(
        "--pdf-dpi", type=int, default=300,
        help="DPI for PDF page rendering (default: 300)",
    )
    parser.add_argument(
        "--no-compile", action="store_true", help="Disable torch.compile",
    )
    parser.add_argument(
        "--no-cudagraph", action="store_true", help="Disable CUDA graph capture",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of input files to discover. Stops scanning early "
             "once the limit is reached (much faster on large mounted buckets). "
             "Applied before PDF page expansion. With --max-samples set, file "
             "ordering is filesystem-dependent rather than sorted.",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=8192,
        help="Engine sequence limit, image tokens included (default: 8192, the "
        "falcon-perception server default; must be a multiple of 128). The engine "
        "default of 4096 cuts dense pages short.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print resolved package versions",
    )

    args = parser.parse_args()

    check_cuda_availability()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Discover files
    if args.max_samples is not None:
        logger.info(
            f"Scanning {input_dir} for up to {args.max_samples} images/PDFs "
            f"(early termination, --max-samples)..."
        )
    else:
        logger.info(f"Scanning {input_dir} for images and PDFs...")
    files = discover_files(input_dir, limit=args.max_samples)
    if not files:
        logger.error(f"No image or PDF files found in {input_dir}")
        sys.exit(1)

    pdf_count = sum(1 for f in files if f.suffix.lower() == ".pdf")
    img_count = len(files) - pdf_count
    logger.info(f"Found {img_count} image(s) and {pdf_count} PDF(s)")

    # Prepare images
    logger.info("Preparing images (rendering PDFs)...")
    items = prepare_images(files, input_dir, output_dir, args.pdf_dpi)
    if not items:
        logger.error("No processable images after preparation")
        sys.exit(1)

    logger.info(f"Total images to OCR: {len(items)}")

    # Load model
    logger.info(f"Loading {MODEL_ID} via falcon-perception engine...")
    from falcon_perception import load_and_prepare_model
    from falcon_perception.data import ImageProcessor
    from falcon_perception.paged_ocr_inference import OCRInferenceEngine

    do_compile = not args.no_compile
    do_cudagraph = not args.no_cudagraph

    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=MODEL_ID,
        device="cuda",
        dtype="bfloat16",
        compile=do_compile,
    )

    image_processor = ImageProcessor(patch_size=16, merge_size=1)
    # max_seq_length counts the image tokens as well as the output. A 1,024 px page uses
    # about 3,000 image tokens, so the engine default (4,096) leaves room for only about
    # 1,000 output tokens and dense pages stop mid-text. 8,192 is the falcon-perception
    # server default.
    engine = OCRInferenceEngine(
        model, tokenizer, image_processor, capture_cudagraph=do_cudagraph,
        max_seq_length=args.max_seq_length,
    )
    logger.info(f"Engine loaded. compile={do_compile}, cudagraph={do_cudagraph}")

    # Process in batches
    errors = 0
    processed = 0
    total = len(items)
    batch_size = args.batch_size

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = items[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        logger.info(f"Batch {batch_num}/{total_batches} ({processed}/{total} done)")

        try:
            batch_images = [img for img, _ in batch]
            texts = engine.generate_plain(images=batch_images, use_tqdm=False)

            for (_, md_path), text in zip(batch, texts):
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(text.strip(), encoding="utf-8")
                processed += 1

        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {e}")
            for _, md_path in batch:
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(f"[OCR ERROR: {e}]", encoding="utf-8")
            errors += len(batch)
            processed += len(batch)

    elapsed = time.time() - start_time
    elapsed_str = f"{elapsed / 60:.1f} min" if elapsed > 60 else f"{elapsed:.1f}s"

    logger.info("=" * 50)
    logger.info(f"Done! Processed {total} images in {elapsed_str}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Errors: {errors}")
    if total > 0:
        logger.info(f"  Speed: {total / elapsed:.2f} images/sec")

    if args.verbose:
        import importlib.metadata

        logger.info("--- Package versions ---")
        for pkg in ["falcon-perception", "torch", "pillow", "pymupdf"]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("Falcon OCR Bucket Script")
        print("=" * 60)
        print(f"\nModel: {MODEL_ID} (0.3B, Apache 2.0)")
        print("OCR images/PDFs from a directory -> markdown files.")
        print("Designed for HF Buckets mounted as volumes.")
        print()
        print("Usage:")
        print("  uv run falcon-ocr-bucket.py INPUT_DIR OUTPUT_DIR")
        print()
        print("Examples:")
        print("  uv run falcon-ocr-bucket.py ./images ./output")
        print()
        print("HF Jobs with bucket volumes:")
        print("  hf jobs uv run --flavor l4x1 -s HF_TOKEN \\")
        print("      -v hf://buckets/user/ocr-input:/input:ro \\")
        print("      -v hf://buckets/user/ocr-output:/output \\")
        print(
            "      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/falcon-ocr-bucket.py \\"
        )
        print("      /input /output")
        print()
        print("For full help: uv run falcon-ocr-bucket.py --help")
        sys.exit(0)

    main()
