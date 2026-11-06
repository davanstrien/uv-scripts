# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0,<5.1",  # 5.1 needs huggingface_hub>=1.31; the image ships an older 1.x on PYTHONPATH, which wins
#     "huggingface-hub",
#     "pillow",
#     "tqdm",
#     "toolz",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:unlimited-ocr"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "a10g-small"
# secrets = ["HF_TOKEN"]
# ///

"""
Convert document images to markdown using Baidu Unlimited-OCR with vLLM.

Unlimited-OCR (baidu/Unlimited-OCR, 3.3B, MIT) is a DeepSeek-OCR / DeepSeek-OCR-2 descendant. This
recipe runs it as an offline vLLM batch job (dataset in -> markdown out), mirroring the proven
deepseek-ocr-vllm.py pattern: llm.generate() with PIL images and the model's
NGramPerReqLogitsProcessor to stop coordinate-token loops on long documents.

One image per row -> one markdown. Output is layout-grounded markdown: text spans are tagged
<|ref|>...<|/ref|> with <|det|>...<|/det|> coordinate boxes (coords normalized 0-1000); tables come
back as HTML and equations as LaTeX. Pass --strip-grounding to drop the tags and keep clean text;
add --grounding-column to keep the raw grounded output (with bboxes) in a second column too.

Multi-page / "long-horizon" parsing (the model's headline feature) is not in this single-image batch
recipe — for multi-page, serve the model and send all pages in one request (see serving-unlimited-ocr.md).
Multi-page *does* work via vLLM serving: on a clean 2-page doc it returned both pages, <PAGE>-separated.
But on hard/degraded scans (dense historical pages, newspaper clippings) vLLM multi-page degraded to
hallucination in our tests, where the model's own SGLang build held up better — so SGLang is the more
robust multi-page path. (vLLM's upstream PR, vllm-project/vllm#46564, benchmarks single-page only.)

IMPORTANT: Unlimited-OCR's architecture is not in a stable vLLM pip wheel, so this script MUST run on
Baidu's dedicated vLLM image (vllm and torch come from the image, not the PEP 723 deps). The
[tool.hf-jobs] header above sets the image, python, PYTHONPATH, flavor and HF_TOKEN secret, so the
run command needs no flags (the header needs `hf` CLI 1.32+):

    hf jobs uv run \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/unlimited-ocr-vllm.py \\
        your-input-dataset your-output-dataset --max-samples 10

On Hopper GPUs (h100/h200) add --image vllm/vllm-openai:unlimited-ocr-cu129 (a CLI flag overrides
the header). On your own GPU, run inside that image: no stable vLLM wheel has this architecture yet.

Model card: https://huggingface.co/baidu/Unlimited-OCR
vLLM recipe: https://recipes.vllm.ai/baidu/Unlimited-OCR
"""

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
from toolz import partition_all
from tqdm.auto import tqdm

# Disable vLLM's FlashInfer sampler: it JIT-compiles a CUDA kernel needing nvcc. Greedy OCR doesn't
# use it; on the dedicated vllm/vllm-openai image it's a harmless no-op.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from vllm.model_executor.models.unlimited_ocr import NGramPerReqLogitsProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = "baidu/Unlimited-OCR"
# Last revision tested with this recipe and image. The 2026-10-14 upstream update
# ("compatibility with transformers") makes the remote code import addict and
# matplotlib at config load, which this image lacks. Re-test before moving it.
MODEL_REVISION = "07dea832e22aefee32ad281d4b80551282e1c168"

# Prompt and no-repeat-ngram knobs straight from the model card / vLLM recipe (single image).
PROMPT = "<image>document parsing."
NGRAM_SIZE = 35
WINDOW_SIZE = 128

# Strip the model's grounding markup to recover clean text:
# drop the <|det|>...<|/det|> coordinate boxes, then unwrap the <|ref|>...<|/ref|> spans.
_DET_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
_REF_RE = re.compile(r"<\|/?ref\|>")


def strip_grounding(text: str) -> str:
    """Remove <|det|> boxes and <|ref|> wrappers, keeping the inner text."""
    text = _DET_RE.sub("", text)
    text = _REF_RE.sub("", text)
    # collapse the blank lines left behind by removed boxes
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def ensure_output_columns_free(dataset, columns, overwrite=False):
    """Fail fast if an output column would collide with an existing input column.

    Adding a column that already exists silently overwrites it (e.g. a ground-truth
    `text`/`markdown` column) or crashes on push with a duplicate-column error only
    *after* inference has run. Catch it up front. With overwrite=True, drop the clashing
    column(s) here instead (logged) so the later add_column is clean.
    """
    clash = [c for c in columns if c in dataset.column_names]
    if not clash:
        return dataset
    if overwrite:
        logger.warning(f"--overwrite: replacing existing column(s) {clash}")
        return dataset.remove_columns(clash)
    logger.error(
        f"Output column(s) {clash} already exist in the input dataset "
        f"(columns: {dataset.column_names})."
    )
    logger.error("Choose a different --output-column, or pass --overwrite to replace them.")
    sys.exit(1)


def to_pil(image: Union[Image.Image, Dict[str, Any], str]) -> Image.Image:
    """Convert various dataset image cell formats to an RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict) and "bytes" in image:
        return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image)}")


def create_dataset_card(
    source_dataset: str,
    output_dataset: str,
    model: str,
    num_samples: int,
    processing_time: str,
    output_column: str,
    strip_grounding_enabled: bool,
    split: str,
) -> str:
    """Create a dataset card documenting the OCR run."""
    if strip_grounding_enabled:
        grounding = "Grounding markup was stripped (`--strip-grounding`); the column holds clean text."
    else:
        grounding = (
            "The column holds the model's raw layout-grounded markdown: text spans tagged "
            "`<|ref|>...<|/ref|>` with `<|det|>...<|/det|>` coordinate boxes (coords 0-1000). "
            "Strip them with "
            "`re.sub(r'<\\|det\\|>.*?<\\|/det\\|>', '', t)` then `re.sub(r'<\\|/?ref\\|>', '', t)`."
        )
    return f"""---
tags:
- ocr
- document-processing
- unlimited-ocr
- baidu
- markdown
- uv-script
- generated
---

# Document OCR using Unlimited-OCR

This dataset contains OCR results for [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
produced by [{model}](https://huggingface.co/{model}) with vLLM.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}
- **Output Column**: `{output_column}`
- **Split**: `{split}`

## Output

{grounding}

Tables are returned as HTML and equations as LaTeX.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{output_dataset}", split="{split}")
print(ds[0]["{output_column}"])
```

## Reproduction

Generated with the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) Unlimited-OCR
vLLM recipe. Unlimited-OCR needs Baidu's dedicated vLLM image, which the script's `[tool.hf-jobs]` header
sets (`hf` CLI 1.32+):

```bash
hf jobs uv run \\
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/unlimited-ocr-vllm.py \\
    {source_dataset} <output-dataset>
```

Generated with [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    model: str = MODEL,
    image_column: str = "image",
    output_column: str = "markdown",
    overwrite: bool = False,
    grounding_column: Optional[str] = None,
    batch_size: int = 8,
    max_model_len: int = 32768,
    max_tokens: int = 8192,
    gpu_memory_utilization: float = 0.8,
    strip_grounding_enabled: bool = False,
    hf_token: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    config: Optional[str] = None,
    create_pr: bool = False,
    verbose: bool = False,
):
    """Process images from an HF dataset through Unlimited-OCR with vLLM."""
    if grounding_column and grounding_column == output_column:
        raise ValueError("--grounding-column must differ from --output-column")
    check_cuda_availability()
    start_time = datetime.now()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)
    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    # Fail fast if the output column would collide with an existing input column
    dataset = ensure_output_columns_free(dataset, [output_column], overwrite=overwrite)

    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    logger.info(f"Initializing vLLM with model: {model}")
    logger.info("This may take a few minutes on first run...")

    # The pin applies to the default model only; another --model loads its main branch.
    revision = MODEL_REVISION if model == MODEL else None
    logger.info(f"Model revision: {revision or 'main'}")
    llm = LLM(
        model=model,
        revision=revision,
        code_revision=revision,
        tokenizer_revision=revision,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 1},
        logits_processors=[NGramPerReqLogitsProcessor],
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        skip_special_tokens=False,
        extra_args=dict(ngram_size=NGRAM_SIZE, window_size=WINDOW_SIZE),
    )

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    all_outputs: List[str] = []
    all_grounded: List[
        str
    ] = []  # raw grounded text, only when --grounding-column is set
    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="Unlimited-OCR",
    ):
        batch_indices = list(batch_indices)
        try:
            model_inputs = [
                {
                    "prompt": PROMPT,
                    "multi_modal_data": {"image": to_pil(dataset[i][image_column])},
                }
                for i in batch_indices
            ]
            outputs = llm.generate(model_inputs, sampling_params)
            for output in outputs:
                raw = output.outputs[0].text.strip()
                all_outputs.append(
                    strip_grounding(raw) if strip_grounding_enabled else raw
                )
                if grounding_column:
                    all_grounded.append(raw)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            all_outputs.extend(["[OCR FAILED]"] * len(batch_indices))
            if grounding_column:
                all_grounded.extend(["[OCR FAILED]"] * len(batch_indices))

    processing_time_str = (
        f"{(datetime.now() - start_time).total_seconds() / 60:.1f} min"
    )

    logger.info(f"Adding '{output_column}' column to dataset")
    if output_column in dataset.column_names:
        logger.warning(f"Column '{output_column}' already exists, replacing it")
        dataset = dataset.remove_columns([output_column])
    dataset = dataset.add_column(output_column, all_outputs)
    if grounding_column:
        logger.info(f"Adding '{grounding_column}' column (raw grounded output)")
        if grounding_column in dataset.column_names:
            logger.warning(f"Column '{grounding_column}' already exists, replacing it")
            dataset = dataset.remove_columns([grounding_column])
        dataset = dataset.add_column(grounding_column, all_grounded)

    # inference_info: append-only log so several models can write into one dataset and be compared.
    inference_entry = {
        "model_id": model,
        "model_name": "Unlimited-OCR",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "strip_grounding": strip_grounding_enabled,
        "grounding_column": grounding_column,
        "script": "unlimited-ocr-vllm.py",
        "script_url": "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/unlimited-ocr-vllm.py",
    }
    if "inference_info" in dataset.column_names:
        logger.info("Updating existing inference_info column")

        def update_inference_info(example):
            try:
                existing = (
                    json.loads(example["inference_info"])
                    if example["inference_info"]
                    else []
                )
            except (json.JSONDecodeError, TypeError):
                existing = []
            existing.append(inference_entry)
            return {"inference_info": json.dumps(existing)}

        dataset = dataset.map(update_inference_info)
    else:
        logger.info("Creating new inference_info column")
        dataset = dataset.add_column(
            "inference_info", [json.dumps([inference_entry])] * len(dataset)
        )

    logger.info(f"Pushing to {output_dataset}")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.warning("Disabling XET (fallback to HTTP upload)")
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            dataset.push_to_hub(
                output_dataset,
                private=private,
                token=HF_TOKEN,
                max_shard_size="500MB",
                **({"config_name": config} if config else {}),
                create_pr=create_pr,
                commit_message=f"Add {model} OCR results ({len(dataset)} samples)"
                + (f" [{config}]" if config else ""),
            )
            break
        except Exception as e:
            logger.error(f"Upload attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                delay = 30 * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error("All upload attempts failed. OCR results are lost.")
                sys.exit(1)

    logger.info("Creating dataset card...")
    card = DatasetCard(
        create_dataset_card(
            source_dataset=input_dataset,
            output_dataset=output_dataset,
            model=model,
            num_samples=len(dataset),
            processing_time=processing_time_str,
            output_column=output_column,
            strip_grounding_enabled=strip_grounding_enabled,
            split=split,
        )
    )
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("✅ OCR conversion complete!")
    logger.info(f"Dataset: https://huggingface.co/datasets/{output_dataset}")
    logger.info(f"Processing time: {processing_time_str}")

    if verbose:
        import importlib.metadata

        logger.info("--- Resolved package versions ---")
        for pkg in ["vllm", "transformers", "torch", "datasets", "pillow"]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")
        logger.info("--- End versions ---")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 80)
        print("Unlimited-OCR to Markdown Converter (vLLM)")
        print("=" * 80)
        print("\nBaidu Unlimited-OCR (3.3B, MIT) — one image per row -> markdown.")
        print("\nMUST run on the dedicated image: vllm/vllm-openai:unlimited-ocr")
        print("(set by the script's [tool.hf-jobs] header, hf CLI 1.32+;")
        print("on Hopper GPUs add --image vllm/vllm-openai:unlimited-ocr-cu129).")
        print("\nExample:")
        print("   hf jobs uv run \\")
        print("     unlimited-ocr-vllm.py my-images my-markdown --max-samples 10")
        print(
            "\nMulti-page documents: serve the model instead (see serving-unlimited-ocr.md)."
        )
        print("\nFor full help, run with --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="OCR images to markdown using Unlimited-OCR (vLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run unlimited-ocr-vllm.py my-images ocr-results

  # Clean text (strip grounding tags)
  uv run unlimited-ocr-vllm.py my-images ocr-results --strip-grounding

  # On HF Jobs (image, flavor and secret come from the [tool.hf-jobs] header; hf CLI 1.32+)
  hf jobs uv run \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/unlimited-ocr-vllm.py \\
      my-dataset my-output --max-samples 10
        """,
    )
    parser.add_argument("input_dataset", help="Input dataset ID from Hugging Face Hub")
    parser.add_argument("output_dataset", help="Output dataset ID for Hugging Face Hub")
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Model to use (default: {MODEL}). Override only for a same-architecture mirror.",
    )
    parser.add_argument(
        "--image-column", default="image", help="Column with images (default: image)"
    )
    parser.add_argument(
        "--output-column",
        default="markdown",
        help="Output column name (default: markdown)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column if it already exists in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )
    parser.add_argument(
        "--strip-grounding",
        action="store_true",
        help="Drop <|det|>/<|ref|> grounding tags from the output column, keeping clean text",
    )
    parser.add_argument(
        "--grounding-column",
        help="Also store the RAW grounded output (boxes + tags) in this extra column "
        "(pair with --strip-grounding to keep clean text AND the layout/bboxes)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Images per batch (default: 8)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Max context length (default: 32768)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max output tokens (default: 8192)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory fraction (default: 0.8)",
    )
    parser.add_argument("--hf-token", help="Hugging Face API token")
    parser.add_argument(
        "--split", default="train", help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Max samples to process (for testing)"
    )
    parser.add_argument(
        "--private", action="store_true", help="Make output dataset private"
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle before processing"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Shuffle seed (default: 42)"
    )
    parser.add_argument(
        "--config",
        help="Config/subset name when pushing (for benchmarking multiple models)",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Push as a PR instead of a direct commit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions after the run",
    )

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        model=args.model,
        image_column=args.image_column,
        output_column=args.output_column,
        overwrite=args.overwrite,
        grounding_column=args.grounding_column,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        strip_grounding_enabled=args.strip_grounding,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        config=args.config,
        create_pr=args.create_pr,
        verbose=args.verbose,
    )
