# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0,<5.1",  # 5.1 needs huggingface_hub>=1.31; the image ships an older 1.x on PYTHONPATH, which wins
#     "huggingface-hub",
#     "transformers",
#     "tqdm",
#     "toolz",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.29.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "a10g-small"
# secrets = ["HF_TOKEN"]
# ///
"""
Extract structured data (JSON / XML / YAML) from text using LiquidAI's LFM2-1.2B-Extract.

LFM2-1.2B-Extract is a compact 1.2B text-only model purpose-built for turning unstructured
documents into structured data: give it a schema, it returns JSON, XML, or YAML. It reports
beating Gemma 3 27B (22x larger) on syntax validity / format accuracy / faithfulness, and
is multilingual (en, ar, zh, fr, de, ja, ko, pt, es).

This is the *text* counterpart to `lfm2-vl-extract.py` (which extracts from images). Pair them:
OCR a page to markdown with one of the OCR recipes, then extract fields from that text here.

Pass `--schema` as inline text/JSON, a URL, or a file path describing the structure to extract:

    --schema '{"invoice_number": "string", "total": "number", "line_items": "array"}'

Model:  https://huggingface.co/LiquidAI/LFM2-1.2B-Extract
Docs:   https://docs.liquid.ai/deployment/gpu-inference/vllm

HF Jobs: the `[tool.hf-jobs]` header above pins the vLLM image (vLLM + torch come from the
image, not the deps), the GPU flavor and the HF_TOKEN secret, so no flags are needed
(requires `hf` CLI 1.32+):

    hf jobs uv run \
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lfm2-extract.py \
        INPUT OUTPUT --text-column text --schema '{"field": "description"}'

On your own GPU, supply vLLM yourself:

    uv run --with vllm==0.29.0 lfm2-extract.py INPUT OUTPUT --schema '{"field": "description"}'

FlashInfer sampling is disabled (see below) so the engine never JIT-compiles a kernel that
needs nvcc.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional
from urllib.request import urlopen

# Disable vLLM's FlashInfer sampler before the engine starts: it JIT-compiles at warmup and
# needs nvcc (absent from the default uv image). Harmless for greedy decoding.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from toolz import partition_all
from tqdm import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "LiquidAI/LFM2-1.2B-Extract"
FORMATS = {"json": "JSON", "xml": "XML", "yaml": "YAML"}


def check_cuda_availability() -> None:
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Run on Hugging Face Jobs with: hf jobs uv run lfm2-extract.py ...")
        sys.exit(1)
    logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name()}")


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


def load_text_arg(value: str) -> str:
    """Resolve --schema (inline text/JSON, URL, or file path) into a string."""
    text = value.strip()
    if text.startswith("http://") or text.startswith("https://"):
        logger.info(f"Loading schema from URL: {text}")
        return urlopen(text).read().decode("utf-8").strip()
    if os.path.exists(text):
        logger.info(f"Loading schema from file: {text}")
        with open(text) as f:
            return f.read().strip()
    return text


def build_system_prompt(schema_text: str, fmt: str) -> str:
    return f"Return data as a {FORMATS[fmt]} object with the following schema:\n\n{schema_text}"


def parse_output(text: str, fmt: str) -> tuple[str, bool]:
    """Strip code fences; for JSON, validate. Returns (cleaned_text, is_valid)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped.rsplit("```", 1)[0]
    stripped = stripped.strip()
    if fmt == "json":
        try:
            return json.dumps(json.loads(stripped), ensure_ascii=False), True
        except (json.JSONDecodeError, ValueError):
            return stripped, False
    return stripped, True  # xml/yaml: store as-is (no strict validator)


def main(
    input_dataset: str,
    output_dataset: str,
    schema: str,
    text_column: str = "text",
    output_column: str = "extraction",
    overwrite: bool = False,
    output_format: str = "json",
    split: str = "train",
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 32,
    model: str = DEFAULT_MODEL,
    max_model_len: int = 8192,
    max_tokens: int = 4096,
    private: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    check_cuda_availability()
    if output_format not in FORMATS:
        logger.error(f"--format must be one of {list(FORMATS)}; got {output_format}")
        sys.exit(1)

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    schema_text = load_text_arg(schema)
    system_prompt = build_system_prompt(schema_text, output_format)

    logger.info(f"Loading dataset: {input_dataset} (split={split})")
    dataset = load_dataset(input_dataset, split=split)

    # Fail fast if the output column would collide with an existing input column
    dataset = ensure_output_columns_free(dataset, [output_column], overwrite=overwrite)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    logger.info(f"Processing {len(dataset)} examples; format={output_format}")

    if text_column not in dataset.column_names:
        logger.error(f"Text column '{text_column}' not found. Columns: {dataset.column_names}")
        sys.exit(1)

    logger.info(f"Loading model: {model}")
    llm = LLM(model=model, max_model_len=max_model_len, enforce_eager=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    all_outputs: List[str] = []
    n_valid = 0
    texts = dataset[text_column]
    for batch in tqdm(list(partition_all(batch_size, texts)), desc="Extracting"):
        batch_messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(doc)},
            ]
            for doc in batch
        ]
        outputs = llm.chat(batch_messages, sampling_params)
        for out in outputs:
            cleaned, ok = parse_output(out.outputs[0].text, output_format)
            n_valid += int(ok)
            all_outputs.append(cleaned)

    logger.info(f"Valid {output_format.upper()}: {n_valid}/{len(all_outputs)}")
    dataset = dataset.add_column(output_column, all_outputs)

    inference_entry = {
        "model": model,
        "column_name": output_column,
        "task": "structured extraction",
        "format": output_format,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": "lfm2-extract.py",
    }
    if "inference_info" in dataset.column_names:
        def update_info(example):
            try:
                existing = json.loads(example["inference_info"]) if example["inference_info"] else []
            except (json.JSONDecodeError, TypeError):
                existing = []
            existing.append(inference_entry)
            return {"inference_info": json.dumps(existing)}
        dataset = dataset.map(update_info)
    else:
        dataset = dataset.add_column(
            "inference_info", [json.dumps([inference_entry])] * len(dataset)
        )

    logger.info(f"Pushing to {output_dataset}")
    dataset.push_to_hub(output_dataset, private=private, token=HF_TOKEN)

    card_text = f"""---
tags:
- uv-script
- extraction
- lfm2
- {output_format}
---

# Structured extraction with LFM2-1.2B-Extract

`{output_format.upper()}` extracted from the `{text_column}` column of
[{input_dataset}](https://huggingface.co/datasets/{input_dataset})
using [{model}](https://huggingface.co/{model}).

- **Source**: `{input_dataset}` (split `{split}`, column `{text_column}`)
- **Model**: `{model}`
- **Format**: `{output_format}`
- **Output column**: `{output_column}`
- **Valid {output_format.upper()}**: {n_valid}/{len(all_outputs)}
- **Date**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

Generated with the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) `lfm2-extract.py` script.
"""
    try:
        DatasetCard(card_text).push_to_hub(output_dataset, token=HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not push dataset card: {e}")

    logger.info("Done! Extraction complete.")
    logger.info(f"Dataset: https://huggingface.co/datasets/{output_dataset}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("LFM2-1.2B-Extract — structured extraction (JSON/XML/YAML) from text")
        print("\nUsage:")
        print("  uv run --with vllm==0.29.0 lfm2-extract.py INPUT OUTPUT --schema SCHEMA [--text-column text] [--format json]")
        print("\nExample:")
        print('  uv run --with vllm==0.29.0 lfm2-extract.py my-docs my-fields \\')
        print('    --text-column markdown \\')
        print('    --schema \'{"title": "the title", "date": "any date", "summary": "one sentence"}\'')
        print("\n  --schema accepts inline text/JSON, a URL, or a file path.")
        print("\nOn HF Jobs (hf CLI 1.32+; image/flavor/secrets come from the script header):")
        print("  hf jobs uv run lfm2-extract.py INPUT OUTPUT --schema SCHEMA")
        print("\nFor full help: uv run --with vllm==0.29.0 lfm2-extract.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Structured extraction (JSON/XML/YAML) from text using LFM2-1.2B-Extract",
    )
    parser.add_argument("input_dataset", help="Input dataset ID (with a text column)")
    parser.add_argument("output_dataset", help="Output dataset ID")
    parser.add_argument(
        "--schema", required=True,
        help="Structure to extract: inline text/JSON, a URL, or a file path",
    )
    parser.add_argument("--text-column", default="text", help="Text column (default: text)")
    parser.add_argument("--output-column", default="extraction", help="Output column (default: extraction)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column if it already exists in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )
    parser.add_argument(
        "--format", dest="output_format", default="json", choices=list(FORMATS),
        help="Output format (default: json)",
    )
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before sampling")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Max context length (default: 8192)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max output tokens (default: 4096)")
    parser.add_argument("--private", action="store_true", help="Make output dataset private")
    parser.add_argument("--hf-token", help="HF token (or set HF_TOKEN)")
    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        schema=args.schema,
        text_column=args.text_column,
        output_column=args.output_column,
        overwrite=args.overwrite,
        output_format=args.output_format,
        split=args.split,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
        batch_size=args.batch_size,
        model=args.model,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        private=args.private,
        hf_token=args.hf_token,
    )
