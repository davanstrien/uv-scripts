# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=3.1.0,<5.1",  # 5.1 needs huggingface_hub 1.x; the image copy (0.x) on PYTHONPATH wins
#     "huggingface-hub",
#     "pillow",
#     "toolz",
#     "numind",
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
Convert document images to markdown OR extract structured JSON using NuExtract3 with vLLM.

NuExtract3 is a 4B Qwen3.5-based VLM for document understanding. It does two things:

1. Document-to-Markdown OCR (default): images -> clean markdown with HTML tables,
   LaTeX math, and <figure> tags.
2. Schema-guided structured extraction: images + a JSON template -> JSON output
   shaped exactly like the template. Useful for invoices, receipts, forms, contracts.

Modes are selected via flags:
- (no flags)         -> markdown OCR
- --mode content     -> plain-content extraction
- --template SOURCE  -> structured extraction with a NuExtract template
- --schema SOURCE    -> structured extraction with a JSON Schema
                        (auto-converted via numind.nuextract_utils)
- --instructions STR -> free-text guidance passed through to the model
                        (output-format rules, branch routing, etc.).
                        Combines with any of the modes above.
                        See https://huggingface.co/numind/NuExtract3#instructions

--template / --schema each accept inline JSON, a URL, or a local file path, so a
schema can be hosted (e.g. on an HF dataset's raw URL) and reused across jobs:
    --template https://huggingface.co/datasets/ORG/REPO/raw/main/card.json

HF Jobs invocation: the [tool.hf-jobs] header above sets the vllm/vllm-openai
image (vLLM + torch come from the image, with pre-built CUDA kernels), the
flavor and the HF_TOKEN secret. Needs `hf` CLI 1.32+.

    hf jobs uv run \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nuextract3.py \\
        INPUT_DATASET OUTPUT_DATASET --max-samples 5 --shuffle --seed 42

On your own GPU: uv run --with vllm==0.29.0 nuextract3.py INPUT_DATASET OUTPUT_DATASET

Model: numind/NuExtract3
License: Apache-2.0
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
from toolz import partition_all
# Disable vLLM's FlashInfer sampler: it JIT-compiles a CUDA kernel needing nvcc, which the
# default uv-script image lacks (engine init then crashes). Greedy OCR doesn't use it; this
# lets the plain default-image command work. On the vllm/vllm-openai image it's a harmless no-op.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DEFAULT = "numind/NuExtract3"
MODEL_NAME = "NuExtract3"


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with a CUDA-capable GPU.")
        sys.exit(1)
    else:
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


def load_template_arg(value: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load a NuExtract template/JSON Schema from inline JSON, a URL, or a file path."""
    if value is None:
        return None
    text = value
    if value.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(value) as resp:  # noqa: S310
            text = resp.read().decode("utf-8")
    elif "{" not in value:
        # Inline JSON often exceeds the OS filename limit, so only probe the
        # filesystem when the value doesn't look like JSON; treat OSError as
        # "not a path".
        try:
            candidate_path = Path(value)
            if candidate_path.is_file():
                text = candidate_path.read_text()
        except OSError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse template/schema as JSON (tried URL/path/inline): {e}"
        ) from e


def resolve_template(
    template_arg: Optional[str],
    schema_arg: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Resolve --template / --schema into a NuExtract template dict, or None."""
    if template_arg and schema_arg:
        raise ValueError("--template and --schema are mutually exclusive.")

    if template_arg is not None:
        return load_template_arg(template_arg)

    if schema_arg is not None:
        schema = load_template_arg(schema_arg)
        try:
            from numind.nuextract_utils import convert_json_schema_to_nuextract_template
        except ImportError as e:
            raise RuntimeError(
                "--schema requires the `numind` package. "
                "It should be listed in this script's PEP 723 dependencies."
            ) from e
        template, dropped = convert_json_schema_to_nuextract_template(schema)
        if dropped:
            logger.warning(
                f"numind dropped {len(dropped)} unsupported branches from the JSON Schema: "
                f"{dropped}"
            )
        return template

    return None


def image_to_data_uri(image: Union[Image.Image, Dict[str, Any], str]) -> str:
    """Normalize an HF dataset image cell to a PNG data URI."""
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    pil_img = pil_img.convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def make_message(image: Union[Image.Image, Dict[str, Any], str]) -> List[Dict]:
    """Build an OpenAI-format chat message containing one image."""
    data_uri = image_to_data_uri(image)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ]


def split_thinking(text: str) -> tuple[Optional[str], str]:
    """Return (reasoning, answer) if <think>...</think> is present, else (None, text)."""
    if "<think>" in text and "</think>" in text:
        reasoning = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        answer = text.split("</think>", 1)[1].strip()
        return reasoning, answer
    return None, text.strip()


def parse_json_output(text: str) -> tuple[Optional[Any], bool]:
    """Parse an extraction output; strip ``` fences as the model card describes.

    Returns (parsed_value, parse_error). On failure, parsed_value is None.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    try:
        return json.loads(stripped), False
    except json.JSONDecodeError:
        return None, True


def create_dataset_card(
    source_dataset: str,
    model: str,
    num_samples: int,
    processing_time: str,
    mode_label: str,
    template: Optional[Dict[str, Any]],
    enable_thinking: bool,
    temperature: float,
    output_column: str,
    image_column: str,
    split: str,
) -> str:
    """Create a dataset card documenting the NuExtract3 run."""
    model_name = model.split("/")[-1]
    template_block = ""
    if template is not None:
        template_block = (
            "\n### Extraction Template\n\n```json\n"
            + json.dumps(template, indent=2)
            + "\n```\n"
        )

    return f"""---
tags:
- ocr
- structured-extraction
- document-processing
- nuextract3
- markdown
- uv-script
- generated
---

# {model_name} on {source_dataset}

This dataset contains outputs from [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) processed with [NuExtract3](https://huggingface.co/{model}), a 4B vision-language model for document understanding.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Mode**: {mode_label}
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `{output_column}`
- **Dataset Split**: `{split}`
- **Temperature**: {temperature}
- **Thinking Mode**: {"enabled" if enable_thinking else "disabled"}
{template_block}
## Dataset Structure

Original columns plus:
- `{output_column}`: NuExtract3 output ({"JSON string" if template else "markdown"})
- `inference_info`: JSON list tracking models applied to this dataset
{"- `" + output_column + "_reasoning`: model's thinking trace (when enabled)" if enable_thinking else ""}

Generated with [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    max_model_len: int = 16384,
    max_tokens: int = 8192,
    gpu_memory_utilization: float = 0.8,
    mode: str = "markdown",
    template_arg: Optional[str] = None,
    schema_arg: Optional[str] = None,
    enable_thinking: bool = False,
    instructions: Optional[str] = None,
    temperature: Optional[float] = None,
    model: str = MODEL_DEFAULT,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    output_column: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = False,
    config: str = None,
    create_pr: bool = False,
):
    """Process images from an HF dataset through NuExtract3."""

    check_cuda_availability()
    start_time = datetime.now()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    template = resolve_template(template_arg, schema_arg)
    extraction_mode = template is not None
    mode_label = "structured-extraction" if extraction_mode else mode

    if output_column is None:
        output_column = "extraction" if extraction_mode else "markdown"

    if temperature is None:
        temperature = 0.6 if enable_thinking else 0.2

    logger.info(f"Using model: {model}")
    logger.info(f"Mode: {mode_label}")
    logger.info(f"Thinking: {enable_thinking}  Temperature: {temperature}")
    if extraction_mode:
        logger.info(f"Template: {json.dumps(template, indent=2)}")

    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)

    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    # Fail fast if an output column would collide with an existing input column.
    # With --enable-thinking the script also writes "{output_column}_reasoning".
    out_cols = [output_column]
    if enable_thinking:
        out_cols.append(f"{output_column}_reasoning")
    dataset = ensure_output_columns_free(dataset, out_cols, overwrite=overwrite)

    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    logger.info("Initializing vLLM with NuExtract3")
    logger.info("This may take a few minutes on first run...")
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    chat_template_kwargs: Dict[str, Any] = {"enable_thinking": enable_thinking}
    if extraction_mode:
        chat_template_kwargs["template"] = json.dumps(template, indent=4)
    else:
        chat_template_kwargs["mode"] = mode
    if instructions:
        chat_template_kwargs["instructions"] = instructions

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Output will be written to column: {output_column}")

    all_outputs: List[str] = []
    all_reasoning: List[Optional[str]] = []
    all_parse_errors: List[bool] = []
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    processed = 0

    for batch_num, batch_indices in enumerate(
        partition_all(batch_size, range(len(dataset))), 1
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        logger.info(
            f"Batch {batch_num}/{total_batches} "
            f"({processed}/{len(dataset)} images done)"
        )

        try:
            batch_messages = [make_message(img) for img in batch_images]
            outputs = llm.chat(
                batch_messages,
                sampling_params,
                chat_template_kwargs=chat_template_kwargs,
                chat_template_content_format="openai",
            )

            for output in outputs:
                raw_text = output.outputs[0].text
                reasoning, answer = split_thinking(raw_text)

                if extraction_mode:
                    parsed, parse_error = parse_json_output(answer)
                    stored = (
                        json.dumps(parsed, ensure_ascii=False)
                        if parsed is not None
                        else answer
                    )
                    all_outputs.append(stored)
                    all_parse_errors.append(parse_error)
                else:
                    all_outputs.append(answer)
                    all_parse_errors.append(False)

                all_reasoning.append(reasoning)

            processed += len(batch_images)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            all_outputs.extend(["[NUEXTRACT3 ERROR]"] * len(batch_images))
            all_reasoning.extend([None] * len(batch_images))
            all_parse_errors.extend([True] * len(batch_images))
            processed += len(batch_images)

    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    if enable_thinking and any(r is not None for r in all_reasoning):
        reasoning_col = f"{output_column}_reasoning"
        logger.info(f"Adding '{reasoning_col}' column to dataset")
        dataset = dataset.add_column(reasoning_col, all_reasoning)

    if extraction_mode:
        parse_error_count = sum(all_parse_errors)
        if parse_error_count:
            logger.warning(
                f"{parse_error_count}/{len(all_parse_errors)} extractions failed to parse as JSON"
            )

    inference_entry = {
        "model_id": model,
        "model_name": MODEL_NAME,
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "mode": mode_label,
        "has_template": extraction_mode,
        "enable_thinking": enable_thinking,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extraction_mode:
        inference_entry["parse_error_rate"] = (
            sum(all_parse_errors) / len(all_parse_errors) if all_parse_errors else 0.0
        )

    if "inference_info" in dataset.column_names:
        logger.info("Updating existing inference_info column")

        def update_inference_info(example):
            try:
                existing_info = (
                    json.loads(example["inference_info"])
                    if example["inference_info"]
                    else []
                )
            except (json.JSONDecodeError, TypeError):
                existing_info = []
            existing_info.append(inference_entry)
            return {"inference_info": json.dumps(existing_info)}

        dataset = dataset.map(update_inference_info)
    else:
        logger.info("Creating new inference_info column")
        inference_list = [json.dumps([inference_entry])] * len(dataset)
        dataset = dataset.add_column("inference_info", inference_list)

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
                commit_message=f"Add {model} {mode_label} results ({len(dataset)} samples)"
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
                logger.error("All upload attempts failed. Results are lost.")
                sys.exit(1)

    logger.info("Creating dataset card")
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        model=model,
        num_samples=len(dataset),
        processing_time=processing_time_str,
        mode_label=mode_label,
        template=template,
        enable_thinking=enable_thinking,
        temperature=temperature,
        output_column=output_column,
        image_column=image_column,
        split=split,
    )
    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("Done! NuExtract3 processing complete.")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")
    logger.info(
        f"Processing speed: {len(dataset) / processing_duration.total_seconds():.2f} images/sec"
    )

    if verbose:
        import importlib.metadata

        logger.info("--- Resolved package versions ---")
        for pkg in [
            "vllm",
            "transformers",
            "torch",
            "datasets",
            "pyarrow",
            "pillow",
            "numind",
        ]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")
        logger.info("--- End versions ---")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("NuExtract3 - Document-to-Markdown + Structured Extraction (4B)")
        print("=" * 70)
        print("\nModes:")
        print("  markdown          - Image -> markdown (default)")
        print("  content           - Image -> plain content")
        print("  --template / --schema  - Image -> JSON shaped like the template")
        print("\nExamples:")
        print("\n1. Markdown OCR:")
        print("   uv run --with vllm==0.29.0 nuextract3.py input-dataset output-dataset")
        print("\n2. Structured extraction with an inline template:")
        print("   uv run --with vllm==0.29.0 nuextract3.py input output \\")
        print('     --template \'{"title": "verbatim-string", "date": "date"}\'')
        print("\n3. Structured extraction from a JSON Schema (e.g. Pydantic):")
        print("   uv run --with vllm==0.29.0 nuextract3.py input output --schema schema.json")
        print("\n   (--template / --schema also accept a URL or a local file path)")
        print("\n4. Reasoning mode for harder documents:")
        print("   uv run --with vllm==0.29.0 nuextract3.py input output --enable-thinking")
        print("\n5. Test with 10 samples:")
        print("   uv run --with vllm==0.29.0 nuextract3.py large-ds test --max-samples 10 --shuffle")
        print("\n6. Running on HF Jobs (image/flavor/secrets from the script header, hf 1.32+):")
        print("   hf jobs uv run \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nuextract3.py \\"
        )
        print("       input-dataset output-dataset --batch-size 16")
        print("\nFor full help: uv run --with vllm==0.29.0 nuextract3.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="NuExtract3: document-to-markdown + schema-guided JSON extraction (4B VLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  (default)     Markdown OCR (image -> clean markdown)
  --mode content
                Plain-content extraction (less structured than markdown)
  --template PATH_OR_JSON
                Structured extraction with a NuExtract template
  --schema PATH_OR_JSON
                Structured extraction from a JSON Schema
                (e.g. Pydantic Model.model_json_schema())

Examples:
  uv run --with vllm==0.29.0 nuextract3.py my-docs analyzed-docs
  uv run --with vllm==0.29.0 nuextract3.py receipts extracted \\
      --template '{"store": "verbatim-string", "total": "number"}'
  uv run --with vllm==0.29.0 nuextract3.py contracts extracted --schema contract_schema.json
  uv run --with vllm==0.29.0 nuextract3.py hard-docs out --enable-thinking
        """,
    )

    parser.add_argument("input_dataset", help="Input dataset ID from Hugging Face Hub")
    parser.add_argument("output_dataset", help="Output dataset ID for Hugging Face Hub")
    parser.add_argument(
        "--image-column",
        default="image",
        help="Column containing images (default: image)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing (default: 16)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Maximum model context length (default: 16384)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
    )
    parser.add_argument(
        "--mode",
        choices=["markdown", "content"],
        default="markdown",
        help="OCR mode when no template/schema is given (default: markdown)",
    )
    parser.add_argument(
        "--template",
        help="NuExtract template: inline JSON, a URL, or a file path",
    )
    parser.add_argument(
        "--schema",
        help="JSON Schema to auto-convert: inline JSON, a URL, or a file path",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable reasoning mode (slower, better on hard documents)",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help=(
            "Free-text instructions passed to NuExtract via "
            "chat_template_kwargs.instructions (e.g. routing guidance across "
            "optional schema branches, output-format rules). "
            "See https://huggingface.co/numind/NuExtract3#instructions"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.2 non-thinking, 0.6 thinking)",
    )
    parser.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        help=f"Model ID (default: {MODEL_DEFAULT})",
    )
    parser.add_argument("--hf-token", help="Hugging Face API token")
    parser.add_argument(
        "--split", default="train", help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make output dataset private"
    )
    parser.add_argument(
        "--config",
        help="Config/subset name when pushing to Hub (for benchmarking multiple models in one repo)",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request instead of pushing directly (for parallel benchmarking)",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle dataset before processing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--output-column",
        default=None,
        help="Column name for output (default: 'markdown' in OCR mode, 'extraction' in template mode)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column if it already exists in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions after processing",
    )

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        mode=args.mode,
        template_arg=args.template,
        schema_arg=args.schema,
        enable_thinking=args.enable_thinking,
        instructions=args.instructions,
        temperature=args.temperature,
        model=args.model,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        output_column=args.output_column,
        overwrite=args.overwrite,
        verbose=args.verbose,
        config=args.config,
        create_pr=args.create_pr,
    )
