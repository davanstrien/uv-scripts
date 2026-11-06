# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0,<5.1",  # 5.1 needs huggingface_hub 1.x; the image copy (0.x) on PYTHONPATH wins
#     "huggingface-hub",
#     "pillow",
#     "tqdm",
#     "toolz",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.24.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "a10g-small"
# secrets = ["HF_TOKEN"]
# ///

"""
Convert document images to markdown using HunyuanOCR-1.5 with vLLM.

HunyuanOCR-1.5 is a lightweight ~1B-parameter, end-to-end OCR-specialized VLM
from Tencent. It keeps the validated 1.0 backbone but extends the max image
resolution to 4K and the context window to 128K, and adds targeted long-tail
capabilities (low-resource / ancient-script OCR, multi-image text QA). Per the
technical report (arXiv:2607.04884) it is faster than dots.ocr / DeepSeek-OCR-2
and top-tier on OmniDocBench v1.6. This script runs it offline via vLLM.

Features:
- 📝 End-to-end document parsing to markdown (tables → HTML, formulas → LaTeX)
- 🧩 Structured / layout-aware parsing
- 📍 Text spotting with coordinates (JSON or Hunyuan format)
- 📐 Formula (LaTeX) and 📊 table (HTML) recognition
- 📈 Chart parsing (Mermaid / Markdown)
- 🌐 Document + general-scene translation (→ zh / → en)
- 🎯 Compact model (~1B parameters)

Model: tencent/HunyuanOCR
  On 2026-07-06 Tencent replaced the repo root in-place with HunyuanOCR-1.5
  (1.0 archived under `v1.0/`, no git tag). So the repo *root* — the default
  here — is now 1.5. The sibling recipe `hunyuan-ocr.py` pins the last 1.0
  commit by revision to keep the 1.0 behavior; this script deliberately tracks
  root (1.5).

vLLM: 0.18.1 (release) is the first stable wheel with native
  `HunYuanVLForConditionalGeneration` support for autoregressive decoding — no
  nightly or patch needed for batch OCR. The [tool.hf-jobs] header pins the
  vllm/vllm-openai:v0.24.0 image (`hf` CLI 1.32+), which also sets the hardware
  and the HF_TOKEN secret. Newer stacks fail: unpinned vLLM with transformers
  >=5.13 does not recognise `hunyuan_vl`, and the v0.29.0 image fails at engine
  start ("Expected 4 multimodal RoPE channels"). To run on your own GPU:
  `uv run --with vllm==0.24.0 --with "transformers<5.13" ...`. The DFlash
  speculative-decoding draft (a per-request *latency* win that needs a vLLM
  nightly) is intentionally NOT implemented: it does not change offline batch
  throughput or output distribution.

trust_remote_code=True per the model card (the processor ships custom code).

License: Tencent Hunyuan Community License (territory excludes EU/UK/South Korea)
https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE

Note: batch_size defaults to 16 (untested on this arch as of writing — 1.5 on
vLLM ≥0.18.1 should batch fine, unlike the 1.0 V1 batching issue; will be
smoke-tested). Lower it if you hit engine errors.

Post-processing note: only the shared tail-repetition cleanup
(`clean_repeated_substrings`, byte-for-byte from the official toolkit) is
ported. The upstream doc_parse-only markdown normalization (10 OmniDocBench
GT-alignment regex passes in `hunyuan_utils.process_one`) is intentionally NOT
ported — it is benchmark-GT alignment, not general OCR, and would bloat this
self-contained recipe. For bench-exact output, use Tencent's toolkit directly.
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
from typing import Any, Dict, List, Union

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
from toolz import partition_all
from tqdm.auto import tqdm

# Disable vLLM's FlashInfer sampler: it JIT-compiles a CUDA kernel needing nvcc, which the
# default uv-script image lacks (engine init then crashes). Greedy OCR doesn't use it; this
# lets the plain default-image command work. On the vllm/vllm-openai image it's a harmless no-op.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# HunyuanOCR-1.5 official task prompts.
# Reproduced VERBATIM from the shipped client's `hunyuan_tasks.py`:
#   https://github.com/Tencent-Hunyuan/HunyuanOCR (inference/*/hunyuan_tasks.py)
# The model card only prints the `doc_parse` prompt; the other 11 live only in
# the client. Prompts are FIXED per task type — upstream deliberately does NOT
# expose free-form prompt editing because hand-tweaked instructions were
# observed to silently degrade quality (users pick a *task*, not a prompt).
# All prompts are Chinese-language, including for English documents — this is
# the officially recommended wording; the card provides no English variants.
# ────────────────────────────────────────────────────────────────

TASK_PROMPTS = {
    # 端到端文档解析
    "doc_parse": "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。",
    # 结构化解析（古文、街景等非文档结构化场景）
    "structured_parse": "提取图中的文字。",
    # Spotting — JSON 格式
    "spotting_json": "检测并识别图中所有的文字行，请按从上到下、从左到右的阅读顺序进行识别。 "
    "输出格式为 JSON 数组，每个元素必须包含："
    '"box": [xmin, ymin, xmax, ymax]（坐标需归一化到 [0, 1000] 范围内）；'
    '"text": "识别出的文字内容"。 '
    "注意：请直接输出 JSON 数组，不要包含任何多余的描述性文字。",
    # Spotting — Hunyuan 模式
    "spotting_hunyuan": "检测并识别图片中的文字，将文本坐标格式化输出。",
    # 版式分析
    "layout": "按照阅读顺序解析图中的版式信息。",
    # 版式分析 + 解析
    "layout_parse": "提取文档图片中所有内容用markdown格式表示，表格用html格式表达，"
    "文档中公式用latex格式表示，请按照阅读顺序组织进行全文解析，并输出版式分析信息。",
    # 图表解析
    "chart_parse": "解析图中的图表，对于流程图使用Mermaid格式表示，其他图表使用Markdown格式表示。",
    # 公式解析
    "formula": "识别图片中的公式，用LaTeX格式表示。",
    # 表格解析
    "table": "把图中的表格解析为HTML。",
    # 文档英译中
    "doc_trans_en2zh": "先解析文档，再将文档内容翻译为中文，其中页眉、页脚忽略，"
    "公式用latex格式表示，表格用html格式表示。",
    # 通用场景翻译 other2en
    "trans_other2en": "按照阅读顺序，提取图中文字，公式用latex格式表示，表格用markdown格式表示，"
    "再将文字内容翻译为英文。",
    # 通用场景翻译 other2zh
    "trans_other2zh": "按照阅读顺序，提取图中文字，公式用latex格式表示，表格用markdown格式表示，"
    "再将文字内容翻译为中文。",
}

# English glosses for --help / the no-args banner (upstream ships Chinese ones).
TASK_DESCRIPTIONS = {
    "doc_parse": "End-to-end doc parse (body→markdown, tables→HTML, formulas→LaTeX, headers/footers ignored). Default.",
    "structured_parse": "Structured parse for non-document scenes (ancient scripts, street signs) — extract all text.",
    "spotting_json": "Text detect+recognize as a JSON array (box normalized to 0-1000 + text).",
    "spotting_hunyuan": "Text detect+recognize in Hunyuan coordinate format.",
    "layout": "Layout analysis in reading order.",
    "layout_parse": "Layout analysis + full-document parse (markdown/HTML/LaTeX).",
    "chart_parse": "Chart parsing (flowcharts→Mermaid, other charts→Markdown).",
    "formula": "Formula recognition → LaTeX.",
    "table": "Table parsing → HTML.",
    "doc_trans_en2zh": "Document translation to Chinese (parse then translate; formulas LaTeX, tables HTML).",
    "trans_other2en": "General-scene extraction + translation to English.",
    "trans_other2zh": "General-scene extraction + translation to Chinese.",
}

DEFAULT_TASK = "doc_parse"

# Sampling params LOCKED by the model card across all official setups so outputs
# are comparable: temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.08.
# Only repetition_penalty is exposed as a flag (the others are fixed for
# deterministic OCR); repetition_penalty is the model's built-in anti-repeat.
DEFAULT_REPETITION_PENALTY = 1.08


def clean_repeated_substrings(text: str, min_repeats: int = 10) -> str:
    """Trim a long repeated suffix as a final safety net against greedy-decoding
    degeneration. Byte-for-byte from the official `hunyuan_utils.py`.
    """
    n = len(text)
    if n < 2000:
        return text
    for length in range(2, n // min_repeats + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i : i + length] == candidate:
            count += 1
            i -= length
        if count >= min_repeats:
            return text[: n - length * (count - 1)]
    return text


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
    logger.error(
        "Choose a different --output-column, or pass --overwrite to replace them."
    )
    sys.exit(1)


def get_prompt(task_type: str) -> str:
    """Return the official prompt for a task type."""
    if task_type not in TASK_PROMPTS:
        raise ValueError(
            f"Unknown task type: {task_type}. Available: {list(TASK_PROMPTS.keys())}"
        )
    return TASK_PROMPTS[task_type]


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str,
) -> List[Dict]:
    """Create the chat messages for one image + prompt.

    Mirrors the official client: an empty system message followed by a user turn
    with the image *before* the text. The empty system content pins "no system
    prompt" (matching how the model is served) rather than letting the chat
    template inject a default.
    """
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Convert to RGB
    pil_img = pil_img.convert("RGB")

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    return [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def create_dataset_card(
    source_dataset: str,
    model: str,
    num_samples: int,
    processing_time: str,
    batch_size: int,
    max_model_len: int,
    max_tokens: int,
    repetition_penalty: float,
    gpu_memory_utilization: float,
    image_column: str = "image",
    output_column: str = "markdown",
    split: str = "train",
    task_type: str = "doc_parse",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    return f"""---
tags:
- ocr
- document-processing
- hunyuan-ocr-1.5
- multilingual
- markdown
- uv-script
- generated
---

# Document OCR using {model_name} (HunyuanOCR-1.5)

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using HunyuanOCR-1.5, a lightweight ~1B end-to-end OCR VLM from Tencent (128K context, 4K max image resolution).

Model license: [Tencent Hunyuan Community License](https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE) (territory excludes EU/UK/South Korea).

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `{output_column}`
- **Dataset Split**: `{split}`
- **Task Type**: `{task_type}`
- **Batch Size**: {batch_size}
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **Repetition Penalty**: {repetition_penalty}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

HunyuanOCR-1.5 is a lightweight end-to-end OCR VLM that excels at:
- 📝 **Document Parsing** - Full markdown extraction in reading order
- 🧩 **Structured / Layout Parsing** - Layout-aware full-document parse
- 📊 **Table Extraction** - HTML format tables
- 📐 **Formula Recognition** - LaTeX format formulas
- 📈 **Chart Parsing** - Mermaid / Markdown format
- 📍 **Text Spotting** - Detection with coordinates (JSON / Hunyuan)
- 🌐 **Translation** - Document and general-scene translation (→ zh / → en)

Per the technical report ([arXiv:2607.04884](https://arxiv.org/pdf/2607.04884)),
1.5 is faster than dots.ocr / DeepSeek-OCR-2 and top-tier on OmniDocBench v1.6.

## Task Types Available

- `doc_parse` - End-to-end document parsing (default)
- `structured_parse` - Non-document structured scenes (ancient scripts, street signs)
- `spotting_json` - Text detection + recognition as JSON array (box 0-1000 + text)
- `spotting_hunyuan` - Text detection + recognition, Hunyuan coordinate format
- `layout` - Layout analysis in reading order
- `layout_parse` - Layout analysis + full-document parse
- `chart_parse` - Chart parsing (flowcharts → Mermaid, others → Markdown)
- `formula` - Formula recognition → LaTeX
- `table` - Table parsing → HTML
- `doc_trans_en2zh` - Document translation to Chinese
- `trans_other2en` - General-scene extraction + translation to English
- `trans_other2zh` - General-scene extraction + translation to Chinese

## Dataset Structure

The dataset contains all original columns plus:
- `{output_column}`: The extracted text (markdown for `doc_parse`, else the task's format)
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Usage

```python
from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("{{output_dataset_id}}", split="{split}")

# Access the extracted text
for example in dataset:
    print(example["{output_column}"])
    break

# View all OCR models applied to this dataset
inference_info = json.loads(dataset[0]["inference_info"])
for info in inference_info:
    print(f"Column: {{info['column_name']}} - Model: {{info['model_id']}}")
```

## Reproduction

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) HunyuanOCR-1.5 script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/hunyuan-ocr-1.5.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --batch-size {batch_size} \\
    --task-type {task_type} \\
    --max-model-len {max_model_len} \\
    --max-tokens {max_tokens} \\
    --gpu-memory-utilization {gpu_memory_utilization}
```

Generated with [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    model: str = "tencent/HunyuanOCR",
    revision: str = None,
    max_model_len: int = 32768,
    max_tokens: int = 8192,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    gpu_memory_utilization: float = 0.8,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    task_type: str = DEFAULT_TASK,
    custom_prompt: str = None,
    output_column: str = "markdown",
    overwrite: bool = False,
    clean_output: bool = True,
    config: str = None,
    create_pr: bool = False,
    verbose: bool = False,
):
    """Process images from an HF dataset through HunyuanOCR-1.5."""

    # Check CUDA availability first
    check_cuda_availability()

    # Context-length invariant (config.json): text max_position_embeddings=131072;
    # the vision processor caps a single image at img_max_token_num=16384. So the
    # default budget holds without resizing: 16384 (image) + prompt + 8192 (output)
    # ≈ 25k ≤ 32768 (default max_model_len). Enforce max_tokens ≤ max_model_len ≤ 131072.
    if max_model_len > 131072:
        logger.error(
            f"--max-model-len {max_model_len} exceeds the model's max context (131072)."
        )
        sys.exit(1)
    if max_tokens > max_model_len:
        logger.error(
            f"--max-tokens ({max_tokens}) cannot exceed --max-model-len ({max_model_len})."
        )
        sys.exit(1)

    # Track processing start time
    start_time = datetime.now()

    # Login to HF if token provided
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Determine prompt to use
    if custom_prompt:
        prompt = custom_prompt
        logger.warning(
            "Using --custom-prompt. Note: upstream deliberately locks prompts per "
            "task type — hand-tweaked instructions can silently degrade quality."
        )
        logger.info(f"Custom prompt: {prompt[:60]}...")
    else:
        prompt = get_prompt(task_type)
        logger.info(f"Using task type: {task_type}")

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)

    # Validate image column
    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    # Fail fast if the output column would collide with an existing input column
    dataset = ensure_output_columns_free(dataset, [output_column], overwrite=overwrite)

    # Shuffle if requested
    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    # Limit samples if requested
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Initialize vLLM model
    logger.info(f"Initializing vLLM with model: {model}")
    logger.info("This may take a few minutes on first run...")

    llm = LLM(
        model=model,
        revision=revision,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        # The encoder cache is sized from max_num_batched_tokens (8192 by default), but one
        # image can reach img_max_token_num=16384 tokens; a 2000 px scan already needs ~8.6k.
        max_num_batched_tokens=16384,
    )

    # Locked sampling per the model card (deterministic OCR); only repetition_penalty
    # is user-tunable.
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        skip_special_tokens=True,
    )

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Output will be written to column: {output_column}")

    # Process images in batches
    all_outputs = []

    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="HunyuanOCR-1.5 processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        try:
            # Create messages for batch
            batch_messages = [make_ocr_message(img, prompt) for img in batch_images]

            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)

            # Extract outputs
            for output in outputs:
                text = output.outputs[0].text.strip()
                # Clean repeated substrings if enabled
                if clean_output:
                    text = clean_repeated_substrings(text)
                all_outputs.append(text)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add error placeholders for failed batch
            all_outputs.extend(["[OCR ERROR]"] * len(batch_images))

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    # Add output column to dataset
    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Handle inference_info tracking (for multi-model comparisons)
    inference_entry = {
        "model_id": model,
        "model_name": "HunyuanOCR-1.5",
        "model_revision": revision or "main",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "task_type": task_type if not custom_prompt else "custom",
        "repetition_penalty": repetition_penalty,
    }

    if "inference_info" in dataset.column_names:
        # Append to existing inference info
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
        # Create new inference_info column
        logger.info("Creating new inference_info column")
        inference_list = [json.dumps([inference_entry])] * len(dataset)
        dataset = dataset.add_column("inference_info", inference_list)

    # Push to hub with retry and XET fallback
    logger.info(f"Pushing to {output_dataset}")
    commit_msg = f"Add HunyuanOCR-1.5 OCR results ({len(dataset)} samples)" + (
        f" [{config}]" if config else ""
    )
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
                commit_message=commit_msg,
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

    # Create and push dataset card (skip when creating PR to avoid conflicts)
    if not create_pr:
        logger.info("Creating dataset card")
        card_content = create_dataset_card(
            source_dataset=input_dataset,
            model=model,
            num_samples=len(dataset),
            processing_time=processing_time_str,
            batch_size=batch_size,
            max_model_len=max_model_len,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            gpu_memory_utilization=gpu_memory_utilization,
            image_column=image_column,
            output_column=output_column,
            split=split,
            task_type=task_type if not custom_prompt else "custom",
        )

        card = DatasetCard(card_content)
        card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("HunyuanOCR-1.5 processing complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")

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
        ]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")
        logger.info("--- End versions ---")


if __name__ == "__main__":
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 80)
        print("HunyuanOCR-1.5 Document Processing")
        print("=" * 80)
        print(
            "\nLightweight ~1B end-to-end OCR VLM from Tencent (128K context, 4K images)"
        )
        print("\nFeatures:")
        print("- 📝 End-to-end document parsing to markdown")
        print("- 📊 Table extraction (HTML format)")
        print("- 📐 Formula recognition (LaTeX format)")
        print("- 📍 Text spotting with coordinates (JSON / Hunyuan)")
        print("- 📈 Chart parsing (Mermaid / Markdown)")
        print("- 🌐 Document + general-scene translation (→ zh / → en)")
        print("\nExample usage:")
        print("\n1. Basic document parsing:")
        print("   uv run hunyuan-ocr-1.5.py input-dataset output-dataset")
        print("\n2. Formula extraction:")
        print("   uv run hunyuan-ocr-1.5.py math-docs formulas --task-type formula")
        print("\n3. Table extraction:")
        print("   uv run hunyuan-ocr-1.5.py docs tables --task-type table")
        print("\n4. Text spotting as JSON (box + text):")
        print("   uv run hunyuan-ocr-1.5.py images spotted --task-type spotting_json")
        print("\n5. Translate a document to Chinese:")
        print(
            "   uv run hunyuan-ocr-1.5.py en-docs zh-docs --task-type doc_trans_en2zh"
        )
        print("\n6. Running on HF Jobs:")
        print("   (image, hardware and HF_TOKEN come from the script's [tool.hf-jobs] header)")
        print("   hf jobs uv run \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/hunyuan-ocr-1.5.py \\"
        )
        print("       input-dataset output-dataset")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run hunyuan-ocr-1.5.py --help")
        sys.exit(0)

    task_help = "\n".join(f"  {k:18s}- {TASK_DESCRIPTIONS[k]}" for k in TASK_PROMPTS)
    parser = argparse.ArgumentParser(
        description="Document OCR using HunyuanOCR-1.5 (lightweight ~1B end-to-end OCR VLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Task Types (official HunyuanOCR-1.5 prompts, all Chinese-language):
{task_help}

Examples:
  # Basic document OCR (default)
  uv run hunyuan-ocr-1.5.py my-docs analyzed-docs

  # Extract formulas as LaTeX
  uv run hunyuan-ocr-1.5.py math-papers formulas --task-type formula

  # Extract tables as HTML
  uv run hunyuan-ocr-1.5.py reports tables --task-type table

  # Text spotting as JSON (box normalized 0-1000 + text)
  uv run hunyuan-ocr-1.5.py images spotted --task-type spotting_json

  # Translate documents to Chinese
  uv run hunyuan-ocr-1.5.py en-docs translated --task-type doc_trans_en2zh

  # Random sampling for testing
  uv run hunyuan-ocr-1.5.py large-dataset test --max-samples 50 --shuffle
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
        help="Batch size for processing (default: 16; lower it if you hit engine errors)",
    )
    parser.add_argument(
        "--model",
        default="tencent/HunyuanOCR",
        help="Model to use (default: tencent/HunyuanOCR — repo root is 1.5)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model repo revision (default: main). Tencent has replaced this repo's "
        "root in-place before (1.0 → 1.5); pin a commit hash for reproducible runs.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum model context length (default: 32768; max 131072). A single "
        "image is capped at ~16384 tokens by the vision processor, so 32768 fits "
        "image + 8192 output; raise for very long outputs.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192; must be ≤ --max-model-len). "
        "Dense pages may need more — raise toward 32768.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
        help=f"Repetition penalty (default: {DEFAULT_REPETITION_PENALTY}, the model card's locked value)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
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
        "--shuffle", action="store_true", help="Shuffle dataset before processing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--task-type",
        choices=list(TASK_PROMPTS.keys()),
        default=DEFAULT_TASK,
        metavar="TASK",
        help=f"Official task type (default: {DEFAULT_TASK}). See the epilog for all types.",
    )
    parser.add_argument(
        "--custom-prompt",
        help="Custom prompt text (overrides --task-type; may degrade quality — upstream "
        "locks prompts per task)",
    )
    parser.add_argument(
        "--output-column",
        default="markdown",
        help="Column name for output text (default: markdown)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column if it already exists in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )
    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Disable cleaning of repeated substrings in output",
    )
    parser.add_argument(
        "--config",
        help="Dataset config name for multi-model benchmarks",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Push results as a pull request instead of direct commit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions at the end of the run",
    )

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        batch_size=args.batch_size,
        model=args.model,
        revision=args.revision,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        gpu_memory_utilization=args.gpu_memory_utilization,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        task_type=args.task_type,
        custom_prompt=args.custom_prompt,
        output_column=args.output_column,
        overwrite=args.overwrite,
        clean_output=not args.no_clean_output,
        config=args.config,
        create_pr=args.create_pr,
        verbose=args.verbose,
    )
