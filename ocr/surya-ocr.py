# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "surya-ocr",
#     "datasets>=3.1.0,<5.1",  # 5.1 needs huggingface_hub 1.x; the image copy (0.x) on PYTHONPATH wins
#     "huggingface-hub",
#     "pillow",
#     "toolz",
#     "tqdm",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.20.1"
# python = "/usr/local/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/site-packages" }
# flavor = "a10g-small"
# secrets = ["HF_TOKEN"]
# ///
"""
Document intelligence on images OR multi-page PDFs with Datalab's **Surya OCR 2**
(`datalab-to/surya-ocr-2`, 650M, Qwen3.5-style).

Surya is *structured* OCR: instead of a flat markdown blob, it returns per-block
HTML with bounding boxes, reading order, and labels (equations in `<math>`). This
recipe writes **both**:

  --output-column   (default `markdown`)  flattened, reading-order text per row
  surya_blocks                            the full structured result as JSON
                                          (bbox / polygon / label / reading_order /
                                          confidence / html per block), one entry
                                          per page.

Three tasks via `--task`:
  ocr     (default)  full-page OCR  -> text + per-block HTML/bboxes
  layout             layout regions -> labelled boxes + reading order
  table              table structure -> HTML (mode `full`) or rows/cols/cells
                     (mode `simple`, via --table-mode)

Input is one document per row:
  --image-column COL   (default `image`)  one image per row
  --pdf-column COL                        PDF bytes per row (multi-page; honors
                                          --page-range). Pages are concatenated in
                                          the text column and kept per-page in
                                          `surya_blocks`.

ENGINE: Surya normally spawns a vLLM **server** (Docker) — which can't run inside
an HF Job. This script instead does **offline batch inference**: it injects a
custom in-process backend into Surya's `SuryaInferenceManager` that runs vLLM's
offline `LLM().chat()` engine (no server, no HTTP). Surya still owns all the
prompting, image preprocessing, and HTML/bbox parsing — we only swap the
transport. Run on the **`vllm/vllm-openai:v0.20.1`** image (Surya's known-good
vLLM build; the model is the recent, version-sensitive `qwen3_5` architecture).

LICENSE NOTE: Surya's *code* is Apache-2.0 but the *weights* are a modified
OpenRAIL-M license — free for research, personal use, and startups under $5M
funding/revenue, but restricted from competitive use against Datalab's API.
Confirm you are within those terms. https://huggingface.co/datalab-to/surya-ocr-2

HF Jobs (the [tool.hf-jobs] header pins the vLLM image, interpreter, PYTHONPATH,
flavor and HF_TOKEN secret; needs `hf` CLI 1.32+):

    hf jobs uv run \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/surya-ocr.py \\
        INPUT_DATASET OUTPUT_DATASET \\
        --max-samples 5 --shuffle --seed 42

On your own GPU: uv run --with vllm==0.20.1 surya-ocr.py INPUT_DATASET OUTPUT_DATASET ...

Model: datalab-to/surya-ocr-2  (package: surya-ocr, https://github.com/datalab-to/surya)
"""

import argparse
import io
import json
import logging
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlopen

from datasets import load_dataset
from huggingface_hub import DatasetCard
from PIL import Image
from toolz import partition_all
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "datalab-to/surya-ocr-2"
# Surya's own vision-tiling bounds (from its vLLM backend), applied to the
# offline engine too so preprocessing matches the server path exactly.
MM_PROCESSOR_KWARGS = {"min_pixels": 3136, "max_pixels": 6291456}
TASKS = ("ocr", "layout", "table")


def check_cuda_availability() -> None:
    """Exit early with a clear message if there's no GPU."""
    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error(
            "Run on Hugging Face Jobs with: hf jobs uv run <script> ... "
            "(the script's [tool.hf-jobs] header sets the GPU + vLLM image; hf CLI 1.32+)"
        )
        sys.exit(1)
    logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def check_vllm_available() -> None:
    """Fail fast (before loading 400 rows) if vLLM isn't importable.

    Surya-2 runs its VLM through vLLM's offline engine, but `vllm` is deliberately
    NOT a PEP723 dependency: the recent hybrid `qwen3_5` architecture is only in the
    pinned `vllm/vllm-openai:v0.20.1` image, which also provides torch/transformers via
    PYTHONPATH. Launched on the bare uv image (no `--image`), the import fails per-batch
    and every row silently gets "[SURYA GENERATE ERROR]". Detect that up front instead.
    """
    import importlib.util

    if importlib.util.find_spec("vllm") is None:
        logger.error("vLLM is not importable — this recipe cannot run on the bare uv image.")
        logger.error(
            "Surya-2 needs the pinned vLLM build. With hf CLI 1.32+ the script's "
            "[tool.hf-jobs] header applies the image + interpreter automatically:"
        )
        logger.error("  hf jobs uv run <script_url> INPUT_DATASET OUTPUT_DATASET ...")
        logger.error(
            "Older hf CLIs: add --flavor a10g-small -s HF_TOKEN --image vllm/vllm-openai:v0.20.1 --python /usr/local/bin/python3 "
            "-e PYTHONPATH=/usr/local/lib/python3.12/site-packages. "
            "On your own GPU: uv run --with vllm==0.20.1 <script> ..."
        )
        sys.exit(1)


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


def parse_page_range(spec: Optional[str]) -> Optional[List[int]]:
    """Turn '0-3,5' into [0,1,2,3,5]. None/empty -> None (all pages)."""
    if not spec:
        return None
    pages: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            pages.extend(range(int(lo), int(hi) + 1))
        else:
            pages.append(int(part))
    return pages or None


def cell_to_bytes(cell: Any) -> bytes:
    """Normalize an HF dataset cell (image or document) to raw file bytes."""
    if isinstance(cell, Image.Image):
        buf = io.BytesIO()
        cell.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return cell["bytes"]
        if cell.get("path"):
            with open(cell["path"], "rb") as f:
                return f.read()
        raise ValueError(
            f"Unsupported image/document dict (no bytes/path): {list(cell)}"
        )
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, str):
        if cell.startswith(("http://", "https://")):
            return urlopen(cell).read()  # noqa: S310
        with open(cell, "rb") as f:
            return f.read()
    raise ValueError(f"Unsupported cell type: {type(cell)}")


def cell_to_pil(cell: Any) -> Image.Image:
    """One image cell -> RGB PIL image."""
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")
    return Image.open(io.BytesIO(cell_to_bytes(cell))).convert("RGB")


def load_pdf_images(
    load_pdf, cell: Any, page_indices: Optional[List[int]], dpi: int
) -> List[Image.Image]:
    """Render one PDF cell into page images via Surya's own pypdfium2 loader."""
    data = cell_to_bytes(cell)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        images, _ = load_pdf(path, page_indices, dpi=dpi)
        return [im.convert("RGB") for im in images]
    finally:
        os.unlink(path)


# --- structured-output shim (vLLM API moved between versions) ---
def build_structured_outputs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """SamplingParams kwargs for guided JSON, across vLLM versions (layout uses this)."""
    try:
        from vllm.sampling_params import StructuredOutputsParams  # vLLM >= 0.12

        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except (ImportError, TypeError):
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams  # older vLLM

        return {"guided_decoding": GuidedDecodingParams(json=schema)}
    except (ImportError, TypeError):
        pass
    logger.warning(
        "Guided JSON unavailable in this vLLM version; relying on the model."
    )
    return {}


def _mean_token_prob(completion_output) -> Optional[float]:
    """Mean exp(logprob) of the sampled tokens -> Surya's per-block `confidence`."""
    lps = getattr(completion_output, "logprobs", None)
    if not lps:
        return None
    probs: List[float] = []
    for tid, lp_dict in zip(completion_output.token_ids, lps):
        if not lp_dict:
            continue
        entry = lp_dict.get(tid)
        if (
            entry is None
        ):  # sampled token not in the returned top-k; use the best we have
            entry = max(lp_dict.values(), key=lambda e: e.logprob)
        probs.append(math.exp(entry.logprob))
    return sum(probs) / len(probs) if probs else None


class OfflineVLLMBackend:
    """Surya `Backend` (duck-typed) that runs vLLM's offline `LLM().chat()` engine.

    Surya's predictors call `manager.generate(batch)` -> `backend.generate(batch)`;
    we satisfy that contract in-process (no server). Surya keeps ownership of the
    prompts (`PROMPT_MAPPING`), image scaling (`scale_to_fit`), and output parsing.
    """

    name = "offline-vllm"

    def __init__(
        self,
        model: str,
        max_model_len: int,
        gpu_memory_utilization: float,
        dtype: str = "bfloat16",
        max_tokens_default: int = 2048,
        logprobs_default: bool = True,
    ):
        self.model = model
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens_default = max_tokens_default
        self.logprobs_default = logprobs_default
        self.llm = None
        self._build_messages = None
        self._scale_to_fit = None
        self._prompt_mapping = None

    def start(self):
        from vllm import LLM

        logger.info(
            f"Loading {self.model} into vLLM offline engine (dtype={self.dtype})..."
        )
        self.llm = LLM(
            model=self.model,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            mm_processor_kwargs=MM_PROCESSOR_KWARGS,
            limit_mm_per_prompt={"image": 1},
        )
        # Reuse Surya's exact request shaping so the offline path matches the server.
        from surya.inference.backends.openai_client import _build_messages
        from surya.inference.prompts import PROMPT_MAPPING
        from surya.inference.util import scale_to_fit

        self._build_messages = _build_messages
        self._scale_to_fit = scale_to_fit
        self._prompt_mapping = PROMPT_MAPPING
        return None

    def stop(self) -> None:
        self.llm = None

    def _sampling_params(self, item):
        from vllm import SamplingParams

        max_tokens = item.max_tokens or self.max_tokens_default
        want_logprobs = item.request_logprobs or self.logprobs_default
        kwargs: Dict[str, Any] = dict(temperature=0.0, top_p=0.1, max_tokens=max_tokens)
        if want_logprobs:
            kwargs["logprobs"] = 1
        if item.guided_json is not None:
            kwargs.update(build_structured_outputs(item.guided_json))
        return SamplingParams(**kwargs)

    def generate(self, batch):
        from surya.inference.schema import BatchOutputItem

        if self.llm is None:
            self.start()
        if not batch:
            return []

        conversations = []
        sampling_params = []
        for item in batch:
            prompt = item.prompt or self._prompt_mapping[item.prompt_type]
            image = self._scale_to_fit(item.image)
            conversations.append(self._build_messages(image, prompt))
            sampling_params.append(self._sampling_params(item))

        outputs = self.llm.chat(
            conversations,
            sampling_params,
            chat_template_content_format="openai",
            use_tqdm=False,
        )

        results = []
        for item, out in zip(batch, outputs):
            comp = out.outputs[0]
            results.append(
                BatchOutputItem(
                    raw=comp.text,
                    token_count=len(comp.token_ids),
                    error=False,
                    mean_token_prob=_mean_token_prob(comp),
                    logprobs=None,
                    metadata=item.metadata,  # carries page_idx/block_idx — must round-trip
                )
            )
        return results


def make_manager(backend: OfflineVLLMBackend):
    """A SuryaInferenceManager wired to our offline backend (bypassing autodetect)."""
    from surya.inference import SuryaInferenceManager

    manager = SuryaInferenceManager.__new__(SuryaInferenceManager)
    manager.method = backend.name
    manager.backend = backend
    return manager


# --- result serialization (text column + structured surya_blocks) ---
def _html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


def serialize_pages(task: str, pages: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, structured-per-page) for one row's page results."""
    structured = [p.model_dump(mode="json") for p in pages]
    page_texts: List[str] = []
    for page in pages:
        if task == "ocr":
            parts = []
            for b in sorted(page.blocks, key=lambda b: b.reading_order):
                if b.skipped or not b.html:
                    continue
                txt = _html_to_text(b.html)
                if txt:
                    parts.append(txt)
            page_texts.append("\n".join(parts))
        elif task == "layout":
            # No OCR text in layout mode — emit a reading-order outline of labels.
            page_texts.append(
                "\n".join(
                    f"{b.position}: {b.label}"
                    for b in sorted(page.bboxes, key=lambda b: b.position)
                )
            )
        else:  # table
            if page.html:  # mode="full"
                page_texts.append(page.html)
            else:  # mode="simple"
                page_texts.append(f"{len(page.rows)} rows x {len(page.cols)} cols")
    return "\n\n".join(page_texts), structured


def create_dataset_card(
    source_dataset: str,
    model: str,
    task: str,
    table_mode: str,
    num_samples: int,
    n_ok: int,
    source_column: str,
    is_pdf: bool,
    page_range: Optional[str],
    output_column: str,
    blocks_column: str,
    split: str,
    processing_time: str,
) -> str:
    input_kind = "PDF documents" if is_pdf else "images"
    col_desc = "PDF" if is_pdf else "image"
    if page_range:
        col_desc += f", pages {page_range}"
    task_desc = {
        "ocr": "full-page OCR (structured HTML + bounding boxes)",
        "layout": "layout analysis (labelled regions + reading order)",
        "table": f"table recognition (mode `{table_mode}`)",
    }[task]
    return f"""---
tags:
- ocr
- document-processing
- surya
- structured
- uv-script
- generated
---

# Surya OCR 2 ({task}) on {source_dataset}

{task_desc.capitalize()} over {input_kind} in
[{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using
[Surya OCR 2](https://huggingface.co/{model}) (650M, Qwen3.5-based) by Datalab, via the
[`surya-ocr`](https://github.com/datalab-to/surya) package, run as **offline vLLM batch
inference** on Hugging Face Jobs.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Task**: `{task}`{f" (table mode `{table_mode}`)" if task == "table" else ""}
- **Input column**: `{source_column}` ({col_desc})
- **Text column**: `{output_column}` (flattened, reading-order text per row)
- **Structured column**: `{blocks_column}` (JSON: per-page blocks with bbox / polygon / label / reading_order / confidence / html)
- **Split**: `{split}`
- **Samples**: {num_samples:,}
- **Processed OK**: {n_ok:,} / {num_samples:,}
- **Processing time**: {processing_time}
- **Date**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## License note

Surya's code is Apache-2.0, but the model **weights** use a modified OpenRAIL-M
license: free for research, personal use, and startups under $5M funding/revenue,
restricted from competitive use against Datalab's API. See the
[model card](https://huggingface.co/{model}).

## Dataset Structure

Original columns plus:
- `{output_column}`: flattened text (OCR), label outline (layout), or table HTML (table)
- `{blocks_column}`: structured result as a JSON string (one entry per page)
- `inference_info`: JSON list tracking models applied to this dataset

Generated with [UV Scripts](https://huggingface.co/uv-scripts).
"""


def main(
    input_dataset: str,
    output_dataset: str,
    task: str = "ocr",
    table_mode: str = "full",
    image_column: str = "image",
    pdf_column: Optional[str] = None,
    output_column: str = "markdown",
    overwrite: bool = False,
    blocks_column: str = "surya_blocks",
    page_range: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    max_model_len: int = 18000,
    gpu_memory_utilization: float = 0.85,
    dtype: str = "bfloat16",
    model: str = DEFAULT_MODEL,
    private: bool = False,
    config: Optional[str] = None,
    create_pr: bool = False,
    hf_token: Optional[str] = None,
    verbose: bool = False,
) -> None:
    # Unlock full Xet bandwidth for the model download (repo convention).
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    # Surya reads settings from env at import; pin the checkpoint and forbid any
    # server autostart (we inject our own offline backend instead).
    os.environ["SURYA_MODEL_CHECKPOINT"] = model
    os.environ["SURYA_INFERENCE_AUTOSTART"] = "False"

    check_cuda_availability()
    check_vllm_available()
    start_time = datetime.now(timezone.utc)

    # Set the env var instead of login(): the old huggingface_hub in this image
    # raises KeyError on a Jobs token in login(). Hub calls read HF_TOKEN.
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN

    # Import Surya only after env is set.
    from surya.input.load import load_pdf
    from surya.settings import settings

    source_column = pdf_column or image_column
    is_pdf = pdf_column is not None
    page_indices = parse_page_range(page_range)
    pdf_dpi = settings.IMAGE_DPI_HIGHRES

    logger.info(
        f"Model: {model}  Task: {task}"
        + (f" (mode {table_mode})" if task == "table" else "")
    )
    logger.info(f"Loading dataset: {input_dataset} (split={split})")
    dataset = load_dataset(input_dataset, split=split)
    if source_column not in dataset.column_names:
        logger.error(
            f"Column '{source_column}' not found. Available: {dataset.column_names}"
        )
        sys.exit(1)
    # Fail fast if the output column would collide with an existing input column
    dataset = ensure_output_columns_free(
        dataset, [output_column, blocks_column], overwrite=overwrite
    )
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    n = len(dataset)
    logger.info(f"Processing {n} documents from column '{source_column}'")

    # Build the offline engine + inject it into a Surya manager, then pick the predictor.
    backend = OfflineVLLMBackend(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )
    manager = make_manager(backend)

    if task == "ocr":
        from surya.recognition import RecognitionPredictor

        predictor = RecognitionPredictor(manager)

        def run(images):
            return predictor(images, full_page=True)
    elif task == "layout":
        from surya.layout import LayoutPredictor

        predictor = LayoutPredictor(manager)

        def run(images):
            return predictor(images)
    else:  # table
        from surya.table_rec import TableRecPredictor

        predictor = TableRecPredictor(manager)

        def run(images):
            return predictor(images, mode=table_mode)

    texts: List[Optional[str]] = [None] * n
    blocks: List[Optional[str]] = [None] * n
    error_flags: List[bool] = [True] * n

    for chunk in tqdm(list(partition_all(batch_size, range(n))), desc=f"Surya {task}"):
        chunk = list(chunk)
        flat_images: List[Image.Image] = []
        spans: List[Tuple[int, int, int]] = []  # (row_idx, start, count)
        for i in chunk:
            try:
                if is_pdf:
                    imgs = load_pdf_images(
                        load_pdf, dataset[i][source_column], page_indices, pdf_dpi
                    )
                else:
                    imgs = [cell_to_pil(dataset[i][source_column])]
            except Exception as e:
                logger.warning(f"Row {i}: failed to load document: {e}")
                texts[i] = f"[SURYA LOAD ERROR] {e}"
                blocks[i] = None
                continue
            if not imgs:
                texts[i] = "[SURYA EMPTY DOCUMENT]"
                continue
            spans.append((i, len(flat_images), len(imgs)))
            flat_images.extend(imgs)

        if not flat_images:
            continue
        try:
            results = run(flat_images)
        except Exception as e:
            logger.error(f"Batch generate failed: {e}")
            for i, _, _ in spans:
                texts[i] = "[SURYA GENERATE ERROR]"
                blocks[i] = None
            continue

        for i, start, count in spans:
            page_results = results[start : start + count]
            text, structured = serialize_pages(task, page_results)
            texts[i] = text
            blocks[i] = json.dumps(structured, ensure_ascii=False)
            error_flags[i] = False

    n_ok = sum(not f for f in error_flags)
    logger.info(f"Processed OK: {n_ok}/{n}")

    dataset = dataset.add_column(output_column, texts)
    dataset = dataset.add_column(blocks_column, blocks)

    inference_entry = {
        "model": model,
        "model_name": "surya-ocr-2",
        "column_name": output_column,
        "blocks_column": blocks_column,
        "task": task,
        "table_mode": table_mode if task == "table" else None,
        "backend": "vllm-offline",
        "page_range": page_range,
        "error_rate": (n - n_ok) / n if n else 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": "surya-ocr.py",
    }
    if "inference_info" in dataset.column_names:

        def update_info(example):
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

        dataset = dataset.map(update_info)
    else:
        dataset = dataset.add_column(
            "inference_info", [json.dumps([inference_entry])] * n
        )

    processing_time = (
        f"{(datetime.now(timezone.utc) - start_time).total_seconds() / 60:.1f} min"
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
                create_pr=create_pr,
                **({"config_name": config} if config else {}),
                commit_message=f"Add Surya OCR 2 {task} results ({n} samples)"
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

    try:
        card = DatasetCard(
            create_dataset_card(
                source_dataset=input_dataset,
                model=model,
                task=task,
                table_mode=table_mode,
                num_samples=n,
                n_ok=n_ok,
                source_column=source_column,
                is_pdf=is_pdf,
                page_range=page_range,
                output_column=output_column,
                blocks_column=blocks_column,
                split=split,
                processing_time=processing_time,
            )
        )
        card.push_to_hub(output_dataset, token=HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not push dataset card: {e}")

    logger.info("Done! Surya OCR 2 complete.")
    logger.info(f"Dataset: https://huggingface.co/datasets/{output_dataset}")
    logger.info(f"Processing time: {processing_time}")

    if verbose:
        import importlib.metadata

        logger.info("--- Resolved package versions ---")
        for pkg in ["surya-ocr", "vllm", "transformers", "torch", "datasets", "pillow"]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(
            "Surya OCR 2 — structured OCR / layout / tables from images & PDFs (650M)"
        )
        print("\nUsage:")
        print("  uv run surya-ocr.py INPUT OUTPUT [--task ocr|layout|table] [options]")
        print("\nExamples:")
        print("  # full-page OCR -> text + structured surya_blocks")
        print("  uv run surya-ocr.py my-images my-ocr")
        print("\n  # layout regions / table structure")
        print("  uv run surya-ocr.py my-images my-layout --task layout")
        print("  uv run surya-ocr.py my-tables my-tables-out --task table")
        print("\n  # multi-page PDFs")
        print("  uv run surya-ocr.py my-pdfs my-ocr --pdf-column pdf --page-range 0-5")
        print("\nRun on the vllm/vllm-openai:v0.20.1 image (offline vLLM batch).")
        print("For full help: uv run surya-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Surya OCR 2 (650M): structured OCR / layout / tables, offline vLLM batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tasks (--task):
  ocr     full-page OCR -> reading-order text + per-block HTML/bboxes (default)
  layout  layout regions -> labelled boxes + reading order
  table   table structure -> HTML (--table-mode full) or rows/cols/cells (simple)

Output columns:
  --output-column   flattened text per row (default: markdown)
  surya_blocks      structured JSON per row (bbox/label/reading_order/confidence/html)

Input (one document per row):
  --image-column COL   one image per row   (default: image)
  --pdf-column COL     PDF bytes per row    (multi-page; honors --page-range)

HF Jobs (hf CLI 1.32+; the [tool.hf-jobs] header sets image/flavor/secrets):
  hf jobs uv run surya-ocr.py INPUT_DATASET OUTPUT_DATASET --max-samples 5
Own GPU:
  uv run --with vllm==0.20.1 surya-ocr.py INPUT_DATASET OUTPUT_DATASET
""",
    )
    parser.add_argument(
        "input_dataset", help="Input dataset ID from the Hugging Face Hub"
    )
    parser.add_argument(
        "output_dataset", help="Output dataset ID for the Hugging Face Hub"
    )
    parser.add_argument(
        "--task", choices=TASKS, default="ocr", help="Task (default: ocr)"
    )
    parser.add_argument(
        "--table-mode",
        choices=["full", "simple"],
        default="full",
        help="Table task: 'full' = HTML, 'simple' = rows/cols/cells (default: full)",
    )
    parser.add_argument(
        "--image-column", default="image", help="Image column (default: image)"
    )
    parser.add_argument(
        "--pdf-column",
        default=None,
        help="PDF column (bytes/path). Mutually exclusive with --image-column.",
    )
    parser.add_argument(
        "--output-column",
        default="markdown",
        help="Text output column (default: markdown)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column if it already exists in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )
    parser.add_argument(
        "--blocks-column",
        default="surya_blocks",
        help="Structured JSON output column (default: surya_blocks)",
    )
    parser.add_argument(
        "--page-range",
        default=None,
        help="Pages from PDFs, e.g. '0-5,7' (PDF column only)",
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Limit number of documents (for testing)"
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle before sampling"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Shuffle seed (default: 42)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Rows (images) per offline llm.chat batch (default: 16)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=18000,
        help="vLLM context length (default: 18000)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory fraction (default: 0.85)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="vLLM dtype (default: bfloat16; use float16 on T4/Turing)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--private", action="store_true", help="Make output dataset private"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config/subset name when pushing (for benchmarking in one repo)",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Push as a pull request instead of directly",
    )
    parser.add_argument("--hf-token", help="Hugging Face API token (or set HF_TOKEN)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions after processing",
    )

    args = parser.parse_args()

    if args.pdf_column and args.image_column != "image":
        parser.error("--image-column and --pdf-column are mutually exclusive.")

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        task=args.task,
        table_mode=args.table_mode,
        image_column=args.image_column,
        pdf_column=args.pdf_column,
        output_column=args.output_column,
        overwrite=args.overwrite,
        blocks_column=args.blocks_column,
        page_range=args.page_range,
        split=args.split,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        model=args.model,
        private=args.private,
        config=args.config,
        create_pr=args.create_pr,
        hf_token=args.hf_token,
        verbose=args.verbose,
    )
