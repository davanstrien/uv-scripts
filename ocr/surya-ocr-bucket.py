# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "surya-ocr==0.20.0",
#     "datasets>=3.1.0,<5.1",  # 5.1 needs huggingface_hub 1.x; the image copy (0.x) on PYTHONPATH wins
#     "huggingface-hub",
#     "pillow",
#     "imagecodecs",
#     "toolz",
#     "tqdm",
# ]
#
# # Pin surya-ocr to the known-good build (has the `surya.inference` engine layout
# # this recipe injects into); an unpinned/loosened resolve backtracks to an ancient
# # surya without it. huggingface-hub is left unpinned: at runtime PYTHONPATH puts the
# # pinned image's hub (with the buckets API) ahead of the venv, so no version tension.
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.20.1"
# python = "/usr/local/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/site-packages" }
# flavor = "a10g-small"
# secrets = ["HF_TOKEN"]
# ///
"""
Structured OCR over a **bucket of document files** (images + PDFs) with Datalab's
**Surya OCR 2** (`datalab-to/surya-ocr-2`, 650M, Qwen3.5-style) — no dataset
round-trip. This is the bucket-native sibling of `surya-ocr.py` (which reads a Hub
dataset column). Point it straight at an HF bucket of `.jp2`/`.png`/`.pdf`/... files.

Like the parent it produces *structured* OCR: per-block HTML + bounding boxes +
reading order + confidence. `--task` switches between `ocr` (full-page text),
`layout` (labelled regions), and `table` (HTML / rows-cols-cells).

INPUT — two interchangeable I/O strategies (`--io-mode`, default `auto`):
  mount  bucket mounted read-only at /in via `-v hf://buckets/<id>:/in:ro`; files
         are read straight off the FUSE mount. Zero ephemeral disk.
  copy   take a bucket id directly; the huggingface_hub library LISTs then batch-
         DOWNLOADS each `--batch-size` chunk to local temp, OCRs it, writes output,
         then deletes the temp batch. Avoids the known FUSE bulk-read stall; peak
         disk = one batch. `auto` picks copy for an `hf://buckets/...` input, mount
         for a local dir.

OUTPUT — one or both (>=1 required):
  --output-bucket  per page a `.md` (flattened reading-order text) AND a `.json`
                   (that page's structured `surya_blocks`), mirroring the input dir
                   structure, into a mounted dir OR an `hf://buckets/...` URL.
                   Streaming / O(1) memory, with resume-by-skip (a file whose
                   `.json` already exists is skipped) — the scalable path.
  --output-dataset a parquet dataset pushed to the Hub (one row per file:
                   file_name / markdown / surya_blocks / inference_info), like the
                   parent recipe. Convenient; buffered in memory (no image bytes by
                   default — use `--include-images` to embed page images).

ENGINE: Surya normally spawns a vLLM **server** (Docker), which can't run inside an
HF Job. This injects a custom in-process backend into Surya's `SuryaInferenceManager`
that runs vLLM's offline `LLM().chat()` engine (no server). Surya still owns all the
prompting, image preprocessing, and HTML/bbox parsing — we only swap the transport.

LICENSE NOTE: Surya's *code* is Apache-2.0 but the *weights* are a modified
OpenRAIL-M license — free for research, personal use, and startups under $5M
funding/revenue, restricted from competitive use against Datalab's API. Confirm you
are within those terms. https://huggingface.co/datalab-to/surya-ocr-2

HF Jobs — the `[tool.hf-jobs]` header above pins the vLLM image + the site-packages
python path (the model is the recent, version-sensitive `qwen3_5` architecture;
v0.20.1 is Surya's known-good build, and it puts python/vLLM under /usr/local, NOT
/usr/bin), plus flavor and HF_TOKEN. The header needs `hf` CLI 1.32+; CLI flags
override it:

    # copy input -> dataset output
    hf jobs uv run \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/surya-ocr-bucket.py \\
        hf://buckets/<ns>/<bucket> --io-mode copy --glob "*.jp2" \\
        --output-dataset <ns>/<out> --private

    # mount input -> per-file bucket output (mirrors dir structure)
    hf jobs uv run \\
        -v hf://buckets/<ns>/<bucket>:/in:ro \\
        -v hf://buckets/<ns>/<out-bucket>:/out \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/surya-ocr-bucket.py \\
        /in --io-mode mount --glob "*.jp2" --output-bucket /out

On your own GPU (no image): `uv run --with vllm==0.20.1 surya-ocr-bucket.py ...`

Model: datalab-to/surya-ocr-2  (package: surya-ocr, https://github.com/datalab-to/surya)
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError
from toolz import partition_all
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "datalab-to/surya-ocr-2"
# Surya's own vision-tiling bounds (from its vLLM backend), applied to the
# offline engine too so preprocessing matches the server path exactly.
MM_PROCESSOR_KWARGS = {"min_pixels": 3136, "max_pixels": 6291456}
TASKS = ("ocr", "layout", "table")
# Extensions read by default. `.jp2`/`.j2k` are first-class: the canonical test
# corpus (Library of Congress / Chronicling America) is all JPEG-2000.
DEFAULT_EXTENSIONS = ".jp2,.j2k,.png,.jpg,.jpeg,.tiff,.tif,.bmp,.webp,.pdf"
JP2_EXTENSIONS = {".jp2", ".j2k"}
PDF_EXTENSION = ".pdf"
BUCKET_PREFIX = "hf://buckets/"


# ---------------------------------------------------------------------------
# GPU / page-range helpers (verbatim from surya-ocr.py)
# ---------------------------------------------------------------------------


def check_cuda_availability() -> None:
    """Exit early with a clear message if there's no GPU."""
    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error(
            "Run on Hugging Face Jobs with: hf jobs uv run surya-ocr-bucket.py ... "
            "(the script's [tool.hf-jobs] header sets image + GPU; hf CLI 1.32+)"
        )
        sys.exit(1)
    logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


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


# ---------------------------------------------------------------------------
# Offline vLLM backend + Surya manager (verbatim from surya-ocr.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Result serialization (verbatim from surya-ocr.py)
# ---------------------------------------------------------------------------


def _html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


def serialize_pages(task: str, pages: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, structured-per-page) for one document's page results."""
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


def serialize_per_page(task: str, pages: List[Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Per-page (text, structured-dict). Reuses `serialize_pages` one page at a time
    so the per-file dataset row and the per-page bucket files share one code path."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    for page in pages:
        text, structured = serialize_pages(task, [page])
        out.append((text, structured[0]))
    return out


# ---------------------------------------------------------------------------
# Bucket-URL helpers (verbatim from pp-doclayout.py)
# ---------------------------------------------------------------------------


def is_bucket_url(s: str) -> bool:
    return s.startswith(BUCKET_PREFIX)


def parse_bucket_url(url: str) -> Tuple[str, str]:
    """Split `hf://buckets/ns/bucket/path/in/bucket` into (`ns/bucket`, `path/in/bucket`)."""
    if not is_bucket_url(url):
        raise ValueError(f"Not a bucket URL: {url}")
    rest = url[len(BUCKET_PREFIX) :].strip("/")
    parts = rest.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"Bucket URL must include namespace and bucket name: {url}")
    bucket_id = f"{parts[0]}/{parts[1]}"
    prefix = parts[2] if len(parts) > 2 else ""
    return bucket_id, prefix


# ---------------------------------------------------------------------------
# Image / PDF loading
# ---------------------------------------------------------------------------


def open_image(path: Path) -> Image.Image:
    """Open one image as RGB. Falls back to imagecodecs for JPEG-2000, which the
    image's bundled Pillow may not decode (no OpenJPEG)."""
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        if path.suffix.lower() in JP2_EXTENSIONS:
            import imagecodecs

            arr = imagecodecs.imread(str(path))
            logger.debug(f"Decoded {path.name} via imagecodecs (Pillow fallback)")
            return Image.fromarray(arr).convert("RGB")
        raise


def load_pages(
    kind: str,
    local_path: Path,
    load_pdf,
    page_indices: Optional[List[int]],
    pdf_dpi: int,
) -> List[Image.Image]:
    """A local document file -> list of RGB page images (1 for an image, N for a PDF)."""
    if kind == "pdf":
        images, _ = load_pdf(str(local_path), page_indices, dpi=pdf_dpi)
        return [im.convert("RGB") for im in images]
    return [open_image(local_path)]


# ---------------------------------------------------------------------------
# File listing + sources (mount vs copy)
# ---------------------------------------------------------------------------


@dataclass
class FileRef:
    """One input document. `key`/`rel` are the source-relative POSIX path (stable
    across runs -> resume) and drive output mirroring. `local_path` is set in mount
    mode; `bucket_file`/`bucket_path` in copy mode."""

    key: str
    rel: PurePosixPath
    kind: str  # "image" | "pdf"
    local_path: Optional[Path] = None
    bucket_file: Any = None
    bucket_path: Optional[str] = None


def classify(path_str: str, exts: set) -> Optional[str]:
    """Map a path to "pdf"/"image"/None using the allowed-extension set."""
    ext = PurePosixPath(path_str).suffix.lower()
    if ext == PDF_EXTENSION and PDF_EXTENSION in exts:
        return "pdf"
    if ext in exts:
        return "image"
    return None


def _shuffle_slice(
    refs: List[FileRef], shuffle: bool, seed: int, max_samples: Optional[int]
) -> List[FileRef]:
    refs.sort(key=lambda r: r.key)
    if shuffle:
        import random

        random.Random(seed).shuffle(refs)
    if max_samples:
        refs = refs[:max_samples]
    return refs


class MountSource:
    """Read files straight off a directory (a bucket mounted read-only at /in)."""

    mode = "mount"

    def __init__(self, root: Path, glob: str, exts: set):
        self.root = root
        self.glob = glob
        self.exts = exts

    def list_refs(
        self, shuffle: bool, seed: int, max_samples: Optional[int]
    ) -> List[FileRef]:
        refs: List[FileRef] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root)
            rel_posix = rel.as_posix()
            kind = classify(rel_posix, self.exts)
            if kind is None or not fnmatch(rel_posix, self.glob):
                continue
            refs.append(
                FileRef(
                    key=rel_posix,
                    rel=PurePosixPath(rel_posix),
                    kind=kind,
                    local_path=path,
                )
            )
        return _shuffle_slice(refs, shuffle, seed, max_samples)

    @contextmanager
    def materialize(
        self, chunk: List[FileRef], load_pdf, page_indices, pdf_dpi
    ) -> Iterator[List[Tuple[FileRef, Optional[List[Image.Image]]]]]:
        loaded: List[Tuple[FileRef, Optional[List[Image.Image]]]] = []
        for ref in chunk:
            loaded.append(
                (
                    ref,
                    _safe_load(
                        ref.kind, ref.local_path, load_pdf, page_indices, pdf_dpi
                    ),
                )
            )
        yield loaded  # nothing to clean up — reads are off the mount


class CopySource:
    """List + batch-download bucket files via huggingface_hub to local temp, then
    delete the batch. The non-FUSE path (sidesteps the bulk-read stall)."""

    mode = "copy"

    def __init__(self, bucket_url: str, glob: str, exts: set, hf_token: Optional[str]):
        from huggingface_hub import HfApi

        self.bucket_id, self.prefix = parse_bucket_url(bucket_url)
        self.glob = glob
        self.exts = exts
        self.hf_token = hf_token
        self.api = HfApi(token=hf_token)

    def list_refs(
        self, shuffle: bool, seed: int, max_samples: Optional[int]
    ) -> List[FileRef]:
        logger.info(
            f"Listing bucket {self.bucket_id}"
            + (f"/{self.prefix}" if self.prefix else "")
        )
        refs: List[FileRef] = []
        for item in self.api.list_bucket_tree(
            self.bucket_id, prefix=self.prefix or None, recursive=True
        ):
            path = getattr(item, "path", None)
            if not path:
                continue
            kind = classify(path, self.exts)
            if kind is None:
                continue
            rel = path[len(self.prefix) :].lstrip("/") if self.prefix else path
            if not fnmatch(rel, self.glob):
                continue
            refs.append(
                FileRef(
                    key=rel,
                    rel=PurePosixPath(rel),
                    kind=kind,
                    bucket_file=item,
                    bucket_path=path,
                )
            )
        logger.info(f"Found {len(refs)} matching file(s) in bucket")
        return _shuffle_slice(refs, shuffle, seed, max_samples)

    @contextmanager
    def materialize(
        self, chunk: List[FileRef], load_pdf, page_indices, pdf_dpi
    ) -> Iterator[List[Tuple[FileRef, Optional[List[Image.Image]]]]]:
        tmp = Path(tempfile.mkdtemp(prefix="surya-copy-"))
        try:
            # Pass the BucketFile objects from list_bucket_tree so download skips the
            # per-file metadata HEAD. Local names are index-keyed to avoid collisions.
            files = []
            locals_: List[Path] = []
            for i, ref in enumerate(chunk):
                local = tmp / f"{i:05d}{PurePosixPath(ref.bucket_path).suffix}"
                files.append((ref.bucket_file, str(local)))
                locals_.append(local)
            self.api.download_bucket_files(
                self.bucket_id, files=files, token=self.hf_token
            )
            loaded: List[Tuple[FileRef, Optional[List[Image.Image]]]] = []
            for ref, local in zip(chunk, locals_):
                if not local.exists():
                    logger.warning(f"Download missing for {ref.key}; skipping")
                    loaded.append((ref, None))
                    continue
                loaded.append(
                    (ref, _safe_load(ref.kind, local, load_pdf, page_indices, pdf_dpi))
                )
            yield loaded
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _safe_load(
    kind: str, local_path: Path, load_pdf, page_indices, pdf_dpi
) -> Optional[List[Image.Image]]:
    try:
        return load_pages(kind, local_path, load_pdf, page_indices, pdf_dpi)
    except Exception as e:  # noqa: BLE001 — a single bad file shouldn't kill the run
        logger.warning(f"Failed to load {local_path.name}: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


class BucketFilesSink:
    """Per page, write `<rel>.md` + `<rel>.json` (PDFs: `<stem>/page_NNN.{md,json}`),
    mirroring the input structure, to a mounted dir OR an `hf://buckets/...` URL.
    Streaming / O(1) memory. Resume-by-skip keys on the `.json` (written last)."""

    def __init__(self, output_target: str, hf_token: Optional[str], resume: bool):
        self.resume = resume
        self.api_mode = is_bucket_url(output_target)
        if self.api_mode:
            from huggingface_hub import HfApi

            self.bucket_id, self.prefix = parse_bucket_url(output_target)
            self.api = HfApi(token=hf_token)
            self.token = hf_token
            self._buffer: List[Tuple[bytes, str]] = []
            self._existing = self._load_existing() if resume else set()
        else:
            self.root = Path(output_target)
            self.root.mkdir(parents=True, exist_ok=True)

    @property
    def label(self) -> str:
        return (
            f"hf://buckets/{self.bucket_id}/{self.prefix}".rstrip("/")
            if self.api_mode
            else str(self.root)
        )

    def _join(self, rel: str) -> str:
        return f"{self.prefix}/{rel}".lstrip("/") if self.prefix else rel

    def _load_existing(self) -> set:
        existing = set()
        try:
            for item in self.api.list_bucket_tree(
                self.bucket_id, prefix=self.prefix or None, recursive=True
            ):
                p = getattr(item, "path", None)
                if p and p.endswith(".json"):
                    existing.add(p)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not pre-list output bucket for resume: {e}")
        if existing:
            logger.info(f"Resume: {len(existing)} output file(s) already present")
        return existing

    def _page_targets(self, ref: FileRef, n_pages: int) -> List[Tuple[str, str]]:
        if ref.kind == "pdf":
            stem = ref.rel.with_suffix("")
            return [
                (
                    str(stem / f"page_{i + 1:03d}.md"),
                    str(stem / f"page_{i + 1:03d}.json"),
                )
                for i in range(n_pages)
            ]
        return [(str(ref.rel.with_suffix(".md")), str(ref.rel.with_suffix(".json")))]

    def is_done(self, ref: FileRef) -> bool:
        # Resume applies to single-image files only; PDFs are re-rendered (idempotent
        # overwrite) since page count isn't known without opening them.
        if not self.resume or ref.kind == "pdf":
            return False
        json_rel = str(ref.rel.with_suffix(".json"))
        if self.api_mode:
            return self._join(json_rel) in self._existing
        return (self.root / json_rel).exists()

    def write_pages(
        self,
        ref: FileRef,
        per_page: List[Tuple[str, Dict[str, Any]]],
        pages: Optional[List[Image.Image]],
    ) -> None:
        targets = self._page_targets(ref, len(per_page))
        for (text, struct), (md_rel, json_rel) in zip(per_page, targets):
            md_bytes = text.encode("utf-8")
            json_bytes = json.dumps(struct, ensure_ascii=False).encode("utf-8")
            if self.api_mode:
                # .md first, .json last so a present .json marks the page complete.
                self._buffer.append((md_bytes, self._join(md_rel)))
                self._buffer.append((json_bytes, self._join(json_rel)))
            else:
                mp = self.root / md_rel
                mp.parent.mkdir(parents=True, exist_ok=True)
                mp.write_bytes(md_bytes)
                (self.root / json_rel).write_bytes(json_bytes)

    def write_error(self, ref: FileRef) -> None:
        # Write nothing on error so the file is retried on the next (resumed) run.
        pass

    def flush(self) -> None:
        if self.api_mode and self._buffer:
            self.api.batch_bucket_files(
                self.bucket_id, add=self._buffer, token=self.token
            )
            self._buffer = []

    def finalize(self, summary: Dict[str, Any]) -> None:
        self.flush()
        logger.info(f"Bucket files written to {self.label}")


class DatasetSink:
    """Buffer one row per file, push a parquet dataset at the end (like surya-ocr.py)."""

    def __init__(
        self,
        repo_id: str,
        *,
        hf_token: Optional[str],
        private: bool,
        config: Optional[str],
        create_pr: bool,
        include_images: bool,
        output_column: str,
        blocks_column: str,
    ):
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.private = private
        self.config = config
        self.create_pr = create_pr
        self.include_images = include_images
        self.output_column = output_column
        self.blocks_column = blocks_column
        self._rows: List[Dict[str, Any]] = []

    def is_done(self, ref: FileRef) -> bool:
        return False  # single push at the end; no per-file resume

    def write_pages(
        self,
        ref: FileRef,
        per_page: List[Tuple[str, Dict[str, Any]]],
        pages: Optional[List[Image.Image]],
    ) -> None:
        row = {
            "file_name": ref.key,
            "num_pages": len(per_page),
            self.output_column: "\n\n".join(t for t, _ in per_page),
            self.blocks_column: json.dumps(
                [s for _, s in per_page], ensure_ascii=False
            ),
        }
        if self.include_images and pages:
            # First page only (keeps a single Image column); documented limitation.
            row["image"] = pages[0]
        self._rows.append(row)

    def write_error(self, ref: FileRef) -> None:
        self._rows.append(
            {
                "file_name": ref.key,
                "num_pages": 0,
                self.output_column: "[SURYA ERROR]",
                self.blocks_column: None,
            }
        )

    def flush(self) -> None:
        pass  # single push at finalize

    def finalize(self, summary: Dict[str, Any]) -> None:
        from datasets import Dataset

        if not self._rows:
            logger.warning("No rows produced; nothing to push to the dataset.")
            return

        inference_entry = {
            "model": summary["model"],
            "model_name": "surya-ocr-2",
            "column_name": self.output_column,
            "blocks_column": self.blocks_column,
            "task": summary["task"],
            "table_mode": summary["table_mode"] if summary["task"] == "table" else None,
            "backend": "vllm-offline",
            "source": summary["source"],
            "io_mode": summary["io_mode"],
            "glob": summary["glob"],
            "page_range": summary["page_range"],
            "error_rate": summary["error_rate"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script": "surya-ocr-bucket.py",
        }
        for row in self._rows:
            row["inference_info"] = json.dumps([inference_entry])

        ds = Dataset.from_list(self._rows)
        if self.include_images and "image" in ds.column_names:
            try:
                from datasets import Image as HFImage

                ds = ds.cast_column("image", HFImage())
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not cast image column: {e}")

        logger.info(f"Pushing {len(ds)} rows to {self.repo_id}")
        push_kwargs = {
            "private": self.private,
            "token": self.hf_token,
            "max_shard_size": "500MB",
            "create_pr": self.create_pr,
            "commit_message": f"Add Surya OCR 2 {summary['task']} results ({len(ds)} files)"
            + (f" [{self.config}]" if self.config else ""),
        }
        if self.config:
            push_kwargs["config_name"] = self.config

        for attempt in range(1, 4):
            try:
                if attempt > 1:
                    logger.warning("Disabling XET (fallback to HTTP upload)")
                    os.environ["HF_HUB_DISABLE_XET"] = "1"
                ds.push_to_hub(self.repo_id, **push_kwargs)
                break
            except Exception as e:  # noqa: BLE001
                logger.error(f"Upload attempt {attempt}/3 failed: {e}")
                if attempt == 3:
                    logger.error("All upload attempts failed.")
                    raise
                time.sleep(30 * (2 ** (attempt - 1)))

        self._push_card(summary, len(ds))
        logger.info(f"Dataset: https://huggingface.co/datasets/{self.repo_id}")

    def _push_card(self, summary: Dict[str, Any], n_rows: int) -> None:
        try:
            from huggingface_hub import DatasetCard

            card = DatasetCard(
                _dataset_card(
                    source=summary["source"],
                    model=summary["model"],
                    task=summary["task"],
                    table_mode=summary["table_mode"],
                    io_mode=summary["io_mode"],
                    n_files=n_rows,
                    n_ok=summary["n_ok"],
                    output_column=self.output_column,
                    blocks_column=self.blocks_column,
                    processing_time=summary["processing_time"],
                )
            )
            card.push_to_hub(self.repo_id, token=self.hf_token)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not push dataset card: {e}")


def _dataset_card(
    source: str,
    model: str,
    task: str,
    table_mode: str,
    io_mode: str,
    n_files: int,
    n_ok: int,
    output_column: str,
    blocks_column: str,
    processing_time: str,
) -> str:
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

# Surya OCR 2 ({task}) on {source}

{task_desc.capitalize()} over document files in the HF bucket
`{source}`, using [Surya OCR 2](https://huggingface.co/{model}) (650M, Qwen3.5-based)
by Datalab, via the [`surya-ocr`](https://github.com/datalab-to/surya) package, run
as **offline vLLM batch inference** on Hugging Face Jobs (`surya-ocr-bucket.py`).

## Processing Details

- **Source bucket**: `{source}`
- **Model**: [{model}](https://huggingface.co/{model})
- **Task**: `{task}`{f" (table mode `{table_mode}`)" if task == "table" else ""}
- **I/O mode**: `{io_mode}`
- **Text column**: `{output_column}` (flattened, reading-order text per file)
- **Structured column**: `{blocks_column}` (JSON: per-page blocks with bbox / polygon / label / reading_order / confidence / html)
- **Files**: {n_files:,}
- **Processed OK**: {n_ok:,} / {n_files:,}
- **Processing time**: {processing_time}
- **Date**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## License note

Surya's code is Apache-2.0, but the model **weights** use a modified OpenRAIL-M
license: free for research, personal use, and startups under $5M funding/revenue,
restricted from competitive use against Datalab's API. See the
[model card](https://huggingface.co/{model}).

## Dataset Structure

One row per source file:
- `file_name`: source-relative path in the bucket
- `num_pages`: pages OCR'd (1 for an image, N for a PDF)
- `{output_column}`: flattened text (OCR), label outline (layout), or table HTML (table)
- `{blocks_column}`: structured result as a JSON string (one entry per page)
- `inference_info`: JSON list tracking models applied

Generated with [UV Scripts](https://huggingface.co/uv-scripts).
"""


# ---------------------------------------------------------------------------
# Predictor + processing loop
# ---------------------------------------------------------------------------


def build_predictor(task: str, table_mode: str, manager):
    """Return a `run(images) -> page_results` closure (verbatim dispatch from parent)."""
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

    return run


def process(
    refs: List[FileRef],
    source,
    run,
    task: str,
    sinks: List[Any],
    batch_size: int,
    load_pdf,
    page_indices: Optional[List[int]],
    pdf_dpi: int,
) -> Tuple[int, int, int, float, float]:
    """Resume-filter, then OCR file-by-file in batches.

    Returns (processed, ok, errors, io_secs, inf_secs). `io_secs` is time spent
    materializing batches (FUSE reads in mount mode; list-skip + batch download in
    copy mode); `inf_secs` is engine time (incl. one-time model load on the first
    batch). The split lets the mount-vs-copy benchmark isolate I/O from inference."""
    pending = [r for r in refs if not all(s.is_done(r) for s in sinks)]
    skipped = len(refs) - len(pending)
    if skipped:
        logger.info(f"Resume: skipping {skipped} already-complete file(s)")
    logger.info(f"Processing {len(pending)} file(s)")

    processed = ok = errors = 0
    io_secs = inf_secs = 0.0
    pbar = tqdm(total=len(pending), desc=f"Surya {task}")
    for chunk in partition_all(batch_size, pending):
        chunk = list(chunk)
        t_io = time.monotonic()
        with source.materialize(chunk, load_pdf, page_indices, pdf_dpi) as loaded:
            io_secs += time.monotonic() - t_io
            entries: List[Tuple[FileRef, List[Image.Image], int, int]] = []
            flat: List[Image.Image] = []
            for ref, pages in loaded:
                if not pages:
                    for s in sinks:
                        s.write_error(ref)
                    errors += 1
                    processed += 1
                    pbar.update(1)
                    continue
                entries.append((ref, pages, len(flat), len(pages)))
                flat.extend(pages)

            if flat:
                t_inf = time.monotonic()
                try:
                    results = run(flat)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Batch generate failed: {e}")
                    results = None
                inf_secs += time.monotonic() - t_inf

                if results is None:
                    for ref, _pages, _start, _count in entries:
                        for s in sinks:
                            s.write_error(ref)
                        errors += 1
                        processed += 1
                        pbar.update(1)
                else:
                    for ref, pages, start, count in entries:
                        per_page = serialize_per_page(
                            task, results[start : start + count]
                        )
                        for s in sinks:
                            s.write_pages(ref, per_page, pages)
                        ok += 1
                        processed += 1
                        pbar.update(1)

        for s in sinks:
            s.flush()
    pbar.close()
    return processed, ok, errors, io_secs, inf_secs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def resolve_io_mode(io_mode: str, input_source: str) -> str:
    if io_mode == "auto":
        return "copy" if is_bucket_url(input_source) else "mount"
    return io_mode


def main(args: argparse.Namespace) -> None:
    # Unlock full Xet bandwidth for the model download (repo convention).
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    # Surya reads settings from env at import; pin the checkpoint and forbid any
    # server autostart (we inject our own offline backend instead).
    os.environ["SURYA_MODEL_CHECKPOINT"] = args.model
    os.environ["SURYA_INFERENCE_AUTOSTART"] = "False"

    check_cuda_availability()
    start_time = datetime.now(timezone.utc)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    # Set the env var instead of login(): the old huggingface_hub in this image
    # raises KeyError on a Jobs token in login(). Hub calls read HF_TOKEN.
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    exts = {e.strip().lower() for e in args.extensions.split(",") if e.strip()}
    io_mode = resolve_io_mode(args.io_mode, args.input_source)

    # ---------- source ----------
    if io_mode == "copy":
        if not is_bucket_url(args.input_source):
            logger.error("--io-mode copy requires an hf://buckets/... input.")
            sys.exit(1)
        source = CopySource(args.input_source, args.glob, exts, hf_token)
    else:
        root = Path(args.input_source)
        if not root.is_dir():
            logger.error(
                f"--io-mode mount requires an existing directory (got {root}). "
                "Mount the bucket with -v hf://buckets/<id>:/in:ro and pass /in."
            )
            sys.exit(1)
        source = MountSource(root, args.glob, exts)
    logger.info(f"I/O mode: {io_mode}  Input: {args.input_source}")

    # ---------- sinks ----------
    sinks: List[Any] = []
    if args.output_bucket:
        sinks.append(
            BucketFilesSink(args.output_bucket, hf_token, resume=not args.no_resume)
        )
    if args.output_dataset:
        sinks.append(
            DatasetSink(
                args.output_dataset,
                hf_token=hf_token,
                private=args.private,
                config=args.config,
                create_pr=args.create_pr,
                include_images=args.include_images,
                output_column=args.output_column,
                blocks_column=args.blocks_column,
            )
        )

    # ---------- import Surya only after env is set ----------
    from surya.input.load import load_pdf
    from surya.settings import settings

    page_indices = parse_page_range(args.page_range)
    pdf_dpi = args.pdf_dpi if args.pdf_dpi else settings.IMAGE_DPI_HIGHRES

    t_list = time.monotonic()
    refs = source.list_refs(args.shuffle, args.seed, args.max_samples)
    list_secs = time.monotonic() - t_list
    if not refs:
        logger.error("No matching files found. Check --glob / --extensions / input.")
        sys.exit(1)
    logger.info(
        f"{len(refs)} file(s) listed in {list_secs:.1f}s | Model: {args.model}  "
        f"Task: {args.task}"
        + (f" (mode {args.table_mode})" if args.task == "table" else "")
    )

    # ---------- engine ----------
    backend = OfflineVLLMBackend(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
    )
    manager = make_manager(backend)
    run = build_predictor(args.task, args.table_mode, manager)

    processed, ok, errors, io_secs, inf_secs = process(
        refs,
        source,
        run,
        args.task,
        sinks,
        args.batch_size,
        load_pdf,
        page_indices,
        pdf_dpi,
    )

    processing_time = (
        f"{(datetime.now(timezone.utc) - start_time).total_seconds() / 60:.1f} min"
    )
    logger.info(
        f"Processed {processed} (ok {ok}, errors {errors}) in {processing_time}"
    )
    # Benchmark breakdown: separate listing + per-batch I/O from engine time so the
    # mount-vs-copy comparison isn't swamped by (identical) inference + model load.
    pages_per_sec = ok / io_secs if io_secs else 0.0
    logger.info(
        f"[timing] io_mode={io_mode} list={list_secs:.1f}s io={io_secs:.1f}s "
        f"inference={inf_secs:.1f}s files={ok} io_files_per_sec={pages_per_sec:.2f}"
    )

    summary = {
        "model": args.model,
        "task": args.task,
        "table_mode": args.table_mode,
        "source": args.input_source,
        "io_mode": io_mode,
        "glob": args.glob,
        "page_range": args.page_range,
        "n_ok": ok,
        "error_rate": (processed - ok) / processed if processed else 0.0,
        "processing_time": processing_time,
    }
    for s in sinks:
        s.finalize(summary)

    logger.info("Done! Surya OCR 2 (bucket) complete.")

    if args.verbose:
        import importlib.metadata

        logger.info("--- Resolved package versions ---")
        for pkg in [
            "surya-ocr",
            "vllm",
            "transformers",
            "torch",
            "datasets",
            "huggingface-hub",
            "pillow",
            "imagecodecs",
        ]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Surya OCR 2 (650M): structured OCR / layout / tables over a bucket of files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
I/O modes (--io-mode):
  auto    copy for an hf://buckets/... input, mount for a local dir (default)
  mount   read off a bucket mounted read-only at /in (-v hf://buckets/<id>:/in:ro)
  copy    list + batch-download via huggingface_hub to temp, OCR, delete the batch

Outputs (at least one required):
  --output-bucket   per-page .md + .json mirroring input structure (mounted dir or
                    hf://buckets/... URL); resumable, O(1) memory
  --output-dataset  parquet dataset push (one row per file)

HF Jobs: the script's [tool.hf-jobs] header pins vllm/vllm-openai:v0.20.1 + the
site-packages python path (qwen3_5 is version-sensitive), flavor and HF_TOKEN, so
`hf jobs uv run surya-ocr-bucket.py ...` needs no flags (hf CLI 1.32+).
Own GPU: uv run --with vllm==0.20.1 surya-ocr-bucket.py ...
""",
    )
    parser.add_argument(
        "input_source",
        help="Mounted dir (e.g. /in) OR hf://buckets/<ns>/<bucket>[/prefix]",
    )
    parser.add_argument(
        "--io-mode",
        choices=["auto", "mount", "copy"],
        default="auto",
        help="Input I/O strategy (default: auto)",
    )
    parser.add_argument(
        "--glob",
        default="*",
        help="fnmatch pattern over the source-relative path (default: '*'; "
        "e.g. '*.jp2'). Applied on top of --extensions.",
    )
    parser.add_argument(
        "--extensions",
        default=DEFAULT_EXTENSIONS,
        help=f"Comma-separated file extensions to read (default: {DEFAULT_EXTENSIONS})",
    )
    parser.add_argument(
        "--output-bucket",
        default=None,
        help="Per-file .md + .json output: a mounted dir OR hf://buckets/<id>[/prefix]",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="Output dataset repo ID (parquet, one row per file)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume-by-skip for --output-bucket (re-OCR everything)",
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
        "--page-range",
        default=None,
        help="Pages from PDFs, e.g. '0-5,7' (PDFs only)",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=None,
        help="DPI for PDF rendering (default: Surya's IMAGE_DPI_HIGHRES)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of input files, not pages (a PDF counts once)",
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
        help="Images per offline llm.chat batch AND per copy-mode download/cleanup unit (default: 16)",
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
        "--output-column",
        default="markdown",
        help="Dataset text column (default: markdown)",
    )
    parser.add_argument(
        "--blocks-column",
        default="surya_blocks",
        help="Dataset structured JSON column (default: surya_blocks)",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Embed the first page image in --output-dataset (memory-heavy)",
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
        help="Push dataset as a pull request instead of directly",
    )
    parser.add_argument("--hf-token", help="Hugging Face API token (or set HF_TOKEN)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions after processing",
    )
    return parser


def _print_banner() -> None:
    print(
        "Surya OCR 2 (bucket) — structured OCR / layout / tables over a bucket of files (650M)"
    )
    print("\nUsage:")
    print(
        "  uv run surya-ocr-bucket.py INPUT [--output-bucket ... | --output-dataset ...] [options]"
    )
    print("\nExamples:")
    print("  # copy a bucket of .jp2 -> a dataset")
    print("  uv run surya-ocr-bucket.py hf://buckets/me/news --io-mode copy \\")
    print("      --glob '*.jp2' --output-dataset me/news-ocr --private")
    print("\n  # mount a bucket -> per-file .md + .json in an output bucket")
    print("  uv run surya-ocr-bucket.py /in --io-mode mount --output-bucket /out")
    print("\nHF Jobs (the [tool.hf-jobs] header sets image + GPU; hf CLI 1.32+):")
    print("  hf jobs uv run \\")
    print("      -v hf://buckets/me/news:/in:ro -v hf://buckets/me/news-ocr:/out \\")
    print(
        "      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/surya-ocr-bucket.py \\"
    )
    print("      /in --io-mode mount --glob '*.jp2' --output-bucket /out")
    print("\nFor full help: uv run surya-ocr-bucket.py --help")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _print_banner()
        sys.exit(0)

    args = build_parser().parse_args()
    if not args.output_bucket and not args.output_dataset:
        build_parser().error(
            "at least one of --output-bucket or --output-dataset is required"
        )
    if args.no_resume and not args.output_bucket:
        logger.warning("--no-resume has no effect without --output-bucket")
    if args.include_images and not args.output_dataset:
        logger.warning("--include-images has no effect without --output-dataset")

    main(args)
