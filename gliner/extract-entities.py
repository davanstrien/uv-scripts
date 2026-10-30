#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gliner>=0.2.16",
#     "transformers<5",  # the default model's tokenizer misloads on transformers 5
#     "sentencepiece",
#     "protobuf",
#     "datasets>=3.0",
#     "huggingface-hub",
#     "tqdm",
# ]
# ///
"""
Extract named entities from a text column of a HuggingFace dataset using GLiNER.

GLiNER is a zero-shot NER model: you pass a list of entity types at inference
time (e.g., "Person", "Organization", "Dataset"), no fine-tuning needed.

Examples:
    # Local CPU run on a small sample
    uv run extract-entities.py \\
        librarian-bots/model_cards_with_metadata \\
        my-username/model-cards-entities \\
        --text-column card \\
        --entity-types Person Organization Dataset Model Framework \\
        --max-samples 100

    # On HF Jobs with a cheap GPU
    hf jobs uv run --flavor t4-small \\
        --secrets HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/gliner/raw/main/extract-entities.py \\
        librarian-bots/model_cards_with_metadata \\
        my-username/model-cards-entities \\
        --text-column card \\
        --entity-types Person Organization Dataset Model Framework \\
        --max-samples 5000

Output schema: original dataset columns + new `entities` column, where each
row's `entities` is a list of dicts:
    {"start": int, "end": int, "text": str, "label": str, "score": float}
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List

import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset
from huggingface_hub import DatasetCard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Bootstrap NER labels with GLiNER over a HuggingFace dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "input_dataset",
        help=(
            "Input. Either a HF dataset ID (e.g. 'org/dataset') or a local path "
            "to parquet/jsonl file(s) — useful when running in HF Jobs with a mounted "
            "bucket ('-v hf://buckets/<ns>/<bucket>:/input') or a local folder synced "
            "automatically ('-v ./data:/input', hub >= 1.22), then pass '/input/cards.parquet'."
        ),
    )
    p.add_argument(
        "output_dataset",
        help=(
            "Output HF dataset ID (e.g. 'user/output'). The script always pushes results "
            "to a HF dataset repo regardless of where input came from."
        ),
    )
    p.add_argument(
        "--text-column",
        default="text",
        help="Name of the text column to run NER over (default: 'text')",
    )
    p.add_argument(
        "--entity-types",
        nargs="+",
        required=True,
        help="Space-separated entity types to extract (e.g., Person Organization Date)",
    )
    p.add_argument(
        "--gliner-model",
        default="urchade/gliner_multi-v2.1",
        help="GLiNER model ID (default: urchade/gliner_multi-v2.1, multilingual)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for entity extraction (default: 0.5)",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split to process (default: train)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process at most N samples (default: all)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for inference (default: auto — uses CUDA if available)",
    )
    p.add_argument(
        "--max-text-chars",
        type=int,
        default=8000,
        help="Truncate texts longer than this many characters (default: 8000)",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Push the output dataset as private",
    )
    return p.parse_args()


def resolve_device(arg: str) -> str:
    if arg == "cpu":
        return "cpu"
    if arg == "cuda":
        if not torch.cuda.is_available():
            log.warning("--device cuda requested but CUDA not available; falling back to CPU")
            return "cpu"
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_entity(ent: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "start": int(ent["start"]),
        "end": int(ent["end"]),
        "text": str(ent["text"]),
        "label": str(ent["label"]),
        "score": float(ent.get("score", 0.0)),
    }


def build_card(args, n_processed: int, n_entities: int, elapsed_s: float, device: str) -> str:
    return f"""---
license: apache-2.0
tags:
  - ner
  - gliner
  - zero-shot
  - bootstrap
  - uv-script
size_categories:
  - n<10K
---

# {args.output_dataset}

Bootstrap NER dataset produced by [`{args.gliner_model}`](https://huggingface.co/{args.gliner_model}) over [`{args.input_dataset}`](https://huggingface.co/datasets/{args.input_dataset}).

Generated using [`uv-scripts/gliner/extract-entities.py`](https://huggingface.co/datasets/uv-scripts/gliner).

## Provenance

| | |
|---|---|
| Source dataset | `{args.input_dataset}` (split `{args.split}`) |
| Text column | `{args.text_column}` |
| Bootstrap model | `{args.gliner_model}` |
| Entity types | `{', '.join(args.entity_types)}` |
| Confidence threshold | {args.threshold} |
| Samples processed | {n_processed} |
| Total entities extracted | {n_entities} |
| Inference device | `{device}` |
| Wall clock | {elapsed_s:.1f}s ({n_processed / max(elapsed_s, 1e-9):.2f} samples/s) |

## Schema

Original `{args.input_dataset}` columns plus an `entities` column:

```python
entities: list of {{
    "start": int,    # character offset, inclusive
    "end": int,      # character offset, exclusive
    "text": str,     # the matched span
    "label": str,    # one of {args.entity_types}
    "score": float,  # GLiNER confidence in [0, 1]
}}
```

## Caveats

- These are **bootstrap labels**, not human-reviewed. Treat low-confidence (< 0.7) entities as candidates for review.
- GLiNER is zero-shot: changing `--entity-types` changes what it extracts, but quality varies by entity type.
- Long texts were truncated at {args.max_text_chars} characters before inference.
"""


def is_local_path(s: str) -> bool:
    """Heuristic: treat as local path if it starts with / or ./ or contains a known data extension."""
    if s.startswith(("/", "./", "../")):
        return True
    if any(s.endswith(ext) for ext in (".parquet", ".jsonl", ".json", ".csv")):
        return True
    if "*" in s and any(ext in s for ext in (".parquet", ".jsonl", ".json", ".csv")):
        return True
    return False


def load_input(spec: str, split: str):
    """Load either a HF dataset by ID, or a local parquet/jsonl path (incl. globs)."""
    if is_local_path(spec):
        ext = ".parquet" if ".parquet" in spec else \
              ".jsonl" if spec.endswith(".jsonl") else \
              ".json" if spec.endswith(".json") else \
              ".csv" if spec.endswith(".csv") else ".parquet"
        loader = {".parquet": "parquet", ".jsonl": "json", ".json": "json", ".csv": "csv"}[ext]
        log.info("Loading local %s file(s): %s", loader, spec)
        return load_dataset(loader, data_files=spec, split="train")
    log.info("Loading HF dataset '%s' split=%s ...", spec, split)
    return load_dataset(spec, split=split, streaming=False)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    ds = load_input(args.input_dataset, args.split)

    if args.text_column not in ds.column_names:
        sys.exit(
            f"--text-column '{args.text_column}' not in {ds.column_names}"
        )

    if args.max_samples is not None:
        n = min(args.max_samples, len(ds))
        ds = ds.select(range(n))
        log.info("Selected %d samples", n)
    else:
        log.info("Processing full split: %d samples", len(ds))

    log.info("Loading GLiNER %s on %s ...", args.gliner_model, device)
    from gliner import GLiNER

    model = GLiNER.from_pretrained(args.gliner_model)
    if device == "cuda":
        model = model.to("cuda")
    model.eval()

    n_entities = 0
    started = time.time()

    def add_entities(batch: Dict[str, List]) -> Dict[str, List]:
        nonlocal n_entities
        texts = [
            (t or "")[: args.max_text_chars] for t in batch[args.text_column]
        ]
        entities_per_row = []
        for text in texts:
            if not text.strip():
                entities_per_row.append([])
                continue
            try:
                ents = model.predict_entities(
                    text, args.entity_types, threshold=args.threshold
                )
            except Exception as e:
                log.warning("predict_entities failed: %s; returning []", e)
                ents = []
            normalized = [normalize_entity(e) for e in ents]
            n_entities += len(normalized)
            entities_per_row.append(normalized)
        return {"entities": entities_per_row}

    log.info("Running inference (batch_size=%d) ...", args.batch_size)
    ds = ds.map(
        add_entities,
        batched=True,
        batch_size=args.batch_size,
        desc="Extracting entities",
    )

    elapsed = time.time() - started
    log.info(
        "Done. %d entities across %d samples in %.1fs (%.2f samples/s)",
        n_entities, len(ds), elapsed, len(ds) / max(elapsed, 1e-9),
    )

    log.info("Pushing to %s ...", args.output_dataset)
    ds.push_to_hub(args.output_dataset, private=args.private)

    DatasetCard(build_card(args, len(ds), n_entities, elapsed, device)).push_to_hub(
        args.output_dataset, repo_type="dataset"
    )

    log.info("Done: https://huggingface.co/datasets/%s", args.output_dataset)


if __name__ == "__main__":
    main()
