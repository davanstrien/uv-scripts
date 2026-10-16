#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gliformer>=0.1.2",
#     "datasets>=3.0",
#     "huggingface-hub",
#     "torch",
# ]
# ///
"""
Extract structured records from a text column of a Hugging Face dataset with GLiFormer.

GLiFormer (Knowledgator) is a DeBERTa encoder with a schema-conditioned structuring head:
you describe the records you want as a JSON schema at inference time — field names, and
optionally nested children — and it fills them with spans grounded in the text. No LLM,
no fine-tuning, and it runs on CPU at a few texts per second. A flat schema is
{"record name": ["field", "field", ...]}; a nested one uses dicts and lists, e.g.
{"paper": {"title": "", "authors": [{"name": "", "affiliation": ""}]}}.

Examples:
    # Pull (model, dataset, metric, score) rows out of ML paper abstracts — CPU is enough
    hf jobs uv run --flavor cpu-basic --secrets HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/gliner/raw/main/extract-structured.py \\
        CShorten/ML-ArXiv-Papers username/arxiv-results \\
        --text-column abstract \\
        --schema '{"result": ["model name", "dataset", "metric", "score"]}' \\
        --max-samples 200

    # Bigger runs: a small GPU, larger batches, schema from a file
    hf jobs uv run --flavor t4-small --secrets HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/gliner/raw/main/extract-structured.py \\
        CShorten/ML-ArXiv-Papers username/arxiv-results \\
        --text-column abstract --schema-file schema.json --batch-size 32

Output schema: original columns + `records` (the extracted JSON, as a string — load it
with `json.loads`) and `n_records` (how many records were found in the row).
Rows where extraction fails get an empty JSON object and a warning in the log.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

# tqdm reads TQDM_DISABLE at import time; Jobs logs have no TTY, so bars arrive as noise.
os.environ.setdefault("TQDM_DISABLE", "1")

import datasets
import huggingface_hub.utils
import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard

# datasets and huggingface_hub keep their own progress-bar switches, independent of TQDM_DISABLE.
datasets.disable_progress_bars()
huggingface_hub.utils.disable_progress_bars()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
# httpx logs every Hub request at INFO; that is noise in a Jobs log.
for noisy in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

SCRIPT_URL = (
    "https://huggingface.co/datasets/uv-scripts/gliner/raw/main/extract-structured.py"
)
DEFAULT_MODEL = "knowledgator/gliformer-base-v1"


def parse_args():
    p = argparse.ArgumentParser(
        description="Schema-driven structured extraction over a Hugging Face dataset with GLiFormer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "input_dataset",
        help=(
            "Input. A HF dataset ID ('org/dataset') or a local path to parquet/jsonl/json/csv "
            "file(s) — e.g. a bucket or folder mounted with '-v ./data:/input', then '/input/x.parquet'."
        ),
    )
    p.add_argument(
        "output_dataset",
        help="Output HF dataset ID ('user/output'). Results are always pushed to the Hub.",
    )
    p.add_argument(
        "--text-column",
        default="text",
        help="Text column to extract from (default: text)",
    )
    schema = p.add_mutually_exclusive_group(required=True)
    schema.add_argument(
        "--schema",
        help=(
            'Extraction schema as a JSON string. Flat: \'{"person": ["name", "role"]}\'. '
            'Nested: \'{"paper": {"title": "", "authors": [{"name": ""}]}}\'.'
        ),
    )
    schema.add_argument("--schema-file", help="Path to a JSON file holding the schema")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"GLiFormer checkpoint (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Span confidence threshold (default: 0.5)",
    )
    p.add_argument("--split", default="train", help="Dataset split (default: train)")
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process at most N rows (default: all)",
    )
    p.add_argument(
        "--batch-size", type=int, default=8, help="Texts per forward pass (default: 8)"
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device (default: auto — CUDA if available)",
    )
    p.add_argument(
        "--max-text-chars",
        type=int,
        default=8000,
        help="Truncate texts longer than this many characters (default: 8000, ~2k tokens; CPU time grows steeply past this)",
    )
    p.add_argument(
        "--private", action="store_true", help="Push the output dataset as private"
    )
    return p.parse_args()


def load_schema(args) -> Dict[str, Any]:
    raw = args.schema if args.schema is not None else open(args.schema_file).read()
    try:
        schema = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"Schema is not valid JSON: {e}\nGot: {raw[:300]}")
    if not isinstance(schema, dict) or not schema:
        sys.exit(
            'Schema must be a non-empty JSON object, e.g. \'{"person": ["name", "role"]}\''
        )
    return schema


def resolve_device(arg: str) -> str:
    if arg == "cpu":
        return "cpu"
    if arg == "cuda" and not torch.cuda.is_available():
        log.warning(
            "--device cuda requested but CUDA not available; falling back to CPU"
        )
        return "cpu"
    return "cuda" if (arg == "cuda" or torch.cuda.is_available()) else "cpu"


def is_local_path(s: str) -> bool:
    if s.startswith(("/", "./", "../")):
        return True
    return any(s.endswith(ext) for ext in (".parquet", ".jsonl", ".json", ".csv"))


def load_input(spec: str, split: str):
    if is_local_path(spec):
        ext = next(
            (e for e in (".parquet", ".jsonl", ".json", ".csv") if e in spec),
            ".parquet",
        )
        loader = {
            ".parquet": "parquet",
            ".jsonl": "json",
            ".json": "json",
            ".csv": "csv",
        }[ext]
        log.info("Loading local %s file(s): %s", loader, spec)
        return load_dataset(loader, data_files=spec, split="train")
    log.info("Loading HF dataset '%s' split=%s ...", spec, split)
    return load_dataset(spec, split=split)


def count_records(result: Any) -> int:
    """Count leaf records in a structuring result: every dict that is not just a container."""
    if isinstance(result, list):
        return sum(count_records(r) for r in result)
    if isinstance(result, dict):
        # A record has at least one scalar field; a pure container has only lists/dicts.
        scalar = any(not isinstance(v, (list, dict)) for v in result.values())
        return int(scalar) + sum(
            count_records(v) for v in result.values() if isinstance(v, (list, dict))
        )
    return 0


def build_card(
    args,
    schema: Dict[str, Any],
    n_rows: int,
    n_records: int,
    n_failed: int,
    elapsed_s: float,
    device: str,
) -> str:
    on_jobs = os.environ.get("JOB_ID") is not None
    hw = os.environ.get("ACCELERATOR") or ""
    origin = (
        (
            "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
            + (f" (`{hw}`)" if hw else "")
        )
        if on_jobs
        else "Generated"
    )
    tags = ["uv-script", "structured-extraction", "gliformer", "zero-shot", "bootstrap"]
    if on_jobs:
        tags.append("hf-jobs")
    tag_lines = "\n".join(f"  - {t}" for t in tags)
    schema_arg = f"--schema '{json.dumps(schema)}'"
    cmd = (
        f"hf jobs uv run --flavor {'cpu-basic' if device == 'cpu' else 't4-small'} --secrets HF_TOKEN \\\n"
        f"    {SCRIPT_URL} \\\n"
        f"    {args.input_dataset} {args.output_dataset} \\\n"
        f"    --text-column {args.text_column} {schema_arg}"
        + (f" \\\n    --max-samples {args.max_samples}" if args.max_samples else "")
    )
    return f"""---
tags:
{tag_lines}
---

# {args.output_dataset}

Structured records extracted from the `{args.text_column}` column of [`{args.input_dataset}`](https://huggingface.co/datasets/{args.input_dataset}) by [`{args.model}`](https://huggingface.co/{args.model}), a schema-conditioned encoder (no LLM). The schema was supplied at inference time; nothing was fine-tuned.

## Schema

```json
{json.dumps(schema, indent=2)}
```

## Provenance

| | |
|---|---|
| Source dataset | `{args.input_dataset}` (split `{args.split}`) |
| Text column | `{args.text_column}` |
| Model | `{args.model}` |
| Threshold | {args.threshold} |
| Rows processed | {n_rows} |
| Records extracted | {n_records} |
| Rows with extraction errors | {n_failed} |
| Device | `{device}` |
| Wall clock | {elapsed_s:.1f}s ({n_rows / max(elapsed_s, 1e-9):.2f} rows/s) |

## Columns

Original columns plus:

- `records` — the extracted JSON for the row, as a string. `json.loads(row["records"])` gives the
  schema shape back: `{{"record name": [{{"field": "span or null", ...}}, ...]}}` for a flat schema,
  the nested shape for a nested one. Missing fields are `null`.
- `n_records` — number of records found in the row (0 if nothing matched or extraction failed).

## Caveats

- These are **bootstrap labels**, not human-reviewed. Field wording, threshold and text length all change what comes out; re-run with a different schema wording before trusting a gap.
- Values are spans copied from the text. The model does not normalise, convert or infer; it can attach a value to the wrong record.
- Texts were truncated at {args.max_text_chars} characters before inference.

## Reproduction

{origin} with the [`extract-structured.py`]({SCRIPT_URL}) recipe from [uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:

```bash
{cmd}
```
"""


def main():
    args = parse_args()
    schema = load_schema(args)
    device = resolve_device(args.device)
    ds = load_input(args.input_dataset, args.split)

    if args.text_column not in ds.column_names:
        sys.exit(f"--text-column '{args.text_column}' not in {ds.column_names}")
    for col in ("records", "n_records"):
        if col in ds.column_names:
            sys.exit(f"Input already has a '{col}' column; rename it before running")

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    log.info("Processing %d rows; schema = %s", len(ds), json.dumps(schema))

    log.info("Loading %s on %s ...", args.model, device)
    from gliformer import GLiFormer

    model = GLiFormer.from_pretrained(args.model, load_tokenizer=True).to(device).eval()

    n_records = 0
    n_failed = 0
    started = time.time()

    def extract(batch: Dict[str, List]) -> Dict[str, List]:
        nonlocal n_records, n_failed
        texts = [(t or "")[: args.max_text_chars] for t in batch[args.text_column]]
        # Empty strings go through as-is; the model returns an empty result for them.
        try:
            with torch.inference_mode():
                results = model.structure(
                    texts, schema, threshold=args.threshold, batch_size=len(texts)
                )
            if not isinstance(results, list):
                results = [results]
        except Exception as e:
            # One bad row should not sink the batch: retry one at a time. Free the failed
            # batch's GPU memory first, or a CUDA OOM repeats on every single row.
            log.warning("batch extraction failed (%s); retrying rows individually", e)
            if device == "cuda":
                torch.cuda.empty_cache()
            results = []
            for text in texts:
                try:
                    with torch.inference_mode():
                        results.append(
                            model.structure(text, schema, threshold=args.threshold)
                        )
                except Exception as e2:
                    log.warning(
                        "row extraction failed (%d chars): %s", len(text), str(e2)[:200]
                    )
                    n_failed += 1
                    results.append({})
                    if device == "cuda":
                        torch.cuda.empty_cache()
        counts = [count_records(r) for r in results]
        n_records += sum(counts)
        return {
            "records": [json.dumps(r, ensure_ascii=False) for r in results],
            "n_records": counts,
        }

    ds = ds.map(extract, batched=True, batch_size=args.batch_size, desc="Extracting")

    elapsed = time.time() - started
    log.info(
        "Done. %d records from %d rows in %.1fs (%.2f rows/s); %d rows failed",
        n_records,
        len(ds),
        elapsed,
        len(ds) / max(elapsed, 1e-9),
        n_failed,
    )
    if n_records == 0:
        log.warning(
            "No records extracted. Check --text-column, the schema wording and --threshold."
        )

    log.info("Pushing to %s ...", args.output_dataset)
    ds.push_to_hub(args.output_dataset, private=args.private)
    DatasetCard(
        build_card(args, schema, len(ds), n_records, n_failed, elapsed, device)
    ).push_to_hub(args.output_dataset, repo_type="dataset")
    log.info("Done: https://huggingface.co/datasets/%s", args.output_dataset)


if __name__ == "__main__":
    main()
