---
viewer: false
license: apache-2.0
tags:
  - uv-script
  - ner
  - zero-shot
  - gliner
  - hf-jobs
---

# GLiNER UV Scripts

Zero-shot extraction over Hugging Face datasets with the GLiNER model family. Pass entity types or a record schema at runtime — no fine-tuning, no LLM, runs on CPU.

For text **classification** with [GLiNER2](https://github.com/fastino-ai/GLiNER2) (a separate library from Fastino): zero-shot, then fine-tuning on your own labels, see [`uv-scripts/classification`](https://huggingface.co/datasets/uv-scripts/classification#zero-shot-first-then-fine-tune-gliner2).

| Script | What it does | Output |
|---|---|---|
| `extract-entities.py` | Extract entities from a text column with a custom set of types ([GLiNER](https://github.com/urchade/GLiNER)) | New `entities` column (list of `{start, end, text, label, score}`) |
| `extract-structured.py` | Fill a JSON record schema from a text column ([GLiFormer](https://github.com/Knowledgator/GLiFormer)) | New `records` column (JSON string in the schema's shape) + `n_records` |

## Quick start

Run on any HF dataset with a text column. No setup — `uv` resolves dependencies inline.

```bash
# Local CPU (small samples)
uv run extract-entities.py \
    librarian-bots/model_cards_with_metadata \
    yourname/model-cards-entities \
    --text-column card \
    --entity-types Person Organization Dataset Model Framework \
    --max-samples 100
```

## On HF Jobs

```bash
# CPU job — fine for small/medium datasets, free or near-free
hf jobs uv run --flavor cpu-basic --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/gliner/raw/main/extract-entities.py \
    librarian-bots/model_cards_with_metadata \
    yourname/model-cards-entities \
    --text-column card \
    --entity-types Person Organization Dataset Model Framework \
    --max-samples 1000

# GPU job — worth it once you're processing >~1000 samples
hf jobs uv run --flavor t4-small --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/gliner/raw/main/extract-entities.py \
    librarian-bots/model_cards_with_metadata \
    yourname/model-cards-entities \
    --text-column card \
    --entity-types Person Organization Dataset Model Framework \
    --device cuda \
    --batch-size 32
```

## Structured extraction (`extract-structured.py`)

[GLiFormer](https://huggingface.co/knowledgator/gliformer-base-v1) is a 264M-parameter DeBERTa encoder (Apache-2.0) with a schema-conditioned structuring head. You describe the records you want as JSON and it fills the fields with spans copied from the text, including nested parent–child records. Same shape as an LLM extraction prompt, at encoder speed.

```bash
# (model, dataset, metric, score) rows from ML paper abstracts — cpu-basic is enough
hf jobs uv run --flavor cpu-basic --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/gliner/raw/main/extract-structured.py \
    CShorten/ML-ArXiv-Papers yourname/arxiv-results \
    --text-column abstract \
    --schema '{"result": ["model name", "dataset", "metric", "score"]}' \
    --max-samples 200
```

Schemas:

- **Flat** — `{"record name": ["field", "field", ...]}` → `{"record name": [{"field": "span or null", ...}, ...]}`
- **Nested** — dicts and lists, e.g. `{"paper": {"title": "", "authors": [{"name": "", "affiliation": ""}]}}`; the output keeps that shape. Put a long schema in a file and pass `--schema-file schema.json`.

The `records` column is a JSON **string** (nested, ragged records don't map cleanly onto a fixed Arrow schema); `json.loads` it downstream. Field wording matters as much as it does for an LLM prompt — "model name" and "model" give different results — so try two wordings on `--max-samples 50` before a full run. Values are spans, not normalised values, and a value can land in the wrong record. Texts are truncated at `--max-text-chars` (default 8,000 characters): attention cost grows steeply with length on CPU, and the structuring head gets vaguer on long inputs, so abstract- or paragraph-sized rows work best. Chunk long documents upstream.

Measured on 200 arXiv abstracts (~1,000 characters each) with the four-field schema above: **cpu-basic 0.74 rows/s** (269 s), **t4-small with `--batch-size 32` 12 rows/s** (17 s), identical records on both. A few thousand short rows is a CPU job; past that, or with paragraph-length rows, take the T4. GPU memory also scales with length: with eager attention a T4 (15 GB) ran out of memory on 12k-character rows, while an A10G handled 20k-character model cards at batch 1. The recipe retries an out-of-memory batch row by row and records `{}` for rows that still fail, so check `n_records == 0` rows and the log before trusting a run.

Quality is best on short, focused texts. Tried on six full model cards (5–20k characters): a compact card like `Qwen/Qwen2.5-7B-Instruct` gave one clean record (name, 7.61B, architecture, 131k context); long cards with benchmark tables came back as many fragmentary records with benchmark/metric/score columns misaligned. Feed it paragraphs or sections rather than whole cards, and review before reuse.

GLiFormer also does NER, classification and relation extraction from the same checkpoint; this recipe covers structuring only. For plain NER use `extract-entities.py` above.

## Reading from local files or a mounted bucket

The `input_dataset` argument also accepts local file paths (parquet, jsonl, json, csv). Useful when the input is staged in a [Storage Bucket](https://huggingface.co/docs/hub/storage-buckets) — typical pattern for multi-stage pipelines where an upstream Job has prepared the data:

```bash
hf jobs uv run --flavor t4-small --secrets HF_TOKEN \
    -v hf://buckets/yourname/working-data:/input \
    https://huggingface.co/datasets/uv-scripts/gliner/raw/main/extract-entities.py \
    /input/data.parquet \
    yourname/output-entities \
    --text-column text --entity-types Person Organization Location \
    --device cuda --batch-size 32
```

Local paths are detected heuristically — anything starting with `/`, `./`, `../`, or ending in a known data extension is treated as a file path; otherwise the argument is interpreted as a HF dataset ID.

## Recommended entity-type vocabularies

GLiNER is open-vocabulary, so any string works. Some starting points:

- **General news/web text**: `Person Organization Location Date Event`
- **ML/AI text (e.g. model cards)**: `Person Organization Dataset Model Framework Metric License`
- **Legal/policy**: `Person Organization Court Statute Date Jurisdiction`
- **Biomedical**: `Drug Disease Gene Protein Symptom`

Quality drops on very abstract or polysemous types — start simple, iterate.

## Models

Default: `urchade/gliner_multi-v2.1` (multilingual, ~600 MB). Override with `--gliner-model`.

Other useful checkpoints:
- `urchade/gliner_small-v2.1` — English, faster
- `urchade/gliner_large-v2.1` — English, larger / higher quality
- `knowledgator/gliner-multitask-large-v0.5` — multitask (NER + classification + relation)

See the [Knowledgator org](https://huggingface.co/knowledgator) and [urchade's models](https://huggingface.co/urchade) for the full set.

## Pairing with Label Studio

Output of this script is a Hugging Face dataset of texts + extracted entities. To put those entities in front of human reviewers, see the `bootstrap-labels` skill (or the workflow it documents): pull this dataset's predictions into a Label Studio project for review, then export a corrected dataset back to the Hub.

## Caveats

- GLiNER predictions are **bootstrap labels** — useful as a starting point, not as ground truth. Plan a review pass before downstream training.
- Texts longer than `--max-text-chars` (default 8000) are truncated. Long-form documents may need chunking + reassembly.
- Entity types are case-sensitive labels in output. Pass them as you want them to appear.
