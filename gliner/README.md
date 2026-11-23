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

<a href="https://huggingface.co/uv-scripts"><img src="https://huggingface.co/spaces/uv-scripts/README/resolve/main/lockup.svg" alt="uv scripts" width="180"></a>

# GLiNER UV Scripts

<a href="https://huggingface.co/uv-scripts"><picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md-dark.svg"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md.svg" alt="Follow uv-scripts on Hugging Face"></picture></a>

Zero-shot named-entity recognition over Hugging Face datasets using [GLiNER](https://github.com/urchade/GLiNER). Pass a list of entity types at runtime — no fine-tuning required.

For text **classification** with [GLiNER2](https://github.com/fastino-ai/GLiNER2) (a separate library from Fastino): zero-shot, then fine-tuning on your own labels, see [`uv-scripts/classification`](https://huggingface.co/datasets/uv-scripts/classification#zero-shot-first-then-fine-tune-gliner2).

| Script | What it does | Output |
|---|---|---|
| `extract-entities.py` | Extract entities from a text column with a custom set of types | New `entities` column (list of `{start, end, text, label, score}`) |

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
