---
viewer: false
tags: [uv-script, classification, fine-tuning, few-shot, zero-shot, decision-model, setfit, gliner2, kev, vllm, structured-outputs, hf-jobs]
---

# Classification Scripts

Text classification on [HF Jobs](https://huggingface.co/docs/hub/jobs): label a dataset with a model that needs no training, or train your own classifier from labelled examples.

If you have seen [Jev](https://docs.typesafe.ai/introduction) and other "System One" models: the
models these scripts train are small, open versions of the same idea. They read a piece of data and
return a label with a probability, and you can train one on your own labels. For example, this
[demo](https://huggingface.co/spaces/davanstrien/hub-task-tagger) suggests task tags for any Hub
dataset; its model was fine-tuned with `train-gliner2.py` in 17 minutes.

To try it on your own account, this command fine-tunes a classifier for British Library book titles
(Fiction / Non-fiction). Training takes about 2 minutes on a `t4-small` and costs about $0.02.
Accuracy goes from 0.767 zero-shot to 0.907 fine-tuned. You get a private model repo, and its card
shows both scores next to the majority-class baseline. Copy and paste it as it is: the model goes
to your own account.

```bash
hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-gliner2.py \
  biglam/blbooksgenre gliner2-blbooks-genre \
  --dataset-config title_genre_classifiction --text-column title
```

You need the `hf` CLI, signed in, and Jobs credit: see the [Jobs quickstart](https://huggingface.co/docs/hub/jobs-quickstart).

| Script | What it does |
|--------|--------------|
| [`classify-gliner2.py`](#zero-shot-first-then-fine-tune-gliner2) | **Label a dataset** with GLiNER2: zero-shot from label names, or with a `train-gliner2.py` model |
| [`train-gliner2.py`](#zero-shot-first-then-fine-tune-gliner2) | **Fine-tune** [GLiNER2](https://huggingface.co/fastino), a small model (74M–287M) that already classifies zero-shot, and report the zero-shot score next to the fine-tuned one |
| [`train-setfit.py`](#few-shot-with-setfit-train-setfitpy) | **Few-shot** train a classifier from 8-64 labels per class with [SetFit](https://huggingface.co/docs/setfit) — runs on CPU or GPU |
| [`train-classifier.py`](#fine-tune-a-classifier-train-classifierpy) | **Fine-tune** an encoder into a classifier (default: [LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M)) and push it to the Hub |
| [`train-kev.py`](#fine-tune-a-kev-decision-model-train-kevpy) | **Fine-tune** the open [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) decision model on your own questions, with Kev's own training code |
| [`classify-dataset.py`](#zero-shot-classification-classify-datasetpy) | **Zero-shot** classify a dataset with an instruction LLM (SmolLM3 + vLLM, structured outputs) |
| [`classify-dataset-sglang.py`](#zero-shot-classification-classify-datasetpy) | Zero-shot variant on SGLang (reasoning-aware `<think>` models) |

Pick by how many labels you have:

| Labels you have | Use | Hardware |
|---|---|---|
| none | `classify-gliner2.py --labels ...` for a cheap first pass; `classify-dataset.py` when the task needs an LLM's reasoning | small GPU (CPU works at ~1.4 rows/s); GPU |
| ~8-64 per class | `train-setfit.py` | CPU supported; GPU for faster training |
| a few hundred to a few thousand | `train-gliner2.py`, which also shows you what zero-shot already gets | small GPU (`t4-small`) |
| a few thousand or more | `train-classifier.py` | GPU |

The rungs chain: bootstrap labels with `classify-gliner2.py` or `classify-dataset.py`, review them,
then train a small dedicated model on what you kept. Each rung is one command, so you can move up
as you collect more labels.

## Zero-shot first, then fine-tune (GLiNER2)

Label a dataset with your own list of labels, see how far zero-shot gets you, then fine-tune a
small model on your labels, in minutes and for a few cents on one GPU. The result is a model
that returns a label and a probability for every row, and is small enough to run on a CPU.

[GLiNER2](https://github.com/fastino-ai/GLiNER2) ([models from Fastino](https://huggingface.co/fastino)) is a small encoder that reads the label names
as part of its input, so it classifies with no training at all, and fine-tuning teaches it what
your labels mean in your data. (For entity extraction with the original GLiNER library, see
[`uv-scripts/gliner`](https://huggingface.co/datasets/uv-scripts/gliner).) Two scripts:

- **`train-gliner2.py`** scores the base model zero-shot, fine-tunes it on your labels, and
  scores it again on the same held-out rows. The model card reports both scores next to the
  majority-class baseline.
- **`classify-gliner2.py`** labels a whole dataset. Pass `--labels` for zero-shot, or `--model`
  with a `train-gliner2.py` output; the tasks and labels are read from the model repo.
  Labels are passed as separate words; quote a label with spaces:
  `--labels World Sports Business "Science and technology"`. A fine-tuned model is passed
  by name: `--model gliner2-blbooks-genre`. A name on its own means your account; use `org/name`
  for an organisation.

### Quick start

```bash
# fine-tune: British Library book titles -> Fiction / Non-fiction
hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-gliner2.py \
  biglam/blbooksgenre gliner2-blbooks-genre \
  --dataset-config title_genre_classifiction --text-column title

# label a dataset with that model
hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-gliner2.py \
  biglam/blbooksgenre blbooks-genre-predictions \
  --dataset-config title_genre_classifiction --text-column title --model gliner2-blbooks-genre

# or skip training: zero-shot from label names
hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-gliner2.py \
  fancyzhx/ag_news ag-news-topics --split test \
  --labels World Sports Business "Science and technology" --task-name topic
```

The second command labels the same rows the model was trained on, so it shows the workflow, not
the model's accuracy; the held-out scores are on the model card. For real use, point it at data
the model has not seen. Outputs are **private by default** (`--public` to opt out).

These commands use the default base model, `fastino/gliner2.5-multi-v1` (multilingual). For
English text, `--base-model fastino/gliner2.5-base-v1` is smaller and faster;
`gliner2.5-small-v1` is the fastest and loses about 4 points on the 52-tag example. For English zero-shot, try
`fastino/GLiNER2.5-Decide`. Sizes and speeds: [Choosing a model size](GLINER2-NOTES.md#choosing-a-model-size).

### Results

Fine-tuning beat zero-shot on every dataset below. The public-dataset runs took 2 to 30 minutes of
training on one GPU and cost $0.01 to $0.20; the 52-tag example took 17 minutes and about $1.50.

| Dataset | Task | Labels | Train rows × epochs | Train time | Train cost | Metric | Majority floor | Zero-shot | Fine-tuned |
|---|---|---|---|---|---|---|---|---|---|
| [`biglam/blbooksgenre`](https://huggingface.co/datasets/biglam/blbooksgenre) (book titles) | single-label | 2 | 1,562 × 5 | 141s | $0.02 | accuracy | 0.747 | 0.767 (0.753–0.782) | **0.907** (0.897–0.925) |
| [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) | single-label | 4 | 2,000 × 2 | 125s | $0.01 | accuracy | 0.268 | 0.718 | **0.852** |
| [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) | multi-label | 28 | 2,000 × 2 | 216s | $0.02 | micro F1 | — | 0.265 | **0.464** |
| [`SetFit/TREC-QC`](https://huggingface.co/datasets/SetFit/TREC-QC), two tasks in one model | single-label ×2 | 6 + 50 | 5,452 × 3 | 1,761s | $0.20 | accuracy | 0.276 / 0.246 | 0.542 / 0.468 | **0.954 / 0.876** |
| same, on `a10g-small`, default batch size, bf16 | single-label ×2 | 6 + 50 | 5,452 × 3 | 509s | $0.14 | accuracy | 0.276 / 0.246 | not run | **0.944 / 0.872** |
| [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) (reviews; 15% truncated at 2,000 characters) | single-label | 2 | 1,000 × 1 | 122s | $0.01 | accuracy | 0.500 | 0.777 | **0.840** |
| Hub dataset task tags ([worked example](#worked-example-tagging-hub-datasets)), `--base-model fastino/gliner2.5-base-v1 --label-augmentation off`, on `rtx-pro-6000` | single-label choice from a fixed set | 52 | 16,000 × 5 | 17 min | ~$1.50 | top-1 in the owner's tags | 0.320 (always "text-generation") | 0.102 | **0.690** (2 seeds: 0.695 / 0.686) |

Most rows are single, deliberately small runs that test the script, not tuned results. Seed ranges,
out-of-memory history and GPU comparisons are in [GLINER2-NOTES.md](GLINER2-NOTES.md).

### Larger or fixed label sets

For tens of labels scored together (a taxonomy, a fixed tag list), or data in a bucket, see
[Larger or fixed label sets](GLINER2-NOTES.md#larger-or-fixed-label-sets) in the notes:
`--labels-file`, `--label-augmentation off`, local `--train-file`/`--eval-file` and `--export-predictions`.

### Worked example: tagging Hub datasets

A GLiNER2.5-base model fine-tuned with this script on 16,000 Hub datasets suggests task tags for
a dataset from its column names and first row, among the 52 tags the Hub offers. Its first
suggestion matches one of the owner's tags 69% of the time on 3,000 newer datasets from owners it
never saw. Owners' tags are a noisy target: in a hand-checked sample, about 1 in 10 datasets was
missing a tag that fits.

[Try the demo](https://huggingface.co/spaces/davanstrien/hub-task-tagger): paste a dataset id and
see the suggested tags, the owner's tags and the exact text the model read. On a free 2-vCPU Space
one prediction takes about 0.7–1 s.
[Model](https://huggingface.co/davanstrien/hub-task-tagger-gliner2.5-base) · [Notes on how it was trained](GLINER2-NOTES.md#a-larger-label-set-52-hub-task-tags)

### Good to know

- **Single-label and multi-label** are auto-detected from the label column; repeat `--label-column` to train several tasks in one model.
- **Label names are part of the prompt.** Real names (`Fiction`, `Sports`) work; integer codes make zero-shot meaningless.
- **Out of GPU memory, it restarts at a smaller batch size** and stops before pushing if even batch size 1 fails. Many labels or long texts want an `a10g-small`.
- **Always pass `--timeout`.** Older `hf` CLIs ignore the scripts' `[tool.hf-jobs]` header and stop the Job after 30 minutes.

More behaviour details, tested commands, findings and dead ends: [GLINER2-NOTES.md](GLINER2-NOTES.md).

## Few-shot with SetFit (`train-setfit.py`)

An alternative when you have only a handful of labelled examples per class (8-64) and want a sentence-transformer model.

Trains a [SetFit](https://huggingface.co/docs/setfit) classifier from a handful of labelled
examples per class. SetFit finetunes a sentence-transformer body on contrastive pairs, then fits a
logistic regression head on the resulting embeddings.

**Runs on CPU or GPU.** CPU is practical for small few-shot experiments. Use a GPU for faster
training, particularly with larger models, longer texts or more classes. The same model and
training settings work on either; the recipe uses the available accelerator automatically.

```bash
# 8 labels per class, on CPU
hf jobs uv run --flavor cpu-basic --timeout 20m --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-setfit.py \
  fancyzhx/ag_news username/ag-news-setfit --num-samples 8

# Same model and training settings on a GPU for faster training
hf jobs uv run --flavor t4-small --timeout 20m --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-setfit.py \
  fancyzhx/ag_news username/ag-news-setfit-gpu --num-samples 8
```

### Measured

8 labels per class, seed 42, evaluated on each dataset's own held-out split (capped at 500
examples, 1000 for banking77):

| Dataset | Classes | Labels used | Body | Flavor | Training | Accuracy | Macro F1 |
|---|---|---|---|---|---|---|---|
| [`SetFit/enron_spam`](https://huggingface.co/datasets/SetFit/enron_spam) | 2 | 16 | MiniLM-L6 | `cpu-basic` | 78s | 0.924 | 0.924 |
| [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) | 4 | 32 | MiniLM-L6 | `cpu-basic` | 118s | 0.804 | 0.807 |
| [`legacy-datasets/banking77`](https://huggingface.co/datasets/legacy-datasets/banking77) | 77 | 616 | MiniLM-L6 | `t4-small` | 18s | 0.803 | 0.789 |
| [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) | 6 | 48 | MiniLM-L6 | `cpu-basic` | 216s | 0.370 | 0.325 |

**Single seed each — these do not rank models or predict your dataset.** Few-shot results vary
substantially with which examples happen to get sampled; SetFit's own benchmarks report mean and
standard deviation across ten seeds for exactly this reason. Run your own task before trusting
any of these numbers.

### Good to know

- **Default body**: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (22M), chosen for CPU speed. Swap it with `--body-model`; set `--max-seq-length` within its context window and add any task prefix it needs yourself. See [Choosing another body or longer context](SETFIT-NOTES.md#choosing-another-body-or-longer-context).
- **Single-label only.** A multi-label column exits with a pointer to `train-classifier.py`.
- **Beat more than the majority baseline.** Every run reports it, but `emotion` beat it by under 2 points; compare with TF-IDF plus logistic regression or zero-shot on the same rows. See [Compare more than the majority baseline](SETFIT-NOTES.md#compare-more-than-the-majority-baseline).
- **Many classes: watch the pair count.** Pairs grow with the square of the training-set size; at 77 classes use `--sampling-strategy undersampling`. See [Many classes](SETFIT-NOTES.md#many-classes-watch-the-pair-count).
- **It refuses runs projected above `--max-minutes`** (default 60). On `biglam/hansard_speech` (2.7M speeches, 28 parties) it refused to train and produced no score, and [the worked failure](SETFIT-NOTES.md#real-world-data-a-worked-failure) shows why.
- **Load it with `SetFitModel.from_pretrained(repo)`**, not `AutoModelForSequenceClassification`: a SetFit model is a sentence-transformer body plus a scikit-learn head.

Evaluation split, metrics, dropped rows and `--private`: [SETFIT-NOTES.md](SETFIT-NOTES.md#behaviour-details).

## Fine-tune a classifier (`train-classifier.py`)

Fine-tunes a text-classification encoder on any Hub dataset and pushes the trained model
back to the Hub — download, train, evaluate, push, and reload-verify in one job.

- **Default model**: [LiquidAI/LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M) — a bidirectional encoder that LiquidAI reports beats ModernBERT-base on GLUE/SuperGLUE, and handles 8,192-token documents. Any Hub encoder works via `--model` (ModernBERT, BERT, DeBERTa, …).
- **Single-label and multi-label**, auto-detected from the label column (`ClassLabel`/string/int → cross-entropy; list of labels → BCE + per-label threshold tuning).
- **Round-trippable artifacts**: standard architectures produce standard models; encoders without a classification head (like LFM2.5) get a generic mean-pooling head pushed as custom code, so `AutoModelForSequenceClassification.from_pretrained(..., trust_remote_code=True)` always works.

```bash
# single-label (ag_news has a ClassLabel column)
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-classifier.py \
  fancyzhx/ag_news username/news-classifier

# multi-label (go_emotions has a list-of-labels column)
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-classifier.py \
  google-research-datasets/go_emotions username/emotion-classifier --label-column labels
```

Key options: `--model`, `--max-length` (512 default; up to 8192 with
`--gradient-checkpointing` and a small `--batch-size` on a10g/a100), `--epochs`, `--lr`,
`--batch-size`, `--max-samples` (smoke runs), `--eval-split` (auto-detects
validation/test, or holds out 10% of train). Run `uv run train-classifier.py --help` for all.

### Worked example: classify dataset cards by task

[`davanstrien/dataset-cards-with-task-categories`](https://huggingface.co/datasets/davanstrien/dataset-cards-with-task-categories)
contains 21k Hub dataset cards (frontmatter stripped) labelled with their `task_categories`
metadata — a real multi-label task over long documents:

```bash
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-classifier.py \
  davanstrien/dataset-cards-with-task-categories username/dataset-card-task-classifier \
  --label-column labels --max-length 1024 --batch-size 8 --grad-accum 2
```

The output model predicts likely task categories from a card's prose — e.g. for suggesting
metadata on datasets that lack it.

### Training a standard encoder instead

`--model answerdotai/ModernBERT-base` (or any encoder with a native classification head)
produces a plain, vLLM-servable model — pair it with
[`uv-scripts/vllm`](https://huggingface.co/datasets/uv-scripts/vllm)'s
`classify-dataset.py` for large-scale batch inference with the model you just trained.

## Fine-tune a Kev decision model (`train-kev.py`)

[Kev](https://github.com/jaredpalmer/kev) is an open decision model by Jared Palmer. It answers
yes/no, multiple-choice and rating questions about a text, with a probability for each answer.
[Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) is a rank-16 LoRA adapter and a small pointer
head on Qwen3.5-4B-Base. This script fine-tunes the released Kev-4B on your own labelled questions.

The script is a thin launcher. It pins one Kev commit and the Kev-4B v1.0 weights, installs Kev,
and runs Kev's own `kev.train`, `kev.benchmark` and `kev.calibrate`. All training and evaluation
code is Kev's. Kev's own recipes run on Modal (the `kev-finetune` skill) or on a GPU you have
([Fine-Tune on Your Own Data](https://github.com/jaredpalmer/kev#fine-tune-on-your-own-data));
use this script when your data and credit are already on the Hub.

First put `train.jsonl` and `development.jsonl` in a bucket folder (the snippet at the end of this
section shows one way). Each line is one record in Kev's format; a record can carry more than one
question:

```json
{"state": "<the text>",
 "questions": {"newsgroup": {"type": "choice",
                             "instructions": "Which newsgroup was this post published in?",
                             "criteria": {"alt.atheism": null, "comp.graphics": null},
                             "label": "comp.graphics"}}}
```

Then run:

```bash
hf jobs uv run --flavor a100-large --timeout 1h --secrets HF_TOKEN \
  -v hf://buckets/<you>/kev-data:/mnt/kev \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-kev.py \
  /mnt/kev/newsgroups /mnt/kev/runs/newsgroups-v1
```

The script checks every record and the output folder before it installs anything. The output
folder must be new or empty. It gets `checkpoint/`, `finetuned-development/` and
`baseline-development/` (Kev benchmark reports for the fine-tuned and the released model),
`finetuned-development/calibration.json` (when `kev.calibrate` succeeds) and a `run.json` summary.
Each result is copied to the bucket as soon as it is written. On Jobs, the script refuses an output
folder that is not on a mounted bucket.

### Measured

| Task | Train / test records | Flavor | Time | Cost | Released Kev-4B | Fine-tuned |
|---|---|---|---|---|---|---|
| [`SetFit/20_newsgroups`](https://huggingface.co/datasets/SetFit/20_newsgroups) (20 classes) | 846 / 426 (1,000 / 500 prepared) | `a100-large` | 31 min | about $1.29 | 62.4% | 72.3% |

One seed, 2026-10-30. The paired difference is +9.9 points (95% CI +6.1 to +13.6). Training took
about 25 of the 31 minutes (846 records, 2 epochs).

**The fine-tuned model is overconfident**: mean confidence 0.82 against accuracy 0.72 (ECE 0.100).
The fine-tuned checkpoint runs at temperature 1.0. `calibration.json` reports what one temperature
fitted on your development records would do; nothing in the checkpoint changes. On 20 Newsgroups,
the out-of-fold fit (`workload_oof`) brings ECE to 0.028 and mean confidence to 0.73, with accuracy
unchanged (ECE change -0.072, 95% CI -0.092 to -0.023). To use the fitted value
(`workload_temperature`, 1.52 here), set `KEV_TEMPERATURE=1.52` when you serve or benchmark the
checkpoint with Kev's commands, or pass `LoadOptions(temperature=1.52)` when you load it in Python.

### Good to know

- **Long records are dropped.** `kev.train`'s default context is 384 state tokens (Kev's `--max_state`
  raises it; this script does not expose it). `kev.train` drops longer records and prints how many; `kev.benchmark` drops them too and lists them in `rejected.json`. On
  20 Newsgroups, after cutting posts at 2,000 characters, this dropped 154 of 1,000 training and 74 of
  500 test posts.
- **Raise `--timeout` before you add data or epochs.** Training time grows with records × epochs.
  If the Job hits the timeout, the results already copied stay in the bucket.
- **Everything is pinned:** the Kev commit, the Kev-4B weights (by commit, because a tag can move)
  and the Qwen3.5 base revision. If the `kev-1.0` tag moves, the script stops with a message.
- **No extra kernels.** Kev's Modal images add `flash-linear-attention` (and, in `modal_app.py`,
  `causal-conv1d`).
  In our `a100-large` runs they gave no speed-up, so this script leaves them out.
- **Smoke test first:** `--max-steps 5` runs the whole path in about 6 minutes.
- **No 20 Newsgroups checkpoint is published.** `SetFit/20_newsgroups` declares no licence.

<details><summary>Turn a labelled dataset into Kev records (the 20 Newsgroups example)</summary>

```python
import json
from pathlib import Path

from datasets import load_dataset

data = load_dataset("SetFit/20_newsgroups")
labels = sorted(set(data["train"]["label_text"]))
out = Path("newsgroups")
out.mkdir(exist_ok=True)
for split, count, name in [("train", 1000, "train.jsonl"), ("test", 500, "development.jsonl")]:
    rows = data[split].filter(lambda row: len(row["text"].strip()) >= 50).shuffle(seed=42).select(range(count))
    with (out / name).open("w") as f:
        for row in rows:
            question = {"type": "choice", "instructions": "Which newsgroup was this post published in?",
                        "criteria": {label: None for label in labels}, "label": row["label_text"]}
            f.write(json.dumps({"state": row["text"].strip()[:2000], "questions": {"newsgroup": question}}) + "\n")
```

Then copy the folder to your bucket: `hf buckets sync ./newsgroups hf://buckets/<you>/kev-data/newsgroups`.

</details>

## Zero-shot classification (`classify-dataset.py`)

Label a dataset with an instruction LLM and no training data. You give a list of labels; the
model picks one per row, and guided decoding (structured outputs) makes sure every answer is one
of your labels. The default model is
[HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B); any instruction
model works via `--model`. The result is your dataset with a new `classification` column.

Two scripts do this. `classify-dataset.py` runs on vLLM and is the one to start with.
`classify-dataset-sglang.py` runs the same task on SGLang, for reasoning models that write
`<think>` traces; its options differ (`--reasoning`, `--save-reasoning`, `--batch-size`,
`--grammar-backend`), so check its `--help`.

### Quick start

A GPU is required. Run on Jobs with the vLLM image:

```bash
hf jobs uv run --flavor l4x1 --image vllm/vllm-openai:latest --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-dataset.py \
  --input-dataset stanfordnlp/imdb \
  --column text \
  --labels "positive,negative" \
  --output-dataset username/imdb-classified \
  --max-samples 100 --shuffle
```

`--max-samples` with `--shuffle` takes a random sample, which matters for datasets sorted by
date or label. Drop both to label the whole split.

### With reasoning and label descriptions

```bash
hf jobs uv run --flavor l4x1 --image vllm/vllm-openai:latest --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-dataset.py \
  --input-dataset user/support-tickets \
  --column content \
  --labels "bug,feature_request,question,other" \
  --label-descriptions "bug:code or product not working as expected,feature_request:asking for new functionality,question:seeking help or clarification,other:general comments or feedback" \
  --enable-reasoning \
  --output-dataset username/tickets-classified
```

With `--enable-reasoning` the model thinks step by step before it answers, and the output also
has `reasoning` and `parsing_success` columns. Reasoning mode turns off structured outputs: the
model must end with `{"label": "..."}`, and rows where that cannot be parsed are marked in
`parsing_success`. It is slower, but you can read why each label was chosen.

### Options

| Option | What it does |
|---|---|
| `--model` | Model to use (default `HuggingFaceTB/SmolLM3-3B`) |
| `--label-descriptions` | `label:description,...` pairs that tell the model what each label means |
| `--enable-reasoning` | Think before answering; adds `reasoning` and `parsing_success` columns |
| `--split` | Split to process (default `train`) |
| `--max-samples` | Label only the first N rows (or N random rows with `--shuffle`) |
| `--shuffle`, `--shuffle-seed` | Shuffle before `--max-samples` (seed default 42) |

Run `uv run classify-dataset.py --help` for all options.

### Good to know

- **Speed**: about 50-100 texts/second for SmolLM3-3B on an A10, and 20-50 for 7B models.
  `l4x1` is a good start; use `a10g-large` or larger for 7B+ models or out-of-memory errors.
- **Text handling**: texts shorter than 3 characters and empty values are skipped; texts are
  truncated to 4,000 characters.
- **Label names matter.** Use clear, distinct names, add `--label-descriptions` when names are
  ambiguous, and try a larger model for nuanced tasks.
- **vLLM version**: `ImportError: cannot import name GuidedDecodingParams` means the vLLM
  version does not match; the script requires `vllm>=0.6.6`.
