---
name: setfit-on-jobs
description: Train and evaluate a single-label SetFit text classifier using Hugging Face Jobs, preparing a small labeled Hub dataset from supplied documents when needed, then return a verified Hub model and inference example. Use for few-shot text classification or when the user explicitly requests SetFit on Jobs.
---

# SetFit on Hugging Face Jobs

Train a Hub-hosted SetFit model with a bounded run and an honest evaluation. Start from labeled text or prepare a small labeled sample from supplied documents. Keep the task to one classification schema and one train/evaluate cycle; taxonomy discovery, automatic active-learning loops and corpus-wide annotation are separate workflows.

## Prepare a small dataset when labels are missing

Inspect representative documents and use the user's categories and definitions. If categories are unspecified or overlap materially, resolve that before assigning labels. Label a bounded sample of actual documents with the agent or a zero-shot teacher; preserve source IDs, original source locations/revisions, label origin and the labeling instructions. Leave genuinely ambiguous examples for review rather than forcing them into a class. Do not substitute invented class-description sentences for real documents or describe agent-labeled training as zero-shot SetFit.

Reserve evaluation documents before labeling or sampling training rows. Prefer independently reviewed evaluation labels. If the agent labels both splits, explicitly report student–agent agreement; do not call those labels human ground truth. The current recipe requires labeled evaluation data: it has no skip-evaluation or unlabeled-inference mode. Without suitable evaluation labels, prepare candidates and explain the missing prerequisite instead of manufacturing an accuracy score.

Prepare a small Hub dataset with `text`, `label` and source IDs, plus explicit train/validation splits and a card describing label provenance. Use the user's namespace and intended visibility; preserve the source corpus. Local files or documents in a bucket need only yield this small training/evaluation sample—the full corpus can stay where it is. Extract text first if inputs are PDFs or other document formats. Bucket mounts alone do not extract documents or create labels; this recipe's supported handoff is a Hub dataset ID and its output is a model, not corpus annotations.

## Check the inputs

Inspect the dataset config, text and label columns, splits and missing values before launching a Job. Compute class counts from the saved rows and check they sum to each split's size. This recipe supports one label per text. Labels may be human or model generated; record their origin. Evaluation against synthetic labels measures agreement with those labels, not independent human accuracy.

Choose a held-out labeled split explicitly. The recipe otherwise prefers validation, then test, then carves training data. Check related documents and duplicates do not cross the evaluation boundary. For repeated experiments, use validation for decisions and reserve test for the final assessment.

Measure text lengths with the selected body's tokenizer and check language suitability. Report truncation only when measured lengths exceed the configured limit; do not turn a possible limitation into an observed finding. The default English MiniLM body and 256-token truncation may not suit other languages or decisions requiring a whole document. Disclose material truncation or choose suitable settings within the budget.

Keep `--max-seq-length` within the chosen body's supported context window. If prepared inputs were already shortened, restore the original text before testing longer context. Follow the body's task-prefix instructions consistently across training, evaluation and inference; the recipe does not add prefixes automatically. Record these preprocessing requirements in the handoff.

## Run the existing recipe

Use the tested recipe rather than recreating SetFit training code. The executable link below pins a tested revision. The canonical recipe is https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-setfit.py; verify it includes the pinned revision's input-validation fixes before switching to the mirror:

https://raw.githubusercontent.com/davanstrien/uv-scripts/11e6ae6250a2ae3dc67cd074679a6da70cac13b0/classification/train-setfit.py

Read its help for flags and check `hf jobs uv run --help` if needed. CPU is practical for small experiments; use a GPU for faster training, particularly with larger models, longer texts or more classes. A small first run can use `cpu-basic`; changing to `t4-small` accelerates the same model and training settings. Choose hardware within the user's budget. Four examples per class is a smoke test, not a quality target. `--num-samples` caps examples per class; it does not mean total training rows.

Example for a small pilot (replace the output namespace):

```bash
hf jobs uv run --flavor cpu-basic --timeout 10m --detach --secrets HF_TOKEN \
  https://raw.githubusercontent.com/davanstrien/uv-scripts/11e6ae6250a2ae3dc67cd074679a6da70cac13b0/classification/train-setfit.py \
  fancyzhx/ag_news YOUR_NAMESPACE/ag-news-setfit \
  --eval-split test --num-samples 8 --max-eval-samples 200 \
  --sampling-strategy undersampling --max-minutes 5 --private
```

Use the requested output visibility. The destination is a model repo. Pass credentials through Jobs secrets. Record the source revision and actual invocation; preserve sampled row IDs and seed when reproducing a comparison.

Inspect both `hf jobs logs` and `hf jobs inspect`: submission or a closed log stream does not establish completion. Keep retries within the authorized budget. A runtime guard refusal is a reason to reduce the workload or explain the limitation; do not silently override it or increase hardware. Stop after the requested run rather than starting an automatic tuning loop.

## Evaluate and hand off

Report actual training examples per class, evaluation size, accuracy, macro F1 and a majority baseline on the same evaluation rows. For an initial usefulness assessment, compare with TF-IDF plus logistic regression when practical. Match the exact training and evaluation rows; a baseline trained on more labels answers a different question. Additional zero-shot comparisons are optional, not part of every SetFit run.

Derive any confusion matrix and error examples from that model's saved predictions and evaluation row IDs. The diagonal divided by the matrix total must match its accuracy. If only aggregate metrics are available, report that limit rather than inventing diagnostics. Inspect a few errors when predictions are available, and disclose dropped rows, missing classes and the limitations of a small single-seed evaluation.

Verify the uploaded model reloads and predicts the expected label names. SetFit has a sentence-transformer body and classification head; load it with `SetFitModel.from_pretrained`, not `AutoModelForSequenceClassification`:

```python
from setfit import SetFitModel

model = SetFitModel.from_pretrained("YOUR_NAMESPACE/ag-news-setfit")
print(model.predict(["The team won the championship."]))
```

Return the model and Job links, exact command, measured results, short inference example and a concrete recommendation about suitability. Verify artifact links exist and, for private outputs, check visibility through authenticated metadata. A working training run and useful classification quality are separate findings.
