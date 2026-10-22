# GLiNER2 scripts: tested runs, findings and dead ends

Notes behind `train-gliner2.py` and `classify-gliner2.py`. The README has what you need to run them;
this file records what was tried, what worked, what did not, and the exact runs the README numbers
come from. Much of it was run and written up by coding agents (Claude) and checked by a person.

## Tested commands

Each README command was run from the branch before merge (`t4-small`, 2026-09-23):

| Command | Result | Time | Cost |
|---|---|---|---|
| train, `biglam/blbooksgenre` `title_genre_classifiction`, defaults | zero-shot 0.793 → fine-tuned 0.925 (174-row carve-out) | 149 s training | ~$0.02 |
| classify with that model, same dataset | 1,736 rows labelled | 20 s | <$0.01 |
| classify zero-shot, `fancyzhx/ag_news` test, 4 labels | 7,600 rows labelled | 96 s | ~$0.02 |
| train, `fancyzhx/ag_news`, `--max-train-samples 2000 --epochs 2` | 0.733 → 0.873 (2,000 test rows) | 137 s training, 298 s whole Job | ~$0.03 |

Re-run after the bump to `gliner2` 2.1.0 (`t4-small`, seed 42, 2026-10-22), same BL books command:

| Run | Fine-tuned (174 rows) | Training |
|---|---|---|
| 2.1.0, defaults | 0.902 | 145 s |
| 2.1.0, `--label-augmentation off` | 0.937 | 135 s |
| 2.0.0, `--label-augmentation off` | 0.931 | 135 s |
| classify with an existing fine-tuned model, 2.1.0 | 500 rows in 13 s | – |

Zero-shot is 0.793 in all three. One seed each, and five 2.0.0 seeds on this split ranged
0.897–0.925, so these differences are within the noise.

The ag_news README row (0.718 → 0.852) and this run (0.733 → 0.873) differ because the 2,000
scored test rows are a different random sample. Expect roughly 0.72 → 0.85–0.87.

## Larger or fixed label sets

For tens of labels that are always scored together (a taxonomy, a fixed tag list), and for data
you keep in a bucket instead of a Hub dataset:

```bash
hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \
  -v hf://buckets/username/my-bucket:/bucket \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-gliner2.py \
  --train-file /bucket/train.jsonl \
  --eval-file calibration=/bucket/calibration.jsonl --eval-file development=/bucket/development.jsonl \
  --labels-file /bucket/labels.json --label-column labels --label-augmentation off \
  --base-model fastino/gliner2.5-base-v1 \
  --no-push --output-dir /bucket/runs/gliner2 --export-predictions /bucket/runs/gliner2/predictions
```

- `--labels-file` fixes the label set and its order for training, zero-shot and evaluation, so a
  label that is rare or missing in the training data is still an option.
- `--label-augmentation off`: gliner2's trainer by default renames labels to "label 1", "label 2",
  … in half the rows and drops up to half of them, which helps a general zero-shot model. With a
  fixed label set it cost 2–3 points of top-1 on the 52-tag example.
- `--eval-file NAME=PATH` (repeatable) scores each split in full and in file order.
- `--export-predictions` writes every row's probability and raw logit for every label, so you can
  fit your own temperature or thresholds on one split and check them on another.
- `--no-push` keeps the model in `--output-dir` instead of creating a Hub repo.

## Measured table notes

On the T4, TREC and IMDB ran out of memory at the default batch size of 16, and the script restarted
them at batch size 4 with 4 gradient accumulation steps. Before that fallback existed, the TREC run
"completed" with 1,006 of 1,020 steps skipped and scored 0.576 / 0.484, barely above zero-shot. Many
labels are also slow: TREC trained at 9 rows/s against 55 rows/s for the 2-label task, because every
label is part of the input. For hundreds of labels, use `train-classifier.py`. Prediction was not the
limit: the 300 IMDB reviews were scored at batch size 32 on the T4 with no fallback.

The same BL books run (seed 42) scored 0.925 on a T4, 0.937 on an A10G in fp32 and 0.931 on an A10G in
bf16 — one or two eval rows apart, so hardware and precision are not a way to gain accuracy, but they
are one more thing to hold constant when you compare runs.

Two upstream options are deliberately absent: in `gliner2` 2.0.0, gradient checkpointing crashes
with the 2.5 models, and a LoRA run trained but its final checkpoint did not load for scoring (LoRA
also did not fix the 56-label out-of-memory case on a T4).

The BL books row is the mean and range of five seeds; the other rows are one seed each. Read the
range before you compare two runs: `--seed` also picks the carve-out rows, and the zero-shot model
never changes, so its 3-point spread is what 174 eval rows do to the number on their own. A
difference smaller than that between two runs is not a result. Use a dataset with a fixed
`--eval-split`, and as many eval rows as you can get, when you want to compare runs.

The ag_news, go_emotions and TREC rows are deliberately small runs (capped training rows, 2–3 epochs) that
test the script, not tuned results. The BL books row trains on the full 1,562 titles; its eval is a 10%
carve-out, so it is not comparable with published numbers for that dataset.

## A larger label set: 52 Hub task tags

`train-gliner2.py` was used to fine-tune a tagger that suggests a Hub dataset's task tags from its
column names and first row (the worked example in the README).

- 16,000 training datasets, 1,000 calibration and 3,000 development datasets; calibration and
  development are newer datasets from owners not in the training data.
- `gliner2.5-base-v1`, `--labels-file` (52 tags), multi-label, `--label-augmentation off`, 5 epochs,
  `rtx-pro-6000`, about 17 minutes of training.
- Development top-1 in the owner's tags: base 0.695 / 0.686 (two seeds), small 0.653, zero-shot
  0.102. Always answering "text-generation" scores 0.320, so zero-shot is below a constant guess.
- `--label-augmentation upstream` (gliner2's default) scored 0.664: turning it off added 2–3 points.
  This was measured on `gliner2` 2.0.0. In that version, label augmentation sometimes trained the
  real label as a negative (fixed in 2.1.0, [GLiNER2#187](https://github.com/fastino-ai/GLiNER2/pull/187)),
  so the gap may be smaller on 2.1.0. Not re-measured; `off` is still the right choice for a fixed label set.
- The two seeds differ by about 1 point, almost all of it on one owner's 33 near-identical
  datasets that the model labels tabular-regression or tabular-classification depending on the seed.
- Owner tags are noisy: in a hand-checked sample, about 1 in 10 datasets was missing a tag that fits.

How it was set up, if you want to do something similar with your own label list:

- **Input text:** the dataset's column names and types, then its first row, built from the dataset
  viewer's preview and cut to about 370 tokens. Keep the exact same builder for training and
  prediction; a small difference in the text is a different input.
- **Labels:** a fixed `--labels-file` of 52 tags, multi-label (`labels` is a list per row), with
  `--label-augmentation off`.
- **Split by time and owner:** the evaluation rows are newer datasets from owners who are not in
  the training data, so the score is not inflated by near-duplicate datasets from the same owner.
- **Two eval files** (`--eval-file calibration=… --eval-file development=…`) and
  `--export-predictions`: thresholds are chosen on one file and checked on the other.

## Choosing a model size

| Base model                             | Params | Use it when                                | Hub-tags top-1 | CPU latency per row (free Space, 2 vCPU) | GPU (L4, fp16) |
| -------------------------------------- | ------ | ------------------------------------------ | -------------- | ---------------------------------------- | -------------- |
| [`fastino/gliner2.5-small-v1`](https://huggingface.co/fastino/gliner2.5-small-v1) | 74M    | speed matters most                         | 0.653          | ~0.3 s                                   | ~19 ms         |
| [`fastino/gliner2.5-base-v1`](https://huggingface.co/fastino/gliner2.5-base-v1) | 194M   | English text; the best accuracy per second | 0.690          | ~0.7–1 s                                 | ~19 ms         |
| [`fastino/gliner2.5-multi-v1`](https://huggingface.co/fastino/gliner2.5-multi-v1) (default) | 287M   | non-English or mixed-language text         | not measured   | —                                        | —              |
| [`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide) | 486M   | English classification; best zero-shot     | not measured   | —                                        | —              |

On a GPU, base and small are equally fast per row; the difference only shows on a CPU. For speed on
a CPU, use plain fp32 PyTorch (see below for what did not work).

GLiNER2.5-Decide on BL books: 0.851 zero-shot (default 0.767), 0.943 fine-tuned (default 0.931).
Train it on `a10g-small`; a `t4-small` ran out of memory.

## Speed and quantization

Per-row latency at batch size 1, fine-tuned 52-label models:

| Model | L4 (fp16) | T4 (fp16) | Apple M1 Pro CPU | Free CPU Space (2 vCPU) |
|---|---|---|---|---|
| gliner2.5-small | ~19 ms | — | — | ~0.3 s |
| gliner2.5-base | ~19 ms | ~27 ms | ~0.19 s | ~0.7–1 s |

- On a GPU, small and base take the same time per row; fixed overhead dominates.
- **Dynamic int8 did not work.** Agreement of the top label with fp32 on 64 rows: ONNX Runtime
  default 6%, `torch.ao` Linear 1.6%, per-channel 58%, MatMul-only per-channel 59%, skipping the
  feed-forward down-projection 91%, attention layers only 94%. Speed-up was at most 1.2×. The fp32
  margins between the top two labels were healthy, so activation outliers in the fine-tuned
  encoder are the likely cause. Calibrated static int8, or keeping sensitive layers in full
  precision, might work; it was not tried.
- **ONNX:** `gliner2` 2.0.0 has no export. Exporting only the encoder matches fp32 exactly, but
  ran 1.8× slower than PyTorch on an Apple M1.
- bf16 on a CPU keeps the same answers but was 6× slower on an M1.
- On HF CPU Jobs, `os.cpu_count()` reports the host's cores, not the Job's vCPUs. Set torch's
  thread count explicitly or the run oversubscribes and stalls.

## Behaviour details

- **Single-label and multi-label**, auto-detected from the label column (a list per row is multi-label). An empty list is kept as a valid "none of these" answer.
- **Several tasks in one model.** Repeat `--label-column` and each column becomes a task; the model answers all of them in one pass. `classify-gliner2.py` then writes one `predicted_<task>` and one `predicted_<task>_confidence` column per task.
- **Evaluation split and metrics match `train-classifier.py` and `train-setfit.py`** (`--eval-split`, else `validation`, else `test`, else a carve-out; accuracy + macro F1), so the rungs are comparable. Multi-label tasks report micro/macro F1 and exact match.
- **Label names are part of the prompt.** Real names (`Fiction`, `Sports`) work; integer codes make zero-shot meaningless, and the script warns. Brackets are stripped from label names because GLiNER2 rejects them at inference.
- **It does not train on nothing.** The GLiNER2 trainer catches a CUDA out-of-memory error, skips the batch and carries on, so an undersized GPU looks like a healthy job that produces an untrained model. After 5 out-of-memory steps the script restarts itself at a quarter of the batch size, with 4× the gradient accumulation, so the effective batch size stays the same. It goes down to batch size 1, then stops before anything is pushed. The model card's reproduce command records the batch size that worked. Memory grows with batch size × number of labels × text length.
- **The launch config ships with the script.** Both scripts carry a [`[tool.hf-jobs]` header](https://huggingface.co/docs/hub/jobs-configuration#define-the-launch-config-in-the-script) (`t4-small`, a 1 hour timeout, the `HF_TOKEN` secret). With `hf` CLI 1.32 or newer, `hf jobs uv run <script-url> <args>` is enough, and `--dry-run` shows what it resolves to. Flags still win, and the examples here keep them so they also work on older CLIs — which ignore the header and stop the Job after 30 minutes, before the model is pushed. Pass `--timeout` explicitly (the examples use `1h`; a large run needs more) whenever you cannot be sure which CLI launches the job.
- **Outputs are private by default.** `train-gliner2.py` creates a private model repo and `classify-gliner2.py` a private dataset; pass `--public` to opt out. If the target repo already exists and is public, both scripts stop before doing any work. (`--private` is still accepted, and does nothing.)
- **Local files, several eval splits, exported predictions.** Instead of a Hub dataset, `train-gliner2.py` can read JSON Lines files, for example from a bucket mounted with `-v`:

  ```bash
  hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \
    -v hf://buckets/username/my-bucket:/bucket \
    https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-gliner2.py \
    --train-file /bucket/train.jsonl \
    --eval-file calibration=/bucket/calibration.jsonl --eval-file development=/bucket/development.jsonl \
    --labels-file /bucket/labels.json --label-column labels --label-augmentation off \
    --no-push --output-dir /bucket/runs/gliner2 --export-predictions /bucket/runs/gliner2/predictions
  ```

  - `--eval-file NAME=PATH` (repeatable): each file is an eval split, scored zero-shot and fine-tuned, in full and in file order.
  - `--labels-file`: a JSON list or one label per line. It fixes the label set and its order for training, zero-shot and evaluation; a label in the data that is not in the file stops the run.
  - `--export-predictions DIR`: writes `DIR/{base,finetuned}-<split>/predictions.jsonl`, one line per row: `{"row": i, "probabilities": {task: {label: p}}, "logits": {task: {label: logit}}}` with every label. Probabilities are gliner2's (softmax for a single-label task, a sigmoid per label for multi-label); logits are the raw per-label scores. Exporting also switches off the `--max-eval-samples` cap for a Hub eval split.
  - `--no-push`: no Hub repo is created or written; the model stays in `--output-dir/final`.
  - `--label-augmentation off`: gliner2's trainer by default renames the labels to "label 1", "label 2", ... in half of the training rows and drops up to half of the labels (`upstream`). With a fixed label set that is always scored in full, `off` trains on the real, complete label set every time; label-order shuffling stays on.
  - A `run_manifest.json` (all arguments except the token, the label-augmentation config, precision, device and package versions, then the results) is written to `--output-dir`, the export directory and the model folder.
- **Pick the GPU by label count and text length.** A `t4-small` fit short texts with 2, 4 and 28 labels at the default batch size. The 56-label TREC task and 2,000-character IMDB reviews both fell back to batch size 4, and IMDB did so on the A10G too. `a10g-small` (24 GB) trains about 2.3× faster and fit the 56-label TREC run at the default batch size, in 509s against 1,761s on the T4 at batch size 4. `--precision auto` uses bf16 on Ampere or newer GPUs (A10G, L4) and fp32 on a T4; on the A10G bf16 and fp32 ran at the same speed (62s and 59s), so bf16 there buys memory, not time.
- **Texts are truncated** to `--max-text-chars` (default 2000), with a count.
- **It is a GLiNER2 checkpoint**, loaded with `gliner2.classification.Classifier.from_pretrained(repo)`, not `AutoModelForSequenceClassification`. `gliner2` pins `transformers<5`, which keeps `huggingface_hub` below 1.0 inside the Job. The `hf` CLI 1.32 or newer that reads the `[tool.hf-jobs]` header is a separate install on your own machine, so the two versions do not conflict.

## Dead ends

- Gradient checkpointing crashes with the GLiNER2.5 models in `gliner2` 2.0.0.
- A LoRA run trained, but its final checkpoint did not load for scoring, and LoRA did not fix the
  56-label out-of-memory case on a T4.
- Before the out-of-memory restart existed, a TREC run "completed" with 1,006 of 1,020 steps
  skipped and scored barely above zero-shot. Check the logs for `OOM at step` when a result looks flat.
