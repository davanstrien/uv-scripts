# timm script: options and behaviour details

Notes behind `train-timm.py`. The README has what you need to run it; this file records the full
options, what the script decides for you, and the timm details found while writing it.

## Options

| Option | Default | Notes |
| --- | --- | --- |
| `--model` | `timm/convnext_tiny.in12k_ft_in1k` | a Hub id (`timm/<name>` or any `org/name` timm checkpoint) or a bare timm name |
| `--mode` | `finetune` | or `probe` |
| `--epochs` | 10 (finetune), 100 (probe head) | |
| `--batch-size` | 64 | per GPU |
| `--lr` | 5e-5 (finetune), 1e-3 (probe) | multiplied by the GPU count in fine-tune mode |
| `--weight-decay` | 0.05 (finetune), 1e-4 (probe) | |
| `--img-size` | the model's pretrained size | recorded in the pushed config |
| `--image-column`, `--label-column` | auto | |
| `--train-split`, `--val-split` | `train`, auto | |
| `--val-size` | 0.1 | used only when the script carves a validation split |
| `--max-train-samples`, `--max-val-samples` | all | for quick tests |
| `--threshold` | 0.5 | multi-label decision threshold |
| `--num-workers` | `CPU_CORES` − 1 on HF Jobs | divided between GPUs in fine-tune mode |
| `--private` | off | create the output repo as private |
| `--skip-reload-check` | off | skip reloading the pushed model |

Everything after a literal `--` goes to timm `train.py` unchanged (fine-tune mode only). See
`python train.py --help` in the timm repo for its options.

## What the script decides

- **Task.** A `ClassLabel` label column gives single-label training (cross-entropy; top-1 and
  top-5). A `List(ClassLabel)` column gives multi-label training (BCE; mAP, micro and macro F1).
- **Columns.** The image column is the first `Image` feature, preferring `image`, `img`, `picture`
  or `photo`. The label column prefers `label`, `labels`, `category`, `class` or `target`.
- **Validation split.** The first of `validation`, `valid`, `val` or `test`. If there is none, a
  seeded 10% split is carved from `train`.
- **Fine-tune defaults.** AdamW, one warmup epoch, then a cosine schedule stepped per update, AMP
  (bfloat16 where the GPU supports it). The best checkpoint on the validation metric is pushed.
- **Several GPUs.** Fine-tune mode runs `train.py` with `torch.distributed.run` on every GPU the
  flavor has. `--batch-size` is per GPU, so the learning rate is multiplied by the GPU count
  (linear scaling for the global batch). Probe mode uses one GPU. Nothing spans several Jobs.

## What gets pushed

- `model.safetensors` and `config.json` with `label_names`, `num_classes` and `task`.
- For multi-label models, `multilabel_thresholds`: per-label thresholds tuned for F1 on the
  validation split. `macro_f1_tuned` on the card uses them on that same split, so it is optimistic.
- A model card with the labels, the validation metrics and the command to reproduce the run.

After the push, the script reloads the model with `timm.create_model("hf-hub:...")` and stops with
an error if the labels or the logits differ from the trained model.

## timm details

- **`train.py` version.** The script fetches `train.py` at the tag of the installed timm release
  (for example `v1.0.31`), so the trainer and the library always match. timm 1.0.31 is the first
  PyPI release with `--task multilabel`.
- **Split slices.** `ReaderHfds` reads `dataset.info.splits[split]`, so `train[:90%]` raises a
  `KeyError`. A carved validation split is written to a local parquet copy first.
- **Lower-case paths.** The hfds reader lower-cases the dataset name. Hub IDs still work through
  the Hub redirect, but a local copy must live on a lower-case path.
- **Multi-label flags.** `--task multilabel` needs an explicit `--num-classes`, and a
  `List(ClassLabel)` column needs `--target-format indices`. The script passes both.
- **Warmup.** `train.py` defaults to 5 warmup epochs and does not shorten them for short runs.
  Without `--sched-on-updates` the learning rate changes once per epoch. With lr 2e-4, 2 epochs
  and no warmup, beans fell to chance; the script passes one warmup epoch and `--sched-on-updates`.
- **Checkpoints.** `load_checkpoint` needs `weights_only=False`, because `train.py` stores its
  argparse `Namespace` in the checkpoint.
- **Metrics.** `map` is timm's `MultiLabelClassificationEvaluator` mAP and can differ from
  scikit-learn's per-class average precision. timm's `sample_f1` scores images with no labels as
  0, so it is low when many images have none.
- **Default backbone.** `convnext_tiny.in12k_ft_in1k` (28M parameters, Apache-2.0). DINOv3 ViT-S
  probes better, but its licence would carry over to every pushed model.
