# timm script: options and behaviour details

Notes behind `train-timm.py`. The README has what you need to run it; this file records the full
options, what the script decides for you, and the timm details found while writing it.

The script is a thin layer over timm's own `train.py`. It does not re-implement training,
evaluation or model building: it prepares the data, runs `train.py`, and pushes what `train.py`
produced.

## Options

| Option | Default | Notes |
| --- | --- | --- |
| `--model` | `timm/convnext_tiny.in12k_ft_in1k` | a Hub ID (`timm/<name>` or any `org/name` timm checkpoint) or a bare timm name |
| `--epochs` | 10 | |
| `--batch-size` | 64 | per GPU |
| `--lr` | 5e-5 | for one GPU's batch; passed as `--lr-base` with `--lr-base-size` = `--batch-size` |
| `--weight-decay` | 0.05 | |
| `--img-size` | the model's pretrained size | fixed-size models (ViT and others) are built at this size |
| `--image-column`, `--label-column` | auto | |
| `--train-split`, `--val-split` | `train`, auto | |
| `--val-size` | 0.1 | used only when the script carves a validation split |
| `--max-train-samples`, `--max-val-samples` | all | for quick tests |
| `--threshold` | 0.5 | multi-label decision threshold (`train.py --multilabel-threshold`) |
| `--num-workers` | (`CPU_CORES` − 1) / GPUs on HF Jobs | per GPU |
| `--private` | off | create the output repo as private |
| `--skip-reload-check` | off | skip reloading the pushed model |

### Passing flags to `train.py`

Everything after a literal `--` goes to timm `train.py` unchanged. On HF Jobs, put another `--`
right after `hf jobs uv run` and its flags; otherwise `hf` takes the only `--` as its own separator
and the script receives `--loss` and the like. See `python train.py --help` in the timm repo for
its options.

Architecture and preprocessing flags (`--gp`, `--mean`, `--std`, `--crop-pct`, `--model-kwargs`,
`--model-ema`, ...) work: the pushed model is rebuilt from the arguments `train.py` saved in its
checkpoint. Flags that clash with what the script sets from the dataset are refused: `--dataset`,
`--data-dir`, the split and key flags, `--num-classes`, `--task`, `--target-format`, `--class-map`
(it would change the label order), `--model`, `--output`, `--experiment`, `-c`/`--config` and `--split-bn`.
`--split-bn` is refused because it changes the checkpoint layout after the model is built.
Abbreviations of these (for example `--mod`) are refused too. `--epochs`, `--batch-size` and `--lr`
after `--` override the script's values, as in `train.py`. Use the script's own options for these,
though. The one warmup epoch is set from `--epochs`, so `-- --epochs 1` would spend the whole run in
warmup. `--lr-base-size` is set from the script's `--batch-size`, so `-- --batch-size 8` also changes
the learning rate. Train.py values that the card reports, such as `--multilabel-threshold`, are read
back from the checkpoint, so a passthrough value shows up correctly.

## What the script decides

- **Task.** A `ClassLabel` label column gives single-label training (cross-entropy; top-1 and
  top-5). A `List(ClassLabel)` column gives multi-label training (BCE; mAP, micro, macro and sample
  F1).
- **Columns.** The image column is the first `Image` feature, preferring `image`, `img`, `picture`
  or `photo`. The label column prefers `label`, `labels`, `category`, `class` or `target`.
- **Validation split.** The first of `validation`, `valid`, `val` or `test`. If there is none, a
  seeded 10% split is carved from `train`, stratified for single-label data. A carved or capped
  split is written to a local parquet copy for `train.py`; a slice such as `train[:90%]` would be
  neither seeded nor stratified.
- **Training defaults.** AdamW, one warmup epoch, then a cosine schedule stepped per update, AMP
  (bfloat16 on GPUs with compute capability 8.0 or higher, float16 otherwise). `train.py` keeps the
  best checkpoint on its validation metric.
- **Several GPUs.** The script runs `train.py` with `torch.distributed.run` on every GPU the flavor
  has, the same as timm's `distributed_train.sh`. It passes `--lr-base` and `--lr-base-size`, so
  timm sets the learning rate for the global batch with its own rule: square-root scaling for
  AdamW. Nothing spans several Jobs.

## What gets pushed

The script saves the model with timm's `save_for_hf` and uploads it with the model card in one
commit:

- `model.safetensors`, and `config.json` with `label_names`, `num_classes`, `task`, the
  preprocessing the model was trained with, and `eval_metrics`. Multi-label models also record
  `multilabel_threshold`.
- A model card with the labels, `train.py`'s validation metrics for the best epoch, and the command
  to reproduce the run.

After the push, the script reloads the model with `timm.create_model("hf-hub:...")` and stops with
an error if the labels or the logits differ from the trained model.

## timm details

- **`train.py` version.** The script fetches `train.py` at the tag of the installed timm release
  (for example `v1.0.31`), so the trainer and the library always match. timm 1.0.31 is the first
  PyPI release with `--task multilabel`.
- **Rebuilding the model.** `model_best.pth.tar` holds `train.py`'s full arguments. The script
  rebuilds the model with `resolve_model_input_args` and `create_model`, as `train.py`
  does (the training-only drop-rate arguments are left out), so fixed-size models at another `--img-size` (ViT, LeViT, VOLO, ...) load correctly. timm
  cannot load PiT's pretrained weights at another size (`pit_ti_224` at 256 stops in `train.py`
  with a `pos_embed` size mismatch, before training starts).
- **Output repo.** The script creates the output repo before training, so a bad name or a token
  that cannot create the repo fails in seconds rather than after the run. (A token that can read
  an existing repo but not write to it still fails only at the upload.)
- **Metrics.** The card's metrics are the `eval_*` columns of `train.py`'s `summary.csv` for the
  best epoch. With `--model-ema`, `train.py` validates the EMA weights and the script pushes them.
  With `--model-ema-force-cpu`, `train.py` validates the regular weights, so the script pushes those. `map` is timm's mean average precision and can differ from scikit-learn's per-class
  average precision. timm's `sample_f1` scores images with no labels as 0.
- **Multi-label flags.** `--task multilabel` needs an explicit `--num-classes`, which the script
  passes, along with `--target-format indices` (timm's default for a `List(ClassLabel)` column).
- **Warmup.** `train.py` defaults to 5 warmup epochs and does not shorten them for short runs.
  Without `--sched-on-updates` the learning rate changes once per epoch. With lr 2e-4, 2 epochs
  and no warmup, beans fell to chance; the script passes one warmup epoch and `--sched-on-updates`.
- **Learning rate and CNNs.** The default `--lr` 5e-5 suits the ConvNeXt and ViT backbones. In
  3 epochs on beans, `resnet50.a1_in1k`, `efficientnet_b0.ra_in1k` and
  `mobilenetv4_conv_small.e2400_r224_in1k` reached only 47-63% top-1 with it (ConvNeXt-tiny and
  ViT-small: 97.7%). Give CNNs a higher `--lr` and more epochs.
- **Default backbone.** `convnext_tiny.in12k_ft_in1k` (28M parameters, Apache-2.0). DINOv3 backbones
  were considered, but their licence would carry over to every trained model.
