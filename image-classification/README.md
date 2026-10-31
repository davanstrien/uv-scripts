# train-timm.py

Train a [timm](https://github.com/huggingface/pytorch-image-models) image classifier on any Hub image dataset and push it back to the Hub. Single-label and multi-label datasets both work.

The script reads the dataset features to decide the task:

| Label column type | Task |
| --- | --- |
| `ClassLabel` | single-label classification (cross-entropy, top-1 / top-5) |
| `List(ClassLabel)` | multi-label classification (BCE, mAP, micro / macro F1) |

It also picks the image column, the label column, the label names and the validation split (`validation`, `valid`, `val` or `test`). If there is no validation split, it carves a seeded 10% split from `train`.

## Quick start

Linear probe (frozen backbone, fast baseline):

```bash
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN --timeout 1h \
    https://huggingface.co/datasets/uv-scripts/image-classification/raw/main/train-timm.py \
    timm/bee-dataset username/bee-probe --mode probe
```

Full fine-tune:

```bash
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN --timeout 2h \
    https://huggingface.co/datasets/uv-scripts/image-classification/raw/main/train-timm.py \
    AI-Lab-Makerere/beans username/beans-convnext --epochs 10
```

Load the result:

```python
import timm

model = timm.create_model("hf-hub:username/beans-convnext", pretrained=True).eval()
print(model.pretrained_cfg["label_names"])
```

## Modes

- `--mode finetune` (default) runs timm's own `train.py` at the pinned timm commit. Defaults: AdamW, lr 5e-5 (warmup + cosine per update), weight decay 0.05, cosine schedule, 10 epochs, batch size 64, AMP, the model's pretrained input size. The best checkpoint on the validation metric is pushed.
- `--mode probe` extracts pooled features once with the frozen backbone and trains a linear head on them. It pushes the backbone plus the head as a normal timm model.

## Passing flags to timm `train.py`

Everything after a literal `--` goes to `train.py` unchanged. For example, the asymmetric loss and mixup for a multi-label run:

```bash
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN --timeout 2h \
    https://huggingface.co/datasets/uv-scripts/image-classification/raw/main/train-timm.py \
    timm/bee-dataset username/bee-asl --epochs 10 -- --loss asl --mixup 0.2
```

See `python train.py --help` in the timm repo for all options.

## Options

| Option | Default | Notes |
| --- | --- | --- |
| `--model` | `convnext_tiny.in12k_ft_in1k` | any timm model name |
| `--mode` | `finetune` | or `probe` |
| `--epochs` | 10 (finetune), 100 (probe head) | |
| `--batch-size` | 64 | |
| `--lr` | 5e-5 (finetune), 1e-3 (probe) | |
| `--img-size` | model default | |
| `--image-column`, `--label-column` | auto | |
| `--train-split`, `--val-split` | `train`, auto | |
| `--val-size` | 0.1 | used only when carving |
| `--max-train-samples`, `--max-val-samples` | all | for quick tests |
| `--threshold` | 0.5 | multi-label decision threshold |
| `--private` | off | create the output repo as private |

## What gets pushed

- `model.safetensors` and `config.json` with `label_names`, `num_classes` and `task`.
- For multi-label models, `multilabel_thresholds`: per-label thresholds tuned for F1 on the validation split.
- A model card with the label list, validation metrics and the command to reproduce the run.

After the push, the script reloads the model with `timm.create_model("hf-hub:...")` and checks that its predictions match the trained model.

## Notes

- A GPU is required. `--allow-cpu` exists only for tiny local debugging.
- Data loader workers come from the `CPU_CORES` environment variable on HF Jobs.
- timm's multi-label support is newer than the latest PyPI release, so the script installs timm from a pinned GitHub commit.
