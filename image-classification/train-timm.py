# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "timm @ https://github.com/huggingface/pytorch-image-models/archive/a6d209b59780f07e519faec54ef69ac51a10a82d.zip",
#     "torch>=2.4",
#     "torchvision",
#     "datasets>=4.0",
#     "huggingface_hub>=0.34",
#     "safetensors",
#     "pillow",
#     "pyyaml",
# ]
# ///
"""
Train a timm image classifier on a Hugging Face dataset and push it to the Hub.

Handles single-label (ClassLabel column) and multi-label (List(ClassLabel) column)
datasets. The task, label names, image column and label column are read from the
dataset features.

Two modes:

- ``--mode finetune`` (default): runs timm's own ``train.py`` (fetched at the pinned
  timm commit) with sensible defaults. Anything after a literal ``--`` goes to
  ``train.py`` unchanged.
- ``--mode probe``: freezes the backbone, extracts pooled features once, and fits a
  linear head. Fast and cheap; a good first baseline.

The pushed model loads with ``timm.create_model("hf-hub:<repo>", pretrained=True)``
and carries the label names in its config.

Default backbone: ``convnext_tiny.in12k_ft_in1k`` - one of the most downloaded timm
checkpoints (28M params, ImageNet-12k pretrain, Apache-2.0), cheap on an A10G.
DINOv3 ViT-S probes better but its licence would carry over to every pushed model.

Examples:

    # Multi-label linear probe (minutes on an A10G)
    hf jobs uv run --flavor a10g-small --secrets HF_TOKEN --timeout 1h \\
        https://huggingface.co/datasets/uv-scripts/image-classification/raw/main/train-timm.py \\
        timm/bee-dataset username/bee-probe --mode probe

    # Single-label fine-tune
    hf jobs uv run --flavor a10g-small --secrets HF_TOKEN --timeout 2h \\
        https://huggingface.co/datasets/uv-scripts/image-classification/raw/main/train-timm.py \\
        AI-Lab-Makerere/beans username/beans-convnext --epochs 10

    # Pass extra flags straight to timm train.py after "--"
    hf jobs uv run --flavor a10g-small --secrets HF_TOKEN --timeout 2h \\
        https://huggingface.co/datasets/uv-scripts/image-classification/raw/main/train-timm.py \\
        timm/bee-dataset username/bee-asl --epochs 10 -- --loss asl --mixup 0.2
"""

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import datasets
import timm
import torch
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import disable_progress_bars
from timm.data import create_transform, resolve_data_config
from timm.models import load_checkpoint, push_to_hf_hub
from timm.task.evaluator import (
    ClassificationEvaluator,
    MultiLabelClassificationEvaluator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train-timm")
datasets.disable_progress_bars()
disable_progress_bars()
# keep job logs readable: httpx logs every Hub request at INFO
logging.getLogger("httpx").setLevel(logging.WARNING)

TIMM_COMMIT = "a6d209b59780f07e519faec54ef69ac51a10a82d"
TRAIN_PY_URL = f"https://raw.githubusercontent.com/huggingface/pytorch-image-models/{TIMM_COMMIT}/train.py"
SCRIPT_URL = "https://huggingface.co/datasets/uv-scripts/image-classification/raw/main/train-timm.py"
DEFAULT_MODEL = "convnext_tiny.in12k_ft_in1k"
VAL_SPLIT_CANDIDATES = ["validation", "valid", "val", "test"]
LABEL_COLUMN_PREFERENCE = [
    "label",
    "labels",
    "category",
    "categories",
    "class",
    "classes",
    "target",
    "targets",
]
IMAGE_COLUMN_PREFERENCE = ["image", "img", "picture", "photo"]
# timm's hfds reader lower-cases the dataset name, so a carved local copy must live on a lower-case path
CARVED_DATA_ROOT = Path("/tmp/train-timm-data")


# ----------------------------------------------------------------------------- args


def split_passthrough(argv):
    """Split argv at a literal '--'. Everything after it goes to timm train.py."""
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1 :]
    return argv, []


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Train a timm image classifier on a Hub dataset and push it to the Hub.",
        epilog="Arguments after a literal '--' are passed verbatim to timm train.py (finetune mode).",
    )
    parser.add_argument(
        "dataset", help="Input dataset id on the Hub, e.g. timm/bee-dataset"
    )
    parser.add_argument(
        "output_repo", help="Output model repo id, e.g. username/my-classifier"
    )
    parser.add_argument("--mode", choices=["finetune", "probe"], default="finetune")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"timm model name (default: {DEFAULT_MODEL})",
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--image-column", default=None, help="Image column (default: auto-detect)"
    )
    data.add_argument(
        "--label-column", default=None, help="Label column (default: auto-detect)"
    )
    data.add_argument("--train-split", default="train")
    data.add_argument(
        "--val-split",
        default=None,
        help="Validation split (default: auto-detect, else carve)",
    )
    data.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction carved from train if no val split",
    )
    data.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap train rows (for quick tests)",
    )
    data.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Cap validation rows (for quick tests)",
    )

    train = parser.add_argument_group("training")
    train.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Default: 10 (finetune), 100 (probe head)",
    )
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Default: 5e-5 (finetune), 1e-3 (probe head)",
    )
    train.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Default: 0.05 (finetune), 1e-4 (probe)",
    )
    train.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Default: the model's pretrained input size",
    )
    train.add_argument(
        "--threshold", type=float, default=0.5, help="Multi-label decision threshold"
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--num-workers", type=int, default=None, help="Default: from CPU_CORES env"
    )

    hub = parser.add_argument_group("hub")
    hub.add_argument(
        "--private", action="store_true", help="Create the output repo as private"
    )
    hub.add_argument(
        "--skip-reload-check",
        action="store_true",
        help="Skip reloading the pushed model",
    )

    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Run without a GPU (tiny local debugging only)",
    )
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------- environment


def pick_device(allow_cpu):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        logger.warning(
            "No CUDA GPU found. Running on CPU because --allow-cpu is set. This is slow."
        )
        return torch.device("cpu")
    sys.exit(
        "ERROR: no CUDA GPU found. Run this on a GPU, for example:\n"
        "  hf jobs uv run --flavor a10g-small --secrets HF_TOKEN train-timm.py ...\n"
        "Use --allow-cpu only for tiny local debugging runs."
    )


def default_num_workers():
    # On HF Jobs, CPU_CORES is the real quota; os.cpu_count() reports the whole node.
    cores = os.environ.get("CPU_CORES")
    if cores:
        return max(1, int(float(cores)) - 1)
    return min(8, max(1, (os.cpu_count() or 2) - 1))


# The smallest Jobs flavor for each GPU, keyed by a fragment of the GPU's name. "L40" comes
# before "L4" because the first match wins.
GPU_NAME_TO_FLAVOR = {
    "T4": "t4-small",
    "A10G": "a10g-small",
    "L40": "l40sx1",
    "L4": "l4x1",
    "A100": "a100-large",
}


def jobs_flavor() -> str:
    """Return the Jobs hardware flavor, or "" when it is not known.

    On HF Jobs, ACCELERATOR holds only "cpu", "gpu" or "neuron", not the flavor name, so the
    flavor is inferred from the GPU name. The result is the smallest flavor with that GPU.
    """
    hardware = os.environ.get("ACCELERATOR") or ""
    looks_like_flavor = "-" in hardware or any(
        character.isdigit() for character in hardware
    )
    if looks_like_flavor:
        return hardware
    if not torch.cuda.is_available():
        return ""
    gpu_name = torch.cuda.get_device_name(0)
    for fragment, flavor in GPU_NAME_TO_FLAVOR.items():
        if fragment in gpu_name:
            return flavor
    return ""


def amp_dtype_for(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ----------------------------------------------------------------------------- dataset inspection


def is_class_label(feature):
    return isinstance(feature, datasets.ClassLabel)


def is_multi_class_label(feature):
    """True for List / Sequence / LargeList of ClassLabel."""
    inner = getattr(feature, "feature", None)
    return inner is not None and is_class_label(inner)


def ordered_by_preference(names, preference):
    preferred = [name for name in preference if name in names]
    rest = [name for name in names if name not in preferred]
    return preferred + rest


def detect_image_column(features, override):
    if override:
        if override not in features:
            sys.exit(
                f"ERROR: image column '{override}' not in dataset columns {list(features)}"
            )
        return override
    candidates = [
        name
        for name, feature in features.items()
        if isinstance(feature, datasets.Image)
    ]
    if not candidates:
        sys.exit(
            f"ERROR: no Image column found in {list(features)}. Use --image-column."
        )
    return ordered_by_preference(candidates, IMAGE_COLUMN_PREFERENCE)[0]


def detect_label_column(features, override):
    if override:
        if override not in features:
            sys.exit(
                f"ERROR: label column '{override}' not in dataset columns {list(features)}"
            )
        feature = features[override]
        if not (is_class_label(feature) or is_multi_class_label(feature)):
            sys.exit(
                f"ERROR: column '{override}' is {feature}, not ClassLabel or List(ClassLabel). "
                "Cast it first, e.g. dataset.class_encode_column(...)."
            )
        return override
    candidates = []
    for name, feature in features.items():
        if is_class_label(feature) or is_multi_class_label(feature):
            candidates.append(name)
    if not candidates:
        sys.exit(
            f"ERROR: no ClassLabel or List(ClassLabel) column found in {dict(features)}. "
            "Use --label-column with a ClassLabel column, or cast one first."
        )
    chosen = ordered_by_preference(candidates, LABEL_COLUMN_PREFERENCE)[0]
    if len(candidates) > 1:
        logger.info(
            f"Several label columns found {candidates}; using '{chosen}' (override with --label-column)"
        )
    return chosen


def label_names_for(feature):
    if is_class_label(feature):
        return list(feature.names)
    return list(feature.feature.names)


def detect_val_split(split_names, override, train_split):
    if override:
        if override not in split_names:
            sys.exit(f"ERROR: split '{override}' not in {split_names}")
        return override
    for candidate in VAL_SPLIT_CANDIDATES:
        if candidate in split_names and candidate != train_split:
            return candidate
    return None


def carve_validation(train_ds, label_column, task, val_size, seed):
    """Seeded train/validation split of the train split, stratified when possible."""
    logger.info(
        f"No validation split found; carving {val_size:.0%} of train (seed={seed})"
    )
    if task == "classification":
        try:
            parts = train_ds.train_test_split(
                test_size=val_size, seed=seed, stratify_by_column=label_column
            )
            return parts["train"], parts["test"]
        except ValueError as error:
            logger.warning(
                f"Stratified split failed ({error}); using a plain random split"
            )
    parts = train_ds.train_test_split(test_size=val_size, seed=seed)
    return parts["train"], parts["test"]


def write_local_copy(train_ds, val_ds, dataset_id):
    """Write the carved splits as parquet so timm train.py can read them with its hfds reader.

    timm's hfds reader looks up split sizes in dataset.info.splits, so split slicing such as
    "train[:90%]" does not work. A local parquet folder gives it real 'train' / 'validation' splits.
    """
    local_dir = CARVED_DATA_ROOT / dataset_id.replace("/", "__").lower()
    data_dir = local_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_ds.to_parquet(str(data_dir / "train-00000-of-00001.parquet"))
    val_ds.to_parquet(str(data_dir / "validation-00000-of-00001.parquet"))
    logger.info(f"Wrote carved splits to {local_dir}")
    return str(local_dir)


def prepare_data(args):
    """Load the dataset, detect columns and task, and settle the train/val splits."""
    logger.info(f"Loading dataset {args.dataset}")
    dataset_dict = datasets.load_dataset(args.dataset)
    split_names = list(dataset_dict.keys())
    if args.train_split not in split_names:
        sys.exit(
            f"ERROR: train split '{args.train_split}' not in {split_names}. Use --train-split."
        )
    train_ds = dataset_dict[args.train_split]
    features = train_ds.features

    image_column = detect_image_column(features, args.image_column)
    label_column = detect_label_column(features, args.label_column)
    label_feature = features[label_column]
    task = "classification" if is_class_label(label_feature) else "multilabel"
    label_names = label_names_for(label_feature)

    # Keep only the two columns we need; it keeps the carved copy and data loading small.
    train_ds = train_ds.select_columns([image_column, label_column])

    val_split = detect_val_split(split_names, args.val_split, args.train_split)
    carved = val_split is None
    if carved:
        train_ds, val_ds = carve_validation(
            train_ds, label_column, task, args.val_size, args.seed
        )
        train_split_name, val_split_name = "train", "validation"
    else:
        val_ds = dataset_dict[val_split].select_columns([image_column, label_column])
        train_split_name, val_split_name = args.train_split, val_split

    if args.max_train_samples:
        train_ds = train_ds.shuffle(seed=args.seed).select(
            range(min(args.max_train_samples, len(train_ds)))
        )
    if args.max_val_samples:
        val_ds = val_ds.shuffle(seed=args.seed).select(
            range(min(args.max_val_samples, len(val_ds)))
        )

    # train.py reads straight from the Hub unless we had to change the rows (probe mode never uses it).
    needs_local_copy = args.mode == "finetune" and bool(
        carved or args.max_train_samples or args.max_val_samples
    )
    train_py_dataset = (
        write_local_copy(train_ds, val_ds, args.dataset)
        if needs_local_copy
        else args.dataset
    )
    if needs_local_copy:
        train_split_name, val_split_name = "train", "validation"

    plan = {
        "task": task,
        "image_column": image_column,
        "label_column": label_column,
        "label_names": label_names,
        "num_classes": len(label_names),
        "train_split": train_split_name,
        "val_split": val_split_name,
        "val_carved": carved,
        "train_py_dataset": train_py_dataset,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
    }
    logger.info(
        "Data plan: "
        + json.dumps({k: v for k, v in plan.items() if k != "label_names"})
    )
    logger.info(f"Labels ({len(label_names)}): {label_names}")
    return plan, train_ds, val_ds


# ----------------------------------------------------------------------------- data loading


class HfImageDataset(torch.utils.data.Dataset):
    """Decodes images from a HF dataset and returns (tensor, target)."""

    def __init__(
        self, hf_dataset, image_column, label_column, transform, task, num_classes
    ):
        self.hf_dataset = hf_dataset
        self.image_column = image_column
        self.label_column = label_column
        self.transform = transform
        self.task = task
        self.num_classes = num_classes

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        row = self.hf_dataset[index]
        image = row[self.image_column].convert("RGB")
        label = row[self.label_column]
        if self.task == "multilabel":
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            for class_index in label:
                target[class_index] = 1.0
        else:
            target = torch.tensor(label, dtype=torch.long)
        return self.transform(image), target


def make_eval_loader(hf_dataset, plan, model, batch_size, num_workers):
    data_config = resolve_data_config(model=model)
    transform = create_transform(**data_config, is_training=False)
    torch_dataset = HfImageDataset(
        hf_dataset,
        plan["image_column"],
        plan["label_column"],
        transform,
        plan["task"],
        plan["num_classes"],
    )
    return torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def run_model(model, loader, device):
    """Return (outputs, targets) for every batch in the loader, on CPU in float32."""
    model.eval()
    outputs = []
    targets = []
    autocast_enabled = device.type == "cuda"
    started = time.time()
    for batch_index, (images, batch_targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype_for(device),
            enabled=autocast_enabled,
        ):
            batch_outputs = model(images)
        outputs.append(batch_outputs.float().cpu())
        targets.append(batch_targets)
        if batch_index % 20 == 0:
            logger.info(
                f"  batch {batch_index + 1}/{len(loader)} ({time.time() - started:.0f}s)"
            )
    return torch.cat(outputs), torch.cat(targets)


# ----------------------------------------------------------------------------- metrics


def best_threshold_for_label(probabilities, targets):
    """Threshold in 0.05..0.95 that gives the best F1 for one label."""
    best_threshold = 0.5
    best_f1 = -1.0
    for step in range(1, 20):
        threshold = step * 0.05
        predicted = probabilities >= threshold
        true_positive = (predicted & targets).sum().item()
        denominator = predicted.sum().item() + targets.sum().item()
        f1 = 2 * true_positive / denominator if denominator else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return round(best_threshold, 2), best_f1


def compute_metrics(logits, targets, plan, threshold):
    """Return (metrics dict, per-label thresholds or None). Metrics are percentages."""
    if plan["task"] == "classification":
        evaluator = ClassificationEvaluator()
        evaluator.update(logits, targets)
        summary = evaluator.compute()
        metrics = {
            "top1": summary["top1"],
            "top5": summary["top5"],
            "loss": summary["loss"],
        }
        return {k: round(v, 4) for k, v in metrics.items()}, None

    evaluator = MultiLabelClassificationEvaluator(threshold=threshold)
    evaluator.update(logits, targets)
    summary = evaluator.compute()
    metrics = {
        "map": summary["map"],
        "micro_f1": summary["micro_f1"],
        "macro_f1": summary["macro_f1"],
        "sample_f1": summary["sample_f1"],
        "loss": summary["loss"],
    }

    probabilities = logits.sigmoid()
    target_bool = targets.bool()
    thresholds = {}
    per_label_f1 = []
    for class_index, name in enumerate(plan["label_names"]):
        best, f1 = best_threshold_for_label(
            probabilities[:, class_index], target_bool[:, class_index]
        )
        thresholds[name] = best
        per_label_f1.append(f1)
    metrics["macro_f1_tuned"] = 100 * sum(per_label_f1) / len(per_label_f1)
    return {k: round(v, 4) for k, v in metrics.items()}, thresholds


# ----------------------------------------------------------------------------- finetune mode


def fetch_train_py(work_dir):
    path = Path(work_dir) / "train.py"
    logger.info(f"Fetching timm train.py at {TIMM_COMMIT[:7]}")
    urllib.request.urlretrieve(TRAIN_PY_URL, path)
    return path


def build_train_command(
    train_py, args, plan, output_dir, device, num_workers, passthrough
):
    epochs = args.epochs or 10
    # LR is stepped per update, so one warmup epoch is fine even for very short runs
    warmup_epochs = 1 if epochs >= 2 else 0
    command = [
        sys.executable,
        str(train_py),
        "--dataset", f"hfds/{plan['train_py_dataset']}",
        "--train-split", plan["train_split"],
        "--val-split", plan["val_split"],
        "--input-key", plan["image_column"],
        "--target-key", plan["label_column"],
        "--num-classes", str(plan["num_classes"]),
        "--model", args.model,
        "--pretrained",
        "--epochs", str(epochs),
        "--batch-size", str(args.batch_size),
        "--opt", "adamw",
        "--lr", str(args.lr or 5e-5),
        "--weight-decay", str(args.weight_decay if args.weight_decay is not None else 0.05),
        "--sched", "cosine",
        "--sched-on-updates",
        "--warmup-epochs", str(warmup_epochs),
        "--workers", str(num_workers),
        "--seed", str(args.seed),
        "--output", str(output_dir),
        "--experiment", "run",
        "--checkpoint-hist", "1",
        "--log-interval", "25",
    ]  # fmt: skip
    if plan["task"] == "multilabel":
        command += ["--task", "multilabel", "--target-format", "indices"]
        command += ["--multilabel-threshold", str(args.threshold)]
    if args.img_size:
        command += ["--img-size", str(args.img_size)]
    if device.type == "cuda":
        command += ["--amp", "--pin-mem"]
        if amp_dtype_for(device) == torch.bfloat16:
            command += ["--amp-dtype", "bfloat16"]
    else:
        command += ["--device", "cpu"]
    return command + passthrough


def run_finetune(args, plan, device, num_workers, passthrough):
    work_dir = Path(tempfile.mkdtemp(prefix="train-timm-"))
    train_py = fetch_train_py(work_dir)
    output_dir = work_dir / "output"
    command = build_train_command(
        train_py, args, plan, output_dir, device, num_workers, passthrough
    )
    logger.info("Running: " + shlex.join(command))
    completed = subprocess.run(
        command, check=False, env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    if completed.returncode != 0:
        sys.exit(f"ERROR: timm train.py failed with exit code {completed.returncode}")

    checkpoint = output_dir / "run" / "model_best.pth.tar"
    if not checkpoint.exists():
        sys.exit(f"ERROR: expected best checkpoint at {checkpoint}, not found")
    logger.info(f"Loading best checkpoint {checkpoint}")
    model = timm.create_model(
        args.model, pretrained=False, num_classes=plan["num_classes"]
    )
    # our own checkpoint, it holds the argparse Namespace so weights_only loading is not possible
    load_checkpoint(model, str(checkpoint), weights_only=False)
    if args.img_size:
        set_input_size(model, args.img_size)
    return model.to(device)


def set_input_size(model, img_size):
    """Record a non-default training size in pretrained_cfg so the pushed config preprocesses the same way."""
    channels = model.pretrained_cfg.get("input_size", (3, 224, 224))[0]
    model.pretrained_cfg["input_size"] = (channels, img_size, img_size)
    model.pretrained_cfg.pop("test_input_size", None)


# ----------------------------------------------------------------------------- probe mode


def fit_linear_head(
    train_features,
    train_targets,
    task,
    num_classes,
    epochs,
    lr,
    weight_decay,
    device,
    seed,
):
    """Logistic regression on frozen features: CE for single-label, BCE for multi-label."""
    torch.manual_seed(seed)
    head = torch.nn.Linear(train_features.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = (
        torch.nn.BCEWithLogitsLoss()
        if task == "multilabel"
        else torch.nn.CrossEntropyLoss()
    )

    features = train_features.to(device)
    targets = train_targets.to(device)
    batch_size = 256
    for epoch in range(epochs):
        order = torch.randperm(features.shape[0], device=device)
        total_loss = 0.0
        for start in range(0, features.shape[0], batch_size):
            batch = order[start : start + batch_size]
            loss = loss_fn(head(features[batch]), targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                f"  head epoch {epoch + 1}/{epochs} loss {total_loss / features.shape[0]:.4f}"
            )
    return head.cpu()


def run_probe(args, plan, train_ds, device, num_workers):
    logger.info(f"Creating frozen backbone {args.model}")
    model = timm.create_model(args.model, pretrained=True, num_classes=0)
    if args.img_size:
        set_input_size(model, args.img_size)
    model = model.to(device)

    logger.info(f"Extracting train features ({len(train_ds)} images)")
    train_loader = make_eval_loader(train_ds, plan, model, args.batch_size, num_workers)
    train_features, train_targets = run_model(model, train_loader, device)
    logger.info(f"Train features: {tuple(train_features.shape)}")

    head = fit_linear_head(
        train_features,
        train_targets,
        plan["task"],
        plan["num_classes"],
        epochs=args.epochs or 100,
        lr=args.lr or 1e-3,
        weight_decay=args.weight_decay if args.weight_decay is not None else 1e-4,
        device=device,
        seed=args.seed,
    )

    # Put the trained head into the model so it is a normal timm classifier.
    model.reset_classifier(plan["num_classes"])
    classifier = model.get_classifier()
    if tuple(classifier.weight.shape) != tuple(head.weight.shape):
        sys.exit(
            f"ERROR: classifier shape {tuple(classifier.weight.shape)} != head {tuple(head.weight.shape)}"
        )
    with torch.no_grad():
        classifier.weight.copy_(head.weight)
        classifier.bias.copy_(head.bias)
    return model.to(device)


# ----------------------------------------------------------------------------- model card + push


def base_model_license(model_name):
    hub_id = f"timm/{model_name}"
    try:
        info = HfApi().model_info(hub_id)
        card = info.card_data.to_dict() if info.card_data else {}
        return hub_id, card.get("license")
    except HfHubHTTPError as error:
        # the base might not live under timm/ (e.g. a custom model name)
        logger.warning(f"Could not read base model info for {hub_id}: {error}")
        return None, None


def reproduction_command(argv):
    """The command that made this model: same flavor, and the script's own args as passed."""
    flags = f"--flavor {jobs_flavor() or 'a10g-small'} --secrets HF_TOKEN --timeout 2h"
    return f"hf jobs uv run {flags} " + shlex.join([SCRIPT_URL, *argv])


def metrics_table(metrics):
    lines = ["| Metric | Value |", "| --- | --- |"]
    for name, value in metrics.items():
        lines.append(
            f"| {name} | {value:.2f} |"
            if name != "loss"
            else f"| {name} | {value:.4f} |"
        )
    return "\n".join(lines)


def label_table(plan, thresholds):
    if thresholds is None:
        lines = ["| Index | Label |", "| --- | --- |"]
        for index, name in enumerate(plan["label_names"]):
            lines.append(f"| {index} | {name} |")
    else:
        lines = ["| Index | Label | Tuned threshold |", "| --- | --- | --- |"]
        for index, name in enumerate(plan["label_names"]):
            lines.append(f"| {index} | {name} | {thresholds[name]} |")
    return "\n".join(lines)


def usage_snippet(repo_id, task):
    if task == "multilabel":
        predict = (
            "probs = model(transform(image).unsqueeze(0)).sigmoid()[0]\n"
            'labels = model.pretrained_cfg["label_names"]\n'
            '# per-label tuned thresholds are in config.json under "multilabel_thresholds"\n'
            "print([name for name, p in zip(labels, probs) if p >= 0.5])"
        )
    else:
        predict = (
            "probs = model(transform(image).unsqueeze(0)).softmax(-1)[0]\n"
            'print(model.pretrained_cfg["label_names"][probs.argmax()])'
        )
    return (
        "```python\n"
        "import timm\n"
        "from PIL import Image\n\n"
        f'model = timm.create_model("hf-hub:{repo_id}", pretrained=True).eval()\n'
        "config = timm.data.resolve_data_config(model=model)\n"
        "transform = timm.data.create_transform(**config)\n"
        'image = Image.open("example.jpg").convert("RGB")\n'
        f"{predict}\n"
        "```"
    )


def build_model_card(args, plan, metrics, thresholds, argv, base_hub_id, license_name):
    on_jobs = os.environ.get("JOB_ID") is not None
    hw = jobs_flavor()
    origin = (
        (
            "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
            + (f" (`{hw}`)" if hw else "")
        )
        if on_jobs
        else "Generated"
    )

    tags = ["uv-script", "timm", "image-classification"]
    if plan["task"] == "multilabel":
        tags.append("multi-label-classification")
    if on_jobs:
        tags.append("hf-jobs")

    frontmatter = ["---", "library_name: timm", "pipeline_tag: image-classification"]
    if license_name:
        frontmatter.append(f"license: {license_name}")
    frontmatter.append("tags:")
    frontmatter += [f"- {tag}" for tag in tags]
    if base_hub_id:
        frontmatter += ["base_model:", f"- {base_hub_id}"]
    frontmatter += ["datasets:", f"- {args.dataset}", "---"]

    mode_text = (
        "fine-tuned end to end with timm `train.py`"
        if args.mode == "finetune"
        else "a linear probe (frozen backbone, trained linear head)"
    )
    task_text = "multi-label" if plan["task"] == "multilabel" else "single-label"
    split_text = (
        f"a seeded {args.val_size:.0%} split carved from `{args.train_split}` (seed {args.seed})"
        if plan["val_carved"]
        else f"the `{plan['val_split']}` split"
    )
    base_link = (
        f"[`{base_hub_id}`](https://huggingface.co/{base_hub_id})"
        if base_hub_id
        else f"`{args.model}`"
    )

    sections = [
        "\n".join(frontmatter),
        f"# {args.output_repo.split('/')[-1]}",
        (
            f"A {task_text} image classifier: {base_link} {mode_text} on "
            f"[`{args.dataset}`](https://huggingface.co/datasets/{args.dataset})."
        ),
        (
            f"- Image column: `{plan['image_column']}`, label column: `{plan['label_column']}`\n"
            f"- Train rows: {plan['num_train']}, validation rows: {plan['num_val']} ({split_text})"
        ),
        "## Labels\n\n" + label_table(plan, thresholds),
        "## Evaluation\n\n"
        + f"Validation metrics (percent) on {split_text}:\n\n"
        + metrics_table(metrics)
        + (
            "\n\n`macro_f1_tuned` uses per-label thresholds tuned on this same validation split, "
            "so it is optimistic. The tuned thresholds are stored as `multilabel_thresholds` in `config.json`. "
            "`sample_f1` follows timm and scores images with no labels as 0, so it is low when many "
            "images have no labels."
            if plan["task"] == "multilabel"
            else ""
        ),
        "## Usage\n\n" + usage_snippet(args.output_repo, plan["task"]),
        (
            "## Reproduction\n\n"
            f"{origin} with the [`train-timm.py`]({SCRIPT_URL}) recipe from "
            "[uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:\n\n"
            "```bash\n"
            f"{reproduction_command(argv)}\n"
            "```"
        ),
    ]
    return "\n\n".join(sections) + "\n"


def push_model(model, args, plan, metrics, thresholds, argv):
    api = HfApi()
    api.create_repo(
        args.output_repo, repo_type="model", private=args.private, exist_ok=True
    )

    model_config = {
        "num_classes": plan["num_classes"],
        "label_names": plan["label_names"],
        "task": plan["task"],
    }
    if plan["task"] == "multilabel":
        model_config["multilabel_threshold"] = args.threshold
        model_config["multilabel_thresholds"] = thresholds
    model_config["eval_metrics"] = metrics

    logger.info(f"Pushing model to {args.output_repo}")
    push_to_hf_hub(
        model.cpu(),
        args.output_repo,
        commit_message=f"Train {args.model} on {args.dataset} ({args.mode})",
        private=args.private,
        model_config=model_config,
        safe_serialization=True,
    )

    base_hub_id, license_name = base_model_license(args.model)
    card = build_model_card(
        args, plan, metrics, thresholds, argv, base_hub_id, license_name
    )
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.output_repo,
        commit_message="Add model card",
    )
    logger.info(f"Pushed: https://huggingface.co/{args.output_repo}")


@torch.no_grad()
def reload_check(args, plan, local_model, val_ds, device):
    """Reload the pushed model from the Hub and compare one prediction with the local model."""
    logger.info("Reload check: timm.create_model('hf-hub:...', pretrained=True)")
    hub_model = (
        timm.create_model(f"hf-hub:{args.output_repo}", pretrained=True)
        .to(device)
        .eval()
    )
    hub_labels = list(hub_model.pretrained_cfg.get("label_names") or [])
    if hub_labels != plan["label_names"]:
        sys.exit(f"ERROR: reloaded label_names {hub_labels} != {plan['label_names']}")

    loader = make_eval_loader(
        val_ds.select(range(1)), plan, hub_model, batch_size=1, num_workers=0
    )
    image, _ = next(iter(loader))
    image = image.to(device)
    local_model = local_model.to(device).eval()
    hub_logits = hub_model(image).float()
    local_logits = local_model(image).float()
    max_diff = (hub_logits - local_logits).abs().max().item()
    if plan["task"] == "multilabel":
        probabilities = hub_logits.sigmoid()[0]
        predicted = [
            name
            for name, p in zip(hub_labels, probabilities.tolist())
            if p >= args.threshold
        ]
    else:
        predicted = hub_labels[hub_logits.argmax(-1).item()]
    logger.info(
        f"Reload OK: prediction on first val image = {predicted}; max |logit diff| = {max_diff:.2e}"
    )
    if max_diff > 1e-3:
        sys.exit(
            f"ERROR: reloaded model logits differ from the trained model by {max_diff}"
        )


# ----------------------------------------------------------------------------- main


def main():
    started = time.time()
    own_argv, passthrough = split_passthrough(sys.argv[1:])
    args = parse_args(own_argv)
    if passthrough and args.mode == "probe":
        sys.exit(
            "ERROR: arguments after '--' are for timm train.py and only apply to --mode finetune"
        )

    device = pick_device(args.allow_cpu)
    num_workers = (
        args.num_workers if args.num_workers is not None else default_num_workers()
    )
    logger.info(
        f"timm {timm.__version__} (commit {TIMM_COMMIT[:7]}), torch {torch.__version__}, device {device}"
    )
    logger.info(f"Data loader workers: {num_workers}")

    plan, train_ds, val_ds = prepare_data(args)

    if args.mode == "finetune":
        model = run_finetune(args, plan, device, num_workers, passthrough)
    else:
        model = run_probe(args, plan, train_ds, device, num_workers)

    logger.info(f"Evaluating on validation ({len(val_ds)} images)")
    val_loader = make_eval_loader(val_ds, plan, model, args.batch_size, num_workers)
    logits, targets = run_model(model, val_loader, device)
    metrics, thresholds = compute_metrics(logits, targets, plan, args.threshold)
    result = {
        "mode": args.mode,
        "model": args.model,
        "task": plan["task"],
        "metrics": metrics,
    }
    if thresholds:
        result["thresholds"] = thresholds
    print("METRICS " + json.dumps(result), flush=True)

    push_model(model, args, plan, metrics, thresholds, sys.argv[1:])
    if not args.skip_reload_check:
        reload_check(args, plan, model, val_ds, device)
    logger.info(f"Done in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
