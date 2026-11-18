# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "timm>=1.0.31",  # first release with multi-label training (--task multilabel)
#     "torch>=2.4",
#     "torchvision",
#     "datasets>=4.0",
#     "huggingface_hub>=0.34",
#     "safetensors",
#     "pillow",
#     "pyyaml",
# ]
#
# [tool.uv-scripts]
# maintenance = "active"
#
# [tool.hf-jobs]
# flavor = "a10g-small"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# ///
"""
Fine-tune a timm image classifier on a Hugging Face dataset and push it to the Hub.

This script runs timm's own ``train.py`` (fetched at the tag of the installed timm
release) and adds what a Hub run on HF Jobs needs around it:

- reads the task, image column, label column, label names and validation split from
  the dataset features (``ClassLabel`` = single-label, ``List(ClassLabel)`` =
  multi-label), and carves a seeded validation split if there is none;
- runs ``train.py`` on every GPU of the flavor with ``torch.distributed.run`` (one Job;
  it does not span several Jobs); the learning rate is a base rate that timm scales
  with the global batch;
- rebuilds the best checkpoint the way ``train.py`` built it, takes the validation
  metrics from ``train.py``'s ``summary.csv``, and pushes the model with its label
  names, its trained preprocessing config and a model card in one commit;
- reloads the pushed model from the Hub and checks it against the trained one.

Anything after a literal ``--`` goes to ``train.py`` unchanged, for example
``-- --loss asl --mixup 0.2``. On HF Jobs, put another ``--`` right after
``hf jobs uv run`` and its flags, so ``hf`` passes the second one on.

The pushed model loads with ``timm.create_model("hf-hub:<repo>", pretrained=True)``.

Default backbone: ``timm/convnext_tiny.in12k_ft_in1k`` - one of the most downloaded timm
checkpoints (28M params, ImageNet-12k pretrain, Apache-2.0), cheap on an A10G. The base
model's licence carries over to the trained model.

Examples (the [tool.hf-jobs] header sets a10g-small, a 2h timeout and HF_TOKEN; hf CLI 1.32+):

    # Single-label, a short first run (about a minute of training)
    hf jobs uv run \\
        https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-timm.py \\
        AI-Lab-Makerere/beans username/beans-convnext --epochs 3

    # Multi-label
    hf jobs uv run \\
        https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-timm.py \\
        timm/bee-dataset username/bee-convnext --epochs 10

    # Four GPUs in one Job
    hf jobs uv run --flavor l4x4 \\
        https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-timm.py \\
        AI-Lab-Makerere/beans username/beans-convnext --epochs 10

    # Extra flags for timm train.py after a second "--"
    hf jobs uv run -- \\
        https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-timm.py \\
        timm/bee-dataset username/bee-asl --epochs 10 -- --loss asl --mixup 0.2
"""

import argparse
import csv
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
from timm.data.model_config import resolve_input_data_config
from timm.models import (
    create_model,
    load_checkpoint,
    resolve_model_input_args,
    save_for_hf,
)
from timm.utils.checkpoint_saver import load_checkpoint_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train-timm")
datasets.disable_progress_bars()
disable_progress_bars()
# keep job logs readable: httpx logs every Hub request at INFO
logging.getLogger("httpx").setLevel(logging.WARNING)

# train.py comes from the tag of the installed timm release, so the trainer matches the library.
TIMM_TAG = f"v{timm.__version__}"
TRAIN_PY_URL = f"https://raw.githubusercontent.com/huggingface/pytorch-image-models/{TIMM_TAG}/train.py"
SCRIPT_URL = (
    "https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-timm.py"
)
DEFAULT_MODEL = "timm/convnext_tiny.in12k_ft_in1k"
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
CARVED_DATA_ROOT = Path("/tmp/train-timm-data")

# train.py flags this script sets from the dataset, or that would change the label order. Every
# other train.py flag passes through after "--": the pushed model is rebuilt from the arguments
# train.py saved in its checkpoint, so architecture and preprocessing flags carry over.
RESERVED_TRAIN_PY_FLAGS = [
    "--dataset", "--data-dir", "--train-split", "--val-split", "--input-key", "--target-key",
    "--num-classes", "--task", "--target-format", "--class-map", "--model", "--output",
    "--experiment", "--config",
]  # fmt: skip


# ----------------------------------------------------------------------------- args


def split_passthrough(argv):
    """Split argv at a literal '--'. Everything after it goes to timm train.py."""
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1 :]
    return argv, []


def reserved_flags_in(passthrough):
    """Reserved train.py flags in the passthrough, including argparse abbreviations and -c."""
    found = []
    for token in passthrough:
        if token.startswith("-c") and not token.startswith("--"):
            found.append("-c (config file)")
            continue
        if not token.startswith("--") or token == "--":
            continue
        name = token.split("=", 1)[0]
        # argparse accepts any unique prefix, so "--mod" can mean "--model"
        found += [flag for flag in RESERVED_TRAIN_PY_FLAGS if flag.startswith(name)]
    return sorted(set(found))


def timm_model_name(model):
    """The name timm.create_model and train.py take: timm/<name> becomes <name>, other Hub ids hf-hub:<id>."""
    if model.startswith("timm/"):
        return model[len("timm/") :]
    if "/" in model and not model.startswith("hf-hub:"):
        return f"hf-hub:{model}"
    return model


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Fine-tune a timm image classifier on a Hub dataset and push it to the Hub.",
        epilog="Arguments after a literal '--' are passed verbatim to timm train.py.",
    )
    parser.add_argument(
        "dataset", help="Input dataset id on the Hub, e.g. timm/bee-dataset"
    )
    parser.add_argument(
        "output_repo", help="Output model repo id, e.g. username/my-classifier"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"timm model: a Hub id such as {DEFAULT_MODEL} (the default), or a bare timm "
            "name such as resnet50.a1_in1k"
        ),
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

    train = parser.add_argument_group("training (passed to train.py)")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=64, help="Per GPU")
    train.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help=(
            "Learning rate for one GPU's batch (default: 5e-5). Passed as train.py --lr-base "
            "with --lr-base-size = --batch-size, so timm scales it for several GPUs."
        ),
    )
    train.add_argument("--weight-decay", type=float, default=0.05)
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
        "--num-workers",
        type=int,
        default=None,
        help="Data loader workers per GPU (default: from CPU_CORES)",
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
    args = parser.parse_args(argv)
    args.model = timm_model_name(args.model)
    return args


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


def gpu_count():
    # device_count() asks NVML and does not create a CUDA context in this parent process
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def gpus_support_bf16():
    """Compute capability 8.0+ (A10G, L4, A100, ...) supports bfloat16. Asks nvidia-smi so this
    process does not hold a CUDA context on GPU 0 while train.py runs."""
    try:
        listed = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        capabilities = [float(line) for line in listed.split() if line.strip()]
        return bool(capabilities) and min(capabilities) >= 8.0
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def default_num_workers(processes):
    # On HF Jobs, CPU_CORES is the real quota; os.cpu_count() reports the whole node.
    cores = os.environ.get("CPU_CORES")
    total = int(float(cores)) if cores else min(8, os.cpu_count() or 2)
    return max(1, (total - 1) // max(1, processes))


def jobs_flavor() -> str:
    """Return the HF Jobs hardware flavor this run is on, or "" when it cannot be told.

    HF Jobs sets ACCELERATOR to "cpu" or "gpu", not the flavor name, so the flavor is looked up
    in the public Jobs hardware list: first by GPU model and count, then by the CPU_CORES and
    MEMORY values that Jobs sets in the container (to tell a10g-small from a10g-large, or
    cpu-basic from cpu-upgrade). Any failure (no network, older huggingface_hub) returns "".
    """
    hardware = os.environ.get("ACCELERATOR") or ""
    if "-" in hardware or any(character.isdigit() for character in hardware):
        return hardware  # already a flavor name
    try:
        from huggingface_hub import HfApi

        gpu_name, gpu_count = "", 0
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
        except ImportError:
            # Torch-free drivers (e.g. the -server recipes): ask nvidia-smi instead.
            import subprocess

            try:
                listed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                ).stdout
                names = [line.strip() for line in listed.splitlines() if line.strip()]
                if names:
                    gpu_name, gpu_count = names[0], len(names)
            except (OSError, subprocess.SubprocessError):
                pass
        if gpu_count == 0 and hardware == "gpu":
            return ""  # a GPU run whose GPU cannot be identified: do not guess a CPU flavor

        # Whole-word match, so "L4" does not match "NVIDIA L40S" ("NVIDIA A100-SXM4-80GB" -> A100).
        gpu_words = gpu_name.upper().replace("-", " ").split()
        candidates = []
        for flavor in HfApi(token=False).list_jobs_hardware():
            accelerator = flavor.accelerator
            if gpu_count == 0 and accelerator is None:
                candidates.append(flavor)
            elif (
                gpu_count > 0
                and accelerator is not None
                and accelerator.model.upper() in gpu_words
                and int(accelerator.quantity) == gpu_count
            ):
                candidates.append(flavor)
        if not candidates:
            return ""

        cores = float(os.environ.get("CPU_CORES") or 0)
        memory_gb = float((os.environ.get("MEMORY") or "0").upper().rstrip("GIB ") or 0)

        def distance(flavor) -> float:
            # Relative gap to the listed vCPUs and RAM; Jobs reports slightly different numbers.
            listed_cores = float(flavor.cpu.split()[0])
            listed_memory = float(flavor.ram.split()[0])
            return (
                abs(listed_cores - cores) / listed_cores
                + abs(listed_memory - memory_gb) / listed_memory
            )

        return min(candidates, key=distance).name
    except Exception:
        return ""


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
    """Write changed splits (carved or capped) as parquet with real 'train' / 'validation' splits.

    train.py could read a slice such as "train[:90%]", but a slice is neither seeded nor
    stratified, so the script writes its own split for train.py's hfds reader.
    """
    local_dir = CARVED_DATA_ROOT / dataset_id.replace("/", "__")
    data_dir = local_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_ds.to_parquet(str(data_dir / "train-00000-of-00001.parquet"))
    val_ds.to_parquet(str(data_dir / "validation-00000-of-00001.parquet"))
    logger.info(f"Wrote the train and validation splits to {local_dir}")
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

    # Keep only the two columns train.py needs; it keeps a local copy small.
    train_ds = train_ds.select_columns([image_column, label_column])

    val_split = detect_val_split(split_names, args.val_split, args.train_split)
    carved = val_split is None
    if carved:
        train_ds, val_ds = carve_validation(
            train_ds, label_column, task, args.val_size, args.seed
        )
    else:
        val_ds = dataset_dict[val_split].select_columns([image_column, label_column])

    if args.max_train_samples:
        train_ds = train_ds.shuffle(seed=args.seed).select(
            range(min(args.max_train_samples, len(train_ds)))
        )
    if args.max_val_samples:
        val_ds = val_ds.shuffle(seed=args.seed).select(
            range(min(args.max_val_samples, len(val_ds)))
        )

    # train.py reads straight from the Hub unless the rows changed.
    if carved or args.max_train_samples or args.max_val_samples:
        train_py_dataset = write_local_copy(train_ds, val_ds, args.dataset)
        train_split_name, val_split_name = "train", "validation"
    else:
        train_py_dataset = args.dataset
        train_split_name, val_split_name = args.train_split, val_split

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
    return plan


# ----------------------------------------------------------------------------- train.py


def fetch_train_py(work_dir):
    path = Path(work_dir) / "train.py"
    logger.info(f"Fetching timm train.py at {TIMM_TAG}")
    urllib.request.urlretrieve(TRAIN_PY_URL, path)
    return path


def build_train_command(train_py, args, plan, output_dir, device, passthrough):
    processes = gpu_count() if device.type == "cuda" else 1
    if processes > 1:
        logger.info(
            f"{processes} GPUs: running train.py with torch.distributed.run; "
            f"global batch {args.batch_size * processes}"
        )
        launcher = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={processes}",
        ]
    else:
        launcher = [sys.executable]
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else default_num_workers(processes)
    )
    # LR is stepped per update, so one warmup epoch is fine even for very short runs
    warmup_epochs = 1 if args.epochs >= 2 else 0
    command = launcher + [
        str(train_py),
        "--dataset", f"hfds/{plan['train_py_dataset']}",
        "--train-split", plan["train_split"],
        "--val-split", plan["val_split"],
        "--input-key", plan["image_column"],
        "--target-key", plan["label_column"],
        "--num-classes", str(plan["num_classes"]),
        "--model", args.model,
        "--pretrained",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--opt", "adamw",
        # timm sets lr = lr-base x (global batch / lr-base-size)^k, with its own k for the optimizer
        "--lr-base", str(args.lr),
        "--lr-base-size", str(args.batch_size),
        "--weight-decay", str(args.weight_decay),
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
        if gpus_support_bf16():
            command += ["--amp-dtype", "bfloat16"]
    else:
        command += ["--device", "cpu"]
    return command + passthrough


def run_train_py(args, plan, device, passthrough):
    """Run train.py and return its run directory (best checkpoint, args.yaml, summary.csv)."""
    work_dir = Path(tempfile.mkdtemp(prefix="train-timm-"))
    train_py = fetch_train_py(work_dir)
    output_dir = work_dir / "output"
    command = build_train_command(train_py, args, plan, output_dir, device, passthrough)
    logger.info("Running: " + shlex.join(command))
    completed = subprocess.run(
        command, check=False, env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    if completed.returncode != 0:
        sys.exit(f"ERROR: timm train.py failed with exit code {completed.returncode}")
    run_dir = output_dir / "run"
    if not (run_dir / "model_best.pth.tar").exists():
        sys.exit(f"ERROR: expected the best checkpoint in {run_dir}, not found")
    return run_dir


def load_trained_model(run_dir):
    """Rebuild the best checkpoint the way train.py built the model, from the arguments it saved.

    Returns (model, train_args, best_epoch). Every model-shaping flag train.py received,
    including passthrough flags such as --gp or --model-kwargs, carries over.
    """
    checkpoint = run_dir / "model_best.pth.tar"
    metadata = load_checkpoint_metadata(str(checkpoint))
    train_args = vars(metadata["args"])
    model = create_model(
        **resolve_model_input_args(
            train_args["model"],
            train_args,
            pretrained=False,
            num_classes=train_args["num_classes"],
            global_pool=train_args["gp"],
            bn_momentum=train_args["bn_momentum"],
            bn_eps=train_args["bn_eps"],
            **train_args["model_kwargs"],
        )
    )
    load_checkpoint(model, str(checkpoint))
    logger.info(
        f"Loaded best checkpoint (epoch {metadata['epoch']}, {metadata['metric']:.4f})"
    )
    return model.eval(), train_args, metadata["epoch"]


def read_best_metrics(run_dir, best_epoch):
    """The eval_* values train.py wrote to summary.csv for the best epoch, as {name: value}."""
    with open(run_dir / "summary.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if int(row["epoch"]) == int(best_epoch):
            return {
                name[len("eval_") :]: round(float(value), 4)
                for name, value in row.items()
                if name.startswith("eval_") and value not in ("", None)
            }
    sys.exit(f"ERROR: no summary.csv row for the best epoch {best_epoch}")


# ----------------------------------------------------------------------------- model card + push


def base_model_license(model_name):
    if model_name.startswith("hf-hub:"):
        hub_id = model_name[len("hf-hub:") :]
    else:
        hub_id = f"timm/{model_name}"
    hub_id = hub_id.split("@")[0]  # hf-hub:org/name@revision
    try:
        info = HfApi().model_info(hub_id)
        card = info.card_data.to_dict() if info.card_data else {}
        return hub_id, card.get("license")
    except (HfHubHTTPError, ValueError) as error:  # ValueError: HFValidationError
        # the base might not live under timm/ (e.g. a custom model name)
        logger.warning(f"Could not read base model info for {hub_id}: {error}")
        return None, None


def reproduction_command(argv, flavor):
    """The command that made this model: same flavor, and the script's own args as passed."""
    flags = f"--flavor {flavor or 'a10g-small'} --secrets HF_TOKEN --timeout 2h"
    return f"hf jobs uv run {flags} -- " + shlex.join([SCRIPT_URL, *argv])


def metrics_table(metrics):
    lines = ["| Metric | Value |", "| --- | --- |"]
    for name, value in metrics.items():
        lines.append(
            f"| {name} | {value:.4f} |"
            if name == "loss"
            else f"| {name} | {value:.2f} |"
        )
    return "\n".join(lines)


def label_table(label_names):
    lines = ["| Index | Label |", "| --- | --- |"]
    for index, name in enumerate(label_names):
        lines.append(f"| {index} | {name} |")
    return "\n".join(lines)


def usage_snippet(repo_id, task, threshold):
    if task == "multilabel":
        predict = (
            "probs = model(transform(image).unsqueeze(0)).sigmoid()[0]\n"
            'labels = model.pretrained_cfg["label_names"]\n'
            f"print([name for name, p in zip(labels, probs) if p >= {threshold}])"
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


def build_model_card(args, plan, metrics, best_epoch, argv):
    base_hub_id, license_name = base_model_license(args.model)
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
    multilabel_note = (
        f"\n\nPredictions use a threshold of {args.threshold}. `map` is timm's mean average "
        "precision, which can differ from scikit-learn's per-class average precision. "
        "`sample_f1` follows timm and scores images with no labels as 0."
        if plan["task"] == "multilabel"
        else ""
    )

    sections = [
        "\n".join(frontmatter),
        f"# {args.output_repo.split('/')[-1]}",
        (
            f"A {task_text} image classifier: {base_link} fine-tuned with timm `train.py` "
            f"(timm {timm.__version__}) on "
            f"[`{args.dataset}`](https://huggingface.co/datasets/{args.dataset})."
        ),
        (
            f"- Image column: `{plan['image_column']}`, label column: `{plan['label_column']}`\n"
            f"- Train rows: {plan['num_train']}, validation rows: {plan['num_val']} ({split_text})"
        ),
        "## Labels\n\n" + label_table(plan["label_names"]),
        "## Evaluation\n\n"
        + f"Validation metrics from timm `train.py` (percent) for the best epoch ({best_epoch}) "
        + f"on {split_text}:\n\n"
        + metrics_table(metrics)
        + multilabel_note,
        "## Usage\n\n" + usage_snippet(args.output_repo, plan["task"], args.threshold),
        (
            "## Reproduction\n\n"
            f"{origin} with the [`train-timm.py`]({SCRIPT_URL}) recipe from "
            "[uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:\n\n"
            "```bash\n"
            f"{reproduction_command(argv, hw)}\n"
            "```"
        ),
    ]
    return "\n\n".join(sections) + "\n"


def push_model(model, train_args, args, plan, metrics, best_epoch, argv):
    """Save with timm's save_for_hf (weights, config with labels and the trained preprocessing),
    add the model card, and upload everything in one commit."""
    model_config = {"label_names": plan["label_names"], "task": plan["task"]}
    if plan["task"] == "multilabel":
        model_config["multilabel_threshold"] = args.threshold
    model_config["eval_metrics"] = metrics

    export_dir = Path(tempfile.mkdtemp(prefix="train-timm-export-"))
    save_for_hf(
        model.cpu(),
        str(export_dir),
        model_config=model_config,
        data_config=resolve_input_data_config(model, train_args),
        safe_serialization=True,
    )
    card = build_model_card(args, plan, metrics, best_epoch, argv)
    (export_dir / "README.md").write_text(card)

    api = HfApi()
    api.create_repo(
        args.output_repo, repo_type="model", private=args.private, exist_ok=True
    )
    logger.info(f"Pushing model to {args.output_repo}")
    api.upload_folder(
        repo_id=args.output_repo,
        folder_path=str(export_dir),
        commit_message=f"Fine-tune {args.model} on {args.dataset}",
    )
    logger.info(f"Pushed: https://huggingface.co/{args.output_repo}")


@torch.no_grad()
def reload_check(args, plan, local_model, device):
    """Reload the pushed model from the Hub; check its labels and its logits on a fixed input."""
    logger.info("Reload check: timm.create_model('hf-hub:...', pretrained=True)")
    hub_model = timm.create_model(f"hf-hub:{args.output_repo}", pretrained=True)
    hub_model = hub_model.to(device).eval()
    hub_labels = list(hub_model.pretrained_cfg.get("label_names") or [])
    if hub_labels != plan["label_names"]:
        sys.exit(f"ERROR: reloaded label_names {hub_labels} != {plan['label_names']}")

    input_size = timm.data.resolve_data_config(model=hub_model)["input_size"]
    generator = torch.Generator().manual_seed(0)
    image = torch.randn(1, *input_size, generator=generator).to(device)
    local_model = local_model.to(device).eval()
    max_diff = (hub_model(image) - local_model(image)).abs().max().item()
    logger.info(
        f"Reload OK: labels match; input {tuple(input_size)}; max |logit diff| = {max_diff:.2e}"
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
    reserved = reserved_flags_in(passthrough)
    if reserved:
        sys.exit(
            f"ERROR: {', '.join(reserved)} after '--' would clash with what the script sets "
            "from the dataset or with the label order. Use the script's own options "
            "(--model, --train-split, --label-column, ...) instead."
        )

    device = pick_device(args.allow_cpu)
    logger.info(
        f"timm {timm.__version__}, torch {torch.__version__}, device {device}, GPUs {gpu_count()}"
    )

    plan = prepare_data(args)
    run_dir = run_train_py(args, plan, device, passthrough)
    model, train_args, best_epoch = load_trained_model(run_dir)
    metrics = read_best_metrics(run_dir, best_epoch)
    print(
        "METRICS "
        + json.dumps(
            {
                "model": args.model,
                "task": plan["task"],
                "best_epoch": best_epoch,
                "metrics": metrics,
            }
        ),
        flush=True,
    )

    push_model(model, train_args, args, plan, metrics, best_epoch, sys.argv[1:])
    if not args.skip_reload_check:
        reload_check(args, plan, model, device)
    logger.info(f"Done in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
