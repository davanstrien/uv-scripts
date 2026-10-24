# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "gliner2[train]==2.1.0",
#     "protobuf",
#     "sentencepiece",
#     "datasets>=4.0.0,<6",
#     "scikit-learn",
#     "huggingface-hub",
# ]
#
# [tool.hf-jobs]
# flavor = "t4-small"
# timeout = "1h"
# secrets = ["HF_TOKEN"]
# ///
"""
Fine-tune GLiNER2 into a text classifier — a small model (74M to 287M parameters, depending on
the base checkpoint) that already works zero-shot.

GLiNER2 reads the label names as part of its input, so it classifies with no training at all.
This script measures that zero-shot score first, fine-tunes on your labels, then measures
again on the same held-out rows. The model card reports both numbers.
One model can answer several questions at once: pass --label-column more
than once and each column becomes a task.

Run on HF Jobs (t4-small is enough for a few thousand short texts):

    hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-gliner2.py \\
        biglam/blbooksgenre username/gliner2-blbooks-genre \\
        --dataset-config title_genre_classifiction --text-column title

The output model repo is PRIVATE unless you pass --public.

The [tool.hf-jobs] header above gives `hf` CLI 1.32+ the defaults (t4-small, a 1 hour timeout,
the HF_TOKEN secret), so there `hf jobs uv run <script> <args>` is enough. Flags always win:
pass `--flavor a10g-small` for more memory and bf16, or `--timeout 3h` for a big run. Older CLIs
ignore the header, and Jobs then stops after 30 minutes; the model is pushed at the end. Pass
`--timeout` explicitly whenever you are not sure which CLI will launch the job.

Local files instead of a Hub dataset (for example files in a bucket mounted at /bucket):

    hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \\
        -v hf://buckets/username/my-bucket:/bucket \\
        https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-gliner2.py \\
        --train-file /bucket/train.jsonl \\
        --eval-file calibration=/bucket/calibration.jsonl \\
        --eval-file development=/bucket/development.jsonl \\
        --labels-file /bucket/labels.json --label-column labels \\
        --no-push --output-dir /bucket/runs/gliner2 \\
        --export-predictions /bucket/runs/gliner2/predictions

- --train-file / --eval-file NAME=PATH read JSON Lines files (repeat --eval-file for several
  eval splits). Every eval file is scored in full, in file order, with no cap.
- --labels-file fixes the label set, and its order, for training, zero-shot and evaluation.
  Every label in the data must be in it. Needs exactly one --label-column.
- --export-predictions DIR writes DIR/{base,finetuned}-{split}/predictions.jsonl, one line per
  eval row: {"row": i, "probabilities": {task: {label: p}}, "logits": {task: {label: logit}}},
  with every label present. "row" is the row's position in its eval file (or split).
- --no-push keeps the model in --output-dir/final and uploads nothing. A run manifest (all
  arguments, the label-augmentation config and package versions) is written to --output-dir,
  the export directory and the model folder.
- --label-augmentation off turns off gliner2's synthetic label names and label dropping during
  training, so the model always sees the real, complete label set (see resolve_sampling_config).

Metrics match `train-classifier.py` and `train-setfit.py` (accuracy + macro F1 on a held-out
split), so the three are directly comparable at equal eval settings.

NOTE: the output is a GLiNER2 checkpoint. It loads with
`gliner2.classification.Classifier.from_pretrained(repo)`, NOT `AutoModelForSequenceClassification`.
Apply it to a whole dataset with the sibling `classify-gliner2.py`.
"""

import argparse
import dataclasses
import importlib.metadata
import json
import logging
import os
import shlex
import sys
import time
from collections import Counter

# tqdm reads TQDM_DISABLE when it is imported, so this must be set before any third-party import
# pulls tqdm in. Jobs logs have no TTY, so progress bars arrive as hundreds of near-identical lines.
# (The gliner2 trainer passes disable=False to its own bar, so its training bar still prints.)
os.environ.setdefault("TQDM_DISABLE", "1")

import datasets
import torch
from datasets import ClassLabel, Dataset, load_dataset
from gliner2 import AutoExtractor
from gliner2.classification import (
    ClassificationConfig,
    ClassificationSchema,
    Classifier,
)
from gliner2.processor import SamplingConfig
from gliner2.training.data import Classification, InputExample
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
from huggingface_hub import HfApi, login
from huggingface_hub.utils import disable_progress_bars
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def configure_logging() -> logging.Logger:
    """Keep Jobs logs readable: root at WARNING, only this script's logger at INFO."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    disable_progress_bars()
    if hasattr(datasets, "disable_progress_bars"):
        datasets.disable_progress_bars()

    script_logger = logging.getLogger("train-gliner2")
    script_logger.setLevel(logging.INFO)
    return script_logger


logger = configure_logging()

SCRIPT_URL = (
    "https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-gliner2.py"
)
DEFAULT_BASE_MODEL = "fastino/gliner2.5-multi-v1"

# The file this script adds to the model repo. It records the tasks and label names the model
# was trained on, so classify-gliner2.py can rebuild the same schema without any flags.
SCHEMA_FILENAME = "classification_schema.json"

# Written next to the model, in --output-dir and in the export directory: the arguments,
# label-augmentation config and package versions of the run.
MANIFEST_FILENAME = "run_manifest.json"

# A column added to every eval split before any row is dropped, so exported predictions can
# name each row's position in the original file or split.
ROW_COLUMN = "__row__"

# After this many out-of-memory training steps, stop and retry smaller. See StopOnRepeatedOOM.
MAX_OOM_STEPS = 5

# GLiNER2 puts label names into the model prompt verbatim. Its inference schema rejects these
# strings, but its trainer accepts them — so a label like "manuscripts (documents)" trains
# without complaint and then cannot be predicted. We clean labels once, before either side.
RESERVED_MARKERS = ("[P]", "[L]", "[C]", "[E]", "[R]", "[DESCRIPTION]", "[EXAMPLE]", "[OUTPUT]")


def clean_label(label: str) -> str:
    """Make one label name safe for the GLiNER2 prompt."""
    cleaned = label.replace("(", " ").replace(")", " ")
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        sys.exit(f"Label {label!r} is empty after cleaning. Rename it in the dataset.")
    for marker in RESERVED_MARKERS:
        if marker in cleaned:
            sys.exit(
                f"Label {label!r} contains {marker!r}, which GLiNER2 reserves for its prompt. "
                "Rename it in the dataset."
            )
    return cleaned


def is_multi_label_column(dataset: Dataset, column: str) -> bool:
    """A list-valued column is a multi-label task."""
    feature = dataset.features.get(column)
    # A Sequence/list feature carries an inner `feature`.
    return getattr(feature, "feature", None) is not None


def label_feature(dataset: Dataset, column: str):
    """Return the ClassLabel that types this column, or None if the labels are plain values."""
    feature = dataset.features.get(column)
    inner = getattr(feature, "feature", None)
    if isinstance(feature, ClassLabel):
        return feature
    if isinstance(inner, ClassLabel):
        return inner
    return None


def decode_label(value, class_label) -> str:
    """Turn one raw label value into its cleaned name."""
    if class_label is not None:
        return clean_label(class_label.int2str(int(value)))
    return clean_label(str(value))


def decode_column(dataset: Dataset, column: str) -> list:
    """Return the gold labels for a column: a name per row, or a sorted list of names per row.

    Decoding uses this split's OWN feature. A named --eval-split can order its ClassLabel
    differently from the train split, and decoding through the train names would silently
    score against the wrong table.
    """
    class_label = label_feature(dataset, column)
    multi = is_multi_label_column(dataset, column)
    decoded = []
    for value in dataset[column]:
        if multi:
            names = {decode_label(item, class_label) for item in (value or [])}
            decoded.append(sorted(names))
        else:
            decoded.append(decode_label(value, class_label))
    return decoded


def drop_unlabelled_rows(dataset: Dataset, columns: list, text_column: str, split_name: str) -> Dataset:
    """Remove rows with no text, or with a missing or blank single-label value.

    An empty LIST in a multi-label column is kept: "none of these labels" is a valid answer,
    and the model needs to see it to learn when to select nothing.
    """
    # In a ClassLabel column, -1 is the Hub convention for "no label" (common in test splits).
    typed_columns = [column for column in columns if isinstance(dataset.features.get(column), ClassLabel)]

    def is_usable(example) -> bool:
        text = example[text_column]
        if text is None or not str(text).strip():
            return False
        for column in columns:
            value = example[column]
            if isinstance(value, list):
                continue
            if value is None:
                return False
            if isinstance(value, str) and not value.strip():
                return False
            if column in typed_columns and value < 0:
                return False
        return True

    kept = dataset.filter(is_usable)
    dropped = len(dataset) - len(kept)
    if dropped:
        logger.warning(
            "Dropped %d %s rows with no text or a missing label (%d remain).",
            dropped, split_name, len(kept),
        )
    if len(kept) == 0:
        sys.exit(
            f"No '{split_name}' rows are left after dropping rows with no text or a missing label "
            f"({len(dataset)} before). Check --text-column and --label-column, or pick another split."
        )
    return kept


def pick_eval_split(dataset_id, config, train_split, requested):
    """Resolve which split to evaluate on, matching train-classifier.py's precedence."""
    if requested:
        if requested == train_split:
            sys.exit(
                f"--eval-split and --train-split are both '{requested}'. Evaluating on the "
                "training data would report a meaningless score."
            )
        return requested

    available = datasets.get_dataset_split_names(dataset_id, config)
    for candidate in ("validation", "test"):
        if candidate in available and candidate != train_split:
            logger.info("Using the '%s' split for evaluation.", candidate)
            return candidate
    return None


def split_train_eval(args, eval_split, first_label_column):
    """Load the train split, and either the named eval split or a carve-out of train."""
    train_data = load_dataset(args.input_dataset, args.dataset_config, split=args.train_split)

    if eval_split:
        eval_data = load_dataset(args.input_dataset, args.dataset_config, split=eval_split)
        return train_data, eval_data

    logger.info("No eval split found; carving %.0f%% off the train split.", args.eval_fraction * 100)
    # Stratify when the first label column is typed, so a rare class cannot vanish from a small carve.
    feature = train_data.features.get(first_label_column)
    stratify = first_label_column if isinstance(feature, ClassLabel) else None
    try:
        parts = train_data.train_test_split(
            test_size=args.eval_fraction, seed=args.seed, stratify_by_column=stratify
        )
    except ValueError as error:
        # Stratification needs at least two members of every class, so it fails on exactly the
        # singleton classes it is meant to protect. An unstratified split is worse but usable.
        logger.warning("Could not stratify the carve-out (%s). Using an unstratified split.", error)
        parts = train_data.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    return parts["train"], parts["test"]


def load_json_file(path: str) -> Dataset:
    """Load one JSON Lines file (a local path, or a path in a mounted bucket) as a Dataset."""
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    return load_dataset("json", data_files=path, split="train")


def load_splits(args):
    """Return the train split, the eval splits as {name: Dataset}, and whether eval was carved out.

    With --train-file, every split comes from a local file and each --eval-file is its own
    eval split. Otherwise one eval split comes from the Hub dataset, as before.
    """
    if args.train_file:
        logger.info("Loading the train file %s", args.train_file)
        train_data = load_json_file(args.train_file)
        eval_sets = {}
        for name, path in args.eval_files.items():
            logger.info("Loading eval split '%s' from %s", name, path)
            eval_sets[name] = load_json_file(path)
        return train_data, eval_sets, False

    logger.info("Loading %s", args.input_dataset)
    eval_split = pick_eval_split(args.input_dataset, args.dataset_config, args.train_split, args.eval_split)
    train_data, eval_data = split_train_eval(args, eval_split, args.label_column[0])
    carved_out = eval_split is None
    return train_data, {eval_split or "eval": eval_data}, carved_out


def add_row_numbers(dataset: Dataset) -> Dataset:
    """Record each row's position, so it survives dropped rows and is exported with predictions."""
    if ROW_COLUMN in dataset.column_names:
        sys.exit(f"The data already has a column named '{ROW_COLUMN}'. Rename it.")
    return dataset.add_column(ROW_COLUMN, list(range(len(dataset))))


def load_labels_file(path: str) -> list:
    """Read the fixed label set: a JSON list, or one label per line."""
    if not os.path.exists(path):
        sys.exit(f"Labels file not found: {path}")
    with open(path) as handle:
        content = handle.read()
    if content.lstrip().startswith("["):
        labels = json.loads(content)
    else:
        labels = [line.strip() for line in content.splitlines() if line.strip()]
    if not labels:
        sys.exit(f"Labels file {path} has no labels.")
    return [str(label) for label in labels]


def check_labels_in_set(tasks: list, gold_by_split: dict) -> None:
    """With --labels-file, every gold label in every split must be one of the fixed labels."""
    for task in tasks:
        allowed = set(task["labels"])
        for split_name, gold_by_task in gold_by_split.items():
            unknown = Counter()
            for value in gold_by_task[task["name"]]:
                row_labels = value if task["multi_label"] else [value]
                for label in row_labels:
                    if label not in allowed:
                        unknown[label] += 1
            if unknown:
                sys.exit(
                    f"Task '{task['name']}', split '{split_name}': labels that are not in "
                    f"--labels-file: {dict(unknown.most_common(20))}"
                )


def prepare_texts(dataset: Dataset, text_column: str, max_text_chars: int, split_name: str) -> list:
    """Return the text for each row, truncated to max_text_chars."""
    texts = []
    truncated = 0
    for value in dataset[text_column]:
        text = str(value)
        if len(text) > max_text_chars:
            text = text[:max_text_chars]
            truncated += 1
        texts.append(text)
    if truncated:
        logger.warning(
            "Truncated %d of %d %s texts to %d characters. Raise --max-text-chars if the label "
            "depends on text past that point.",
            truncated, len(texts), split_name, max_text_chars,
        )
    return texts


def build_tasks(train_data: Dataset, label_columns: list, task_names: list, fixed_labels=None) -> list:
    """Describe one classification task per label column.

    GLiNER2 reads the task name as part of its prompt, next to the label names, so --task-name
    lets you call the task "genre" rather than "label". Do not expect much from it: on BL book
    titles the zero-shot accuracy was 0.79 with "label" and 0.78 with "genre".

    The label list comes from the TRAIN split. A label that appears only in the eval split
    cannot be predicted, and evaluate() counts it as an error rather than hiding it.
    With --labels-file (fixed_labels), the label list is that file, in its order, instead.
    """
    tasks = []
    for column, task_name in zip(label_columns, task_names):
        multi = is_multi_label_column(train_data, column)
        class_label = label_feature(train_data, column)
        if fixed_labels is not None:
            labels = [clean_label(name) for name in fixed_labels]
        elif class_label is not None:
            labels = [clean_label(name) for name in class_label.names]
        else:
            seen = set()
            for value in decode_column(train_data, column):
                if multi:
                    seen.update(value)
                else:
                    seen.add(value)
            labels = sorted(seen)

        if fixed_labels is not None:
            raw_names = list(fixed_labels)
        elif class_label is not None:
            raw_names = list(class_label.names)
        else:
            raw_names = []
            for value in train_data[column]:
                raw_names.extend((value or []) if multi else [value])
            raw_names = sorted({str(name) for name in raw_names})
        renamed = {name: clean_label(name) for name in raw_names if clean_label(name) != name}
        if renamed:
            logger.warning(
                "Column '%s': GLiNER2 does not allow brackets in label names, so the model will "
                "predict the cleaned names: %s", column, renamed,
            )

        # Check raw -> cleaned before trusting `labels`: for a plain string column the labels were
        # cleaned on the way into a set, so two different raw labels could already have merged.
        raw_by_cleaned = {}
        for name in raw_names:
            raw_by_cleaned.setdefault(clean_label(name), []).append(name)
        merged = {cleaned: raws for cleaned, raws in raw_by_cleaned.items() if len(raws) > 1}
        if merged:
            sys.exit(
                f"Column '{column}': different labels become identical after cleaning "
                f"(brackets are removed): {merged}. Rename them in the dataset."
            )
        if len(set(labels)) != len(labels):
            sys.exit(f"Column '{column}': two labels are identical after cleaning: {labels}")
        if len(labels) < 2:
            sys.exit(f"Column '{column}' has fewer than two labels: {labels}")
        if all(label.lstrip("-").isdigit() for label in labels):
            logger.warning(
                "Column '%s' has numeric labels %s. GLiNER2 reads label NAMES, so the zero-shot "
                "score will be meaningless and fine-tuning starts from nothing. A ClassLabel or "
                "string column with real names will do better.",
                column, labels[:6],
            )
        tasks.append({"name": clean_label(task_name), "column": column, "labels": labels, "multi_label": multi})
        logger.info(
            "Task '%s' (column '%s'): %d labels, %s.",
            task_name, column, len(labels), "multi-label" if multi else "single-label",
        )
    return tasks


def build_training_examples(texts: list, gold_by_task: dict, tasks: list) -> list:
    examples = []
    for row, text in enumerate(texts):
        classifications = []
        for task in tasks:
            classifications.append(
                Classification(
                    task=task["name"],
                    labels=task["labels"],
                    true_label=gold_by_task[task["name"]][row],
                    # Only auto-inferred when a row has 2+ true labels, so state it.
                    multi_label=task["multi_label"],
                )
            )
        examples.append(InputExample(text=text, classifications=classifications))
    return examples


def build_schema(tasks: list) -> ClassificationSchema:
    schema = ClassificationSchema()
    for task in tasks:
        if task["multi_label"]:
            schema.multi(task["name"], task["labels"])
        else:
            schema.single(task["name"], task["labels"])
    return schema


class TrainingOutOfMemory(Exception):
    """Raised when the GPU keeps running out of memory during training."""


class StopOnRepeatedOOM(logging.Handler):
    """Abort training when the gliner2 trainer keeps hitting CUDA out-of-memory.

    The gliner2 trainer catches an OOM, skips that batch, logs a warning and carries on. On a
    GPU that is too small for the batch, EVERY step is skipped: the job runs to the end, looks
    healthy, and produces a model that never trained. (Seen on t4-small with 56 labels at batch
    size 16: 1,006 of 1,020 steps skipped.) The trainer has no option to raise instead, so this
    handler watches its log. An exception raised in emit() propagates out of the trainer's own
    logger.warning() call, which stops trainer.train().
    """

    def __init__(self, limit: int):
        super().__init__(level=logging.WARNING)
        self.limit = limit
        self.oom_steps = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "OOM at step" not in record.getMessage():
            return
        self.oom_steps += 1
        if self.oom_steps >= self.limit:
            raise TrainingOutOfMemory()


def resolve_precision(requested: str) -> str:
    """Pick the training precision: bf16 where the GPU does it in hardware, else fp32.

    gliner2 itself defaults the 2.5 models to bf16. torch.cuda.is_bf16_supported() also says yes
    on a T4, where bf16 is emulated and slow, so this checks the compute capability instead
    (8.0+ = Ampere and newer: A10G, L4, A100, ...). fp16 is not offered: it overflowed on T4.
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return "bf16"
    return "fp32"


def resolve_sampling_config(mode: str) -> SamplingConfig:
    """Pick the gliner2 training-time label augmentation.

    "upstream" is gliner2's default SamplingConfig (unchanged from 2.0.0 to 2.1.0). For each
    classification task in each training row, it replaces the real label names with "label 1", "label 2", ... half of the
    time (synthetic_label_prob=0.5), and drops a random share of up to half of the labels
    (remove_classification_label_prob=0.5; the true label is then put back only half of the
    time). That teaches a general zero-shot model to cope with unseen label sets.

    "off" is for a FIXED label set that is always scored in full: the model then always trains
    on the real names and the complete label set, the same prompt it gets at inference.
    Label-order shuffling stays on, and so does task-order shuffling: neither changes which
    labels the model sees, and both stop it tying a label to a position in the prompt.
    The other options only touch entities, relations and JSON structures, or label
    descriptions and examples, which this script does not use.
    """
    if mode == "upstream":
        return SamplingConfig()
    return SamplingConfig(synthetic_label_prob=0.0, remove_classification_label_prob=0.0)


def package_versions() -> dict:
    versions = {}
    for package in ("gliner2", "torch", "transformers", "datasets", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_manifest(args, sampling_config: SamplingConfig, precision: str) -> dict:
    """Everything needed to tell two runs apart. The HF token is left out."""
    settings = {key: value for key, value in vars(args).items() if key != "hf_token"}
    return {
        "script": SCRIPT_URL,
        "args": settings,
        "label_augmentation": args.label_augmentation,
        "sampling_config": dataclasses.asdict(sampling_config),
        "precision": precision,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "job_id": os.environ.get("JOB_ID"),
        "versions": package_versions(),
    }


def write_manifest(manifest: dict, directories: list) -> None:
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, MANIFEST_FILENAME)
        with open(path, "w") as handle:
            json.dump(manifest, handle, indent=2)
        logger.info("Wrote %s", path)


def train_with_batch_fallback(args, examples: list, precision: str, sampling_config: SamplingConfig) -> int:
    """Train, and if the GPU runs out of memory, restart the script at a quarter of the batch size.

    Gradient accumulation grows by the same factor, so the effective batch size (and the
    number of optimizer steps) stays the same: the fallback costs time, not comparability.

    The restart is a whole new process (os.execv). Retrying inside this process was tried and
    does not work: after a failed run the gliner2 trainer's model and optimizer state stay on
    the GPU (about 4.6 GB per attempt), so each retry starts with less memory than the last.
    A new process gets a clean GPU. It parses the new --batch-size / --grad-accum itself, so
    the model card's reproduce command describes the run that produced the model. The restart
    also repeats the zero-shot scoring; that gives the same number and takes under a minute.

    Returns the number of training steps that were skipped for lack of memory.
    """
    model = AutoExtractor.from_pretrained(args.base_model)
    # The trainer uses model.processor, and the processor reads sampling_config for every
    # training row, so setting it here is what changes the training prompts.
    model.processor.sampling_config = sampling_config
    logger.info(
        "Label augmentation '%s': %s",
        args.label_augmentation, json.dumps(dataclasses.asdict(model.processor.sampling_config)),
    )
    config = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        seed=args.seed,
        # We score the held-out split ourselves, before and after, with the same code.
        eval_strategy="no",
        fp16=False,
        bf16=(precision == "bf16"),
        logging_steps=20,
    )
    oom_guard = StopOnRepeatedOOM(limit=MAX_OOM_STEPS)
    logging.getLogger("gliner2.training.trainer").addHandler(oom_guard)
    try:
        result = ExtractorTrainer(model, config).train(train_data=examples)
        # The trainer skips a batch that runs out of memory. On a short run every batch can be
        # skipped without reaching MAX_OOM_STEPS, which would push an untrained model.
        updates = result.get("total_steps") if isinstance(result, dict) else None
        if updates != 0:
            return oom_guard.oom_steps
        if oom_guard.oom_steps == 0:
            sys.exit("Stopped: training finished without a single optimizer update, so nothing was pushed.")
        logger.warning("No optimizer update succeeded: every batch ran out of memory.")
    except (TrainingOutOfMemory, torch.cuda.OutOfMemoryError):
        # gliner2 catches out-of-memory in the forward and backward pass, but not in the
        # optimizer step (for example while allocating optimizer state), so catch that here too.
        pass

    if args.batch_size == 1:
        sys.exit(
            "Stopped: the GPU ran out of memory even at batch size 1, so nothing was pushed. "
            "Memory grows with number of labels x text length. Lower --max-text-chars, or use a "
            "GPU with more memory (`--flavor a10g-small` has 24 GB, `--flavor a100-large` 80 GB)."
        )
    smaller = max(1, args.batch_size // 4)
    grad_accum = args.grad_accum * (args.batch_size // smaller)
    on_t4 = "T4" in torch.cuda.get_device_name(0)
    logger.warning(
        "The GPU ran out of memory at batch size %d. Restarting at batch size %d with %d gradient "
        "accumulation steps (same effective batch).%s",
        args.batch_size, smaller, grad_accum,
        " `--flavor a10g-small` (24 GB) would be faster." if on_t4 else "",
    )
    sys.stdout.flush()
    sys.stderr.flush()
    # argparse keeps the LAST value of a repeated flag, so appending these overrides the originals.
    os.execv(
        sys.executable,
        [sys.executable, *sys.argv, "--batch-size", str(smaller), "--grad-accum", str(grad_accum)],
    )


def score_task(task: dict, gold: list, results: list) -> dict:
    """Accuracy-style metrics for one task, from decoded results."""
    name = task["name"]
    if task["multi_label"]:
        predicted = [sorted(result.selected(name)) for result in results]
        # Labels seen only in eval still count: the binarizer covers the union.
        all_labels = sorted(set(task["labels"]) | {label for row in gold for label in row})
        binarizer = MultiLabelBinarizer(classes=all_labels)
        gold_matrix = binarizer.fit_transform(gold)
        predicted_matrix = binarizer.transform(predicted)
        return {
            "f1_micro": round(f1_score(gold_matrix, predicted_matrix, average="micro", zero_division=0), 4),
            "f1_macro": round(f1_score(gold_matrix, predicted_matrix, average="macro", zero_division=0), 4),
            "exact_match": round(accuracy_score(gold_matrix, predicted_matrix), 4),
        }
    predicted = [result.value(name) for result in results]
    # The majority-class rate is the floor any classifier must clear to be worth having.
    majority = Counter(gold).most_common(1)[0][1] / len(gold)
    return {
        "accuracy": round(accuracy_score(gold, predicted), 4),
        "f1_macro": round(f1_score(gold, predicted, average="macro", zero_division=0), 4),
        "majority_baseline": round(majority, 4),
    }


def write_predictions(path: str, rows: list, scores: list, results: list, tasks: list) -> None:
    """One JSON line per eval row, in order, with the probability and raw logit of EVERY label.

    Logits are the model's per-label scores before any activation (ClassificationScores.tasks).
    Probabilities are gliner2's own: softmax over the labels for a single-label task, a
    sigmoid per label for a multi-label task.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        for row, score, result in zip(rows, scores, results):
            record = {"row": row, "probabilities": {}, "logits": {}}
            for task in tasks:
                name = task["name"]
                probabilities = result.probabilities(name)
                logits = score.tasks[name]
                record["probabilities"][name] = {label: float(probabilities[label]) for label in task["labels"]}
                record["logits"][name] = {label: float(logits[label]) for label in task["labels"]}
            handle.write(json.dumps(record) + "\n")
    logger.info("Wrote %d predictions to %s", len(rows), path)


def evaluate(model_path: str, eval_sets: dict, tasks: list, batch_size: int, export_dir=None, export_name="") -> dict:
    """Load a GLiNER2 checkpoint once, predict every task on every eval split, and score each task.

    eval_sets maps a split name to {"texts", "gold", "rows"}. With export_dir, each split's
    predictions are also written to export_dir/<export_name>-<split>/predictions.jsonl.
    Returns {split name: metrics}.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # from_pretrained(device=...) does not move the weights in gliner2 2.0.0; .to() does.
    classifier = Classifier.from_pretrained(model_path).to(device=device).eval()
    parameters = sum(parameter.numel() for parameter in classifier.model.parameters())
    schema = build_schema(tasks)
    config = ClassificationConfig(batch_size=batch_size)

    metrics_by_split = {}
    for split_name, eval_set in eval_sets.items():
        started = time.time()
        # batch_classify() is batch_score() + decode(); calling both here keeps the raw logits.
        scores = classifier.batch_score(eval_set["texts"], schema, config=config)
        results = [classifier.decode(score, schema, config=config) for score in scores]
        elapsed = time.time() - started

        metrics = {}
        for task in tasks:
            metrics[task["name"]] = score_task(task, eval_set["gold"][task["name"]], results)
        metrics_by_split[split_name] = {
            "tasks": metrics,
            "eval_examples": len(eval_set["texts"]),
            "predict_seconds": round(elapsed, 1),
            "parameters": parameters,
        }

        if export_dir:
            path = os.path.join(export_dir, f"{export_name}-{split_name}", "predictions.jsonl")
            write_predictions(path, eval_set["rows"], scores, results, tasks)

    # Release the GPU before training starts.
    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics_by_split


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
                    capture_output=True, text=True, timeout=10, check=True,
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
            return abs(listed_cores - cores) / listed_cores + abs(listed_memory - memory_gb) / listed_memory

        return min(candidates, key=distance).name
    except Exception:
        return ""


def build_reproduce_command(args) -> str:
    """Rebuild the exact invocation, so the card's command produces the card's model."""
    flavor = jobs_flavor() or "t4-small"
    # The timeout is the [tool.hf-jobs] default, spelled out because older CLIs ignore the header.
    parts = [f"hf jobs uv run --flavor {flavor} --timeout 1h --secrets HF_TOKEN \\"]
    if args.train_file:
        # Local files: the job needs them mounted at the same paths.
        parts.insert(0, "# Mount the data files at the paths below, e.g. -v hf://buckets/<owner>/<bucket>:/bucket")
    positionals = [shlex.quote(value) for value in (args.input_dataset, args.output_repo) if value]
    if positionals:
        parts.append(f"  {SCRIPT_URL} \\")
        parts.append("  " + " ".join(positionals))
    else:
        parts.append(f"  {SCRIPT_URL}")
    flags = []
    if args.train_file:
        flags.append(f"--train-file {shlex.quote(args.train_file)}")
        for name, path in args.eval_files.items():
            flags.append(f"--eval-file {shlex.quote(f'{name}={path}')}")
    if args.labels_file:
        flags.append(f"--labels-file {shlex.quote(args.labels_file)}")
    if args.dataset_config:
        flags.append(f"--dataset-config {shlex.quote(args.dataset_config)}")
    if args.text_column != "text":
        flags.append(f"--text-column {shlex.quote(args.text_column)}")
    if args.label_column != ["label"]:
        for column in args.label_column:
            flags.append(f"--label-column {shlex.quote(column)}")
    if args.task_name != args.label_column:
        for task_name in args.task_name:
            flags.append(f"--task-name {shlex.quote(task_name)}")
    if args.base_model != DEFAULT_BASE_MODEL:
        flags.append(f"--base-model {shlex.quote(args.base_model)}")
    if args.train_split != "train":
        flags.append(f"--train-split {shlex.quote(args.train_split)}")
    if args.eval_split:
        flags.append(f"--eval-split {shlex.quote(args.eval_split)}")
    # These change which rows are trained on or scored, so a command without them reproduces
    # a different model and a different number.
    if args.eval_fraction != 0.1:
        flags.append(f"--eval-fraction {args.eval_fraction}")
    if args.max_train_samples:
        flags.append(f"--max-train-samples {args.max_train_samples}")
    if args.max_eval_samples != 2000:
        flags.append(f"--max-eval-samples {args.max_eval_samples}")
    if args.max_text_chars != 2000:
        flags.append(f"--max-text-chars {args.max_text_chars}")
    if args.epochs != 5:
        flags.append(f"--epochs {args.epochs}")
    if args.batch_size != 16:
        flags.append(f"--batch-size {args.batch_size}")
    if args.grad_accum != 1:
        flags.append(f"--grad-accum {args.grad_accum}")
    if args.encoder_lr != 1e-5:
        flags.append(f"--encoder-lr {args.encoder_lr}")
    if args.task_lr != 5e-4:
        flags.append(f"--task-lr {args.task_lr}")
    if args.seed != 42:
        flags.append(f"--seed {args.seed}")
    if args.precision != "auto":
        flags.append(f"--precision {args.precision}")
    if args.skip_zero_shot:
        flags.append("--skip-zero-shot")
    if args.label_augmentation != "upstream":
        flags.append(f"--label-augmentation {args.label_augmentation}")
    if args.no_push:
        flags.append(f"--no-push --output-dir {shlex.quote(args.output_dir)}")
    if args.export_predictions:
        flags.append(f"--export-predictions {shlex.quote(args.export_predictions)}")
    if args.public:
        flags.append("--public")

    if flags:
        parts[-1] += " \\"
        parts.append("  " + " ".join(flags))
    return "\n".join(parts)


def results_table(tasks: list, zero_shot, fine_tuned) -> str:
    """One row per task and metric, with the zero-shot score next to the fine-tuned one."""
    lines = ["| Task | Metric | Zero-shot | Fine-tuned |", "|---|---|---|---|"]
    for task in tasks:
        name = task["name"]
        for metric, value in fine_tuned["tasks"][name].items():
            if metric == "majority_baseline":
                continue
            before = zero_shot["tasks"][name][metric] if zero_shot else "not run"
            lines.append(f"| `{name}` | {metric} | {before} | **{value}** |")
    return "\n".join(lines)


def build_card(args, tasks, zero_shot, fine_tuned, train_size, train_seconds, carved_out, oom_steps) -> str:
    """Model card following the uv-scripts conventions (org credit, Jobs claim gated on JOB_ID)."""
    on_jobs = os.environ.get("JOB_ID") is not None
    hardware = jobs_flavor()
    if on_jobs:
        provenance = "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
        if hardware:
            provenance += f" (`{hardware}`)"
    else:
        provenance = "Produced"
    provenance += " with [`uv-scripts/classification`](https://huggingface.co/datasets/uv-scripts/classification)."

    tags = ["gliner2", "text-classification", "uv-script"]
    if on_jobs:
        tags.append("hf-jobs")
    tag_lines = "\n".join(f"- {tag}" for tag in tags)

    caveats = []
    if carved_out:
        caveats.append(
            f"**No held-out split existed, so {args.eval_fraction:.0%} was carved out of train.** "
            "These numbers are not comparable with published results on this dataset."
        )
    for split_name, split_metrics in fine_tuned.items():
        for task in tasks:
            if not task["multi_label"]:
                floor = split_metrics["tasks"][task["name"]]["majority_baseline"]
                caveats.append(
                    f"`{task['name']}` on `{split_name}`: always answering the most common label scores "
                    f"`{floor}` accuracy. Read the accuracy against that floor."
                )
    if oom_steps:
        caveats.append(
            f"**{oom_steps} training step(s) were skipped** because the GPU ran out of memory. "
            "The model saw less data than the example count above suggests."
        )
    caveats.append("Single seed. Small differences between runs are not evidence of anything.")
    caveat_block = "\n".join(f"- {caveat}" for caveat in caveats)

    label_lines = []
    for task in tasks:
        kind = "multi-label" if task["multi_label"] else "single-label"
        names = ", ".join(f"`{label}`" for label in task["labels"])
        label_lines.append(f"- **`{task['name']}`** ({kind}): {names}")
    label_block = "\n".join(label_lines)

    schema_lines = ["schema = ClassificationSchema()"]
    for task in tasks:
        method = "multi" if task["multi_label"] else "single"
        schema_lines.append(f"schema.{method}({task['name']!r}, {task['labels']!r})")
    schema_code = "\n".join(schema_lines)
    first_task = tasks[0]
    read_result = "selected" if first_task["multi_label"] else "value"

    result_sections = []
    for split_name, split_metrics in fine_tuned.items():
        split_zero_shot = zero_shot[split_name] if zero_shot else None
        result_sections.append(
            f"`{split_name}`: {split_metrics['eval_examples']} held-out examples.\n\n"
            + results_table(tasks, split_zero_shot, split_metrics)
        )
    result_block = "\n\n".join(result_sections)

    first_split = next(iter(fine_tuned.values()))
    size = f"{first_split['parameters'] / 1e6:.0f}M parameters"
    if args.input_dataset:
        source = f"[`{args.input_dataset}`](https://huggingface.co/datasets/{args.input_dataset})"
        dataset_metadata = f"datasets:\n- {args.input_dataset}\n"
    else:
        source = f"the local file `{os.path.basename(args.train_file)}`"
        dataset_metadata = ""
    if args.output_repo:
        title = args.output_repo.split("/")[-1]
        model_ref = args.output_repo
    else:
        title = os.path.basename(os.path.abspath(args.output_dir))
        model_ref = os.path.join(args.output_dir, "final")

    return f"""---
tags:
{tag_lines}
library_name: gliner2
pipeline_tag: text-classification
base_model: {args.base_model}
{dataset_metadata}---

# {title}

[GLiNER2](https://github.com/fastino-ai/GLiNER2) text classifier ({size}), fine-tuned from
[`{args.base_model}`](https://huggingface.co/{args.base_model}) on {train_size} examples from
{source}.

{provenance}

## Results

"Zero-shot" is the base model given only the label names, before any training, on the same
examples.

{result_block}

Training took {round(train_seconds)} seconds.

## Read this before trusting the numbers

{caveat_block}

## Tasks and labels

{label_block}

## Use it

```python
# pip install "gliner2[local]==2.0.0" protobuf sentencepiece
from gliner2.classification import ClassificationSchema, Classifier

classifier = Classifier.from_pretrained("{model_ref}").eval()

{schema_code}

result = classifier.batch_classify(["some text to classify"], schema)[0]
print(result.{read_result}({first_task["name"]!r}), result.confidence({first_task["name"]!r}))
```

To label a whole Hub dataset with this model:

```bash
hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN \\
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-gliner2.py \\
  <input-dataset> <output-dataset> --model {shlex.quote(model_ref)} --text-column {shlex.quote(args.text_column)}
```

## Reproduction

Produced by [`train-gliner2.py`]({SCRIPT_URL}) from
[`uv-scripts/classification`](https://huggingface.co/datasets/uv-scripts/classification):

```bash
{build_reproduce_command(args)}
```
"""


def in_own_account(api: HfApi, repo_id: str) -> str:
    """A bare name ("my-model") means a repo in your own account: return "<username>/my-model"."""
    if "/" in repo_id:
        return repo_id
    return f"{api.whoami()['name']}/{repo_id}"


def ensure_output_repo(api: HfApi, repo_id: str, private: bool) -> None:
    """Create the model repo, and refuse to train if a private run would push to a public repo.

    create_repo(exist_ok=True) leaves an existing repo's visibility alone, so a repo that
    already exists as public would silently receive a "private" model.
    """
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    if private and not api.repo_info(repo_id, repo_type="model").private:
        sys.exit(
            f"{repo_id} already exists and is public. Pass --public to push there anyway, or choose "
            "a new repo name."
        )


def main(args) -> None:
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    elif not args.no_push:
        sys.exit("No HF token. Pass --hf-token or run with --secrets HF_TOKEN (or pass --no-push).")

    if not torch.cuda.is_available():
        if not args.allow_cpu:
            sys.exit(
                "No GPU found. GLiNER2 fine-tuning needs one: run with `--flavor t4-small` on HF "
                "Jobs. Pass --allow-cpu to run anyway (only sensible with a tiny --max-train-samples)."
            )
        logger.warning("No GPU found; training on CPU because --allow-cpu was passed.")
    else:
        logger.info(
            "GPU: %s (ACCELERATOR=%s)", torch.cuda.get_device_name(0), os.environ.get("ACCELERATOR")
        )

    # Prove we can write the output repo BEFORE paying for training.
    api = HfApi(token=token)
    if args.no_push:
        logger.info("--no-push: the model will stay in %s.", os.path.join(args.output_dir, "final"))
    else:
        args.output_repo = in_own_account(api, args.output_repo)
        ensure_output_repo(api, args.output_repo, private=not args.public)

    precision = resolve_precision(args.precision)
    sampling_config = resolve_sampling_config(args.label_augmentation)
    manifest = build_manifest(args, sampling_config, precision)
    manifest_dirs = [args.output_dir]
    if args.export_predictions:
        manifest_dirs.append(args.export_predictions)
    # Written now, so a run that dies still records what it was; rewritten with results at the end.
    write_manifest(manifest, manifest_dirs)

    train_data, raw_eval_sets, carved_out = load_splits(args)

    for split_name, data in [("train", train_data)] + list(raw_eval_sets.items()):
        for column in [args.text_column] + args.label_column:
            if column not in data.column_names:
                sys.exit(f"Column '{column}' not found in '{split_name}'. Columns are: {data.column_names}.")

    train_data = drop_unlabelled_rows(train_data, args.label_column, args.text_column, "train")
    if args.max_train_samples and len(train_data) > args.max_train_samples:
        train_data = train_data.shuffle(seed=args.seed).select(range(args.max_train_samples))
    logger.info("Train examples: %d.", len(train_data))

    fixed_labels = load_labels_file(args.labels_file) if args.labels_file else None
    tasks = build_tasks(train_data, args.label_column, args.task_name, fixed_labels)
    train_texts = prepare_texts(train_data, args.text_column, args.max_text_chars, "train")
    train_gold = {task["name"]: decode_column(train_data, task["column"]) for task in tasks}
    logger.info("Example input: %s", train_texts[0][:300])

    # Local eval files, and any eval split whose predictions are exported, are scored in full
    # and in order. Only a Hub eval split that is not exported keeps the old shuffled cap.
    full_eval = bool(args.train_file or args.export_predictions)
    eval_sets = {}
    for split_name, data in raw_eval_sets.items():
        data = add_row_numbers(data)
        data = drop_unlabelled_rows(data, args.label_column, args.text_column, split_name)
        if not full_eval and len(data) > args.max_eval_samples:
            data = data.shuffle(seed=args.seed).select(range(args.max_eval_samples))
        eval_sets[split_name] = {
            "texts": prepare_texts(data, args.text_column, args.max_text_chars, split_name),
            "gold": {task["name"]: decode_column(data, task["column"]) for task in tasks},
            "rows": list(data[ROW_COLUMN]),
        }
        logger.info("Eval split '%s': %d examples.", split_name, len(data))

    if fixed_labels is not None:
        gold_by_split = {"train": train_gold}
        for split_name, eval_set in eval_sets.items():
            gold_by_split[split_name] = eval_set["gold"]
        check_labels_in_set(tasks, gold_by_split)

    zero_shot = None
    if not args.skip_zero_shot:
        logger.info("Scoring the base model zero-shot, before any training.")
        zero_shot = evaluate(
            args.base_model, eval_sets, tasks, args.eval_batch_size, args.export_predictions, "base"
        )
        for split_name, split_metrics in zero_shot.items():
            logger.info("Zero-shot on '%s': %s", split_name, json.dumps(split_metrics["tasks"]))

    examples = build_training_examples(train_texts, train_gold, tasks)
    logger.info("Training precision: %s", precision)
    logger.info("Training for %d epochs on %d examples.", args.epochs, len(examples))
    started = time.time()
    oom_steps = train_with_batch_fallback(args, examples, precision, sampling_config)
    train_seconds = time.time() - started
    logger.info("Training took %.0f seconds.", train_seconds)
    if oom_steps:
        logger.warning(
            "%d training step(s) were skipped after running out of GPU memory. The model trained "
            "on the rest. Lower --batch-size to avoid this.", oom_steps,
        )

    # Score the checkpoint that will actually be uploaded.
    final_dir = os.path.join(args.output_dir, "final")
    fine_tuned = evaluate(
        final_dir, eval_sets, tasks, args.eval_batch_size, args.export_predictions, "finetuned"
    )
    for split_name, split_metrics in fine_tuned.items():
        logger.info("Fine-tuned on '%s': %s", split_name, json.dumps(split_metrics["tasks"]))

    schema_record = {
        "text_column": args.text_column,
        "tasks": [
            {"name": task["name"], "labels": task["labels"], "multi_label": task["multi_label"]}
            for task in tasks
        ],
    }
    with open(os.path.join(final_dir, SCHEMA_FILENAME), "w") as handle:
        json.dump(schema_record, handle, indent=2)
    card = build_card(
        args, tasks, zero_shot, fine_tuned, len(examples), train_seconds, carved_out, oom_steps
    )
    with open(os.path.join(final_dir, "README.md"), "w") as handle:
        handle.write(card)

    summary = {"zero_shot": zero_shot, "fine_tuned": fine_tuned, "train_seconds": round(train_seconds)}
    manifest["tasks"] = schema_record["tasks"]
    manifest["train_examples"] = len(examples)
    manifest["oom_steps"] = oom_steps
    manifest["results"] = summary
    write_manifest(manifest, manifest_dirs + [final_dir])

    if args.no_push:
        logger.info("--no-push: nothing uploaded. The model is in %s", final_dir)
    else:
        api.upload_folder(repo_id=args.output_repo, folder_path=final_dir, repo_type="model")
        logger.info("Pushed to https://huggingface.co/%s", args.output_repo)
    print("SUMMARY_JSON " + json.dumps(summary))


def parse_eval_files(values: list) -> dict:
    """Turn repeated --eval-file NAME=PATH values into {name: path}, in the order given."""
    eval_files = {}
    for value in values or []:
        name, separator, path = value.partition("=")
        name = name.strip()
        if not separator or not name or not path:
            sys.exit(f"--eval-file wants NAME=PATH, got {value!r}.")
        if "/" in name or name in eval_files:
            sys.exit(f"--eval-file name {name!r} must be unique and contain no '/'.")
        eval_files[name] = path
    return eval_files


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_dataset", nargs="?",
        help="Input dataset ID. Leave out with --train-file; a single positional is then the output repo.",
    )
    parser.add_argument("output_repo", nargs="?", help="Output model repo: a name for your own account (my-model) or a full ID (org/my-model). Not needed with --no-push.")
    parser.add_argument("--train-file", help="Train on a local JSON Lines file (e.g. under a mounted /bucket) instead of a Hub dataset")
    parser.add_argument(
        "--eval-file", action="append",
        help="NAME=PATH of a local JSON Lines eval split. Repeat for several splits. Scored in full, in file order.",
    )
    parser.add_argument(
        "--labels-file",
        help="Fixed label set (a JSON list, or one label per line), used in this order for training, "
        "zero-shot and eval. Every label in the data must be in it. Needs exactly one --label-column.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help=f"GLiNER2 checkpoint to start from (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--dataset-config", help="Dataset config name")
    parser.add_argument("--text-column", default="text", help="Text column (default: text)")
    parser.add_argument(
        "--label-column", action="append",
        help="Label column (default: label). Repeat for several tasks in one model. A column of "
        "lists is treated as multi-label.",
    )
    parser.add_argument(
        "--task-name", action="append",
        help="Name of the task, one per --label-column in the same order (default: the column "
        "name). The model reads it as part of its prompt, and it names the output columns.",
    )
    parser.add_argument("--train-split", default="train", help="Train split (default: train)")
    parser.add_argument("--eval-split", help="Eval split (default: validation or test if present, else a carve-out of train)")
    parser.add_argument("--eval-fraction", type=float, default=0.1, help="Eval fraction if no eval split (default: 0.1)")
    parser.add_argument("--max-train-samples", type=int, help="Cap training examples (smoke runs)")
    parser.add_argument(
        "--max-eval-samples", type=int, default=2000,
        help="Cap Hub eval examples (default: 2000). Not applied to --eval-file splits or with --export-predictions.",
    )
    parser.add_argument("--max-text-chars", type=int, default=2000, help="Truncate texts to this many characters (default: 2000)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Prediction batch size (default: 32)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--encoder-lr", type=float, default=1e-5, help="Encoder learning rate (default: 1e-5)")
    parser.add_argument("--task-lr", type=float, default=5e-4, help="Task-head learning rate (default: 5e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument(
        "--precision", choices=["auto", "fp32", "bf16"], default="auto",
        help="Training precision (default: auto = bf16 on Ampere or newer GPUs such as A10G and L4, fp32 on T4 and CPU)",
    )
    parser.add_argument(
        "--label-augmentation", choices=["upstream", "off"], default="upstream",
        help="upstream (default) = gliner2's synthetic label names and label dropping during training; "
        "off = always train on the real, complete label set (for a fixed schema)",
    )

    parser.add_argument("--skip-zero-shot", action="store_true", help="Skip the zero-shot score of the base model")
    parser.add_argument("--allow-cpu", action="store_true", help="Train without a GPU (slow)")
    parser.add_argument("--output-dir", default="./output", help="Local checkpoint directory; the model is saved in <dir>/final (default: ./output)")
    parser.add_argument(
        "--export-predictions",
        help="Directory for per-row predictions: <dir>/{base,finetuned}-<split>/predictions.jsonl",
    )
    parser.add_argument("--no-push", action="store_true", help="Do not create or upload a Hub repo; keep the model in --output-dir")
    parser.add_argument("--public", action="store_true", help="Make the output model repo public (default: private)")
    parser.add_argument("--private", action="store_true", help="Accepted for older commands; private is now the default")
    parser.add_argument("--hf-token", help="HF token (or set HF_TOKEN)")
    args = parser.parse_args()
    if not args.label_column:
        args.label_column = ["label"]
    if not args.task_name:
        args.task_name = list(args.label_column)
    if len(args.task_name) != len(args.label_column):
        parser.error("Pass one --task-name per --label-column, in the same order.")
    if len(set(args.task_name)) != len(args.task_name):
        parser.error("Each --task-name must be different.")
    if args.public and args.private:
        parser.error("Pass --public or --private, not both.")

    if args.train_file:
        # With local files there is no input dataset, so one positional is the output repo.
        if args.input_dataset and not args.output_repo:
            args.output_repo = args.input_dataset
            args.input_dataset = None
        if args.input_dataset:
            parser.error("Pass either an input dataset or --train-file, not both.")
        if not args.eval_file:
            parser.error("--train-file needs at least one --eval-file NAME=PATH.")
        if args.eval_split or args.dataset_config:
            parser.error("--eval-split and --dataset-config apply to Hub datasets, not --train-file.")
    else:
        if not args.input_dataset:
            parser.error("Pass an input dataset ID, or --train-file with --eval-file.")
        if args.eval_file:
            parser.error("--eval-file needs --train-file.")
    args.eval_files = parse_eval_files(args.eval_file)
    if not args.no_push and not args.output_repo:
        parser.error("Pass an output repo, or --no-push.")
    if args.labels_file and len(args.label_column) != 1:
        parser.error("--labels-file needs exactly one --label-column.")
    return args


if __name__ == "__main__":
    main(parse_args())
