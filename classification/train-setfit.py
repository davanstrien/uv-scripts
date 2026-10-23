# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "setfit>=1.2.0",
#     "datasets>=4.0.0",
#     "scikit-learn",
#     "huggingface-hub",
#     "torch",
#     "transformers",
# ]
# ///
"""
Few-shot text classification with SetFit — train on 8-64 labelled examples per class, on CPU or GPU.

SetFit fine-tunes a sentence-transformer body with contrastive pairs, then fits a logistic
regression head on the embeddings. It supports small labelled datasets between zero-shot LLM
labelling and a full encoder fine-tune (`train-classifier.py`). CPU is practical for small
experiments; use a GPU for faster training, particularly with larger models, longer texts or
more classes.

Run a small experiment on HF Jobs:

    hf jobs uv run --flavor cpu-basic --secrets HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-setfit.py \\
        fancyzhx/ag_news username/ag-news-setfit \\
        --num-samples 8

For faster training with the same model and settings, change --flavor to t4-small.

Metrics match `train-classifier.py` (accuracy + macro F1 on a held-out split) so the two are
directly comparable at equal eval settings.

NOTE: a SetFit model is a sentence-transformer body plus a scikit-learn head. It loads with
`SetFitModel.from_pretrained(repo)`, NOT `AutoModelForSequenceClassification`.
"""

import argparse
import logging
import os
import random
import sys
import time
from collections import Counter
from math import ceil, comb, isnan

# tqdm reads TQDM_DISABLE when it is imported, so this must be set before any third-party import
# pulls tqdm in — setting it later has no effect. Jobs logs have no TTY, so progress bars arrive
# as hundreds of carriage-return frames that bury the lines you actually want.
os.environ.setdefault("TQDM_DISABLE", "1")

import datasets
import torch
import transformers
from datasets import ClassLabel, Dataset, Value, load_dataset
from huggingface_hub import HfApi, ModelCard, login
from huggingface_hub.utils import disable_progress_bars
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.metrics import accuracy_score, f1_score


def configure_logging() -> logging.Logger:
    """Keep Jobs logs readable.

    `basicConfig(level=INFO)` sets the ROOT logger, which switches on every library's INFO
    output — on Jobs that means one line per HTTP request. Root stays at WARNING here and only
    this script's logger is verbose. Progress bars are disabled because Jobs logs have no TTY:
    tqdm's carriage-return frames arrive as hundreds of near-identical lines.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    disable_progress_bars()
    transformers.utils.logging.disable_progress_bar()
    if hasattr(datasets, "disable_progress_bars"):
        datasets.disable_progress_bars()

    script_logger = logging.getLogger("train-setfit")
    script_logger.setLevel(logging.INFO)
    return script_logger


logger = configure_logging()

SCRIPT_URL = (
    "https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-setfit.py"
)

# Fixed threshold for prompting review of a small gain over the majority baseline.
# This is a heuristic, not an estimate of seed variance or statistical significance.
NOISE_BAND = 0.05

# Applied to the measured step time. Covers what the measurement omits — optimizer update,
# pair-batch assembly, data loading. Raw shortfall against real runs of the same config:
# ~5% on cpu-basic (two independent configs agreed) and 37% on t4-small.
MEASUREMENT_MARGIN = 1.35


# MiniLM-L6 is the default because it makes the CPU path viable: ~4.4x faster than
# paraphrase-mpnet-base-v2 on cpu-basic (207s vs 915s for 8 examples/class on ag_news). It is
# NOT chosen on accuracy — on ag_news's test split the two scored 0.804 and 0.788 in single-seed
# runs, which do not establish a reliable ranking. Pass --body-model to try a larger body.
DEFAULT_BODY = "sentence-transformers/all-MiniLM-L6-v2"


def check_label_column(dataset: Dataset, label_column: str) -> None:
    """Fail early and clearly on a missing or multi-label column."""
    if not len(dataset):
        sys.exit("Dataset split is empty. Supply a non-empty labelled split.")
    if label_column not in dataset.column_names:
        sys.exit(
            f"Label column '{label_column}' not found. Columns are: {dataset.column_names}. "
            "Pass --label-column."
        )

    # A Sequence/list feature carries an inner `feature`; that is the multi-label shape.
    feature = dataset.features.get(label_column)
    if getattr(feature, "feature", None) is not None:
        sys.exit(
            f"Label column '{label_column}' is multi-label (a list per row). "
            "train-setfit.py is single-label only — use train-classifier.py, which "
            "auto-detects multi-label and tunes per-label thresholds."
        )
    if isinstance(dataset[label_column][0], list):
        sys.exit(
            f"Label column '{label_column}' holds lists (multi-label). "
            "Use train-classifier.py instead."
        )


def normalise_label_column(dataset: Dataset, label_column: str) -> Dataset:
    """Make the label column safe for SetFit's positional label mapping.

    SetFit maps an integer prediction through `model.labels` BY POSITION, so integers are only
    safe when they really are indices — which is true for a ClassLabel column and nothing else.
    Any other integer column holds arbitrary values (1-indexed, sparse, or negative), so it is
    stringified and the head learns the label text directly. Without this, a -1/0/1 column
    decodes through Python's negative indexing and silently mislabels everything while the
    metrics still look correct.
    """
    feature = dataset.features.get(label_column)
    if isinstance(feature, ClassLabel):
        return dataset
    # cast_column, not map: map re-casts its output back to the column's EXISTING feature, so
    # returning str() from a map over an int64 column silently converts straight back to int64.
    return dataset.cast_column(label_column, Value("string"))


def drop_unlabelled_rows(dataset: Dataset, label_column: str, split_name: str) -> Dataset:
    """Remove rows whose label is missing or blank.

    Real-world catalogue data carries missing values, and a blank string is silently a valid
    class name: biglam/hansard_speech trains a "" party class unless this runs. Dropping is the
    right default — an unlabelled row is not a class, and keeping it teaches the model to
    predict "no label".
    """
    feature = dataset.features.get(label_column)

    def has_label(value) -> bool:
        if value is None:
            return False
        if isinstance(feature, ClassLabel) and value == -1:
            return False
        if isinstance(value, float) and isnan(value):
            return False
        return not (isinstance(value, str) and not value.strip())

    kept = dataset.filter(has_label, input_columns=[label_column])
    dropped = len(dataset) - len(kept)
    if dropped:
        logger.warning(
            "Dropped %d %s rows with a missing or blank '%s' (%d remain).",
            dropped, split_name, label_column, len(kept),
        )
    return kept


def prepare_split(dataset, text_column, label_column, split_name):
    """Validate both splits before loading a model or paying for training."""
    check_label_column(dataset, label_column)
    if text_column not in dataset.column_names:
        sys.exit(
            f"Text column '{text_column}' not found in {split_name}. "
            f"Columns are: {dataset.column_names}. Pass --text-column."
        )
    # Check missing labels before casting: a float NaN otherwise becomes the class "nan".
    dataset = drop_unlabelled_rows(dataset, label_column, split_name)
    if not len(dataset):
        sys.exit(f"No labelled rows remain in {split_name} after removing missing labels.")
    def has_text(text):
        if text is None:
            return False
        if not isinstance(text, str):
            sys.exit(
                f"Text column '{text_column}' in {split_name} contains non-string values. "
                "Clean the text column before training."
            )
        return bool(text.strip())

    kept = dataset.filter(has_text, input_columns=[text_column])
    if len(kept) < len(dataset):
        logger.warning(
            "Dropped %d %s rows with missing or blank '%s' (%d remain).",
            len(dataset) - len(kept), split_name, text_column, len(kept),
        )
    if not len(kept):
        sys.exit(f"No usable text rows remain in {split_name} after removing missing texts.")
    return normalise_label_column(kept, label_column)


def resolve_label_names(dataset: Dataset, label_column: str) -> list[str]:
    """Return the class names, in the order SetFit should map predictions through."""
    feature = dataset.features.get(label_column)
    if isinstance(feature, ClassLabel):
        return list(feature.names)
    return sorted(set(dataset[label_column]))


def pick_eval_split(dataset_id, config, train_split, requested):
    """Resolve which split to evaluate on, matching train-classifier.py's precedence."""
    if requested:
        if requested == train_split:
            sys.exit(
                f"--eval-split and --train-split are both '{requested}'. Evaluating on the "
                "training data would report a meaningless score."
            )
        return requested

    # Same auto-detect order as the sibling, so both scripts evaluate on the same split by
    # default and their reported metrics really are comparable.
    available = datasets.get_dataset_split_names(dataset_id, config)
    for candidate in ("validation", "test"):
        if candidate in available and candidate != train_split:
            logger.info("Using the '%s' split for evaluation.", candidate)
            return candidate
    return None


def split_train_eval(dataset_id, config, train_split, eval_split, eval_fraction, seed, label_column):
    """Load the train split, and either the named eval split or a stratified carve-out."""
    train_data = load_dataset(dataset_id, config, split=train_split)

    if eval_split:
        eval_data = load_dataset(dataset_id, config, split=eval_split)
        return train_data, eval_data

    logger.info("No eval split found; carving %.0f%% off the train split.", eval_fraction * 100)
    check_label_column(train_data, label_column)
    train_data = drop_unlabelled_rows(train_data, label_column, "train")
    if len(train_data) < 2:
        sys.exit("Need at least two labelled rows to carve out an evaluation split.")
    # Encode plain labels as well, so string/int columns get the same stratification guarantee.
    train_data = normalise_label_column(train_data, label_column)
    if not isinstance(train_data.features[label_column], ClassLabel):
        train_data = train_data.class_encode_column(label_column)

    try:
        parts = train_data.train_test_split(
            test_size=eval_fraction, seed=seed, stratify_by_column=label_column
        )
    except ValueError as error:
        # Stratification needs at least two members of every class, so it fails on exactly the
        # singleton classes it is meant to protect. An unstratified split is worse but usable;
        # crashing is not.
        logger.warning(
            "Could not stratify the carve-out (%s). Falling back to an unstratified split — a "
            "very rare class may be absent from either split.",
            error,
        )
        parts = train_data.train_test_split(test_size=eval_fraction, seed=seed)
    return parts["train"], parts["test"]


def evaluate(model, eval_data, text_column, label_column) -> dict:
    """Predict on the eval set and report the same metrics as train-classifier.py."""
    # None in the text column would crash model.encode after training has already been paid for.
    texts = [str(text) for text in eval_data[text_column]]
    gold_raw = list(eval_data[label_column])

    # The model predicts label NAMES. Decode the gold side using the EVAL set's own feature —
    # a named --eval-split can order its ClassLabel differently from the train split, and
    # decoding through train-derived names would silently score against the wrong table.
    feature = eval_data.features.get(label_column)
    if isinstance(feature, ClassLabel):
        gold = [feature.int2str(int(value)) for value in gold_raw]
    else:
        gold = [str(value) for value in gold_raw]

    started = time.time()
    predictions = model.predict(texts)
    elapsed = time.time() - started

    # SetFit returns a tensor for int labels and a list for string labels.
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()

    # The majority-class rate is the floor any classifier must clear to be worth having. Without
    # it a number like 0.37 reads as "a model"; against a 0.35 floor it reads as "nothing learned".
    majority = Counter(gold).most_common(1)[0][1] / len(gold)

    return {
        "accuracy": round(accuracy_score(gold, predictions), 4),
        "majority_baseline": round(majority, 4),
        "f1_macro": round(f1_score(gold, predictions, average="macro", zero_division=0), 4),
        "eval_examples": len(gold),
        "predict_seconds": round(elapsed, 1),
    }


def warn_on_truncation(model, texts, max_seq_length: int) -> None:
    """Say how much of the corpus is being cut off.

    Truncation is silent and its consequence is not uniform: for short utterances it never fires,
    while for long documents it can remove the very span that carries the label. The 256-token
    default is right for the former and wrong for the latter, so measure and report rather than
    letting it be discovered in the scores.
    """
    tokenizer = model.model_body.tokenizer
    # No internal cap: the caller decides the sample, and it deliberately mixes train and eval.
    # An earlier version re-sliced to the first 200 here, which meant the eval texts appended by
    # the caller were never actually looked at.
    sample = list(texts)
    lengths = [
        len(tokenizer.encode(text, truncation=False, add_special_tokens=True)) for text in sample
    ]
    over = [n for n in lengths if n > max_seq_length]
    if not over:
        return

    median_over = sorted(over)[len(over) // 2]
    logger.warning(
        "%d of %d sampled documents exceed --max-seq-length %d (median of those: %d tokens). "
        "Everything past the limit is discarded before training and before prediction. If the "
        "signal for your labels sits late in the document, raise --max-seq-length within the "
        "body model's supported context window, or choose a longer-context --body-model.",
        len(over), len(sample), max_seq_length, median_over,
    )


def measure_step_seconds(model, texts, batch_size: int) -> float:
    """Time a real forward+backward on real texts, on the hardware that will train.

    Earlier versions timed an ENCODE and multiplied by a constant standing in for the backward
    pass. That constant had to be fitted per device (5 on CPU, 3 on GPU) and each value rested on
    a single observation — the same one-datapoint reasoning that produced two other wrong guards
    today. A training step is a forward and a backward over 2 x batch_size texts, so time exactly
    that instead and delete the constant.

    The loss here is a stand-in, not SetFit's CoSENTLoss: cost is dominated by the transformer
    forward and backward over the batch, not by the scalar reduction on top.
    """
    body = model.model_body
    device = body.device
    pool = list(texts)
    rng = random.Random(0)

    def draw() -> list:
        # A real step always sees 2 x batch_size texts because pairs are drawn WITH repetition.
        # Sampling min(2*batch_size, len(pool)) instead measured a short batch whenever the
        # few-shot set was smaller than a batch — a 2-class 8-shot run at the default batch size
        # measured half a step and projected it as a whole one.
        return rng.choices(pool, k=2 * batch_size)

    def one_step(sample) -> None:
        features = body.tokenize(sample)
        # tokenize() does not return tensors for every key, so move only what can move.
        features = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in features.items()
        }
        embeddings = body(features)["sentence_embedding"]
        embeddings.pow(2).mean().backward()
        body.zero_grad(set_to_none=True)

    was_training = body.training
    body.train()
    try:
        one_step(draw())  # warmup: first pass pays kernel/thread setup, not per-step cost
        timings = []
        for _ in range(3):
            # Draw a FRESH batch each time. Reusing one sample collapses timing noise but not
            # batch-composition noise, and padded length drives cost — on a corpus mixing short
            # interjections with long speeches, one unlucky draw sets the whole estimate.
            sample = draw()
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # CUDA is async; without this we time the launch only
            started = time.time()
            one_step(sample)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append(time.time() - started)
    finally:
        body.zero_grad(set_to_none=True)
        if not was_training:
            body.eval()

    return sorted(timings)[1]


def project_training_seconds(model, texts, batch_size: int, steps: int, max_seq_length: int) -> float:
    """Project total training time from a measured step.

    Step count alone cannot bound runtime: measured cost per step ranged from 0.07s (short
    utterances on a T4) to 11.2s (long speeches on CPU), a 160x spread driven by hardware and
    document length. A 2,055-step job cleared a 5,000-step budget and then ran for six hours.
    """
    try:
        measured = measure_step_seconds(model, texts, batch_size)
        # The measurement covers forward+backward, which is most of a step but not all of it: the
        # optimizer update, pair-batch assembly and data loading are not included. Raw shortfall
        # against real runs of the same config, measured AFTER the full-batch fix:
        #   2-class cpu-basic  8.01 vs 8.41 actual   -5%
        #   4-class cpu-basic  2.39 vs 2.51 actual   -5%
        #   4-class t4-small   0.055 vs 0.0875       -37%
        # The margin leaves CPU over-reading by ~28%, which is the side a refusal gate should err
        # on, and leaves GPU under-reading by ~15%. That GPU looseness is accepted, but NOT
        # because GPU runs never approach the budget — 200 classes at 8/class is ~159k steps,
        # nearly four hours on a T4, so they certainly do. It is accepted because a 15% under-read
        # only changes the verdict within 15% of the boundary, and the failure there is a job that
        # runs modestly over the budget the user set, not the multi-hour runaway this exists to
        # catch. Far from the boundary the answer is the same either way.
        per_step = measured * MEASUREMENT_MARGIN
        logger.info(
            "Measured %.3fs per training step (forward+backward); using %.3fs with margin.",
            measured, per_step,
        )
    except torch.cuda.OutOfMemoryError:
        # A step that will not fit now will not fit in training either. Fail here, cheaply and
        # clearly, rather than proceeding on a fragmented allocator and OOMing mid-run.
        torch.cuda.empty_cache()
        sys.exit(
            f"Out of GPU memory timing a single training step at --batch-size {batch_size} and "
            f"--max-seq-length {max_seq_length}. Training would fail the same way. Lower "
            "--batch-size or --max-seq-length, or use a larger flavor."
        )
    except Exception as error:
        # Any other failure of the guard itself must not block a legitimate run.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.warning("Could not time a training step (%s); skipping the time budget.", error)
        return 0.0
    return per_step * steps


def estimate_training_steps(per_class, batch_size, num_epochs, strategy) -> tuple[int, int]:
    """Return (contrastive pairs, optimizer steps) for one run, before any training happens.

    SetFit builds pairs from every combination of training examples, so the count grows with the
    SQUARE of the training-set size — and the training set is num_samples x number of classes.
    A 77-class dataset at 8 examples per class is 374k pairs under the default strategy, which
    is hours of CPU time. Knowing that before the job starts is worth a few lines of arithmetic.
    """
    counts = list(per_class.values())
    total = sum(counts)

    # SetFit's shuffle_combinations defaults to replacement=True, i.e. np.triu_indices(n, 0) —
    # the DIAGONAL is included, and those `total` self-pairs all count as positive. Negatives are
    # cross-class, so they exclude the diagonal and must be computed from the combinations
    # WITHOUT it. Verified against SetFit's own reported "Num unique pairs", which is the
    # POST-strategy total, so the strategy matters when reading these:
    #   oversampling:  2x4 -> 40 · 2x8 -> 144 · 3x8 -> 384 · 4x8 -> 768 · 77x8 -> 374,528
    #   undersampling: 4x8 -> 288
    #   unique:        4x8 -> 528
    same_class_pairs = sum(comb(count, 2) for count in counts)
    positive = same_class_pairs + total
    negative = comb(total, 2) - same_class_pairs

    if strategy == "oversampling":
        pairs = 2 * max(positive, negative)
    elif strategy == "undersampling":
        pairs = 2 * min(positive, negative)
    else:  # "unique"
        pairs = positive + negative

    steps = ceil(pairs / batch_size) * num_epochs
    return pairs, steps


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
    """Rebuild recipe options for Jobs, preserving the recorded accelerator when available.

    Only non-default flags are appended, keeping the command short while staying faithful.
    Outside Jobs, the hardware flavor is a suggested default rather than an exact record.
    """
    flavor = "t4-small" if torch.cuda.is_available() else "cpu-basic"
    if os.environ.get("JOB_ID"):
        flavor = jobs_flavor() or flavor
    parts = [
        f"hf jobs uv run --flavor {flavor} --secrets HF_TOKEN \\",
        f"  {SCRIPT_URL} \\",
        f"  {args.input_dataset} {args.output_repo} \\",
    ]

    flags = []
    if args.body_model != DEFAULT_BODY:
        flags.append(f"--body-model {args.body_model}")
    if args.dataset_config:
        flags.append(f"--dataset-config {args.dataset_config}")
    if args.text_column != "text":
        flags.append(f"--text-column {args.text_column}")
    if args.label_column != "label":
        flags.append(f"--label-column {args.label_column}")
    if args.train_split != "train":
        flags.append(f"--train-split {args.train_split}")
    if args.eval_split:
        flags.append(f"--eval-split {args.eval_split}")
    if args.num_samples != 8:
        flags.append(f"--num-samples {args.num_samples}")
    if args.num_epochs != 1:
        flags.append(f"--num-epochs {args.num_epochs}")
    if args.batch_size != 16:
        flags.append(f"--batch-size {args.batch_size}")
    if args.max_seq_length != 256:
        flags.append(f"--max-seq-length {args.max_seq_length}")
    if args.sampling_strategy != "oversampling":
        flags.append(f"--sampling-strategy {args.sampling_strategy}")
    if args.seed != 42:
        flags.append(f"--seed {args.seed}")
    # These three change which rows are trained on or scored, so a command without them
    # reproduces a different model and a different number.
    if args.max_train_pool != 200_000:
        flags.append(f"--max-train-pool {args.max_train_pool}")
    if args.max_eval_samples != 2000:
        flags.append(f"--max-eval-samples {args.max_eval_samples}")
    if args.eval_fraction != 0.1:
        flags.append(f"--eval-fraction {args.eval_fraction}")
    # Without these the published command either stops at the refusal gate or publishes to a
    # different visibility than the run it describes.
    if args.max_minutes != 60:
        flags.append(f"--max-minutes {args.max_minutes}")
    if args.allow_slow_training:
        flags.append("--allow-slow-training")
    if args.private:
        flags.append("--private")

    # --num-samples is the defining knob, so always show it even at its default.
    if f"--num-samples {args.num_samples}" not in flags:
        flags.insert(0, f"--num-samples {args.num_samples}")

    parts.append("  " + " ".join(flags))
    return "\n".join(parts)


def build_card(args, label_names, metrics, per_class, train_seconds, eval_split) -> str:
    """Model card following the uv-scripts conventions (org credit, Jobs claim gated on JOB_ID)."""
    on_jobs = os.environ.get("JOB_ID") is not None
    provenance = (
        "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) "
        "with [`uv-scripts/classification`](https://huggingface.co/datasets/uv-scripts/classification)."
        if on_jobs
        else "Produced with [`uv-scripts/classification`](https://huggingface.co/datasets/uv-scripts/classification)."
    )

    tags = ["setfit", "text-classification", "few-shot", "uv-script"]
    if on_jobs:
        tags.append("hf-jobs")
    tag_lines = "\n".join(f"- {tag}" for tag in tags)

    metric_lines = "\n".join(f"| {name} | {value} |" for name, value in metrics.items())
    train_size = sum(per_class.values())
    counts = ", ".join(f"`{name}`: {count}" for name, count in per_class.items())

    # Disclose the two things that most often make a headline number misleading.
    caveats = []
    short = {name: n for name, n in per_class.items() if n < args.num_samples}
    if short:
        caveats.append(
            f"**{len(short)} of {len(per_class)} classes had fewer than {args.num_samples} "
            f"examples available** ({', '.join(f'`{k}`: {v}' for k, v in short.items())}). "
            "The few-shot budget was not met for those classes."
        )
    if not eval_split:
        caveats.append(
            f"**No held-out split existed, so {args.eval_fraction:.0%} was carved out of train.** "
            "These numbers are not comparable with published results on this dataset."
        )
    caveats.append(
        f"Accuracy is reported against a majority-class baseline of "
        f"`{metrics['majority_baseline']}`. Also compare with a simple trained baseline on "
        "the same rows, and consider a zero-shot comparison where suitable. A small "
        "single-seed gain does not establish reliable improvement."
    )
    caveat_block = "\n".join(f"- {c}" for c in caveats)

    return f"""---
tags:
{tag_lines}
library_name: setfit
pipeline_tag: text-classification
base_model: {args.body_model}
datasets:
- {args.input_dataset}
---

# {args.output_repo.split("/")[-1]}

Few-shot text classifier trained with [SetFit](https://github.com/huggingface/setfit) on
**up to {args.num_samples} examples per class** ({train_size} training examples in total) from
[`{args.input_dataset}`](https://huggingface.co/datasets/{args.input_dataset}).

{provenance}

## Results

| Metric | Value |
|---|---|
{metric_lines}
| training seconds | {round(train_seconds, 1)} |

## Training examples per class

{counts}

## Read this before trusting the numbers

{caveat_block}

## Labels

{", ".join(f"`{name}`" for name in label_names)}

## Use it

```python
from setfit import SetFitModel

model = SetFitModel.from_pretrained("{args.output_repo}")
model.predict(["some text to classify"])
```

## Reproduction

Produced by [`train-setfit.py`]({SCRIPT_URL}) from
[`uv-scripts/classification`](https://huggingface.co/datasets/uv-scripts/classification):

```bash
{build_reproduce_command(args)}
```
"""


def main(args) -> None:
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("No HF token. Pass --hf-token or run with --secrets HF_TOKEN.")
    login(token=token)

    # Prove we can write the output repo BEFORE paying for training. A permissions failure
    # after trainer.train() costs the whole run and leaves no artifact behind.
    api = HfApi(token=token)
    api.create_repo(
        args.output_repo, repo_type="model", private=args.private, exist_ok=True
    )
    if args.private and not api.model_info(args.output_repo).private:
        sys.exit(
            f"Output repo '{args.output_repo}' is public. --private does not change an existing "
            "repo's visibility. Choose a new output repo or make that repo private before training."
        )

    logger.info("Loading %s", args.input_dataset)
    eval_split = pick_eval_split(
        args.input_dataset, args.dataset_config, args.train_split, args.eval_split
    )
    train_pool, eval_data = split_train_eval(
        args.input_dataset,
        args.dataset_config,
        args.train_split,
        eval_split,
        args.eval_fraction,
        args.seed,
        args.label_column,
    )

    # Cap before final validation (a carved split already needed a pass over labels).
    # Both caps are O(1) selects while the blank-label filter is a full
    # scan, and on a 2.4M-row corpus that ordering was eight minutes of preflight before a single
    # training step. sample_dataset also calls .to_pandas() on whatever pool it is handed, which
    # would OOM a cpu-basic job outright. The cap samples randomly, so a very rare class can be
    # thinned by it.
    if args.max_train_pool and len(train_pool) > args.max_train_pool:
        train_pool = train_pool.shuffle(seed=args.seed).select(range(args.max_train_pool))
        logger.info("Capped train pool at %d rows before sampling.", args.max_train_pool)
    if args.max_eval_samples and len(eval_data) > args.max_eval_samples:
        eval_data = eval_data.shuffle(seed=args.seed).select(range(args.max_eval_samples))
        logger.info("Capped eval set at %d examples.", args.max_eval_samples)

    train_pool = prepare_split(train_pool, args.text_column, args.label_column, "train")
    eval_data = prepare_split(eval_data, args.text_column, args.label_column, "eval")

    label_names = resolve_label_names(train_pool, args.label_column)
    logger.info("Found %d classes: %s", len(label_names), label_names)
    if len(set(train_pool[args.label_column])) < 2:
        sys.exit(f"Fewer than two observed classes in '{args.label_column}'. A classifier needs two or more.")

    train_data = sample_dataset(
        train_pool, label_column=args.label_column, num_samples=args.num_samples, seed=args.seed
    )
    # sample_dataset takes AT MOST num_samples per class, so report what was actually drawn.
    # Keys go through the class names, or a ClassLabel column reports bare indices.
    counts = sorted(Counter(train_data[args.label_column]).items())
    feature = train_data.features.get(args.label_column)
    if isinstance(feature, ClassLabel):
        # Keep the full positional label table, but disclose declared classes with no examples.
        per_class = dict.fromkeys(label_names, 0)
        per_class.update({feature.int2str(int(value)): count for value, count in counts})
    else:
        per_class = {str(value): count for value, count in counts}
    logger.info("Sampled %d training examples; per class: %s", len(train_data), per_class)

    pairs, steps = estimate_training_steps(
        per_class, args.batch_size, args.num_epochs, args.sampling_strategy
    )
    logger.info(
        "Contrastive pairs: %d -> %d optimizer steps (%s).", pairs, steps, args.sampling_strategy
    )
    logger.info("Loading body model %s", args.body_model)
    model = SetFitModel.from_pretrained(args.body_model, labels=label_names)
    model.model_body.max_seq_length = args.max_seq_length

    # SetFit picks the accelerator itself; report what it chose so a run's logs are self-describing.
    if torch.cuda.is_available():
        logger.info("DEVICE: cuda (%s)", torch.cuda.get_device_name(0))
    else:
        logger.info("DEVICE: cpu")
    logger.info("DEVICE: body model is on %s", model.model_body.device)

    # Sample the POOL and the eval set, not the few-shot training slice — that slice can be
    # as small as 16 texts, and eval documents are truncated at predict time too.
    warn_on_truncation(
        model,
        list(train_pool[args.text_column][:200]) + list(eval_data[args.text_column][:200]),
        args.max_seq_length,
    )

    projected = project_training_seconds(
        model, train_data[args.text_column], args.batch_size, steps, args.max_seq_length
    )
    if projected:
        logger.info("Projected training time: %.0f min (%d steps).", projected / 60, steps)
    if projected and projected / 60 > args.max_minutes and not args.allow_slow_training:
        _, cheaper_steps = estimate_training_steps(
            per_class, args.batch_size, args.num_epochs, "undersampling"
        )
        cheaper_minutes = projected / 60 * cheaper_steps / max(steps, 1)
        if args.sampling_strategy == "undersampling":
            suggestion = "  (already on the cheapest sampling strategy)"
        elif cheaper_minutes <= args.max_minutes:
            suggestion = (
                f"  --sampling-strategy undersampling  ->  {cheaper_steps} steps "
                f"(~{cheaper_minutes:.0f} min, within budget)"
            )
        else:
            suggestion = (
                f"  --sampling-strategy undersampling  ->  {cheaper_steps} steps "
                f"(~{cheaper_minutes:.0f} min — still over budget on this hardware)"
            )
        sys.exit(
            f"Refusing to start: projected {projected / 60:.0f} min of training exceeds "
            f"--max-minutes ({args.max_minutes}).\n"
            f"Measured on this hardware with your actual texts, so it accounts for both the pair "
            f"count and how long your documents are.\n"
            f"{suggestion}\n"
            f"  or lower --num-samples / --max-seq-length, use a GPU flavor, "
            f"or pass --allow-slow-training."
        )

    training_args = TrainingArguments(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        seed=args.seed,
        sampling_strategy=args.sampling_strategy,
        report_to="none",
        show_progress_bar=False,
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        column_mapping={args.text_column: "text", args.label_column: "label"},
    )

    started = time.time()
    trainer.train()
    train_seconds = time.time() - started
    logger.info("Training finished in %.1fs", train_seconds)

    metrics = evaluate(model, eval_data, args.text_column, args.label_column)
    logger.info("Metrics: %s", metrics)

    # A majority baseline is a useful first comparison. A small gain triggers review;
    # the fixed threshold does not determine whether the difference is significant.
    lift = metrics["accuracy"] - metrics["majority_baseline"]
    if lift <= 0:
        logger.warning(
            "BELOW FLOOR: accuracy %.3f does not beat always predicting the majority class "
            "(%.3f). This model is not worth deploying.",
            metrics["accuracy"], metrics["majority_baseline"],
        )
    elif lift < NOISE_BAND:
        logger.warning(
            "SMALL GAIN OVER BASELINE: accuracy %.3f versus a %.3f majority class is only %.1f "
            "points, below the %.0f-point review threshold. This threshold is a heuristic, "
            "not a significance test. Evaluate other seeds and matched baselines before "
            "drawing conclusions.",
            metrics["accuracy"], metrics["majority_baseline"], lift * 100, NOISE_BAND * 100,
        )
    else:
        logger.info("Beats the majority-class floor by %.1f points.", lift * 100)


    logger.info("Pushing to %s (private=%s)", args.output_repo, args.private)
    model.push_to_hub(args.output_repo, private=args.private, token=token)
    card = build_card(args, label_names, metrics, per_class, train_seconds, eval_split)
    ModelCard(card).push_to_hub(args.output_repo, token=token)

    logger.info("Verifying reload from the Hub")
    reloaded = SetFitModel.from_pretrained(args.output_repo, token=token)
    sample_texts = eval_data[args.text_column][:4]
    logger.info("Reloaded predictions: %s", reloaded.predict(sample_texts))
    logger.info("Done: https://huggingface.co/%s", args.output_repo)


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot text classification with SetFit")
    parser.add_argument("input_dataset", help="Input dataset ID")
    parser.add_argument("output_repo", help="Output model repo ID (username/model-name)")
    parser.add_argument("--body-model", default=DEFAULT_BODY, help=f"Sentence-transformer body (default: {DEFAULT_BODY})")
    parser.add_argument("--dataset-config", help="Dataset config name")
    parser.add_argument("--text-column", default="text", help="Text column (default: text)")
    parser.add_argument("--label-column", default="label", help="Label column (default: label)")
    parser.add_argument("--train-split", default="train", help="Train split (default: train)")
    parser.add_argument(
        "--eval-split",
        help="Eval split. Default: validation, else test, else carve --eval-fraction off "
             "train. A slice such as train[:10%%] is NOT checked for overlap with training.",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.1, help="Eval fraction if no eval split (default: 0.1)")
    parser.add_argument("--max-eval-samples", type=int, default=2000, help="Cap eval examples (default: 2000)")
    parser.add_argument(
        "--max-train-pool", type=int, default=200_000,
        help="Cap the pool before per-class sampling (default: 200000)",
    )
    parser.add_argument("--num-samples", type=int, default=8, help="Labelled examples per class (default: 8)")
    parser.add_argument("--num-epochs", type=int, default=1, help="Epochs (default: 1)")
    parser.add_argument("--sampling-strategy", default="oversampling",
                        choices=["oversampling", "undersampling", "unique"],
                        help="Contrastive pair sampling (default: oversampling)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Max sequence length (default: 256)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument("--max-minutes", type=int, default=60,
                        help="Refuse to start if projected training exceeds this (default: 60)")
    parser.add_argument("--allow-slow-training", action="store_true",
                        help="Override the --max-minutes refusal")
    parser.add_argument("--private", action="store_true", help="Make the output model repo private")
    parser.add_argument("--hf-token", help="HF token (or set HF_TOKEN)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
