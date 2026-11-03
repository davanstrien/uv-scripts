# /// script
# requires-python = ">=3.11"
# dependencies = []
#
# [tool.hf-jobs]
# flavor = "a100-large"
# timeout = "1h"
# secrets = ["HF_TOKEN"]
# ///
"""
Fine-tune a Kev decision model on your own labelled questions, on one HF Jobs GPU.

Kev (https://github.com/jaredpalmer/kev) answers yes/no, multiple-choice and rating questions
about a text, with a probability for each answer. This script is a thin launcher around Kev's
own commands. It pins one Kev commit and one set of released weights, installs Kev, and then runs:

1. `kev.train`: a LoRA + pointer-head delta from the released Kev-4B, on your training records.
2. `kev.benchmark`: the fine-tuned model and the released model, on your development records.
3. `kev.calibrate`: a calibration report for the fine-tuned model (report only, nothing changes).

All training and evaluation code is Kev's. Kev's own recipes run on Modal (the `kev-finetune`
skill) or on a GPU you have; this script is for people whose data and credit are on the Hub.

Input: a folder with `train.jsonl` and `development.jsonl` in Kev's record format (see the README).
Output: an empty or new folder on a mounted bucket. Each result is copied there as soon as it is
written: the checkpoint, both benchmark reports, the calibration report and `run.json`.

Example, with input and output in a bucket:

    hf jobs uv run --flavor a100-large --timeout 1h --secrets HF_TOKEN \\
        -v hf://buckets/<you>/kev-data:/mnt/kev \\
        https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-kev.py \\
        /mnt/kev/newsgroups /mnt/kev/runs/newsgroups-v1

Tested on 2026-10-30 (a100-large, 31 min, about $1.29): 20 Newsgroups, 846 training and 426 test
posts after Kev's length limit. Accuracy 72.3% fine-tuned vs 62.4% for the released Kev-4B. One seed.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

KEV_REPO = "https://github.com/jaredpalmer/kev"
KEV_TAG = "kev-1.0"
KEV_COMMIT = "6b719c3c3f367295f6ef336f4f751cf5ff970abc"  # the kev-1.0 tag on 2026-11-03
# jaredpalmer/kev-4b: the commit `@v1.0` resolved to on 2026-10-30 and 2026-11-03 (same weights as
# the v1.0 tag). Pinned by commit, because a tag can move.
INIT_MODEL = "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"  # main on 2026-11-03 (last change 2026-04-23)
QUESTION_TYPES = ("noul", "choice", "score")
MAX_OPTIONS = 255  # kev/api.py
WORK_DIR = Path("/tmp/kev")
SCRATCH = Path("/tmp/out")


def run(command, cwd=WORK_DIR):
    print(f"\n$ {' '.join(command)}", flush=True)
    started = time.time()
    subprocess.run(command, cwd=cwd, check=True)
    print(f"[{time.time() - started:.0f}s]", flush=True)


def check_gpu():
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    except FileNotFoundError:
        listing = None
    if listing is None or listing.returncode != 0 or "GPU" not in listing.stdout:
        sys.exit("No GPU found. Run this on a GPU flavor, for example --flavor a100-large.")
    print(listing.stdout.strip(), flush=True)


def question_problem(question):
    """Why a labelled question is not in Kev's format, or None. Rules from kev/api.py and Kev's README:
    noul labels are true/false; choice criteria is an object of options and the label is one of them;
    score criteria is a list of levels and the label is a level's position from 0."""
    if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
        return f"needs a type: one of {', '.join(QUESTION_TYPES)}"
    if "label" not in question:
        return "needs a label"
    kind, label, criteria = question["type"], question["label"], question.get("criteria")
    if kind == "noul" and not isinstance(label, bool):
        return "is a noul question, so its label must be true or false"
    if kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_OPTIONS:
            return f"is a choice question, so its criteria must be an object with 1 to {MAX_OPTIONS} options"
        if label not in criteria:
            return f"has label {label!r}, which is not one of its criteria"
    if kind == "score":
        if not isinstance(criteria, list) or not 1 <= len(criteria) <= MAX_OPTIONS:
            return f"is a score question, so its criteria must be a list of 1 to {MAX_OPTIONS} levels"
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(criteria):
            return f"is a score question, so its label must be a level position from 0 to {len(criteria) - 1}"
    return None


def check_records(path):
    """Fail before any GPU time is spent if a record is not in Kev's labelled format."""
    if not path.is_file():
        sys.exit(f"{path} not found. The input folder needs train.jsonl and development.jsonl.")
    count = 0
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            where = f"{path.name} line {line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                sys.exit(f"{where}: not valid JSON ({error}).")
            if not isinstance(record, dict) or record.get("state") is None:
                sys.exit(f'{where}: a record needs a "state" (text, or any JSON object or list).')
            questions = record.get("questions")
            if not isinstance(questions, dict) or not questions:
                sys.exit(f'{where}: a record needs a non-empty "questions" object.')
            for name, question in questions.items():
                problem = question_problem(question)
                if problem:
                    sys.exit(f'{where}: question "{name}" {problem}.')
            count += 1
    if count == 0:
        sys.exit(f"{path} has no records.")
    print(f"{path.name}: {count} records", flush=True)
    return count


def mount_of(path, mounts_file=Path("/proc/self/mounts")):
    """(mount point, filesystem type, device) of the mount that holds `path`."""
    best = ("", "", "")
    for line in mounts_file.read_text().splitlines():
        device, mount_point, fstype = line.split()[:3]
        holds_path = path == Path(mount_point) or Path(mount_point) in path.parents
        if holds_path and len(mount_point) > len(best[0]):
            best = (mount_point, fstype, device)
    return best


def prepare_output(output_dir):
    """Fail before any GPU time is spent if the results would be lost or mixed with an old run."""
    if os.environ.get("JOB_ID"):
        # on Jobs, only a mounted bucket outlives the container (it shows as device "hf-mount")
        mount_point, fstype, device = mount_of(output_dir)
        if device != "hf-mount":
            sys.exit(f"{output_dir} is not on a mounted bucket (it is on {mount_point}, {fstype or 'unknown'}). "
                     "Mount one with -v hf://buckets/<you>/<bucket>:/mnt/kev and write under /mnt/kev.")
    if output_dir.exists() and any(output_dir.iterdir()):
        sys.exit(f"{output_dir} is not empty. Use a new folder for each run.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.json").write_text(json.dumps({"status": "started"}))  # also checks that the folder is writable


def install_kev():
    run(["git", "clone", "-q", "--depth", "1", "--branch", KEV_TAG, KEV_REPO, str(WORK_DIR)], cwd="/tmp")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=WORK_DIR, check=True, capture_output=True, text=True)
    if head.stdout.strip() != KEV_COMMIT:
        sys.exit(f"{KEV_TAG} now points to {head.stdout.strip()}, not {KEV_COMMIT}. Check the Kev release notes.")
    run(["uv", "sync", "--no-dev", "-q"])
    cuda = subprocess.run(["uv", "run", "python", "-c", "import torch; print(torch.cuda.is_available())"],
                          cwd=WORK_DIR, capture_output=True, text=True)
    if cuda.stdout.strip() != "True":
        sys.exit(f"PyTorch in Kev's environment cannot use the GPU.\n{cuda.stderr[-2000:]}")


def save(name, output_dir):
    """Copy one finished result to the output folder, so a later failure cannot lose it."""
    destination = output_dir / name
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-r", f"{SCRATCH / name}/.", str(destination)], check=True)
    print(f"saved {destination}", flush=True)


def train(args):
    command = [
        "uv", "run", "python", "-m", "kev.train",
        "--data", str(args.input_dir / "train.jsonl"),
        "--base", BASE_MODEL,
        "--base_revision", BASE_REVISION,
        "--init_from", args.init_from,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch", "1", "--accum", "8", "--dtype", "bf16", "--checkpointing", "1",
        "--device", "cuda",
        "--out", str(SCRATCH / "checkpoint"),
    ]
    if args.max_steps:
        command += ["--max_steps", str(args.max_steps)]
    run(command)


def benchmark(model, data, name):
    run(["uv", "run", "python", "-m", "kev.benchmark", "--run", str(model), "--data", str(data), "--out", str(SCRATCH / name)])


def calibrate():
    """Kev's calibration report. It needs enough development records for its folds, so a failure
    here is reported and the run goes on: the checkpoint and benchmarks are already saved."""
    try:
        run(["uv", "run", "python", "-m", "kev.calibrate", "--rows", str(SCRATCH / "finetuned-development" / "rows.json")])
        report = json.loads((SCRATCH / "finetuned-development" / "calibration.json").read_text())
    except (subprocess.CalledProcessError, OSError, json.JSONDecodeError) as error:
        print(f"kev.calibrate failed ({error}); see the messages above. The other results are saved.", flush=True)
        return None
    return report.get("workload_temperature")


def headline(name):
    clean = json.loads((SCRATCH / name / "report.json").read_text())["clean"]
    return {"n": clean["n"], "acc": round(clean["acc"], 4), "ece": round(clean["ece"], 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Folder with train.jsonl and development.jsonl in Kev's format")
    parser.add_argument("output_dir", type=Path, help="New or empty folder for the results, on a mounted bucket")
    parser.add_argument("--init-from", default=INIT_MODEL, help="Released Kev model to start from (repo@revision)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many optimizer steps, for a smoke test")
    args = parser.parse_args()
    # Kev's commands run from its own checkout, so relative paths must not reach them
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    check_gpu()
    check_records(args.input_dir / "train.jsonl")
    check_records(args.input_dir / "development.jsonl")
    prepare_output(args.output_dir)
    started = time.time()
    install_kev()

    development = args.input_dir / "development.jsonl"
    train(args)
    save("checkpoint", args.output_dir)
    benchmark(SCRATCH / "checkpoint", development, "finetuned-development")
    save("finetuned-development", args.output_dir)
    benchmark(args.init_from, development, "baseline-development")
    save("baseline-development", args.output_dir)
    temperature = calibrate()
    if temperature is not None:
        # copy only the new file: the saved reports stay as they are
        calibration = SCRATCH / "finetuned-development" / "calibration.json"
        subprocess.run(["cp", str(calibration), str(args.output_dir / "finetuned-development")], check=True)
        print(f"saved {args.output_dir / 'finetuned-development' / 'calibration.json'}", flush=True)

    summary = {
        "status": "completed",
        "kev_commit": KEV_COMMIT,
        "init_from": args.init_from,
        "base_revision": BASE_REVISION,
        "finetuned": headline("finetuned-development"),
        "released": headline("baseline-development"),
        "workload_temperature": temperature,
        "seconds": round(time.time() - started),
    }
    (args.output_dir / "run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
