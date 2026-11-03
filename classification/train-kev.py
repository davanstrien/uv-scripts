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

Kev (https://github.com/jaredpalmer/kev) answers fixed-choice questions about a text with a
calibrated probability per option. This script is a thin launcher around Kev's own commands. It
pins one Kev commit and one set of released weights, installs Kev, and then runs:

1. `kev.train`: a LoRA + head delta from the released Kev-4B, on your training records.
2. `kev.benchmark`: the fine-tuned model and the released model, on your development records.
3. `kev.calibrate`: a calibration report for the fine-tuned model (report only, nothing changes).

All training and evaluation code is Kev's. Kev's own docs use Modal; this script is for people
whose data and credit are already on the Hub.

Input: a folder with `train.jsonl` and `development.jsonl` in Kev's record format (see the README).
Output: the checkpoint, both benchmark reports, the calibration report and `run.json`.

Example, with input and output in a bucket:

    hf jobs uv run -v hf://buckets/<you>/kev-data:/mnt/kev \\
        https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-kev.py \\
        /mnt/kev/newsgroups /mnt/kev/runs/newsgroups-v1

Tested on 2026-10-30 (a100-large, 31 min, about $1.29): 1,000 20 Newsgroups training posts and
426 test posts. Accuracy 72.3% fine-tuned vs 62.4% for the released Kev-4B. One seed.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

KEV_REPO = "https://github.com/jaredpalmer/kev"
KEV_TAG = "kev-1.0"
KEV_COMMIT = "6b719c3c3f367295f6ef336f4f751cf5ff970abc"  # the kev-1.0 tag on 2026-11-03
# jaredpalmer/kev-4b at tag v1.0 (2026-11-03). Pinned by commit, because a tag can move.
INIT_MODEL = "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"  # main on 2026-11-03 (last change 2026-04-23)
WORK_DIR = Path("/tmp/kev")


def run(command, cwd=WORK_DIR):
    print(f"\n$ {' '.join(command)}", flush=True)
    started = time.time()
    subprocess.run(command, cwd=cwd, check=True)
    print(f"[{time.time() - started:.0f}s]", flush=True)


def check_gpu():
    if shutil.which("nvidia-smi") is None:
        sys.exit("No GPU found. Run this on a GPU flavor, for example --flavor a100-large.")


def check_input(input_dir):
    for name in ("train.jsonl", "development.jsonl"):
        if not (input_dir / name).is_file():
            sys.exit(f"{input_dir / name} not found. The input folder needs train.jsonl and development.jsonl.")


def install_kev():
    run(["git", "clone", "-q", "--depth", "1", "--branch", KEV_TAG, KEV_REPO, str(WORK_DIR)], cwd="/tmp")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=WORK_DIR, check=True, capture_output=True, text=True)
    if head.stdout.strip() != KEV_COMMIT:
        sys.exit(f"{KEV_TAG} now points to {head.stdout.strip()}, not {KEV_COMMIT}. Check the Kev release notes.")
    run(["uv", "sync", "--no-dev", "-q"])


def train(args, output):
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
        "--out", str(output),
    ]
    if args.max_steps:
        command += ["--max_steps", str(args.max_steps)]
    run(command)


def benchmark(model, data, output):
    run(["uv", "run", "python", "-m", "kev.benchmark", "--run", str(model), "--data", str(data), "--out", str(output)])


def calibrate(benchmark_dir):
    # writes calibration.json next to rows.json
    run(["uv", "run", "python", "-m", "kev.calibrate", "--rows", str(benchmark_dir / "rows.json")])


def headline(benchmark_dir):
    report = json.loads((benchmark_dir / "report.json").read_text())
    clean = report["clean"]
    return {"n": clean["n"], "acc": round(clean["acc"], 4), "ece": round(clean["ece"], 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Folder with train.jsonl and development.jsonl in Kev's format")
    parser.add_argument("output_dir", type=Path, help="Folder for the checkpoint and the reports")
    parser.add_argument("--init-from", default=INIT_MODEL, help="Released Kev model to start from (repo@revision)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many optimizer steps, for a smoke test")
    args = parser.parse_args()

    check_gpu()
    check_input(args.input_dir)
    started = time.time()
    install_kev()

    scratch = Path("/tmp/out")
    development = args.input_dir / "development.jsonl"
    train(args, scratch / "checkpoint")
    benchmark(scratch / "checkpoint", development, scratch / "finetuned-development")
    benchmark(args.init_from, development, scratch / "baseline-development")
    calibrate(scratch / "finetuned-development")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-r", f"{scratch}/.", str(args.output_dir)], check=True)
    summary = {
        "kev_commit": KEV_COMMIT,
        "init_from": args.init_from,
        "base_revision": BASE_REVISION,
        "finetuned": headline(scratch / "finetuned-development"),
        "released": headline(scratch / "baseline-development"),
        "seconds": round(time.time() - started),
    }
    (args.output_dir / "run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
