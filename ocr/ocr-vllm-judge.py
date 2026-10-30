#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "rich",
#     "tqdm",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.29.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "a100-large"
# timeout = "1h"
# secrets = ["HF_TOKEN"]
# ///
"""
Offline vLLM judge for OCR benchmark evaluation.

Runs pairwise OCR quality comparisons using a local VLM judge via vLLM's
offline LLM() pattern. Supports jury mode (multiple models vote sequentially
on the same GPU) with majority voting.

Advantages over API-based judging (ocr-jury-bench.py):
- Zero network failures — everything runs locally
- vLLM structured output guarantees valid JSON (no parse retries)
- Batch processing is faster than sequential API calls
- Any vLLM-supported VLM can be used as judge

Usage:
    # Single judge
    uv run ocr-vllm-judge.py davanstrien/ocr-bench-nls-50 --from-prs \\
        --judge-model Qwen/Qwen2.5-VL-7B-Instruct --max-samples 3

    # Jury of 2 models (sequential on same GPU)
    uv run ocr-vllm-judge.py davanstrien/ocr-bench-nls-50 --from-prs \\
        --judge-model Qwen/Qwen3-VL-8B-Instruct \\
        --judge-model Qwen/Qwen2.5-VL-7B-Instruct \\
        --max-samples 50

    # Via HF Job. The [tool.hf-jobs] header pins vllm/vllm-openai:v0.29.0 (the
    # default uv image has no nvcc, which vLLM needs at warmup) and a100-large
    # (the default 7B judge does not leave room for the KV cache on a 24 GB L4).
    hf jobs uv run --timeout 1h \\
        ocr-vllm-judge.py davanstrien/ocr-bench-nls-50 --from-prs \\
        --judge-model Qwen/Qwen3-VL-8B-Instruct --max-samples 50
"""

import argparse
import base64
import gc
import io
import json
import logging
import random
import sys
from collections import Counter
from itertools import combinations
from typing import NamedTuple

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from PIL import Image
from rich.console import Console
from rich.table import Table

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
console = Console()

# --- ELO ---
INITIAL_ELO = 1500
K = 32


def update_elo(elo_a: float, elo_b: float, winner: str) -> tuple[float, float]:
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    if winner == "A":
        score_a = 1.0
    elif winner == "B":
        score_a = 0.0
    else:  # tie
        score_a = 0.5
    elo_a += K * (score_a - expected_a)
    elo_b += K * ((1 - score_a) - (1 - expected_a))
    return elo_a, elo_b


# --- Image helpers ---
def image_to_base64(image: Image.Image) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    max_dim = 1024
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --- Judge prompt ---
PAIRWISE_PROMPT = """You are an expert OCR quality evaluator. You are given a document image and TWO OCR outputs (A and B) extracted from that same image.

Compare them and decide which extraction is better overall.

Evaluation criteria (in priority order):

1. Faithfulness: The output must ONLY contain text from the document. Any added commentary, interpretation, or notes (e.g. "it appears the text says...", "the document contains...") is a serious error. Penalize heavily.

2. Completeness: ALL visible text must be captured — headers, footers, marginalia, stamps, handwritten notes. Missing any section of text is a significant penalty.

3. Accuracy: Correct characters, no hallucinated words or garbled text.

4. Reading order: Text flows naturally as a human would read the document.

5. Formatting: Clean structure. Ignore bounding box tags like <|ref|> <|det|> if present. Do NOT prefer fancier markdown formatting — plain accurate text is better than nicely formatted but incomplete text.

If both outputs capture the same text with similar accuracy, respond with "tie". Only pick a winner when there is a clear quality difference.

Output A:
---
{ocr_text_a}
---

Output B:
---
{ocr_text_b}
---

Respond with JSON only (no markdown fences, no extra text):
{{"winner": "A", "reason": "brief explanation"}}
Use "A", "B", or "tie" for the winner field."""

# JSON schema for structured output
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["winner", "reason"],
}


class Comparison(NamedTuple):
    """A single pairwise comparison to evaluate."""

    sample_idx: int
    model_a: str
    model_b: str
    col_a: str
    col_b: str
    swapped: bool  # True if A/B labels were swapped for position-bias mitigation
    messages: list  # Chat messages for vLLM


# --- Data loading (adapted from ocr-jury-bench.py) ---
def discover_ocr_columns(dataset) -> dict[str, str]:
    columns = {}
    try:
        info_raw = dataset[0].get("inference_info")
        if not info_raw:
            return columns
        info = json.loads(info_raw)
        if not isinstance(info, list):
            info = [info]
        for entry in info:
            col = entry.get("column_name", "")
            model = entry.get("model_id", entry.get("model_name", "unknown"))
            if col and col in dataset.column_names:
                columns[col] = model
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning(f"Could not parse inference_info: {e}")

    if not columns:
        for col in dataset.column_names:
            if "markdown" in col.lower() or "ocr" in col.lower() or "text" == col:
                columns[col] = col

    model_counts = {}
    for model in columns.values():
        model_counts[model] = model_counts.get(model, 0) + 1

    disambiguated = {}
    for col, model in columns.items():
        if model_counts[model] > 1:
            short_model = model.split("/")[-1] if "/" in model else model
            disambiguated[col] = f"{short_model} ({col})"
        else:
            disambiguated[col] = model

    return disambiguated


def load_benchmark_dataset(args):
    """Load dataset from configs/PRs/flat mode. Returns (dataset, ocr_columns)."""
    api = HfApi()
    pr_revisions = {}

    if args.from_prs or args.merge_prs:
        console.print(f"\n[bold]Checking PRs on:[/bold] {args.dataset}")
        discussions = api.get_repo_discussions(
            args.dataset,
            repo_type="dataset",
            discussion_type="pull_request",
            discussion_status="open",
        )
        prs = list(discussions)

        if not prs:
            console.print("[yellow]No open PRs found.[/yellow]")
        else:
            for pr in prs:
                title = pr.title
                config_name = None
                if "[" in title and title.endswith("]"):
                    config_name = title.rsplit("[", 1)[1].rstrip("]")

                if config_name:
                    console.print(
                        f"  PR #{pr.num}: {title} -> config [cyan]{config_name}[/cyan]"
                    )
                    pr_revisions[config_name] = f"refs/pr/{pr.num}"

                    if args.merge_prs:
                        console.print(f"    Merging PR #{pr.num}...")
                        api.merge_pull_request(
                            args.dataset,
                            pr.num,
                            repo_type="dataset",
                            comment="Auto-merged by ocr-vllm-judge.py",
                        )
                        pr_revisions[config_name] = "main"
                else:
                    console.print(
                        f"  PR #{pr.num}: {title} (skipped — no config name in title)"
                    )

        if args.from_prs and not args.configs:
            args.configs = list(pr_revisions.keys())
            if not args.configs:
                console.print("[red]No config PRs found. Nothing to evaluate.[/red]")
                sys.exit(1)
            console.print(f"\n  Auto-discovered configs: {', '.join(args.configs)}")

    if args.configs:
        console.print(
            f"\n[bold]Loading dataset (config-per-model):[/bold] {args.dataset}"
        )
        console.print(f"  Configs: {', '.join(args.configs)}")

        config_datasets = {}
        ocr_columns = {}

        for config_name in args.configs:
            revision = pr_revisions.get(config_name)
            rev_label = f" (from {revision})" if revision and revision != "main" else ""
            console.print(f"  Loading config: {config_name}{rev_label}")
            cds = load_dataset(
                args.dataset,
                name=config_name,
                split=args.split,
                revision=revision,
            )

            info_raw = cds[0].get("inference_info")
            if info_raw:
                info = json.loads(info_raw)
                entry = info[0] if isinstance(info, list) else info
                model_id = entry.get("model_id", config_name)
            else:
                model_id = config_name
            ocr_columns[config_name] = model_id
            config_datasets[config_name] = cds

        base = config_datasets[args.configs[0]]
        rows = []
        for i in range(len(base)):
            row = {"image": base[i]["image"]}
            for cn in args.configs:
                row[cn] = config_datasets[cn][i].get("markdown", "") or ""
            rows.append(row)
        ds = Dataset.from_list(rows)
        console.print(f"  Unified rows: {len(ds)}")
    else:
        console.print(f"[bold]Loading dataset:[/bold] {args.dataset}")
        ds = load_dataset(args.dataset, split=args.split)
        console.print(f"  Columns: {ds.column_names}")
        console.print(f"  Rows: {len(ds)}")

        if args.columns:
            ocr_columns = {col: col for col in args.columns}
        else:
            ocr_columns = discover_ocr_columns(ds)

    if len(ocr_columns) < 2:
        console.print(
            "[red]Need at least 2 OCR columns for pairwise comparison. "
            "Use --columns or --configs to specify them.[/red]"
        )
        sys.exit(1)

    return ds, ocr_columns


# --- vLLM structured output compatibility ---
def make_sampling_params(schema: dict, max_tokens: int = 512):
    """Create SamplingParams with structured output, handling vLLM version differences."""
    from vllm import SamplingParams

    # Try StructuredOutputsParams first (vLLM >= 0.12)
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            structured_outputs=StructuredOutputsParams(json=schema),
        )
    except (ImportError, TypeError):
        pass

    # Try GuidedDecodingParams (older vLLM)
    try:
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            guided_decoding=GuidedDecodingParams(json=schema),
        )
    except (ImportError, TypeError):
        pass

    # Fallback: no structured output, rely on prompt-based JSON
    logger.warning(
        "Structured output not available in this vLLM version. "
        "Falling back to prompt-based JSON parsing."
    )
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )


def parse_judge_output(text: str) -> dict:
    """Parse judge output, handling both structured and free-form JSON."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
        winner = result.get("winner", "tie").upper().strip()
        if winner not in ("A", "B", "TIE"):
            winner = "tie"
        return {"winner": winner, "reason": result.get("reason", "")}
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse judge output: {text[:200]}")
        return {}


# --- Build comparisons ---
def build_comparisons(
    dataset,
    ocr_columns: dict[str, str],
    max_samples: int | None,
    seed: int,
) -> list[Comparison]:
    """Build all pairwise comparison prompts upfront (CPU only)."""
    model_names = list(ocr_columns.values())
    col_names = list(ocr_columns.keys())
    pairs = list(combinations(range(len(col_names)), 2))

    indices = list(range(len(dataset)))
    if max_samples and max_samples < len(indices):
        random.seed(seed)
        indices = random.sample(indices, max_samples)

    rng = random.Random(seed)
    comparisons = []

    for idx in indices:
        row = dataset[idx]
        image = row["image"]
        b64 = image_to_base64(image)

        for i, j in pairs:
            col_a, col_b = col_names[i], col_names[j]
            model_a, model_b = model_names[i], model_names[j]
            text_a = row[col_a] or ""
            text_b = row[col_b] or ""

            if not text_a.strip() or not text_b.strip():
                continue

            # Randomize order to reduce position bias
            swapped = rng.random() < 0.5
            if swapped:
                prompt = PAIRWISE_PROMPT.format(
                    ocr_text_a=text_b[:2500], ocr_text_b=text_a[:2500]
                )
            else:
                prompt = PAIRWISE_PROMPT.format(
                    ocr_text_a=text_a[:2500], ocr_text_b=text_b[:2500]
                )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            comparisons.append(
                Comparison(
                    sample_idx=idx,
                    model_a=model_a,
                    model_b=model_b,
                    col_a=col_a,
                    col_b=col_b,
                    swapped=swapped,
                    messages=messages,
                )
            )

    return comparisons


# --- GPU cleanup ---
def cleanup_vllm():
    """Free GPU memory between judge models."""
    try:
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except ImportError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# --- Run judge ---
def run_judge(
    model_id: str,
    comparisons: list[Comparison],
    max_model_len: int,
    gpu_memory_utilization: float,
) -> list[dict]:
    """Run a single judge model on all comparisons via vLLM batch inference."""
    from vllm import LLM

    short_name = model_id.split("/")[-1] if "/" in model_id else model_id
    console.print(f"\n[bold]Loading judge:[/bold] {model_id}")

    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = make_sampling_params(JUDGE_SCHEMA, max_tokens=512)

    console.print(f"  Running {len(comparisons)} comparisons...")
    all_messages = [comp.messages for comp in comparisons]

    results = []
    try:
        outputs = llm.chat(all_messages, sampling_params)
        for output in outputs:
            text = output.outputs[0].text
            results.append(parse_judge_output(text))
        console.print(
            f"  [green]{short_name}: {sum(1 for r in results if r)}/{len(results)} valid responses[/green]"
        )
    except Exception as e:
        logger.error(f"vLLM batch inference failed for {short_name}: {e}")
        results = [{}] * len(comparisons)
    finally:
        del llm
        cleanup_vllm()

    return results


# --- Aggregate and score ---
def aggregate_jury_votes(
    all_judge_results: list[list[dict]],
    comparisons: list[Comparison],
    judge_names: list[str],
) -> list[dict]:
    """Aggregate votes from multiple judges using majority voting."""
    n_comparisons = len(comparisons)
    final_results = []

    for i in range(n_comparisons):
        votes = []
        reasons = []
        for j, judge_results in enumerate(all_judge_results):
            result = judge_results[i]
            if result:
                winner = result.get("winner", "tie").upper().strip()
                votes.append(winner)
                reasons.append(f"[{judge_names[j]}] {result.get('reason', '')[:60]}")

        if not votes:
            final_results.append({})
            continue

        counts = Counter(votes)
        winner, count = counts.most_common(1)[0]
        agreement = f"{count}/{len(votes)}"

        final_results.append(
            {
                "winner": winner,
                "reason": f"Jury ({agreement}): " + " | ".join(reasons),
                "votes": dict(counts),
                "agreement": agreement,
            }
        )

    return final_results


def compute_elo(
    comparisons: list[Comparison],
    results: list[dict],
    model_names: list[str],
) -> tuple[dict, dict, dict, dict, list]:
    """Compute ELO ratings from comparison results."""
    elo = {model: INITIAL_ELO for model in model_names}
    wins = {model: 0 for model in model_names}
    losses = {model: 0 for model in model_names}
    ties = {model: 0 for model in model_names}
    comparison_log = []

    for comp, result in zip(comparisons, results):
        if not result:
            continue

        winner_raw = result.get("winner", "tie").upper().strip()

        # Unswap if positions were randomized
        if comp.swapped:
            if winner_raw == "A":
                winner_raw = "B"
            elif winner_raw == "B":
                winner_raw = "A"

        model_a, model_b = comp.model_a, comp.model_b

        if winner_raw == "A":
            elo[model_a], elo[model_b] = update_elo(elo[model_a], elo[model_b], "A")
            wins[model_a] += 1
            losses[model_b] += 1
        elif winner_raw == "B":
            elo[model_a], elo[model_b] = update_elo(elo[model_a], elo[model_b], "B")
            losses[model_a] += 1
            wins[model_b] += 1
        else:
            elo[model_a], elo[model_b] = update_elo(elo[model_a], elo[model_b], "tie")
            ties[model_a] += 1
            ties[model_b] += 1

        comparison_log.append(
            {
                "sample_idx": comp.sample_idx,
                "model_a": model_a,
                "model_b": model_b,
                "winner": winner_raw,
                "reason": result.get("reason", ""),
                "agreement": result.get("agreement", "1/1"),
            }
        )

    return elo, wins, losses, ties, comparison_log


def print_elo_leaderboard(elo, wins, losses, ties):
    table = Table(title="OCR ELO Leaderboard (vLLM Judge)", show_lines=True)
    table.add_column("Rank", style="bold", justify="center")
    table.add_column("Model", style="cyan")
    table.add_column("ELO", style="bold green", justify="right")
    table.add_column("W", justify="right")
    table.add_column("L", justify="right")
    table.add_column("T", justify="right")
    table.add_column("Win%", justify="right")

    ranked = sorted(elo.items(), key=lambda x: x[1], reverse=True)
    for rank, (model, rating) in enumerate(ranked, 1):
        total = wins[model] + losses[model] + ties[model]
        win_pct = f"{wins[model] / total * 100:.0f}%" if total > 0 else "N/A"
        table.add_row(
            str(rank),
            model.split("/")[-1] if "/" in model else model,
            f"{rating:.0f}",
            str(wins[model]),
            str(losses[model]),
            str(ties[model]),
            win_pct,
        )

    console.print()
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Offline vLLM judge for OCR benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single judge, 3 samples
  uv run ocr-vllm-judge.py my-bench --from-prs \\
      --judge-model Qwen/Qwen2.5-VL-7B-Instruct --max-samples 3

  # Jury of 2 models
  uv run ocr-vllm-judge.py my-bench --from-prs \\
      --judge-model Qwen/Qwen3-VL-8B-Instruct \\
      --judge-model Qwen/Qwen2.5-VL-7B-Instruct \\
      --max-samples 50

  # Via HF Job
  hf jobs uv run --timeout 1h \\
      ocr-vllm-judge.py my-bench --from-prs \\
      --judge-model Qwen/Qwen3-VL-8B-Instruct
        """,
    )
    parser.add_argument(
        "dataset", help="HF dataset with OCR outputs from multiple models"
    )
    parser.add_argument(
        "--judge-model",
        action="append",
        dest="judge_models",
        default=None,
        help="Judge model ID (repeatable for jury mode)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--columns", nargs="+", default=None, help="Specific OCR columns to compare"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Load these configs from the dataset repo",
    )
    parser.add_argument(
        "--from-prs", action="store_true", help="Auto-discover configs from open PRs"
    )
    parser.add_argument(
        "--merge-prs", action="store_true", help="Merge open PRs before loading"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM context length (default: 4096)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory fraction (default: 0.85)",
    )
    parser.add_argument(
        "--save-results",
        default=None,
        help="Push judge results to this HF dataset repo (e.g. davanstrien/ocr-bench-rubenstein-judge)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create the --save-results repo as private"
    )
    args = parser.parse_args()

    # --- CUDA check ---
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")

    # --- Judge models ---
    judge_models = args.judge_models or ["Qwen/Qwen2.5-VL-7B-Instruct"]

    console.print(
        f"\n[bold]Judge panel ({len(judge_models)} model{'s' if len(judge_models) > 1 else ''}):[/bold]"
    )
    for m in judge_models:
        console.print(f"  - {m}")

    # --- Load data ---
    ds, ocr_columns = load_benchmark_dataset(args)

    console.print("\n[bold]OCR columns found:[/bold]")
    for col, model in ocr_columns.items():
        console.print(f"  {col} -> {model}")

    model_names = list(ocr_columns.values())

    # --- Build comparisons (CPU, no GPU needed) ---
    console.print("\n[bold]Building comparison prompts...[/bold]")
    comparisons = build_comparisons(ds, ocr_columns, args.max_samples, args.seed)

    pairs = list(combinations(range(len(model_names)), 2))
    console.print(
        f"  Models: {len(model_names)}, Pairs per sample: {len(pairs)}, "
        f"Total comparisons: {len(comparisons)}"
    )

    if not comparisons:
        console.print("[red]No valid comparisons to evaluate.[/red]")
        sys.exit(1)

    # --- Run judges ---
    all_judge_results = []
    judge_short_names = []

    for model_id in judge_models:
        short_name = model_id.split("/")[-1] if "/" in model_id else model_id
        judge_short_names.append(short_name)

        results = run_judge(
            model_id,
            comparisons,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        all_judge_results.append(results)

    # --- Aggregate votes ---
    if len(judge_models) > 1:
        console.print(
            f"\n[bold]Aggregating jury votes ({len(judge_models)} judges)...[/bold]"
        )
        final_results = aggregate_jury_votes(
            all_judge_results, comparisons, judge_short_names
        )
    else:
        final_results = all_judge_results[0]

    # --- Compute ELO ---
    elo, wins, losses, ties, comparison_log = compute_elo(
        comparisons, final_results, model_names
    )
    print_elo_leaderboard(elo, wins, losses, ties)

    # --- Save results ---
    if args.save_results:
        import datetime

        console.print(f"\n[bold]Saving results to:[/bold] {args.save_results}")

        # Comparisons config
        comp_ds = Dataset.from_list(comparison_log)
        comp_ds.push_to_hub(args.save_results, config_name="comparisons", private=args.private)
        console.print(f"  Pushed [cyan]comparisons[/cyan] ({len(comparison_log)} rows)")

        # Leaderboard config
        ranked = sorted(elo.items(), key=lambda x: x[1], reverse=True)
        leaderboard_rows = []
        for model, rating in ranked:
            total = wins[model] + losses[model] + ties[model]
            leaderboard_rows.append(
                {
                    "model": model,
                    "elo": round(rating),
                    "wins": wins[model],
                    "losses": losses[model],
                    "ties": ties[model],
                    "win_pct": round(wins[model] / total * 100) if total > 0 else 0,
                }
            )
        Dataset.from_list(leaderboard_rows).push_to_hub(
            args.save_results, config_name="leaderboard", private=args.private
        )
        console.print(
            f"  Pushed [cyan]leaderboard[/cyan] ({len(leaderboard_rows)} rows)"
        )

        # Metadata config
        metadata_row = {
            "source_dataset": args.dataset,
            "judge_models": json.dumps(judge_models),
            "seed": args.seed,
            "max_samples": args.max_samples or len(ds),
            "total_comparisons": len(comparisons),
            "valid_comparisons": len(comparison_log),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "from_prs": args.from_prs,
        }
        Dataset.from_list([metadata_row]).push_to_hub(
            args.save_results, config_name="metadata", private=args.private
        )
        console.print("  Pushed [cyan]metadata[/cyan]")
        console.print(f"  [green]Results saved to: {args.save_results}[/green]")

    # --- Sample comparisons ---
    console.print("\n[bold]Sample comparisons:[/bold]")
    for entry in comparison_log[:5]:
        winner_model = (
            entry["model_a"]
            if entry["winner"] == "A"
            else (entry["model_b"] if entry["winner"] == "B" else "tie")
        )
        agreement = entry.get("agreement", "1/1")
        console.print(
            f"  Sample {entry['sample_idx']}: {entry['winner']} wins "
            f"({winner_model.split('/')[-1]}) [{agreement}] — {entry['reason'][:80]}"
        )

    console.print("\n[bold green]Evaluation complete![/bold green]")
    console.print(
        f"  Judge{'s' if len(judge_models) > 1 else ''}: {', '.join(judge_short_names)}"
    )
    console.print(f"  Comparisons evaluated: {len(comparison_log)}/{len(comparisons)}")


if __name__ == "__main__":
    main()
