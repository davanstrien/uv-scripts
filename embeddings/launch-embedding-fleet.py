# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "huggingface-hub>=1.12",
# ]
# ///
"""
Fan one embedding run out across N Hugging Face Jobs, then consolidate.

Runs generate-embeddings.py once per shard (each worker gets RANK / NUM_SHARDS /
RUN_ID / OUTPUT_BUCKET via env), workers write parquet shards + progress heartbeats
to the run bucket (object PUTs — no repo-commit contention), and when all workers
finish a consolidation Job merges the shards into the final Hub dataset with one
commit. Runs locally on your laptop; only the workers/consolidator run on Jobs.

Examples:
    # Smoke: 2 small-GPU jobs over a 20k-row slice
    uv run launch-embedding-fleet.py stanfordnlp/imdb your-name/imdb-emb \\
        --max-samples 20000 --num-shards 2 --flavor t4-small --timeout 20m

    # Mid-size: 8 L4s over a few million rows
    uv run launch-embedding-fleet.py your-name/corpus your-name/corpus-emb \\
        --num-shards 8 --flavor l4x1 --timeout 1h

    # Re-run one failed shard, then consolidate an existing run
    uv run launch-embedding-fleet.py ... --run-id 20260709-1200-abc123 --retry-rank 3
    uv run launch-embedding-fleet.py ... --run-id 20260709-1200-abc123 --consolidate-only

Resume model: every worker writes only runs/<run-id>/data/<rank>.parquet and its own
status file, so re-running a rank is idempotent. The launcher exiting early never
orphans a run — --retry-rank / --consolidate-only pick it back up.
"""
import argparse
import json
import logging
import os
import secrets as pysecrets
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("launch-embedding-fleet")

SCRIPT_BASE = "https://huggingface.co/datasets/uv-scripts/embeddings/raw/main"


def put_json(bucket, path, obj, token=None):
    from huggingface_hub import batch_bucket_files
    batch_bucket_files(bucket, add=[(json.dumps(obj, indent=2).encode(), path)], token=token)


def rows_in_split(input_dataset, config, split):
    """Row count WITHOUT downloading: builder metadata, else the dataset-viewer size API."""
    try:
        from datasets import load_dataset_builder
        b = load_dataset_builder(input_dataset, config) if config else load_dataset_builder(input_dataset)
        n = b.info.splits[split].num_examples
        if n:
            return n
    except Exception as e:
        logger.info(f"builder metadata unavailable ({e}); trying dataset-viewer size API")
    try:
        import huggingface_hub
        r = huggingface_hub.get_session().get(
            "https://datasets-server.huggingface.co/size", params={"dataset": input_dataset}
        )
        r.raise_for_status()
        for s in r.json()["size"]["splits"]:
            if s["split"] == split and (config is None or s["config"] == config):
                return s["num_rows"]
    except Exception as e:
        logger.info(f"size API unavailable ({e})")
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dataset", help="Input dataset repo id")
    p.add_argument("output_dataset", help="Output dataset repo id (written by the consolidation Job; with --run-id the run's manifest value is used)")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model each worker loads (default: sentence-transformers/all-MiniLM-L6-v2)")
    p.add_argument("--column", default="text", help="Column to embed (default: text)")
    p.add_argument("--split", default="train", help="Dataset split to read (default: train)")
    p.add_argument("--config", default=None, help="Dataset config name")
    p.add_argument("--max-samples", type=int, default=None, help="Embed only the first N rows (for testing; not with --streaming)")
    p.add_argument("--num-shards", type=int, default=8, help="Number of worker Jobs; each embeds one shard (default: 8)")
    p.add_argument("--flavor", default="l4x1", help="Hardware flavor for each worker Job (default: l4x1)")
    p.add_argument("--timeout", default="1h", help="Per-worker timeout — also the hard cost ceiling")
    p.add_argument("--bucket", default=None,
                   help="Run bucket (default: <namespace>/embedding-runs)")
    p.add_argument("--script", default=f"{SCRIPT_BASE}/generate-embeddings.py",
                   help="Worker script: raw URL (default) or a local .py path for dev")
    p.add_argument("--consolidate-script", default=f"{SCRIPT_BASE}/consolidate-shards.py",
                   help="Consolidator script: raw URL (default) or local path for dev")
    p.add_argument("--consolidate-flavor", default="cpu-upgrade",
                   help="Consolidation job flavor. cpu-upgrade has 50 GB disk — use cpu-xl "
                        "(1 TB) when total shard size approaches ~20 GB.")
    p.add_argument("--rows-total", type=int, default=None,
                   help="Override when the dataset has no split metadata")
    p.add_argument("--streaming", action="store_true",
                   help="Workers stream + shard at the FILE level (no full-split download per "
                        "rank) — for very big datasets. Text only; incompatible with --max-samples.")
    p.add_argument("--private", action="store_true", help="Final output dataset is private")
    p.add_argument("--embed-args", nargs=argparse.REMAINDER, default=[],
                   help="Everything after --embed-args is passed through to generate-embeddings.py "
                        "verbatim — put it LAST (any launcher flags after it are swallowed too).")
    p.add_argument("--run-id", default=None, help="Attach to an existing run (with --resume/--retry-rank/--consolidate-only)")
    p.add_argument("--resume", action="store_true",
                   help="Converge an existing --run-id: find ranks without a 'done' status, re-run "
                        "exactly those, then consolidate. Idempotent — safe to run repeatedly.")
    p.add_argument("--retry-rank", type=int, default=None, help="Re-spawn a single failed shard of --run-id")
    p.add_argument("--consolidate-only", action="store_true", help="Just run consolidation for --run-id")
    p.add_argument("--no-wait", action="store_true",
                   help="Spawn workers and exit (run --consolidate-only later)")
    args = p.parse_args()

    from huggingface_hub import (JobStage, create_bucket, download_bucket_files, get_token,
                                 run_uv_job, wait_for_job, whoami)

    # Workers need a real token as a secret (bucket writes + output push); env may be empty
    # when the user authenticated via `hf auth login` (keyring/hub cache).
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        p.error("No HF token found — set HF_TOKEN or run `hf auth login`.")
    namespace = whoami(token=token)["name"]
    bucket = args.bucket or f"{namespace}/embedding-runs"

    if (args.retry_rank is not None or args.consolidate_only or args.resume) and not args.run_id:
        p.error("--resume / --retry-rank / --consolidate-only need --run-id")
    if args.num_shards < 1:
        p.error(f"--num-shards must be >= 1 (got {args.num_shards})")
    if args.streaming and args.max_samples:
        p.error("--streaming is incompatible with --max-samples (use row mode for capped tests)")

    def read_manifest(run_id):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "run.json"
            download_bucket_files(bucket, [(f"runs/{run_id}/run.json", dst)],
                                  raise_on_missing_files=True, token=token)
            return json.loads(dst.read_text())

    # Workers are always spawned from the run manifest, never from current CLI flags —
    # a --retry-rank months later must reproduce the original slice/model/args exactly.
    def spawn_worker(rank, manifest):
        script_args = [manifest["input_dataset"], manifest["output_dataset"],
                       "--model", manifest["model"], "--column", manifest["column"],
                       "--split", manifest["split"]]
        if manifest.get("config"):
            script_args += ["--config", manifest["config"]]
        if manifest.get("max_samples"):
            script_args += ["--max-samples", str(manifest["max_samples"])]
        script_args += list(manifest.get("embed_args") or [])
        env = {
            "RANK": str(rank),
            "NUM_SHARDS": str(manifest["num_shards"]),
            "RUN_ID": manifest["run_id"],
            "OUTPUT_BUCKET": bucket,
        }
        if manifest.get("revision"):
            env["REVISION"] = manifest["revision"]
        if manifest.get("streaming"):
            env["STREAMING"] = "1"
        job = run_uv_job(
            args.script,
            script_args=script_args,
            flavor=manifest["flavor"],
            timeout=manifest["timeout"],
            env=env,
            secrets={"HF_TOKEN": token},
            labels={"embedding-fleet-run": manifest["run_id"], "rank": str(rank)},
            token=token,
        )
        logger.info(f"  rank {rank} → job {job.id} ({manifest['flavor']})")
        return job

    def spawn_consolidator(run_id):
        job = run_uv_job(
            args.consolidate_script,
            script_args=[
                "--bucket", bucket, "--run-id", run_id,
            ] + (["--private"] if args.private else []),
            flavor=args.consolidate_flavor,
            timeout="2h",
            secrets={"HF_TOKEN": token},
            labels={"embedding-fleet-run": run_id, "role": "consolidate"},
            token=token,
        )
        logger.info(f"Consolidation job {job.id} ({args.consolidate_flavor}) — "
                    f"merges shards → {args.output_dataset}")
        return job

    def execute_workers(ranks, manifest):
        """Spawn the given ranks, wait, auto-retry failures ONCE, return still-failed ranks."""
        for attempt in (1, 2):
            jobs = {rank: spawn_worker(rank, manifest) for rank in ranks}
            infos = wait_for_job([j.id for j in jobs.values()], token=token)
            ranks = [rank for (rank, job), info in zip(jobs.items(), infos)
                     if info.status.stage != JobStage.COMPLETED]
            if not ranks:
                return []
            if attempt == 1:
                logger.warning(f"{len(ranks)} worker(s) failed; auto-retrying once: {ranks}")
        return ranks

    def done_ranks(run_id, n):
        """Ranks whose bucket status reports state == 'done' (works for both shard modes)."""
        import tempfile
        from pathlib import Path
        done = set()
        with tempfile.TemporaryDirectory() as td:
            pairs = [(f"runs/{run_id}/status/{i:05d}.json", Path(td) / f"{i}.json") for i in range(n)]
            download_bucket_files(bucket, pairs, token=token)
            for i, (_, dst) in enumerate(pairs):
                if dst.exists() and json.loads(dst.read_text()).get("state") == "done":
                    done.add(i)
        return done

    # --- attach-to-existing-run paths ---
    if args.resume:
        manifest = read_manifest(args.run_id)
        n = manifest["num_shards"]
        # Don't double-spawn ranks that are still running — wait for them, then diff.
        from huggingface_hub import list_jobs
        try:
            in_flight = [j for j in list_jobs(labels={"embedding-fleet-run": args.run_id},
                                              namespace=namespace, token=token)
                         if j.status.stage in (JobStage.RUNNING, JobStage.SCHEDULING)
                         and (j.labels or {}).get("rank") is not None]
        except Exception as e:
            logger.warning(f"in-flight check skipped ({e})")
            in_flight = []
        if in_flight:
            ranks = sorted({j.labels["rank"] for j in in_flight})
            logger.info(f"{len(in_flight)} worker(s) still in flight (ranks {ranks}) — waiting before resuming.")
            wait_for_job([j.id for j in in_flight], token=token)
        todo = sorted(set(range(n)) - done_ranks(args.run_id, n))
        if todo:
            logger.info(f"Resume {args.run_id}: {n - len(todo)}/{n} shards done; re-running {todo}")
            still_failed = execute_workers(todo, manifest)
            if still_failed:
                logger.error(f"Ranks still failing after retry: {still_failed} — investigate, then --resume again.")
                sys.exit(1)
        else:
            logger.info(f"Resume {args.run_id}: all {n} shards already done — consolidating.")
        job = spawn_consolidator(args.run_id)
        info = wait_for_job(job.id, token=token)
        if info.status.stage != JobStage.COMPLETED:
            logger.error(f"Consolidation failed (job {job.id}); run --resume again.")
            sys.exit(1)
        logger.info(f"✅ https://huggingface.co/datasets/{manifest['output_dataset']}")
        return

    if args.retry_rank is not None:
        manifest = read_manifest(args.run_id)
        if not 0 <= args.retry_rank < manifest["num_shards"]:
            p.error(f"--retry-rank must be in [0, {manifest['num_shards']}) for run {args.run_id}")
        logger.info(f"Re-spawning rank {args.retry_rank} of run {args.run_id} (config from manifest)")
        job = spawn_worker(args.retry_rank, manifest)
        info = wait_for_job(job.id, token=token)
        if info.status.stage != JobStage.COMPLETED:
            logger.error(f"Retry of rank {args.retry_rank} did not complete "
                         f"(stage={info.status.stage}, job {job.id}).")
            sys.exit(1)
        logger.info("Retry completed; run --consolidate-only when all shards are done.")
        return

    if args.consolidate_only:
        job = spawn_consolidator(args.run_id)
        info = wait_for_job(job.id, token=token)
        sys.exit(0 if info.status.stage == JobStage.COMPLETED else 1)

    # --- fresh run ---
    rows_total = args.rows_total or rows_in_split(args.input_dataset, args.config, args.split)
    if rows_total is None and not args.streaming:
        p.error("Couldn't determine the split's row count — pass --rows-total.")
    if rows_total is not None:
        if args.max_samples:
            rows_total = min(rows_total, args.max_samples)
        if not args.streaming and args.num_shards > rows_total:
            p.error(f"--num-shards {args.num_shards} exceeds the row count ({rows_total}) — "
                    f"some shards would be empty.")

    # Pin the input snapshot: every rank (and any later --retry-rank) must slice the IDENTICAL
    # revision, or a mid-run commit to the input dataset silently breaks the exact partition.
    from huggingface_hub import dataset_info
    revision = dataset_info(args.input_dataset, token=token).sha
    logger.info(f"Pinned input revision: {revision[:12]}")

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + pysecrets.token_hex(3)
    create_bucket(bucket, private=True, exist_ok=True, token=token)

    manifest = {
        "run_id": run_id,
        "input_dataset": args.input_dataset,
        "output_dataset": args.output_dataset,
        "model": args.model,
        "column": args.column,
        "split": args.split,
        "config": args.config,
        "max_samples": args.max_samples,
        "num_shards": args.num_shards,
        "rows_total": rows_total,
        "flavor": args.flavor,
        "timeout": args.timeout,
        "private": args.private,
        "embed_args": list(args.embed_args),
        "streaming": args.streaming,
        "revision": revision,
        "started_at": time.time(),
        "job_ids": [],
    }
    put_json(bucket, f"runs/{run_id}/run.json", manifest, token=token)
    if rows_total is not None:
        logger.info(f"Run {run_id}: {rows_total:,} rows → {args.num_shards} shards "
                    f"(~{rows_total // args.num_shards:,} rows each) on {args.flavor}")
    else:
        logger.info(f"Run {run_id}: streaming file-shards × {args.num_shards} on {args.flavor} "
                    f"(row count unknown upfront)")

    jobs = [spawn_worker(rank, manifest) for rank in range(args.num_shards)]
    manifest["job_ids"] = [j.id for j in jobs]
    put_json(bucket, f"runs/{run_id}/run.json", manifest, token=token)

    logger.info(f"Manifest: hf://buckets/{bucket}/runs/{run_id}/run.json")
    logger.info(f"Dashboard: https://huggingface.co/spaces/davanstrien/embedding-fleet-dashboard?run={run_id}")

    if args.no_wait:
        logger.info(f"--no-wait: consolidate later with --run-id {run_id} --consolidate-only")
        return

    logger.info("Waiting for workers…")
    infos = wait_for_job([j.id for j in jobs], token=token)
    failed = [rank for rank, (j, i) in enumerate(zip(jobs, infos))
              if i.status.stage != JobStage.COMPLETED]
    if failed:
        logger.warning(f"{len(failed)} worker(s) did not complete; auto-retrying: {failed}")
        still_failed = execute_workers(failed, manifest)
        if still_failed:
            logger.error(f"Ranks still failing after retries: {still_failed}")
            logger.error(f"Investigate (job logs / heartbeat age), then converge with: "
                         f"--run-id {run_id} --resume")
            sys.exit(1)

    job = spawn_consolidator(run_id)
    info = wait_for_job(job.id, token=token)
    if info.status.stage != JobStage.COMPLETED:
        logger.error(f"Consolidation failed (job {job.id}); retry with --consolidate-only --run-id {run_id}")
        sys.exit(1)
    logger.info(f"✅ https://huggingface.co/datasets/{args.output_dataset}")


if __name__ == "__main__":
    main()
