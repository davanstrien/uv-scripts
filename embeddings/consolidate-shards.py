# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface-hub>=1.12",
#     "pyarrow",
# ]
# ///
"""
Assemble the parquet shards of a fan-out embedding run (see launch-embedding-fleet.py)
into the final Hub dataset, one file at a time, with a provenance card.

Pass-through design: worker shards are ALREADY valid parquet in final order, so nothing
is loaded or re-serialized — each file is downloaded, its footer row count read, and the
file uploaded to the dataset repo as data/train-XXXXX-of-YYYYY.parquet. Peak disk = one
shard, so any total run size fits on a small CPU flavor. Deliberately the ONLY step in
the fleet that writes to a dataset repo — workers write bucket objects, so N-way commit
contention (412s) can't happen by construction.

Accepts both worker output namings: <rank>.parquet (row mode) and <rank>.partNNNN.parquet
(streaming mode). Re-running is idempotent for same-named files; if a re-run has FEWER
files than a previous consolidation of the same repo, stale data/ files from the earlier
attempt are removed first.

Usually spawned as a cpu Job by the launcher; can also run locally:
    uv run consolidate-shards.py --bucket you/embedding-runs --run-id 20260709-1200-abc123

To re-run it on Jobs by hand (the launcher's defaults):
    hf jobs uv run --flavor cpu-upgrade --timeout 2h -s HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/embeddings/raw/main/consolidate-shards.py \\
        --bucket you/embedding-runs --run-id 20260709-1200-abc123
"""
import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("consolidate-shards")


ROW_GROUP_ROWS = 25_000  # ~100MB groups with an embeddings column; viewer needs random access


def normalize_embeddings_column(local, out_col):
    """Rewrite the file in place if its embeddings column isn't fixed_size_list<float32>
    OR its row groups are too big for the dataset viewer.

    Shards written by different script versions can disagree (old: list<double>, new:
    fixed_size_list<float32>) — mixed schemas break load_dataset. And pyarrow-default
    ~1M-row groups are multi-GB with embeddings, which the viewer rejects with
    "Scan size limit exceeded". Returns the file's row count."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(local)
    meta = pf.metadata
    schema = pf.schema_arrow
    if out_col not in schema.names:
        return meta.num_rows  # nothing to normalize (unexpected, but not fatal here)
    field = schema.field(out_col)
    schema_ok = pa.types.is_fixed_size_list(field.type) and field.type.value_type == pa.float32()
    groups_ok = meta.num_row_groups > 0 and all(
        meta.row_group(i).num_rows <= 4 * ROW_GROUP_ROWS for i in range(meta.num_row_groups))
    if schema_ok and groups_ok:
        return meta.num_rows
    t = pq.read_table(local)
    if not schema_ok:
        col = t[out_col].combine_chunks()
        dim = len(col[0].as_py())
        values = pc.cast(pc.list_flatten(col), pa.float32())
        fixed = pa.FixedSizeListArray.from_arrays(values, dim)
        idx = t.schema.get_field_index(out_col)
        t = t.set_column(idx, pa.field(out_col, fixed.type), fixed)
    logger.info(f"  normalized {Path(local).name}: schema_ok={schema_ok} groups_ok={groups_ok} "
                f"→ {t.schema.field(out_col).type}, {ROW_GROUP_ROWS}-row groups + page index")
    pq.write_table(t, local, row_group_size=ROW_GROUP_ROWS, write_page_index=True)
    return t.num_rows


def upload_with_retry(api, local, repo_id, path_in_repo, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.warning("Disabling XET (fallback to HTTP upload)")
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            api.upload_file(path_or_fileobj=local, path_in_repo=path_in_repo,
                            repo_id=repo_id, repo_type="dataset")
            return
        except Exception as e:
            logger.error(f"Upload attempt {attempt}/{max_retries} for {path_in_repo} failed: {e}")
            if attempt < max_retries:
                delay = 30 * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error("All upload attempts failed. Shards remain in the bucket — re-run consolidation.")
                sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", required=True, help="Run bucket, e.g. you/embedding-runs")
    p.add_argument("--run-id", required=True, help="Fleet run id; shards are read from runs/<run-id>/ in --bucket")
    p.add_argument("--private", action="store_true",
                   help="Make the output dataset private (also set when the run's manifest says private)")
    args = p.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import (DatasetCard, HfApi, download_bucket_files,
                                 list_bucket_tree, login)

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    api = HfApi()

    prefix = f"runs/{args.run_id}"
    workdir = Path(tempfile.mkdtemp(prefix=f"consolidate-{args.run_id}-"))

    download_bucket_files(args.bucket, [(f"{prefix}/run.json", workdir / "run.json")],
                          raise_on_missing_files=True)
    run = json.loads((workdir / "run.json").read_text())
    n = run["num_shards"]
    out_repo = run["output_dataset"]

    # Collect shard files: <rank>.parquet (row mode) or <rank>.partNNNN.parquet (streaming).
    pat = re.compile(r"^(\d{5})(?:\.part(\d{4}))?\.parquet$")
    shard_files = []  # (rank, part, bucket_path)
    for f in list_bucket_tree(args.bucket, prefix=f"{prefix}/data/", recursive=True):
        name = Path(getattr(f, "path", "")).name
        m = pat.match(name)
        if m:
            shard_files.append((int(m.group(1)), int(m.group(2) or 0), f.path))
    shard_files.sort()
    ranks_present = {r for r, _, _ in shard_files}
    missing = sorted(set(range(n)) - ranks_present)
    if missing:
        logger.error(f"Missing output from {len(missing)}/{n} rank(s): {missing}")
        logger.error("Re-run those ranks (launch-embedding-fleet.py --retry-rank), then consolidate again.")
        sys.exit(1)
    total_files = len(shard_files)
    logger.info(f"{total_files} shard file(s) across {n} rank(s)")

    private = args.private or run.get("private", False)
    api.create_repo(out_repo, repo_type="dataset", private=private, exist_ok=True)

    # Remove stale data/ files from any earlier consolidation with a different file count.
    expected = {f"data/train-{i:05d}-of-{total_files:05d}.parquet" for i in range(total_files)}
    try:
        existing = [f for f in api.list_repo_files(out_repo, repo_type="dataset")
                    if f.startswith("data/") and f.endswith(".parquet") and f not in expected]
        for stale in existing:
            logger.warning(f"Deleting stale {stale} from a previous consolidation")
            api.delete_file(stale, repo_id=out_repo, repo_type="dataset")
    except Exception as e:
        logger.warning(f"stale-file check skipped: {e}")

    # Embedding column name: default unless overridden via the run's embed_args.
    ea = run.get("embed_args") or []
    out_col = ea[ea.index("--output-column") + 1] if "--output-column" in ea else "embeddings"

    # Pass each file through: download → normalize schema if needed → upload → delete local.
    rows_total = 0
    t0 = time.time()
    for seq, (rank, part, bucket_path) in enumerate(shard_files):
        local = workdir / Path(bucket_path).name
        download_bucket_files(args.bucket, [(bucket_path, local)], raise_on_missing_files=True)
        rows = normalize_embeddings_column(local, out_col)
        rows_total += rows
        dest = f"data/train-{seq:05d}-of-{total_files:05d}.parquet"
        logger.info(f"[{seq + 1}/{total_files}] rank {rank} part {part}: {rows:,} rows → {dest}")
        upload_with_retry(api, local, out_repo, dest)
        local.unlink()

    logger.info(f"Assembled {rows_total:,} rows in {time.time() - t0:.0f}s "
                f"(manifest says {run.get('rows_total')})")
    if run.get("rows_total") and rows_total != run["rows_total"]:
        logger.warning("Row count differs from manifest — check for a re-run with different settings.")

    # Timing/throughput line from final worker statuses (best effort).
    stats_line = ""
    try:
        status_files = [(f"{prefix}/status/{i:05d}.json", workdir / f"status-{i:05d}.json") for i in range(n)]
        download_bucket_files(args.bucket, status_files)
        statuses = [json.loads(dst.read_text()) for _, dst in status_files if dst.exists()]
        if statuses:
            wall = max(s["updated_at"] for s in statuses) - min(s["started_at"] for s in statuses)
            rps = sum(s.get("rows_per_sec") or 0 for s in statuses)
            stats_line = f"- Fleet wall-clock: ~{wall / 60:.1f} min · aggregate ~{rps:,.0f} rows/s\n"
    except Exception as e:
        logger.info(f"status stats skipped: {e}")

    script_url = "https://huggingface.co/datasets/uv-scripts/embeddings/raw/main/generate-embeddings.py"
    launcher_url = "https://huggingface.co/datasets/uv-scripts/embeddings/raw/main/launch-embedding-fleet.py"
    on_jobs = os.environ.get("HF_JOB_ID") is not None
    origin = ("Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
              if on_jobs else "Generated")
    jobs_tag = "\n- hf_jobs" if on_jobs else ""
    mode = "streaming file-shards" if run.get("streaming") else "row shards"
    rev_line = f"- Input revision: `{run['revision']}`\n" if run.get("revision") else ""
    job_list = "".join(f"  - shard {i}: `{jid}`\n" for i, jid in enumerate(run.get("job_ids", [])))
    card = DatasetCard(
        f"---\ntags:\n- embeddings\n- uv-script\n- generated{jobs_tag}\n---\n\n"
        f"# {out_repo}\n\n"
        f"Embeddings of [`{run['input_dataset']}`](https://huggingface.co/datasets/{run['input_dataset']}) "
        f"column `{run['column']}`, computed by a fleet of {n} parallel Jobs ({mode}).\n\n"
        f"- Model: [`{run['model']}`](https://huggingface.co/{run['model']})\n"
        f"- Rows: {rows_total:,}\n"
        f"- Fleet: {n} × `{run['flavor']}` · run `{run['run_id']}`\n"
        f"{rev_line}{stats_line}"
        f"- Worker jobs:\n{job_list}\n"
        f"## Reproduction\n\n"
        f"{origin} with the [`generate-embeddings.py`]({script_url}) recipe from "
        f"[uv-scripts](https://huggingface.co/uv-scripts), fanned out with "
        f"[`launch-embedding-fleet.py`]({launcher_url}):\n\n"
        f"```bash\nuv run {launcher_url} \\\n"
        f"    {run['input_dataset']} <output-dataset> --column {run['column']} "
        f"--model {run['model']} --num-shards {n} --flavor {run['flavor']}\n```\n"
    )
    try:
        card.push_to_hub(out_repo, repo_type="dataset")
    except Exception as e:
        logger.warning(f"card push skipped: {e}")
    logger.info(f"✅ https://huggingface.co/datasets/{out_repo}")


if __name__ == "__main__":
    main()
