# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "sentence-transformers>=5.0.0",
#     "torch",
#     "numpy",
#     "pillow",
#     "einops",
#     "huggingface-hub>=1.12",
# ]
# ///
"""
Generate embeddings for a Hugging Face dataset (text OR images) with sentence-transformers,
and push the result back to the Hub as a new dataset with an `embeddings` column.

This is the simple, ergonomic default. It runs as one command on the bare uv image, on CPU
or any GPU flavor. For maximum throughput on large *decoder* embedding models (e.g.
Qwen3-Embedding), see the vLLM variant; to get a searchable vector index as a Hub dataset,
see the Lance variant.

PROMPTS (retrieval correctness — read this):
    Many embedding models need a DIFFERENT prefix/instruction for documents vs queries, and
    getting it wrong silently degrades retrieval. This script embeds a *document corpus* by
    default, via sentence-transformers' native encode_document()/encode_query() (which also
    route Router models by task), picking the right document convention for you:
      1. the model's REGISTERED prompt if it ships one (e.g. Qwen3-Embedding) — selected
         natively by encode_document/encode_query, else
      2. a small built-in table of well-known families (e5, nomic, bge), else
      3. no prefix.
    Heads-up: current sentence-transformers injects a placeholder prompts dict
    {"query": "", "document": ""} even for models that register NOTHING — so e5 ("passage: "),
    nomic ("search_document: ") etc. look prompt-less via `.prompts`; their real prefixes live
    only in the model card. The built-in table handles that. Override with --prompt '<prefix>'
    or --prompt-name <registered-name>; embed a query set with --query-mode; force no prefix
    with --prompt ''. The chosen prompt is logged and recorded in the dataset card.

Benchmarks (20k rows, seq-cap 512): all-MiniLM-L6-v2 ~900 rows/s on an L4 (~$0.24/1M rows);
bge-base-en-v1.5 ~120 rows/s. L4 is the cheapest flavor for these encoder models.

Examples:
    # Text (default). Document convention auto-picked.
    hf jobs uv run --flavor l4x1 -s HF_TOKEN generate-embeddings.py \\
        stanfordnlp/imdb  your-name/imdb-embeddings \\
        --column text --model sentence-transformers/all-MiniLM-L6-v2

    # e5: docs auto-get "passage: ". (--prompt 'passage: ' would be the explicit form.)
    hf jobs uv run --flavor l4x1 -s HF_TOKEN generate-embeddings.py \\
        stanfordnlp/imdb  your-name/imdb-e5 --model intfloat/multilingual-e5-large

    # Images (CLIP) — prompts don't apply.
    hf jobs uv run --flavor l4x1 -s HF_TOKEN generate-embeddings.py \\
        your-name/photos  your-name/photos-embeddings \\
        --modality image --column image --model clip-ViT-B-32

    # Test on a small slice first, keep the output private
    hf jobs uv run --flavor l4x1 -s HF_TOKEN generate-embeddings.py \\
        stanfordnlp/imdb  your-name/imdb-emb --max-samples 100 --private

FAN-OUT (multi-job) MODE:
    Split one embedding run across N Jobs with --num-shards/--shard-index (or the RANK /
    NUM_SHARDS / RUN_ID / OUTPUT_BUCKET env vars, so a launcher only varies RANK per job —
    see launch-embedding-fleet.py). Each shard takes an exact, non-overlapping contiguous
    slice, writes its rows to the run's bucket as runs/<run-id>/data/<rank>.parquet (object
    PUTs — no repo-commit contention), and heartbeats progress to
    runs/<run-id>/status/<rank>.json. consolidate-shards.py merges shards into the final
    dataset with one commit. Re-running a failed rank overwrites only its own files (resume).
    Note: --max-samples applies BEFORE sharding, so a capped run still partitions exactly.
    Each rank still downloads the full split before slicing — fine at few-million-row scale;
    for larger corpora shard at the file level or stream instead.
"""
import argparse
import logging
import os
import re
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate-embeddings")


def find_batch_size(model, sample, normalize, candidates=(32, 64, 128, 256)):
    """Probe for the fastest batch that fits (used by --batch-size auto). Throughput is NOT
    monotonic in batch size, so we time a few on a warmup sample and keep the fastest that doesn't
    OOM. Why bigger isn't better: for text, larger batches pad to the longest member + add overhead;
    for images, the ViT forward already saturates the GPU by ~batch 32. Works for text and images."""
    import time
    import torch
    warm = sample[: min(1024, len(sample))]
    try:  # one untimed warmup so cudnn autotune doesn't penalise the first probe
        model.encode(warm[:32], batch_size=32, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=normalize)
    except Exception:
        pass
    best_bs, best_rps = candidates[0], 0.0
    for bs in candidates:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            t = time.perf_counter()
            model.encode(warm, batch_size=bs, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=normalize)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rps = len(warm) / (time.perf_counter() - t)
            logger.info(f"  auto-batch probe bs={bs}: {rps:.0f} rows/s")
            if rps > best_rps:
                best_rps, best_bs = rps, bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.info(f"  auto-batch bs={bs} OOM → stopping probe")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            raise
    logger.info(f"auto-batch chose bs={best_bs} ({best_rps:.0f} rows/s on warmup)")
    return best_bs


def known_convention(model_id):
    """Best-effort (query_prefix, doc_prefix) for common families whose convention is
    documented in the model card but NOT registered in config_sentence_transformers.json.
    Returns None if unknown. Overridable with --prompt / --no-auto-prompt.

    Verified 2026-07-03 on HF Jobs: of e5 / nomic / bge-en / bge-m3 / Qwen3-Embedding, only
    Qwen3-Embedding registers real ST prompts; the rest ship none and rely on manual prefixes.
    """
    m = model_id.lower()
    # Instruction-style embedders (e5-*-instruct, gte-Qwen, ...): prefer the model's REGISTERED
    # prompt or an explicit --prompt; don't guess a literal prefix.
    if "instruct" in m:
        return None
    if "nomic-embed-text" in m:
        return ("search_query: ", "search_document: ")
    if "bge-m3" in m:  # bge-m3 uses no prompts
        return ("", "")
    # e5 family (e5-base/large/small, multilingual-e5-*), boundaried so e.g. "table5" or a
    # model with "e5" mid-word can't silently pick up "query:/passage:" prefixes.
    if re.search(r"(^|[/_-])e5([_-]|$)", m):
        return ("query: ", "passage: ")
    if "bge" in m and "-en" in m:  # English bge retrieval: query instruction, docs raw
        return ("Represent this sentence for searching relevant passages: ", "")
    return None


def resolve_prompt(model, model_id, is_query, args):
    """Decide the EXPLICIT prefix to pass to encode_query()/encode_document(), or None to let the
    native method choose. sentence-transformers' encode_query/encode_document already select the
    model's REGISTERED query/document prompt and set the Router task — we lean on that, and only
    supply a prefix ourselves for (a) explicit --prompt/--prompt-name, (b) the known-family table
    covering models that register nothing (e5, nomic, bge-en — their prefixes live only in the
    model card, so the native fallback would silently apply NO prefix)."""
    registered = dict(getattr(model, "prompts", {}) or {})
    # Current sentence-transformers injects a placeholder {"query":"","document":""} for models
    # with no config prompts; only non-empty values are real conventions.
    real = {k: v for k, v in registered.items() if v}
    logger.info(f"Registered prompts: {registered} · real (non-empty): {real or 'none'} · "
                f"default_prompt_name={getattr(model, 'default_prompt_name', None)}")
    side = "query" if is_query else "document"

    if args.prompt is not None:  # includes --prompt '' to force no prefix
        logger.info(f"Prompt: raw --prompt → {args.prompt!r}")
        return args.prompt
    if args.prompt_name:
        if args.prompt_name not in registered:
            logger.error(f"--prompt-name {args.prompt_name!r} not registered ({list(registered)}); "
                         f"use --prompt '<raw prefix>' instead.")
            sys.exit(1)
        logger.info(f"Prompt: registered prompt_name={args.prompt_name!r} → {registered[args.prompt_name]!r}")
        return registered[args.prompt_name]
    native_keys = ("query",) if is_query else ("document", "passage", "corpus")
    if any(real.get(k) for k in native_keys):
        # Model ships a real prompt for this side (e.g. Qwen3 query) → encode_query/encode_document
        # selects it natively (and routes Router models by task).
        logger.info(f"Prompt: model-registered — selected natively by encode_{side}()")
        return None

    kc = known_convention(model_id)
    if kc is not None:
        chosen = kc[0] if is_query else kc[1]
        if args.no_auto_prompt:
            if chosen:
                logger.warning(f"--no-auto-prompt set: NOT applying the known {side} prefix {chosen!r} for "
                               f"{model_id}. Retrieval may degrade unless you pass --prompt.")
            return ""
        logger.info(f"Prompt: known-family {side} prefix → {chosen!r} (override with --prompt)"
                    if chosen else f"Prompt: known-family → no {side} prefix needed")
        return chosen

    logger.info(f"Prompt: none registered or known for {model_id} — encode_{side}() applies no prefix. "
                f"If it's a retrieval model needing a query/document prefix, pass --prompt.")
    return None


def sniff_token_lengths(model, texts, max_seq_len, sample=512):
    """Tokenize a sample to report the token-length distribution + how much --max-seq-len truncates,
    and return the median length (used to pick the auto-batch candidate range: short texts under-use
    the GPU at small batch, long texts waste compute on padding). Text only; returns None on failure."""
    try:
        tok = model.tokenizer
    except Exception:
        return None
    s = texts[: min(sample, len(texts))]
    lens = sorted(len(tok.encode(t, add_special_tokens=True)) for t in s)
    n = len(lens)
    if not n:
        return None
    median, p90, mx = lens[n // 2], lens[min(n - 1, int(n * 0.9))], lens[-1]
    pct_over = 100 * sum(1 for L in lens if L > max_seq_len) / n
    note = (f" → {pct_over:.0f}% exceed --max-seq-len {max_seq_len} and are truncated "
            f"(raise it to keep more, at higher cost/slower)" if pct_over >= 5
            else f" (all within --max-seq-len {max_seq_len})")
    logger.info(f"Token lengths (sample {n}): median {median}, p90 {p90}, max {mx}{note}")
    return median


def put_bucket_files(bucket_id, add, max_retries=3):
    """batch_bucket_files with a short retry — bucket PUTs are cheap object writes but can
    flake transiently; shard data must not be lost to a blip."""
    from huggingface_hub import batch_bucket_files
    for attempt in range(1, max_retries + 1):
        try:
            batch_bucket_files(bucket_id, add=add)
            return True
        except Exception as e:
            if attempt == max_retries:
                raise
            logger.warning(f"bucket PUT attempt {attempt}/{max_retries} failed: {e}; retrying in {5 * attempt}s")
            time.sleep(5 * attempt)


class StatusReporter:
    """Fan-out worker heartbeat: PUTs runs/<run-id>/status/<rank>.json to the run bucket,
    throttled to one write per ~30s (last-writer-wins per key, so N workers never contend).
    A failed status write must never kill a paid embedding run — errors are logged, not raised."""

    def __init__(self, bucket, run_id, rank, rows_total, tokens_per_row=None):
        self.bucket, self.run_id, self.rank = bucket, run_id, rank
        self.rows_total, self.tokens_per_row = rows_total, tokens_per_row
        self.started_at = time.time()
        self.rows_done = 0
        self._last_write = 0.0

    def report(self, rows_done=None, state="running", force=False):
        import json
        if rows_done is not None:
            self.rows_done = rows_done
        now = time.time()
        if not force and now - self._last_write < 30:
            return
        elapsed = max(now - self.started_at, 1e-6)
        payload = {
            "run_id": self.run_id,
            "rank": self.rank,
            "state": state,
            "rows_done": self.rows_done,
            "rows_total": self.rows_total,
            "tokens_done_est": int(self.rows_done * self.tokens_per_row) if self.tokens_per_row else None,
            "rows_per_sec": round(self.rows_done / elapsed, 1),
            "started_at": self.started_at,
            "updated_at": now,
            "job_id": os.environ.get("JOB_ID"),
        }
        dest = f"runs/{self.run_id}/status/{self.rank:05d}.json"
        try:
            put_bucket_files(self.bucket, [(json.dumps(payload).encode(), dest)], max_retries=2)
            self._last_write = now
        except Exception as e:
            logger.warning(f"status write skipped ({e})")


def run_streaming_shard(ds, model, prompt_str, args):
    """Streaming fan-out worker: iterate this rank's FILE-shard, encode in chunks, and flush
    parquet PARTS to the bucket every ~250k rows, so memory and disk stay bounded no matter
    how big the shard is. Output keys: runs/<run-id>/data/<rank>.part<p>.parquet — the
    consolidator accepts both this and row mode's single <rank>.parquet naming."""
    import itertools
    import pyarrow as pa
    import pyarrow.parquet as pq

    encode_fn = model.encode_query if args.query_mode else model.encode_document
    encode_kwargs = {"prompt": prompt_str} if prompt_str is not None else {}

    def clean(t):
        return t if isinstance(t, str) and t.strip() else " "

    # Buffer a head sample for token sniffing + the auto-batch probe, then chain it back.
    it = iter(ds)
    head = list(itertools.islice(it, 1024))
    if not head:
        logger.error(f"File-shard {args.shard_index} yielded no rows.")
        sys.exit(1)
    if args.column not in head[0]:
        logger.error(f"Column {args.column!r} not in rows. Available: {sorted(head[0])}")
        sys.exit(1)
    if args.output_column in head[0]:
        logger.error(f"Output column {args.output_column!r} already exists — choose another --output-column.")
        sys.exit(1)
    head_texts = [clean(r[args.column]) for r in head]
    median_tok = sniff_token_lengths(model, head_texts, args.max_seq_len)
    if str(args.batch_size).lower() == "auto":
        if median_tok is None or median_tok >= 256:
            candidates = (32, 64, 128, 256)
        elif median_tok >= 64:
            candidates = (64, 128, 256, 512)
        else:
            candidates = (128, 256, 512, 1024)
        batch_size = find_batch_size(model, head_texts, args.normalize, candidates=candidates)
    else:
        batch_size = int(args.batch_size)

    reporter = StatusReporter(args.output_bucket, args.run_id, args.shard_index,
                              rows_total=None, tokens_per_row=median_tok)
    reporter.report(0, force=True)

    chunk_rows, part_rows = 25_000, 250_000
    prog = {"part_idx": 0, "rows_done": 0}
    buf_rows, buf_texts, part_buf = [], [], []

    def flush_part():
        if not part_buf:
            return
        path = f"/tmp/part-{args.shard_index:05d}-{prog['part_idx']:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(part_buf), path,
                       row_group_size=25_000, write_page_index=True)
        dest = f"runs/{args.run_id}/data/{args.shard_index:05d}.part{prog['part_idx']:04d}.parquet"
        logger.info(f"Uploading {len(part_buf):,}-row part → {args.output_bucket}/{dest}")
        put_bucket_files(args.output_bucket, [(path, dest)])
        os.remove(path)
        part_buf.clear()
        prog["part_idx"] += 1

    def encode_chunk():
        if not buf_rows:
            return
        emb = encode_fn(buf_texts, batch_size=batch_size, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=args.normalize, **encode_kwargs)
        for r, e in zip(buf_rows, emb):
            r[args.output_column] = e.tolist()
        part_buf.extend(buf_rows)
        prog["rows_done"] += len(buf_rows)
        buf_rows.clear()
        buf_texts.clear()
        reporter.report(prog["rows_done"])
        if len(part_buf) >= part_rows:
            flush_part()

    t0 = time.perf_counter()
    try:
        for row in itertools.chain(head, it):
            buf_rows.append(dict(row))
            buf_texts.append(clean(row[args.column]))
            if len(buf_rows) >= chunk_rows:
                encode_chunk()
        encode_chunk()
        flush_part()
    except Exception:
        reporter.report(state="error", force=True)
        raise
    secs = time.perf_counter() - t0
    logger.info(f"Embedded {prog['rows_done']:,} rows in {secs:.0f}s "
                f"({prog['rows_done'] / max(secs, 1e-6):.0f} rows/s), {prog['part_idx']} part(s)")
    reporter.report(prog["rows_done"], state="done", force=True)
    logger.info(f"✅ streaming shard {args.shard_index + 1}/{args.num_shards} of run {args.run_id} uploaded")


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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dataset", help="Input dataset ID on the Hugging Face Hub")
    p.add_argument("output_dataset", help="Output dataset ID to create on the Hub")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="sentence-transformers model (text or CLIP image model)")
    p.add_argument("--modality", choices=["text", "image"], default="text")
    p.add_argument("--column", default="text", help="Input column (text string, or image)")
    p.add_argument("--output-column", default="embeddings")
    p.add_argument("--config", default=None, help="Dataset config name (e.g. wikipedia needs one)")
    p.add_argument("--split", default="train")
    p.add_argument("--max-samples", type=int, default=None, help="Limit rows (for testing)")
    p.add_argument("--batch-size", default="auto",
                   help="'auto' probes for the fastest batch that fits, or pass an int")
    p.add_argument("--prompt", default=None,
                   help="Raw prefix to prepend to every text (e.g. 'passage: '). Highest precedence. "
                        "Use --prompt '' to force NO prefix.")
    p.add_argument("--prompt-name", default=None,
                   help="Name of a prompt REGISTERED by the model (e.g. 'query'); errors if not registered.")
    p.add_argument("--query-mode", action="store_true",
                   help="Embed inputs as QUERIES, not documents (flips the auto-picked convention).")
    p.add_argument("--no-auto-prompt", action="store_true",
                   help="Disable the built-in known-family prefix table (still honours registered prompts).")
    p.add_argument("--max-seq-len", type=int, default=512,
                   help="Truncate text to this many tokens (predictable cost; RAG-typical)")
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--private", action="store_true", help="Make the output dataset private")
    # Fan-out mode (see docstring). Env-defaulted so a launcher drives workers purely via env.
    # Defaults stay raw strings; int conversion happens after parse so a malformed env var
    # reaches p.error() instead of blowing up argparse construction.
    p.add_argument("--num-shards", default=os.environ.get("NUM_SHARDS"),
                   help="Fan-out: total number of shards (env NUM_SHARDS). Enables shard mode.")
    p.add_argument("--shard-index", default=os.environ.get("RANK"),
                   help="Fan-out: this worker's shard index, 0-based (env RANK).")
    p.add_argument("--output-bucket", default=os.environ.get("OUTPUT_BUCKET"),
                   help="Fan-out: bucket for shard parquets + status (env OUTPUT_BUCKET).")
    p.add_argument("--run-id", default=os.environ.get("RUN_ID"),
                   help="Fan-out: run identifier grouping shards under runs/<run-id>/ (env RUN_ID).")
    p.add_argument("--streaming", action="store_true",
                   default=os.environ.get("STREAMING") == "1",
                   help="Fan-out: stream the dataset and shard at the FILE level (env STREAMING=1). "
                        "Each rank downloads only its own files — use for very big datasets. "
                        "Text modality only; incompatible with --max-samples.")
    p.add_argument("--revision", default=os.environ.get("REVISION"),
                   help="Input dataset revision (commit sha). Pin this in fan-out runs so every "
                        "rank slices the identical snapshot (env REVISION).")
    args = p.parse_args()

    def int_or_error(val, name):
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            p.error(f"{name} must be an integer (got {val!r}).")

    args.num_shards = int_or_error(args.num_shards, "--num-shards / NUM_SHARDS")
    args.shard_index = int_or_error(args.shard_index, "--shard-index / RANK")

    sharded = args.num_shards is not None
    if sharded:
        if args.num_shards < 1:
            p.error(f"--num-shards must be >= 1 (got {args.num_shards}).")
        if args.shard_index is None or not 0 <= args.shard_index < args.num_shards:
            p.error(f"--shard-index must be in [0, {args.num_shards}) when --num-shards is set "
                    f"(got {args.shard_index}).")
        if not (args.output_bucket and args.run_id):
            p.error("fan-out mode needs --output-bucket and --run-id (env OUTPUT_BUCKET / RUN_ID).")
    elif args.shard_index is not None:
        p.error("--shard-index requires --num-shards.")

    if args.streaming:
        if not sharded:
            p.error("--streaming is a fan-out mode — it needs --num-shards/--shard-index.")
        if args.modality != "text":
            p.error("--streaming currently supports text modality only.")
        if args.max_samples:
            p.error("--max-samples is incompatible with --streaming (use row mode for capped test runs).")

    import torch
    from datasets import load_dataset
    from huggingface_hub import DatasetCard, login
    from sentence_transformers import SentenceTransformer

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    if not torch.cuda.is_available():
        logger.warning("No CUDA — running on CPU (much slower). Prefer a GPU flavor, e.g. --flavor l4x1.")

    logger.info(f"Loading {args.input_dataset} [{args.split}]"
                + (f" @ {args.revision[:12]}" if args.revision else "")
                + (" (streaming)" if args.streaming else ""))
    load_kwargs = {"split": args.split, "streaming": args.streaming}
    if args.revision:
        load_kwargs["revision"] = args.revision
    ds = (load_dataset(args.input_dataset, args.config, **load_kwargs) if args.config
          else load_dataset(args.input_dataset, **load_kwargs))
    if ds.column_names is not None:
        if args.column not in ds.column_names:
            logger.error(f"Column {args.column!r} not found. Available: {ds.column_names}")
            sys.exit(1)
        if args.output_column in ds.column_names:
            logger.error(f"Output column {args.output_column!r} already exists — choose another --output-column.")
            sys.exit(1)
    if args.max_samples and not args.streaming:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    if sharded and args.streaming:
        # File-level split: each rank reads ONLY its own data files. Sizes vary per rank and
        # rows_total is unknown upfront; correctness (exact, deterministic, idempotent) holds
        # as long as every rank pins the same --revision.
        if ds.n_shards < args.num_shards:
            logger.error(f"Dataset has {ds.n_shards} file shard(s) < --num-shards {args.num_shards}. "
                         f"Lower --num-shards or use row mode.")
            sys.exit(1)
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_index)
        logger.info(f"Fan-out file-shard {args.shard_index + 1}/{args.num_shards} of run {args.run_id} "
                    f"({ds.n_shards} file(s) for this rank)")
    elif sharded:
        # Contiguous slices keep row order reconstructable at consolidation. Note the whole
        # split was still downloaded above — acceptable at few-M rows, not at corpus scale.
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_index, contiguous=True)
        logger.info(f"Fan-out shard {args.shard_index + 1}/{args.num_shards} of run {args.run_id}")
        if len(ds) == 0:
            logger.error(f"Shard {args.shard_index} is empty — num_shards exceeds the row count. "
                         f"Lower --num-shards (or raise --max-samples).")
            sys.exit(1)
    if not args.streaming:
        logger.info(f"{len(ds)} rows; modality={args.modality}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device, trust_remote_code=True)
    if args.modality == "text" and getattr(model, "max_seq_length", None):
        model.max_seq_length = min(model.max_seq_length, args.max_seq_len)
    dim = model.get_sentence_embedding_dimension()
    logger.info(f"Model {args.model} on {device}; dim={dim}")

    # Prompt handling — many retrieval models need a query vs document/passage prefix (text only).
    prompt_str = None  # None = let encode_query/encode_document choose natively
    if args.modality == "text":
        prompt_str = resolve_prompt(model, args.model, is_query=args.query_mode, args=args)
        if args.streaming:
            run_streaming_shard(ds, model, prompt_str, args)
            return
        items = [t if isinstance(t, str) and t.strip() else " " for t in ds[args.column]]
    else:
        if args.prompt or args.prompt_name:
            logger.warning("--prompt/--prompt-name ignored for image modality.")
        items = [im.convert("RGB") if hasattr(im, "convert") else im for im in ds[args.column]]

    median_tok = sniff_token_lengths(model, items, args.max_seq_len) if args.modality == "text" else None

    if str(args.batch_size).lower() == "auto":
        # Pick the probe range from the data shape. Images: the ViT forward saturates the GPU by
        # ~batch 32, so bigger only adds memory — probe low. Text: short texts under-use the GPU at
        # small batch (probe bigger); long texts pad-waste at big batch (stay modest). Probe verifies.
        if args.modality == "image":
            candidates = (32, 64, 128)
        elif median_tok is None or median_tok >= 256:
            candidates = (32, 64, 128, 256)
        elif median_tok >= 64:
            candidates = (64, 128, 256, 512)
        else:
            candidates = (128, 256, 512, 1024)
        logger.info(f"Finding batch size (--batch-size auto; candidates {candidates})...")
        batch_size = find_batch_size(model, items, args.normalize, candidates=candidates)
    else:
        batch_size = int(args.batch_size)

    # Text goes through encode_query/encode_document (native registered-prompt selection + Router
    # task routing); our resolved prefix, when not None, overrides via prompt=. Images use encode().
    if args.modality == "text":
        encode_fn = model.encode_query if args.query_mode else model.encode_document
        encode_kwargs = {"prompt": prompt_str} if prompt_str is not None else {}
    else:
        encode_fn = model.encode
        encode_kwargs = {}
    reporter = None
    if sharded:
        reporter = StatusReporter(args.output_bucket, args.run_id, args.shard_index,
                                  rows_total=len(items), tokens_per_row=median_tok)
        reporter.report(0, force=True)

    t0 = time.perf_counter()
    try:
        if reporter is None:
            emb = encode_fn(items, batch_size=batch_size, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=args.normalize, **encode_kwargs)
        else:
            # Macro-chunks so the heartbeat can report rows_done between encode calls.
            # Chunking doesn't change results — batching happens at batch_size regardless.
            import numpy as np
            chunk_rows = 25_000
            parts = []
            for start in range(0, len(items), chunk_rows):
                chunk = items[start:start + chunk_rows]
                parts.append(encode_fn(chunk, batch_size=batch_size, show_progress_bar=True,
                                       convert_to_numpy=True, normalize_embeddings=args.normalize,
                                       **encode_kwargs))
                reporter.report(start + len(chunk))
            emb = np.concatenate(parts) if len(parts) > 1 else parts[0]
    except Exception:
        if reporter:
            reporter.report(state="error", force=True)
        raise
    secs = time.perf_counter() - t0
    logger.info(f"Embedded {len(items)} in {secs:.1f}s ({len(items)/secs:.0f} rows/s), dim={dim}")

    if sharded:
        # Shard mode: no repo commit here (N workers committing → 412 contention). Write this
        # rank's parquet to the run bucket; consolidate-shards.py makes the single final commit.
        #
        # Build the embedding column as zero-copy float32 Arrow — NEVER as Python floats.
        # `[e.tolist() for e in emb]` on an 800k-row shard is ~10 GB of PyFloat objects and
        # swap-thrashed L4 workers into multi-hour silent stalls (wiki fleet, 2026-07-09).
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        reporter.report(len(items), state="writing", force=True)
        try:
            emb32 = np.ascontiguousarray(emb, dtype=np.float32)
            emb_col = pa.FixedSizeListArray.from_arrays(pa.array(emb32.ravel()), emb32.shape[1])
            table = ds.with_format("arrow")[:].append_column(args.output_column, emb_col)
            out_path = f"/tmp/shard-{args.shard_index:05d}.parquet"
            dest = f"runs/{args.run_id}/data/{args.shard_index:05d}.parquet"
            # Small row groups + page index or the dataset viewer can't random-access:
            # pyarrow's default ~1M-row groups ≈ multi-GB with an embeddings column
            # ("Scan size limit exceeded"). ~25k rows ≈ ~100MB here.
            pq.write_table(table, out_path, row_group_size=25_000, write_page_index=True)
            logger.info(f"Uploading shard parquet → {args.output_bucket}/{dest}")
            put_bucket_files(args.output_bucket, [(out_path, dest)])
        except Exception:
            reporter.report(state="error", force=True)
            raise
        reporter.report(len(items), state="done", force=True)
        logger.info(f"✅ shard {args.shard_index + 1}/{args.num_shards} of run {args.run_id} uploaded")
        return

    ds = ds.add_column(args.output_column, [e.tolist() for e in emb])

    # For the card: record the effective prefix (explicit, else the model's registered one).
    side_keys = ("query",) if args.query_mode else ("document", "passage", "corpus")
    effective = prompt_str if prompt_str is not None else next(
        (v for k in side_keys if (v := (getattr(model, "prompts", {}) or {}).get(k))), "")
    prompt_line = f"`{effective}`" if effective else "(none)"
    # Canonical provenance stamp (see AGENTS.md): Jobs claim gated on JOB_ID, set by HF Jobs in-container.
    script_url = "https://huggingface.co/datasets/uv-scripts/embeddings/raw/main/generate-embeddings.py"
    on_jobs = os.environ.get("JOB_ID") is not None
    hw = jobs_flavor()  # flavor name, e.g. "a10g-small"; "" on CPU
    origin = (
        "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
        + (f" (`{hw}`)" if hw else "")
    ) if on_jobs else "Generated"
    jobs_tag = "\n- hf-jobs" if on_jobs else ""
    card = DatasetCard(
        f"---\ntags:\n- embeddings\n- uv-script\n- generated{jobs_tag}\n---\n\n"
        f"# {args.output_dataset}\n\n"
        f"Embeddings of [`{args.input_dataset}`](https://huggingface.co/datasets/{args.input_dataset}) "
        f"column `{args.column}`.\n\n"
        f"- Model: [`{args.model}`](https://huggingface.co/{args.model}) (dim {dim})\n"
        f"- Column: `{args.output_column}`  ·  normalized: {args.normalize}\n"
        f"- Prompt prepended ({'query' if args.query_mode else 'document'} side): {prompt_line}\n\n"
        f"## Reproduction\n\n"
        f"{origin} with the [`generate-embeddings.py`]({script_url}) recipe "
        f"from [uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:\n\n"
        f"```bash\nhf jobs uv run {script_url} \\\n"
        f"    {args.input_dataset} <output-dataset> --column {args.column} --model {args.model}\n```\n"
    )
    # Retry the push with an XET-disable fallback: a transient upload failure here would
    # otherwise lose the whole (paid) embedding run.
    logger.info(f"Pushing to {args.output_dataset} (private={args.private})")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.warning("Disabling XET (fallback to HTTP upload)")
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            ds.push_to_hub(args.output_dataset, private=args.private)
            break
        except Exception as e:
            logger.error(f"Upload attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                delay = 30 * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error("All upload attempts failed. Results are lost.")
                sys.exit(1)
    try:
        card.push_to_hub(args.output_dataset, repo_type="dataset")
    except Exception as e:
        logger.warning(f"card push skipped: {e}")
    logger.info(f"✅ https://huggingface.co/datasets/{args.output_dataset}")


if __name__ == "__main__":
    main()
