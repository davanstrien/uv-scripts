# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]==0.3.1",
#     "huggingface-hub>=1.12",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.30.0"
# flavor = "a10g-small"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# ///
"""
Generate embeddings for a text column of a Hugging Face dataset with a vLLM server, and write
them to a new dataset on the Hub.

This is the fast variant for large runs. The script starts a vLLM server on the Job's GPU and
sends it one text per request with `saturate`, which sets the number of requests in flight.
vLLM groups the requests into batches itself. For a first try, for image embeddings, or for a
model vLLM does not serve, use generate-embeddings.py (sentence-transformers) instead.

    hf jobs uv run \\
        https://huggingface.co/datasets/uv-scripts/embeddings/raw/main/generate-embeddings-saturate.py \\
        stanfordnlp/imdb your-username/imdb-embeddings --column text --max-samples 100

Measured on one A10G with 20,000 IMDb reviews of up to 300 words (one run each):
Qwen/Qwen3-Embedding-0.6B 187 rows/s, against 96 rows/s for generate-embeddings.py.
BAAI/bge-m3 276 rows/s, against 71 rows/s for generate-embeddings.py at full precision.
The server was ready 100 to 120 seconds after it started, so a run of a few hundred rows is
quicker with generate-embeddings.py.

On Jobs, the [tool.hf-jobs] header sets the vllm/vllm-openai:v0.30.0 image, the a10g-small
flavor, a 2h timeout and the HF_TOKEN secret (`hf` CLI 1.32+). --flavor and --timeout before
the script override it.

PROMPTS: many retrieval models need a different prefix for documents and for queries. The
script embeds documents by default and picks the prefix in this order: --prompt, then the
"document" prompt the model registers in config_sentence_transformers.json, then a small table
of known families (e5, nomic), then no prefix. --query-mode uses the query side. The prefix is
logged and written to the dataset card.

LONG TEXTS: each text is cut at the model's own maximum length, which the server reports
(for example 8,192 tokens for BAAI/bge-m3). Long documents cost more to embed. Pass
--max-seq-len 512 for a shorter cut and a predictable cost; a value above the model's maximum
is lowered to it. generate-embeddings.py cuts at 512 tokens by default. The cut is logged and
written to the dataset card.

OUTPUT: one record per input row, with `id`, the text column, the embedding column, `model`,
`prompt_tokens` and `error`. A row that failed has a message in `error` and no embedding.
Run the same command again to resume: rows already written are skipped. Add --retry-errors to
try the failed rows again. A retry adds a second record for the row and keeps the old error
record, so after a --retry-errors run drop the records that have an `error`, or read the
output with `saturate.read_output`, which keeps one record per id (a success over an earlier
error).

BUCKETS: the input can be a glob of files in a Storage Bucket instead of a dataset id, for
example hf://buckets/you/corpus/data/*.parquet (also .jsonl, .json, .csv, and .txt or .md,
where each file is one row). The output can be a bucket path, for example
hf://buckets/you/embeddings/run1: the parts land there and no dataset card is written.

KEEP COLUMNS: --keep language date copies those source columns into the output, so the
vectors join back to their metadata. Kept values are written as strings (lists and dicts as
JSON). --id-column sets the row id; use a column with stable unique values for large runs.

FAN-OUT: --shard 2/4 makes this Job process the rows whose id hashes to shard 2 of 4. Start
4 Jobs with --shard 0/4 ... 3/4 and the same output; together they cover the input once.
Each Job still reads the whole input and keeps its share. Use a bucket output for fan-out:
each flush to a dataset repo is a commit. With a dataset output, only shard 0 writes the
card, and its counts cover shard 0 only.

--skip-empty leaves out rows whose text is empty or only whitespace (they get no record).
"""
import argparse
import json
import os
import re
import shlex
import sys
import time
import urllib.request

SCRIPT_URL = "https://huggingface.co/datasets/uv-scripts/embeddings/raw/main/generate-embeddings-saturate.py"


def known_family_prefixes(model_id):
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


# File extension -> `datasets` builder, for bucket inputs.
BUCKET_FORMATS = {".parquet": "parquet", ".jsonl": "json", ".json": "json", ".csv": "csv",
                  ".txt": "text", ".md": "text"}


def parse_shard(text):
    """'2/4' -> (2, 4). The rank counts from 0."""
    try:
        rank_text, world_text = text.split("/")
        rank, world = int(rank_text), int(world_text)
    except ValueError:
        sys.exit(f"--shard takes RANK/WORLD, for example 0/4; got {text!r}")
    if world < 1 or not 0 <= rank < world:
        sys.exit(f"--shard {text}: the rank must be from 0 to {world - 1}")
    return rank, world


def bucket_id(uri):
    """'hf://buckets/ns/name/any/prefix' -> 'ns/name'."""
    return "/".join(uri.removeprefix("hf://buckets/").split("/")[:2])


def keep_value(value):
    """Kept columns are written as strings; lists and dicts as JSON."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def report_memory(every_s=60):
    """Print this process's resident memory every minute, to stderr (Linux only)."""
    import threading

    def loop():
        while True:
            try:
                with open("/proc/self/status") as status:
                    for line in status:
                        if line.startswith("VmRSS"):
                            print(f"[rss] {int(line.split()[1]) // 1024} MB", file=sys.stderr, flush=True)
            except OSError:
                return
            time.sleep(every_s)

    threading.Thread(target=loop, daemon=True).start()


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


def registered_prompt(model_id, side):
    """Return the prompt the model registers for "document" or "query", or None when it has none."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    try:
        path = hf_hub_download(model_id, "config_sentence_transformers.json")
    except (HfHubHTTPError, OSError):
        return None  # the model has no such file
    with open(path) as file:
        prompts = json.load(file).get("prompts") or {}
    return prompts.get(side) or None


def choose_prefix(args):
    """Return (prefix, where it came from) for the text side this run embeds."""
    side = "query" if args.query_mode else "document"
    if args.prompt is not None:
        return args.prompt, "--prompt"
    registered = registered_prompt(args.model, side)
    if registered:
        return registered, "registered by the model"
    family = known_family_prefixes(args.model)
    if family:
        query_prefix, document_prefix = family
        return (query_prefix if args.query_mode else document_prefix), "known model family"
    return "", "none"


def model_max_tokens(endpoint, model_id):
    """Return the most tokens the server accepts for one text of this model."""
    with urllib.request.urlopen(f"{endpoint}/models", timeout=60) as response:
        served = json.load(response)["data"]
    for entry in served:
        if entry["id"] == model_id:
            return entry["max_model_len"]
    sys.exit(f"The server does not list {model_id}. It serves: {[entry['id'] for entry in served]}")


def embedding_dimension(endpoint, model_id):
    """Return the length of the model's vectors, from one request to the server."""
    request = urllib.request.Request(
        f"{endpoint}/embeddings",
        data=json.dumps({"model": model_id, "input": "dimension check"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return len(json.load(response)["data"][0]["embedding"])


def reproduce_options(args):
    """Return the options that give this dataset again, as a shell-quoted string."""
    options = ["--column", args.column, "--model", args.model]
    if args.output_column != "embeddings":
        options += ["--output-column", args.output_column]
    if args.config:
        options += ["--config", args.config]
    if args.split != "train":
        options += ["--split", args.split]
    if args.id_column:
        options += ["--id-column", args.id_column]
    if args.max_samples:
        options += ["--max-samples", str(args.max_samples)]
    if args.max_seq_len:
        options += ["--max-seq-len", str(args.max_seq_len)]
    if args.prompt is not None:
        options += ["--prompt", args.prompt]
    if args.query_mode:
        options += ["--query-mode"]
    if args.keep:
        options += ["--keep", *args.keep]
    if args.skip_empty:
        options += ["--skip-empty"]
    return " ".join(shlex.quote(option) for option in options)


def write_card(args, prefix, prefix_source, dimension, max_tokens, stats):
    """Write the dataset card, with the canonical uv-scripts reproduction section."""
    from huggingface_hub import HfApi

    on_jobs = os.environ.get("JOB_ID") is not None
    flavor = jobs_flavor()
    origin = (
        "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
        + (f" (`{flavor}`)" if flavor else "")
    ) if on_jobs else "Generated"
    jobs_tag = "\n- hf-jobs" if on_jobs else ""
    side = "query" if args.query_mode else "document"
    prefix_line = f"`{prefix}` ({prefix_source})" if prefix else "(none)"
    # A resumed run processes only the rows that were missing, so the card gives both counts.
    earlier = f", {stats.rows_done_prior} already written by an earlier run" if stats.rows_done_prior else ""
    rank, world = parse_shard(args.shard)
    shard_note = f" (shard {rank} of {world} only; the other shards wrote the rest)" if world > 1 else ""
    card = (
        f"---\ntags:\n- embeddings\n- uv-script\n- generated{jobs_tag}\n---\n\n"
        f"# {args.output_dataset}\n\n"
        f"Embeddings of [`{args.input_dataset}`](https://huggingface.co/datasets/{args.input_dataset}) "
        f"column `{args.column}`, made with a vLLM server.\n\n"
        f"- Model: [`{args.model}`](https://huggingface.co/{args.model}) (dim {dimension})\n"
        f"- Column: `{args.output_column}`, normalized\n"
        f"- Prompt prepended ({side} side): {prefix_line}\n"
        f"- Texts cut at {max_tokens} tokens\n"
        f"- Rows in the last run{shard_note}: {stats.rows_processed} embedded, {stats.rows_failed} failed{earlier}. "
        f"A failed row has a message in `error`.\n\n"
        f"## Reproduction\n\n"
        f"{origin} with the [`generate-embeddings-saturate.py`]({SCRIPT_URL}) recipe "
        f"from [uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:\n\n"
        f"```bash\nhf jobs uv run {SCRIPT_URL} \\\n"
        f"    {args.input_dataset} <output-dataset> {reproduce_options(args)}\n```\n"
    )
    HfApi().upload_file(
        path_or_fileobj=card.encode(), path_in_repo="README.md",
        repo_id=args.output_dataset, repo_type="dataset",
    )


def main():
    parser = argparse.ArgumentParser(description="Embed a text column with a vLLM server via saturate")
    parser.add_argument("input_dataset",
                        help="Input dataset ID, or a bucket glob such as hf://buckets/you/b/data/*.parquet")
    parser.add_argument("output_dataset",
                        help="Output dataset ID (created if missing), or a bucket path hf://buckets/you/b/run1")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="An embedding model that vLLM serves")
    parser.add_argument("--column", default="text", help="Text column to embed")
    parser.add_argument("--output-column", default="embeddings")
    parser.add_argument("--config", default=None, help="Dataset config name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--id-column", default=None, help="Column to use as the row id (default: the row's position)")
    parser.add_argument("--max-samples", type=int, default=None, help="Embed only the first N rows (for testing)")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Cut each text at this many tokens (default: the model's own maximum)")
    parser.add_argument("--prompt", default=None,
                        help="Prefix to put before every text, such as 'passage: '. Pass '' for no prefix.")
    parser.add_argument("--query-mode", action="store_true", help="Embed the texts as queries, not documents")
    parser.add_argument("--private", action="store_true", help="Make the output dataset private")
    parser.add_argument("--retry-errors", action="store_true", help="Try again the rows whose only record is an error")
    parser.add_argument("--keep", nargs="+", default=[], metavar="COLUMN",
                        help="Source columns to copy into the output (written as strings)")
    parser.add_argument("--skip-empty", action="store_true", help="Leave out rows whose text is empty")
    parser.add_argument("--shard", default="0/1", metavar="RANK/WORLD",
                        help="Process only shard RANK of WORLD (fan-out over several Jobs), e.g. 0/4")
    args = parser.parse_args()
    shard = parse_shard(args.shard)
    bucket_output = args.output_dataset.startswith("hf://buckets/")
    bucket_input = args.input_dataset.startswith("hf://buckets/")
    if args.output_column in {"id", "error", "model", "prompt_tokens", args.column, *args.keep}:
        sys.exit(f"--output-column {args.output_column!r} clashes with another output column")

    import pyarrow as pa
    from huggingface_hub import HfApi
    from saturate import Engine, dataset_rows, pump

    prefix, prefix_source = choose_prefix(args)
    print(f"Prompt prefix: {prefix!r} ({prefix_source})", file=sys.stderr)

    if args.id_column is None:
        ids = "index"
    elif args.id_column in ("index", "content"):
        # saturate reads these two names as id strategies, so a column with one of them
        # needs a callable. It returns the raw value, so saturate still refuses a null id.
        def ids(example):
            return example[args.id_column]
    else:
        ids = args.id_column

    if bucket_output:
        HfApi().create_bucket(bucket_id(args.output_dataset), private=args.private, exist_ok=True)
        output_uri = args.output_dataset
    else:
        HfApi().create_repo(args.output_dataset, repo_type="dataset", private=args.private, exist_ok=True)
        output_uri = f"hf://datasets/{args.output_dataset}/data"

    columns = [args.column, *args.keep]
    if bucket_input:
        from datasets import load_dataset

        suffix = os.path.splitext(args.input_dataset)[1].lower()
        if suffix not in BUCKET_FORMATS:
            sys.exit(f"A bucket input is a glob ending in one of {sorted(BUCKET_FORMATS)}, "
                     f"for example hf://buckets/you/b/data/*.parquet; got {args.input_dataset!r}")
        builder = BUCKET_FORMATS[suffix]
        extra = {"sample_by": "document"} if builder == "text" else {}
        loaded = load_dataset(builder, data_files=args.input_dataset, split="train", streaming=True, **extra)
        rows = dataset_rows(loaded, split="train", columns=columns, ids=ids, limit=args.max_samples, shard=shard)
    else:
        rows = dataset_rows(
            args.input_dataset, config=args.config, split=args.split,
            columns=columns, ids=ids, limit=args.max_samples, shard=shard,
        )
    if args.skip_empty:
        rows = ((row_id, row) for row_id, row in rows if (row[args.column] or "").strip())
    report_memory()

    cut = {"max_tokens": None}  # set once the server reports the model's maximum

    def to_request(row):
        return {
            "model": args.model,
            "input": prefix + (row[args.column] or ""),
            "truncate_prompt_tokens": cut["max_tokens"],
        }

    # Declared, so a part holding only error rows has the same columns as the rest.
    schema = pa.schema([
        ("id", pa.string()), ("error", pa.string()), (args.column, pa.string()),
        (args.output_column, pa.list_(pa.float32())), ("model", pa.string()), ("prompt_tokens", pa.int64()),
    ] + [(column, pa.string()) for column in args.keep])  # fmt: skip
    def parse(row, body):
        return {
            args.column: row[args.column],
            args.output_column: body["data"][0]["embedding"],
            "model": args.model,
            "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens"),
            **{column: keep_value(row.get(column)) for column in args.keep},
        }

    # The readiness check sends one embedding request, so the server is known to embed before the run starts.
    engine = Engine(
        args.model, engine="vllm", extra_args=["--runner", "pooling"],
        ready_route="/embeddings", ready_payload={"model": args.model, "input": "ready"},
    )
    with engine as endpoint:
        model_maximum = model_max_tokens(endpoint, args.model)
        cut["max_tokens"] = min(args.max_seq_len or model_maximum, model_maximum)
        print(f"Texts are cut at {cut['max_tokens']} tokens (the model's maximum is {model_maximum})", file=sys.stderr)
        dimension = embedding_dimension(endpoint, args.model)
        started = time.monotonic()
        stats = pump(
            rows, to_request, parse, endpoint, output_uri,
            route="/embeddings", retry_errors=args.retry_errors, schema=schema, shard=shard,
        )
        elapsed = time.monotonic() - started

    if not bucket_output and shard[0] == 0:
        write_card(args, prefix, prefix_source, dimension, cut["max_tokens"], stats)
    where = (f"https://huggingface.co/buckets/{bucket_id(args.output_dataset)}" if bucket_output
             else f"https://huggingface.co/datasets/{args.output_dataset}")
    print(
        f"{where} shard {shard[0]}/{shard[1]} "
        f"({stats.rows_processed} ok, {stats.rows_failed} error rows, "
        f"{stats.rows_processed / max(elapsed, 1e-6):.0f} rows/s, dim {dimension})",
        file=sys.stderr,
    )
    print("GENERATE_EMBEDDINGS_SATURATE " + stats.to_json(), flush=True)
    sys.stderr.flush()
    # saturate README: a process can hang or abort at interpreter exit after a Hub Parquet
    # stream stopped early (--max-samples); everything is written, so leave now.
    os._exit(0)


if __name__ == "__main__":
    main()
