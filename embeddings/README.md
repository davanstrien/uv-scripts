---
viewer: false
tags:
- uv-script
- embeddings
- sentence-transformers
- vector-search
---

# Embeddings

Generate embeddings for a Hugging Face dataset — text or images — with one command, on a
cloud GPU, no infra. The output lands back on the Hub as a new dataset (or, with the Lance
variant, as a **searchable vector index you can query over `hf://` without downloading**).

There is one simple default and a few variants; they are separate single-file scripts because
their dependencies (sentence-transformers vs vLLM vs Lance) are too different to share one env.

| Script | Use it for | Engine |
|---|---|---|
| `generate-embeddings.py` | The default. Text or images. Simple, fast, runs anywhere. | sentence-transformers |
| `generate-embeddings-saturate.py` | **Large text runs.** A vLLM server in the Job, driven by [saturate](https://github.com/davanstrien/saturate). One row out per row in, resumable. | vLLM server + saturate |
| `generate-embeddings-vllm.py` | The older vLLM path (offline engine, unpinned `vllm`). Use `generate-embeddings-saturate.py` instead. | vLLM pooling |
| `embed-to-lance.py` | Get a **searchable vector index as a Hub dataset** (the "vector DB" path). | sentence-transformers + Lance |
| `launch-embedding-fleet.py` | **Fan one run out across N parallel Jobs** (+ `consolidate-shards.py`). | run_uv_job + Buckets |

## Quick start

```bash
# Text — pick a model from the MTEB leaderboard
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/embeddings/raw/main/generate-embeddings.py \
  stanfordnlp/imdb  your-name/imdb-embeddings --column text

# Images (CLIP)
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/embeddings/raw/main/generate-embeddings.py \
  your-name/photos your-name/photos-embeddings --modality image --column image --model clip-ViT-B-32
```

Always try `--max-samples 100 --private` first.

## Large runs: `generate-embeddings-saturate.py`

```bash
hf jobs uv run \
  https://huggingface.co/datasets/uv-scripts/embeddings/raw/main/generate-embeddings-saturate.py \
  stanfordnlp/imdb your-name/imdb-embeddings --column text --max-samples 100
```

The script's header sets the `vllm/vllm-openai:v0.30.0` image, `a10g-small` and the `HF_TOKEN`
secret, so the command needs no job flags. Measured on one A10G with 20,000 IMDb reviews of up
to 300 words, one run each, embedding step only:

| Model | `generate-embeddings.py` | `generate-embeddings-saturate.py` |
|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | 96 rows/s | 187 rows/s |
| BAAI/bge-m3 | 71 rows/s | 276 rows/s |

The server is ready 100 to 120 seconds after it starts, so `generate-embeddings.py` is quicker
for a few hundred rows. Texts are cut at the model's own maximum length unless you pass
`--max-seq-len`. Prompts follow the same rule as below: `--prompt`, then the model's registered
document prompt, then the known-family table.

## Fan-out: N Jobs in parallel

One L4 does ~900 rows/s with `all-MiniLM-L6-v2`; a fleet of 8 does ~7k. `launch-embedding-fleet.py`
splits the run into exact, non-overlapping shards (one Job each), workers stream shard parquets +
progress heartbeats to a [Bucket](https://huggingface.co/docs/hub/storage-buckets) (object writes —
no repo-commit contention), and a final CPU Job merges everything into the output dataset in one
commit, with full provenance on the card.

```bash
# 8 L4s over a corpus; runs from your laptop, workers run on Jobs
uv run https://huggingface.co/datasets/uv-scripts/embeddings/raw/main/launch-embedding-fleet.py \
  your-name/corpus your-name/corpus-embeddings --num-shards 8 --flavor l4x1 --timeout 1h
```

Every shard is idempotent (a rank overwrites only its own files), so failures never cost you the
run. Workers that fail are auto-retried once; past that, one command converges any run — no need
to know which rank failed or why:

```bash
uv run launch-embedding-fleet.py <in> <out> --run-id <id> --resume   # re-runs ONLY missing shards, then merges
```

(`--retry-rank`/`--consolidate-only` remain for surgical control.) Each worker's timeout gives a
**hard cost ceiling**: a fleet can never cost more than `N × flavor-rate × timeout`.

Watch a run live — progress, ETA, live ~$ vs ceiling, GPU utilization, replica health — in the
[fleet dashboard Space](https://huggingface.co/spaces/davanstrien/embedding-fleet-dashboard)
(the launcher prints your run's deep link).

Scale note: by default workers shard row-wise after loading the split, so each rank downloads the
full split first — fine up to a few tens of millions of rows. Past that, add `--streaming`: workers
then shard at the **file** level and each rank streams only its own files (text only; needs
`num_files ≥ num_shards`). The launcher pins the input dataset revision either way, so every rank
— including a `--retry-rank` weeks later — slices the identical snapshot.

## Which model?

**Find the *current* best — don't trust a fixed list** (embedding quality moves fast). Check the
[MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard), or from the CLI:

```bash
hf models ls --filter sentence-transformers --sort trending_score --limit 20   # what's hot now
hf models ls --filter sentence-transformers --sort downloads --limit 20        # proven workhorses
```
(Sort by `trending_score`/`downloads`, not `created_at` — the newest list is mostly test repos.)

See **[HEURISTICS.md](./HEURISTICS.md)** for the full "which model / GPU / batch for your data" guide
(measured). The table below is examples benchmarked 2026-07, not a permanent answer:

| Model | Params | Dim | Note |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | 384 | Fastest; safe default |
| `BAAI/bge-base-en-v1.5` | 109M | 768 | Strong English quality/speed balance |
| `BAAI/bge-m3` | 568M | 1024 | Multilingual + long context (slower) |
| `Qwen/Qwen3-Embedding-0.6B` | 596M | 1024 | Top open MTEB; decoder → use the vLLM variant / A100 |

Images: `clip-ViT-B-32` (fast) or `clip-ViT-L-14` (higher quality).

## Prompts (retrieval correctness — read this if you're building search)

Many retrieval models need a **different prefix for documents vs queries**, and getting it
wrong silently degrades results. Worse, you can't trust `model.prompts`: current
sentence-transformers injects a placeholder `{"query": "", "document": ""}` even for models
that register **nothing**, so e5 / nomic / bge look "prompt-less" via that attribute while
their real prefixes live only in the model card.

`generate-embeddings.py` handles this. It embeds a **document corpus** by default and picks the
document convention in this order: (1) the model's **registered** prompt if it ships a real one
(e.g. Qwen3-Embedding), else (2) a small **built-in family table**, else (3) no prefix. The
chosen prefix is logged and written into the output dataset card.

| Family | Query prefix | Document prefix |
|---|---|---|
| e5 (`intfloat/e5-*`, `multilingual-e5-*`, non-instruct) | `query: ` | `passage: ` |
| nomic (`nomic-embed-text-*`) | `search_query: ` | `search_document: ` |
| bge English (`bge-*-en-*`) | `Represent this sentence for searching relevant passages: ` | (none) |
| bge-m3 | (none) | (none) |
| Qwen3-Embedding | registered by the model | (none) |
| anything else | — | — (pass `--prompt` if it needs one) |

Override the auto-pick:

- `--query-mode` — embed inputs as **queries**, not documents (flips the convention)
- `--prompt 'passage: '` — force a raw prefix (highest precedence; `--prompt ''` forces none)
- `--prompt-name query` — use a prompt the model registered, by name
- `--no-auto-prompt` — turn off the family table (still honours registered prompts)

Instruct-style models (`e5-*-instruct`, `gte-Qwen…`) are deliberately left to their registered
prompt or your explicit `--prompt`, since the instruction is task-specific.

## Batch size (auto by default)

`--batch-size auto` (the default) times a few batch sizes on a warmup sample and keeps the
fastest that fits — bigger isn't always faster, because variable-length text wastes compute on
padding. Pass `--batch-size 128` to pin it.

## Which GPU? (measured, 20k rows, seq-cap 512)

Throughput (rows/s) and cost per 1M rows:

| Model | L4 ($0.80/hr) | A10G ($1.50/hr) | A100 ($2.50/hr) |
|---|---|---|---|
| all-MiniLM-L6-v2 | 912 · **$0.24/1M** | 1099 · $0.38/1M | 1372 · $0.51/1M |
| bge-base-en-v1.5 | 119 · **$1.87/1M** | 206 · $2.02/1M | 261 · $2.66/1M |
| Qwen3-Embedding-0.6B | 59 · $3.77/1M | 93 · $4.48/1M | 250 · **$2.78/1M** |

**Default to `l4x1`** — cheapest per 1M rows for encoder models. For **decoder** embedders
(Qwen3-Embedding) the A100 is both faster *and* cheaper per 1M (they use the extra compute),
and the vLLM variant roughly doubles throughput again (Qwen3-Embedding-0.6B: ~121 rows/s on an
L4 via `generate-embeddings-vllm.py`, ~2× the sentence-transformers path).

Images embed much faster than text: `clip-ViT-B-32` runs ~395 img/s on an L4 at the auto-picked batch (bs=32; ~455 on an A10G). Full-resolution photos land nearer ~215 img/s — decode/resize is a real CPU tax on fast models.

## The vector-DB path (`embed-to-lance.py`)

Writes a [Lance](https://huggingface.co/docs/hub/datasets-lance) table with a vector index and
pushes it as a Hub dataset. You (or anyone you share it with) can then search it directly over
`hf://` **without downloading it**:

```python
import lance
ds = lance.dataset("hf://datasets/your-name/my-vecdb/vecdb.lance")   # opens in ~1s, no download
hits = ds.to_table(nearest={"column": "vector", "q": query_vec, "k": 5})
```

> **Query prompts:** embed `query_vec` with the model's *query* prefix (e5 → `"query: "`,
> nomic → `"search_query: "`; the run prints the right one). Documents and queries use
> different prefixes on these models — mismatching them silently degrades retrieval.

End-to-end this is fast and cheap: **all 241,787 Simple-English-Wikipedia articles → a
searchable Lance vector DB on the Hub in ~4.5 min for ~$0.07 on a single L4** (load → embed →
index → push, with `all-MiniLM-L6-v2`; pass `--model` to trade speed for quality).

Best for share-and-search over a corpus; for high-QPS serving, pull the dataset local first.
