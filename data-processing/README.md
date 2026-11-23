---
viewer: false
tags:
  - uv-script
  - data-processing
  - parquet
  - buckets
  - webhooks
---

<a href="https://huggingface.co/uv-scripts"><img src="https://huggingface.co/spaces/uv-scripts/README/resolve/main/lockup.svg" alt="uv scripts" width="180"></a>

# Data processing

<a href="https://huggingface.co/uv-scripts"><picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md-dark.svg"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md.svg" alt="Follow uv-scripts on Hugging Face"></picture></a>

General data processing recipes: convert, clean and prepare data files.

| Script | What it does |
|---|---|
| [`optimize-parquet.py`](#optimize-parquetpy-optimized-parquet-from-bucket-uploads) | Converts CSV, JSON and Parquet files uploaded to a bucket into optimized Parquet, triggered by a bucket webhook |

## optimize-parquet.py: optimized Parquet from bucket uploads

Upload a CSV, JSON or Parquet file to a [Storage Bucket](https://huggingface.co/docs/hub/storage-buckets) and get an optimized Parquet version in a second bucket, automatically. A bucket [webhook](https://huggingface.co/docs/hub/webhooks) starts a [Job](https://huggingface.co/docs/hub/jobs) for each upload, and the Job converts only the files that changed.

```
input bucket  ──upload──▶  webhook  ──▶  Job (optimize-parquet.py)  ──▶  output bucket
data.csv                                                                 data.csv/data/train-00000-of-00001.parquet
```

The output is written by [`datasets`](https://huggingface.co/docs/datasets), so it gets the same [optimizations](https://huggingface.co/docs/hub/datasets-libraries#optimized-parquet-files) as `push_to_hub`: content-defined chunking for Xet deduplication, a page index for fast filtering and random access, and row groups of at most 100 MB.

### Setup

You need two buckets: one you upload to, and one for the output. The Job writes to a different bucket so that its own output does not trigger it again.

```bash
hf buckets create my-raw-files --private
hf buckets create my-parquet --private
```

**1. Create a base Job for the webhook to re-run.** With no webhook payload, this first run exits straight away:

```bash
hf jobs run --flavor cpu-upgrade --timeout 2h -e OUTPUT_BUCKET=<user>/my-parquet \
    ghcr.io/astral-sh/uv:python3.12-bookworm \
    uv run https://huggingface.co/datasets/uv-scripts/data-processing/raw/main/optimize-parquet.py
```

Use `hf jobs run ... uv run <url>` here, not `hf jobs uv run <url>`. `hf jobs uv run` uploads the script as a volume, and webhook runs don't keep volumes.

**2. Create a webhook on the input bucket that re-runs this Job:**

```bash
hf webhooks create --job-id <job id from step 1> \
    --watch bucket:<user>/my-raw-files --domain repo --secrets HF_TOKEN
```

`--secrets HF_TOKEN` stores a token with the webhook, encrypted, and every triggered run receives it as a secret to read and write the buckets. The value comes from `HF_TOKEN` in your environment, or from the token you logged in with; pass `--secrets HF_TOKEN=hf_…` to use a different one. Use a fine-grained token, not your main one. The base Job needs no `--secrets`, because a webhook run does not inherit the Job's own secrets.

**3. Upload a file:**

```bash
hf buckets cp data.csv hf://buckets/<user>/my-raw-files/data.csv
```

After about a minute, the output is in `<user>/my-parquet/data.csv/`: the Parquet file(s) under `data/`, plus a README written by `datasets`.

### Options

| Environment variable | Default | Meaning |
|---|---|---|
| `OUTPUT_BUCKET` | required | Bucket to write the Parquet files to. Must differ from the watched bucket. |
| `STREAM_ABOVE_BYTES` | 1/3 of free disk | Files larger than this are streamed instead of loaded to disk. |

Files that fit on the Job's disk are loaded in full. Larger files are streamed, so they don't need to fit on the disk (50 GB on `cpu-upgrade`); raise `--timeout` for very large files. Supported inputs: `.csv`, `.json`, `.jsonl`, `.parquet`. Other files are skipped, and deleted files are ignored.

### Notes

- **Cost:** a small file takes about 20 seconds on `cpu-upgrade` ($0.03/hour). In testing, one `hf buckets sync` of several files sent one webhook event, so it started one Job.
- **Pin the script:** each webhook run downloads the script again. To stop changes to this recipe from reaching your webhook, replace `main` in the URL with a commit hash.
- **Limits:** a webhook can trigger at most 1,000 times per 24 hours. Above 10,000 changed files in one event, the payload list is truncated; those files are not converted.
