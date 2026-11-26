# For coding agents

This repo is a curated collection of ready-to-run OCR scripts — each one self-contained
via UV inline metadata, runnable over the network via `hf jobs uv run`. No clone, no
install, no setup.

## Don't rely on this doc — discover the current state

This file will go stale. Prefer these sources of truth:

- `hf jobs uv run --help` — job submission flags (volumes, secrets, flavors, timeouts)
- `hf jobs hardware` — current GPU flavors and pricing
- `hf auth whoami` — check HF token is set
- `hf jobs ps` / `hf jobs logs <id>` — monitor running jobs
- `ls` the repo to see which scripts actually exist (bucket variants especially)
- [README.md](./README.md) — the table of scripts with model sizes and notes

## Picking a script

The [README.md](./README.md) table lists every script with model size, backend, and
a short note. Axes that matter:

- **Model size** vs accuracy vs GPU cost. Smaller = cheaper per doc.
- **Backend**: vLLM scripts are usually fastest at scale. `transformers` and
  `falcon-perception` are alternatives for specific models.
- **Task support**: most scripts do plain text; some expose `--task-mode`
  (table, formula, layout, etc.) — check the script's own docstring.
- **Language coverage**: [`models.json`](./models.json) maps every script to its model,
  params, backend, required image pins, and what the model card claims about languages —
  with an `evidence` level (`per-language-benchmark` beats a bare `multilingual` tag).
  Human-readable version: [LANGUAGES.md](./LANGUAGES.md).

For the authoritative benchmark numbers on any model in the table, query the model
card programmatically — every OCR model publishes eval results on its card:

    from huggingface_hub import HfApi
    info = HfApi().model_info("tiiuae/Falcon-OCR", expand=["evalResults"])
    for r in info.eval_results:
        print(r.dataset_id, r.value)

See the [leaderboard data guide](https://huggingface.co/docs/hub/en/leaderboard-data-guide)
for the full API. This is more reliable than any markdown table that might drift.

## Getting help from a specific script

Each script has a docstring at the top with a description and usage examples, and its
argument definitions near the end. Read the whole file before you run it; the first 100
lines usually stop before the arguments. To read it without installing anything:

    curl -s https://huggingface.co/datasets/uv-scripts/ocr/raw/main/<script>.py

Or open the URL in a browser. Running `uv run <url> --help` locally may fail if the
script has GPU-only dependencies — reading the file is more reliable.

## The main pattern: dataset → dataset

Most scripts take an input HF dataset ID and push results to an output HF dataset ID:

    hf jobs uv run --flavor l4x1 -s HF_TOKEN \
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/<script>.py \
      <input-dataset-id> <output-dataset-id> [--max-samples N] [--shuffle]

The script adds a `markdown` column to the input dataset and pushes the merged result
to the output dataset ID on the Hub.

## Alternative: directory → directory (bucket variants)

A couple of scripts have `-bucket.py` variants (currently `falcon-ocr-bucket.py` and
`glm-ocr-bucket.py`) that read from a mounted directory and write one `.md` per image
(or per PDF page). Useful with HF Buckets via `-v`:

    hf jobs uv run --flavor l4x1 -s HF_TOKEN \
      -v hf://buckets/<user>/<input>:/input:ro \
      -v hf://buckets/<user>/<output>:/output \
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/<script>-bucket.py \
      /input /output

`ls` the repo to check whether a `-bucket.py` variant exists for the model you want
before assuming it's available.

The `-v` source can also be a **local directory** (`huggingface_hub` ≥ 1.22) — the CLI
syncs it to a private bucket and mounts it, so local images/PDFs need no upload step:

    hf jobs uv run --flavor l4x1 -s HF_TOKEN \
      -v ./pages:/input \
      -v hf://buckets/<user>/<output>:/output \
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/<script>-bucket.py \
      /input /output

Re-runs only sync new/changed files. Local mounts are read-only by default; a local
*output* dir works too with `:rw` — pull results back with the `hf buckets sync`
command the CLI prints at launch.

## Common flags across dataset-mode scripts

Most scripts support: `--max-samples`, `--shuffle`, `--seed`, `--split`, `--image-column`,
`--output-column`, `--private`, `--config`, `--create-pr`, `--verbose`. Read the script's
docstring for the authoritative list — individual scripts may add model-specific options
like `--task-mode`.

## Gotchas

- **Secrets**: pass `-s HF_TOKEN` to forward the user's token into the job.
- **GPU required**: all scripts exit if CUDA isn't available. `l4x1` is the cheapest
  GPU flavor and works for models up to ~3B. Check `hf jobs hardware` for current options.
- **First run is slow**: model download + `torch.compile` / vLLM warmup dominates small
  runs. Cost per doc drops sharply past a few hundred images — test with `--max-samples 10`
  first, then scale.
- **Don't poll jobs**: jobs run async. Submit once, check status later with
  `hf jobs ps` or `hf jobs logs <id>`.
