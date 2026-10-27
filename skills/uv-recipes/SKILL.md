---
name: uv-recipes
description: "Run self-contained UV-script recipes over Hugging Face datasets on Hugging Face Jobs (`hf jobs uv run SCRIPT_URL`): OCR & document extraction, audio transcription, object detection & segmentation, NER/entity extraction, text & image classification, embeddings & dataset maps, synthetic data, and batch LLM/VLM inference. Each recipe reads a Hub dataset and writes a new one, so recipes chain into pipelines. Use when the user wants to batch-process a dataset at scale, pre-label data for human review, run a model over many images/audio/documents/rows, or build a data pipeline on Hugging Face — locally with `uv run` or on a managed GPU with `hf jobs uv run`. Recipes live at huggingface.co/uv-scripts (one dataset repo per task family); discover them and read a recipe's header before running."
---

# uv-recipes

A recipe is one self-contained Python file (a [PEP 723](https://peps.python.org/pep-0723/) [UV script](https://docs.astral.sh/uv/guides/scripts/)) that runs straight from a URL. Most take `INPUT_DATASET OUTPUT_DATASET`; directory recipes take input and output paths, and training recipes may produce a model. Read the dependency block, docstring and argument definitions to establish what the selected recipe accepts.

## Requires

- **`uv`** — `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **The `hf` CLI** — `curl -LsSf https://hf.co/cli/install.sh | bash -s` (or `uv tool install "huggingface_hub[cli]"`). Authenticate with `hf auth login`, or set `HF_TOKEN`. Jobs is pay-as-you-go (no subscription needed) — it just needs a Hugging Face account with credit.
- **Strongly recommended — install the Hugging Face skills:** `hf skills add`. This installs the **`hf-cli`** skill (the full `hf` surface: jobs, auth, repos, datasets, buckets, webhooks) and stays current via `hf skills update`. The two compose: `uv-recipes` picks and runs a recipe, `hf-cli` drives everything else on the Hub.

## Run a recipe

**Jobs is recommended, not required.** It gives a managed GPU — pay-per-second, no local CUDA — which matters because most recipes need a GPU you may not have locally. Run a recipe on Jobs:

```bash
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
  INPUT_DATASET OUTPUT_DATASET --max-samples 10
```

- `--secrets HF_TOKEN` forwards the user's token so the job can write the output dataset back.
- `--flavor` picks hardware: `cpu-basic`, `t4-small`, `l4x1` (good default for ≤3B vision models), `a10g-large`, `a100-large`. Run `hf jobs hardware` for live prices.
- **Default timeout is 30 min.** For training or large batches add `--timeout 2h`.
- Default image is `astral-sh/uv:python3.12-bookworm`. Some vLLM recipes need `--image vllm/vllm-openai` — **the recipe's docstring has the exact command; read it first.**
- Track a run: `hf jobs logs <job-id>`, `hf jobs ps`, `hf jobs inspect <job-id>`.
- Jobs concepts, hardware flavors, and pricing: https://huggingface.co/docs/hub/jobs

**Or run locally** — the same file works with `uv run <url> INPUT OUTPUT` when your machine has the hardware it needs (usually a CUDA GPU). Inspect the source first: `uv run <url> --help` resolves dependencies, and recipes such as GLM-OCR import GPU libraries before parsing arguments. This can fail on a laptop.

## Discover recipes (no fixed list — read it live)

```bash
# task families — one dataset repo per task
curl -s "https://huggingface.co/api/datasets?author=uv-scripts" | jq -r '.[].id'

# recipes inside a family
curl -s "https://huggingface.co/api/datasets/uv-scripts/ocr/tree/main" \
  | jq -r '.[].path | select(endswith(".py"))'

# read a recipe without installing its dependencies
curl -fsSL https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py
```

Repo name is the task family (e.g. `ocr`, `transcription`, `sam3`, `gliner`, `classification`, `build-atlas`); a few repos are utilities, not recipes. Prefer a smaller model first (0.3–1B on `l4x1`); scale up only if quality demands it.

## Adapt a recipe

A recipe is one readable file, so you can **copy and modify it** instead of only running it as-is — change the prompt, add or rename an output column, swap the model id, add a CLI flag or a row filter, or add a dependency to the PEP 723 block at the top. Fetch it, edit, and run the local copy (`hf jobs uv run` takes a local path or a URL; a local file is packaged into a temporary job repo):

```bash
curl -sLO https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py
# edit glm-ocr.py — prompt, columns, model, deps …
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN ./glm-ocr.py INPUT OUTPUT --max-samples 10
```

Keep the inline dependency block valid (new deps install at runtime); update the docstring if you change behaviour; test on a small input first, using `--max-samples 10` when the recipe supports it.

## Compose a pipeline

Each recipe's output dataset is the next recipe's input (handoff through the Hub): e.g. `dataset-creation` (PDFs→dataset) → `ocr/glm-ocr.py` (→`markdown`) → `deduplication` → `build-atlas` (embed + map). A pipeline can also end in a trained model. For a small single-label text classifier, use the focused [setfit-on-jobs skill](../setfit-on-jobs/SKILL.md): it can prepare an agent-labeled sample, run SetFit on Jobs, and evaluate and reload the model. One pipeline ships pre-assembled as its own skill: `object-detection` carries a `SKILL.md` covering the full no-labels→detector loop (zero-shot teacher labels → validate → convert → train a small Apache-licensed detector) — fetch it from the repo when that is the task.

## Pre-label, then review

A recipe's prediction column is a set of **suggestions, not final labels**. To put a human in the loop, **build a small, task-specific review UI on the spot** — a single-file Gradio app or static HTML that shows each input beside its predicted label with an accept / edit control, writing corrections back to a Hub dataset. Tailor it to the task: image + markdown side-by-side for OCR, image with box overlay for detection, text with highlighted spans for NER. Keep it minimal.

To spend review effort well (active learning), order the queue by informativeness first — by model **uncertainty** (low confidence from a `classification` recipe) or by **diversity** (embed with `build-atlas`, sample to cover the space). Review the most-informative first; optionally fine-tune on the verified labels (the `classification` repo's `train-classifier.py` fine-tunes an encoder from a labelled Hub dataset in one job) and re-predict, repeating until predictions stop changing. A **scheduled job** or **webhook** can re-run pre-labelling as new data lands.

## Rules that keep jobs working

- Jobs disk is ephemeral — write outputs to the Hub, never local paths. The default pattern is **a dataset in, a dataset out** (`push_to_hub`). **Local input files** don't need an upload step: `-v ./scans:/input` syncs the folder to a private bucket and mounts it in the Job (re-runs only sync changed files; `huggingface_hub` ≥ 1.22). For data you rewrite often (mutable, incremental writes, checkpoints, one file per item), use a [storage bucket](https://huggingface.co/docs/hub/storage-buckets) — mount one with `-v hf://buckets/<user>/<bucket>:/mnt`, or read/write `hf://buckets/…` paths via fsspec. See [buckets.md](buckets.md).
- GPU recipes check `torch.cuda.is_available()` and exit clearly if missing — match `--flavor` to the recipe.
- For OCR, `uv-scripts/ocr-demo` is a seven-image input with `train`/`image`; the matching PDF is in `hf://buckets/uv-scripts/ocr-demo/demo`. For larger dataset inputs that support it, test with `--max-samples 10`; check the input split and column names first. For `glm-ocr-bucket.py`, which has no sample-limit flag, use a folder with a few images or a short PDF. Create any local output folder before mounting it with `:rw`.

## Verify the result

Check the final Job status and retrieve the output before reporting success. For dataset output, inspect the expected column and row count; for folder output, use the CLI's printed sync command and check the expected files. GLM-OCR recipes can catch batch failures and exit normally with `[OCR ERROR]` markers, so check for errors or empty output and compare a few results with the inputs. Report any failed pages or batches separately.

## If a recipe fails

Most failures are environment or usage, not recipe bugs — triage before reporting:

- **401 / auth** → add `--secrets HF_TOKEN`, or run `hf auth login`.
- **OOM / CUDA error** → flavor too small; bump `--flavor` (`l4x1` → `a10g-large` → `a100-large`).
- **Hit the 30-min timeout** → add `--timeout 2h`.
- **`nvcc` / engine-init crash** → try the recipe's documented `--image vllm/vllm-openai` fallback (see its header).
- **Model gated / license error** → accept the terms on the model's Hub page.
- **Transient build failure on a nightly wheel** → wait and retry.
- **Check the docs** — the recipe's `--help` / header and the [Jobs docs](https://huggingface.co/docs/hub/jobs) for the exact invocation and known constraints.

If it still fails, reproduce minimally: use `--max-samples 10` when supported, or prepare a small input folder for directory recipes. Use **small public inputs appropriate to the task** (e.g. `uv-scripts/ocr-demo`, seven page images, for OCR recipes; a public audio set for transcription, a public text set for NER) to confirm it's the recipe, not your data.

**Only if that is a genuine, reproducible defect in the recipe itself**, open an issue at https://github.com/davanstrien/uv-scripts/issues — but first **search existing issues** (one per distinct failure) and **confirm with the user**. The report must give a **full reproducer the maintainer can run as-is: a public input dataset** (one appropriate to the task) and the exact command — never the user's private data. **Redact every token/secret** from the command and logs. Include the recipe URL, flavor + image, the error tail, the job ID/URL, and what you already tried. Do **not** open issues for usage errors, flavor/OOM, gated models, or transient infra.
