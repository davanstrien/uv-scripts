# uv-scripts-for-ai

<a href="https://huggingface.co/uv-scripts"><picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md-dark.svg"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md.svg" alt="Follow uv-scripts on Hugging Face"></picture></a>
<a href="https://huggingface.co/davanstrien"><picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-md-dark.svg"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-md.svg" alt="Follow davanstrien on Hugging Face"></picture></a>

> **A UV script is a single Python file that declares its own dependencies inline — a *portable* unit you run with `uv run` where you have the hardware, or hand to `hf jobs uv run` on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) for a GPU. Chain several into a pipeline.**

Each script carries its own dependencies, so people and agents can run one without cloning a repo, making a virtualenv, or installing a `requirements.txt` first.

A **recipe** here is one such script. Most read and write the [Hugging Face Hub](https://huggingface.co/datasets), so one script's output dataset becomes the next one's input.

![Demo: OCR a whole image dataset with one command — no clone, no environment, straight from a Hub URL](ocr/demo.gif)

## Quickstart

Before starting, [install the `hf` CLI and sign in](https://huggingface.co/docs/hub/jobs-quickstart). Jobs needs a Hugging Face account with [pay-as-you-go credit](https://huggingface.co/pricing). Run `hf jobs hardware` for current hardware and prices.

**Try OCR on seven scanned pages** from [NASA’s *Food for Space Flight* booklet](https://huggingface.co/datasets/uv-scripts/ocr-demo). Replace `your-username` with your Hugging Face username; the results will be saved as a new dataset in your namespace:

```bash
hf jobs uv run --flavor a10g-small --timeout 15m --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
  uv-scripts/ocr-demo your-username/ocr-demo-results
```

`--secrets HF_TOKEN` forwards your token so the Job can save the output dataset. The script adds a `markdown` column containing the OCR text. Dependency installation and model loading can take a few minutes. Follow [Get and check your results](ocr/README.md#get-and-check-your-results), or [try the same pages as a PDF from a Bucket](ocr/README.md#try-the-same-pages-as-a-pdf).

**Got local scans or PDFs?** Put a few images or a short PDF in `./my-scans` for your first run, then create the output directory:

```bash
mkdir -p ./ocr-output
hf jobs uv run --flavor a10g-small --timeout 15m --secrets HF_TOKEN \
  -v ./my-scans:/input -v ./ocr-output:/output:rw \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr-bucket.py \
  /input /output
```

The CLI syncs these local folders to a private bucket. Local folder mounts are read-only unless you add `:rw`; the CLI prints the `hf buckets sync` command to retrieve the output after the Job finishes. See the [OCR walkthrough](ocr/README.md#use-your-own-documents) for inputs and results.

**Prefer your own machine?** With [uv](https://docs.astral.sh/uv/getting-started/installation/) and the required hardware installed, use `uv run` with the same script URL and arguments. Most recipes need a CUDA GPU. To inspect a recipe without installing its dependencies, open its [source](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/glm-ocr.py).

## What's a UV script?

A normal Python file with a metadata block at the top that lists its dependencies:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "transformers", "torch"]
# ///
```

Normally, running someone's Python script means cloning their repo, making a virtual environment, and `pip install`-ing a `requirements.txt` first — and if your versions don't match theirs, it can still break. Here the dependencies live inside the file, in that comment block, so `uv` (and `hf jobs uv run`) reads them, installs exactly those versions into a throwaway environment, and runs the file — straight from a URL, with nothing to set up. This is the standard [PEP 723](https://peps.python.org/pep-0723/) inline-script-metadata format; see the [uv scripts guide](https://docs.astral.sh/uv/guides/scripts/) to learn more.

## Why UV scripts

A self-contained, pinned script is easy to run and reuse, for a few reasons:

- **Discrete & single-purpose** — one script, one job. That job can be a two-second transform or a multi-hour fine-tune; either way it's one self-contained unit you pick by reading a header instead of a whole codebase.
- **Self-describing** — the [PEP 723](https://peps.python.org/pep-0723/) dependency block, the docstring, and `--help` tell you what it needs and how to call it.
- **Reproducible** — dependencies are pinned *in the file*, so there's no env drift and no "works on my machine."
- **Composable** — recipes hand off through the Hub (usually a dataset in, a dataset or model out), so you can chain them into a pipeline.
- **Portable** — one self-contained file; run it with `uv run` where you have the hardware (most recipes need a GPU), or `hf jobs uv run` it on a managed GPU.

**Built for agents, too.** Every recipe takes its arguments in the same `input output` order and runs from a URL, so an AI agent can pick a tool from its header and run it with no setup. On Jobs the agent runs in a sandbox: a throwaway disk, access limited to what the token's repo permissions allow, and a cost cap per job — not arbitrary code on your machine. (Hugging Face also ships an [`hf` CLI skill for agents](https://huggingface.co/docs/hub/agents-cli) for driving Jobs from an editor.) This repo also ships a ready-to-use **[`uv-recipes` agent skill](skills/uv-recipes/)** — point your agent at it to discover, run, and adapt recipes.

For few-shot text classification, the focused **[`setfit-on-jobs` skill](skills/setfit-on-jobs/)** prepares a small labeled sample when needed, runs SetFit on Jobs, and verifies the resulting model.

## Recipes

| Domain | What it does | On the Hub |
|---|---|---|
| **ocr** ⭐ | OCR / document → text & structured data — GLM, PaddleOCR-VL, Nanonets, olmOCR, dots, … (30+ models) | [`uv-scripts/ocr`](https://huggingface.co/datasets/uv-scripts/ocr) |
| **vision** | Zero-shot detection & segmentation over image datasets; `object-detection` also ships a [`SKILL.md`](object-detection/SKILL.md) for the full no-labels→trained-detector loop | [`sam3`](https://huggingface.co/datasets/uv-scripts/sam3) · [`object-detection`](https://huggingface.co/datasets/uv-scripts/object-detection) · [`vlm-object-detection`](https://huggingface.co/datasets/uv-scripts/vlm-object-detection) |
| **audio** | Transcription (incl. 102 languages via BuzzASR), speaker diarization & speech translation | [`transcription`](https://huggingface.co/datasets/uv-scripts/transcription) |
| **video** | Caption videos with timestamped events + temporal grounding (Marlin-2B, chunked for long films) | [`video`](https://huggingface.co/datasets/uv-scripts/video) |
| **embeddings** | Embed text/images (auto-batch, prompt-aware); build a searchable Lance vector DB on the Hub | [`embeddings`](https://huggingface.co/datasets/uv-scripts/embeddings) |
| **embeddings & atlas** | Embed a dataset; build an interactive map | [`build-atlas`](https://huggingface.co/datasets/uv-scripts/build-atlas) |
| **data processing** | Filter / dedup / stats over large datasets; turn bucket uploads into optimized Parquet with a webhook | [`dataset-stats`](https://huggingface.co/datasets/uv-scripts/dataset-stats) · [`deduplication`](https://huggingface.co/datasets/uv-scripts/deduplication) · [`data-processing`](https://huggingface.co/datasets/uv-scripts/data-processing) |
| **classification** | Fine-tune an encoder classifier (LFM2.5-Encoder, ModernBERT, …), few-shot train one with SetFit on CPU or GPU, fine-tune or zero-shot label with GLiNER2 (~300M), or zero-shot classify with an LLM | [`classification`](https://huggingface.co/datasets/uv-scripts/classification) |
| **dataset creation** | Turn PDFs / image URLs into Hub datasets | [`dataset-creation`](https://huggingface.co/datasets/uv-scripts/dataset-creation) · [`iiif-tiles`](https://huggingface.co/datasets/uv-scripts/iiif-tiles) |
| **synthetic data** | Generate datasets with LLMs | [`synthetic-data`](https://huggingface.co/datasets/uv-scripts/synthetic-data) |
| **inference** | Run any open LLM / VLM over a dataset | [`vllm`](https://huggingface.co/datasets/uv-scripts/vllm) · [`openai-oss`](https://huggingface.co/datasets/uv-scripts/openai-oss) · [`transformers-inference`](https://huggingface.co/datasets/uv-scripts/transformers-inference) |
| **entity extraction** | Zero-shot NER (GLiNER) and schema-driven structured records (GLiFormer) over text, CPU-friendly | [`gliner`](https://huggingface.co/datasets/uv-scripts/gliner) |
| ***…and more*** | *Training, evaluation, RAG indexing — migrating as they mature* | [`training`](https://huggingface.co/datasets/uv-scripts/training) · [`transformers-training`](https://huggingface.co/datasets/uv-scripts/transformers-training) |

Most recipes now live in this repo; the rest link to the [`uv-scripts`](https://huggingface.co/uv-scripts) Hugging Face org where they run today, and migrate here over time. (each folder mirrors to its Hub dataset repo.)

**What fits here:** any self-contained UV script for data or ML work on the Hub. OCR and dataset work are the current focus, but inference, evaluation, RAG indexing, and **training** (fine-tuning with TRL / `transformers`, producing a model) are all in scope. If it's one pinned script that reads from or writes to the Hub, it belongs.

## Compose a pipeline

Because recipes hand off through the Hub, you can chain them — each step's output dataset is the next step's input. A document-collection pipeline, end to end:

```
PDFs / scans          →   OCR to markdown      →   dedup + stats        →   embed + visualise
dataset-creation          ocr/glm-ocr.py           deduplication            build-atlas
```

Each arrow is a Hub dataset; each box is one `hf jobs uv run` (or `uv run`), and every box runs today from its Hub URL, even before it's migrated into this repo. A pipeline can also end in a *trained model* instead of another dataset. You can write the chain as a shell script, or an agent can generate it — the scripts are the same.

## Portable: run it locally or on Jobs

A recipe is the same file wherever you run it — on a machine with the hardware it needs, or on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) for a managed GPU. Same file, same arguments:

```bash
SCRIPT=https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py

# locally — needs the right hardware (a GPU for most recipes)
uv run $SCRIPT uv-scripts/ocr-demo your-username/ocr-demo-results

# on a managed GPU — pick hardware with --flavor; --secrets forwards your write token
hf jobs uv run --flavor a10g-small --timeout 15m --secrets HF_TOKEN $SCRIPT uv-scripts/ocr-demo your-username/ocr-demo-results
```

Why reach for [Jobs](https://huggingface.co/docs/hub/jobs):

- **Pay by the second** — billed only while the job runs. Run `hf jobs hardware`, or see the [flavors](https://huggingface.co/docs/huggingface_hub/guides/jobs#select-the-hardware) and [pricing](https://huggingface.co/docs/hub/jobs).
- **No infra** — `hf jobs uv run <url>` and you're done. See the [`hf jobs` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-jobs).
- **Hub-native** — read and write datasets, models, and [storage buckets](https://huggingface.co/docs/hub/storage-buckets) directly. Running from the `https://huggingface.co/datasets/uv-scripts/…` URL also attributes usage to the recipe.

## Model licenses

These scripts are orchestration code: they download third-party models from the Hugging Face Hub at runtime and run inference. **This repo does not redistribute any model weights.** Each model you run carries its own license (MIT, Apache-2.0, OpenRAIL-M, and some with non-commercial or other use-based terms); those terms govern your use of the *model*, not this repo's code. **You are responsible for checking each model's license** — on its Hugging Face model card — before using it, especially in production.

## License

The code and documentation in this repository are licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

---

*Recipes mirror to the [`uv-scripts`](https://huggingface.co/uv-scripts) Hugging Face org via GitHub Actions. See [CONTRIBUTING.md](CONTRIBUTING.md) to add one.*
