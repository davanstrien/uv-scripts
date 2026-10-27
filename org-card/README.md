---
title: README
emoji: 📚
colorFrom: red
colorTo: indigo
sdk: static
pinned: false
---

# UV Scripts

**Run a data or ML task over a Hugging Face dataset in one command: OCR, transcription, classification, embeddings, deduplication and more.** Run it on your own machine, or on a pay-per-second GPU with [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-overview). For people and coding agents.

Each recipe is one Python file with its dependencies declared inside it ([UV script](https://docs.astral.sh/uv/guides/scripts/)), so it runs straight from its URL: no clone, no virtualenv, no `pip install`. Most recipes read a Hub dataset and write a new one, so one recipe's output can be the next one's input.

## Quickstart

First [install the `hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli#getting-started). **See the recipe families** (no GPU, no account):

```bash
hf datasets ls --author uv-scripts
```

List a family's scripts without downloading them:

```bash
hf download uv-scripts/ocr --repo-type dataset --include "*.py" --dry-run
```

**OCR seven scanned pages on a GPU** from a [NASA booklet](https://huggingface.co/datasets/uv-scripts/ocr-demo). [Sign in](https://huggingface.co/docs/hub/jobs-quickstart) with a token that can write to your account; Jobs needs [pre-paid credits](https://huggingface.co/settings/billing). Replace `your-username` with your Hugging Face username (`hf auth whoami` shows it):

```bash
hf jobs uv run --flavor a10g-small --timeout 15m --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
  uv-scripts/ocr-demo your-username/ocr-demo-results
```

This takes about 4 minutes on an `a10g-small` ($1.00/hour), so about $0.07. Most of that is startup. The Job saves a new dataset with a `markdown` column. Follow the [OCR walkthrough](https://huggingface.co/datasets/uv-scripts/ocr#use-your-own-documents) to use your own scans or PDFs and to check the results.

## Use it with your coding agent

A recipe runs from a URL and documents itself in its header, so a coding agent (Claude Code, Codex, Cursor, …) can pick one and run it with no setup. Paste this to start:

```
List the uv-scripts recipe families and tell me which fit my task:
  hf datasets ls --author uv-scripts
List a family's scripts with:
  hf download uv-scripts/<family> --repo-type dataset --include "*.py" --dry-run
Read the matching family's README on the Hub. Before running a script, read its header and docstring for the arguments and hardware.
Use my username from `hf auth whoami` for output datasets, and always set --timeout.
```

<details>
<summary><b>More prompts: run a job, build a dataset</b></summary>

**Try it now.** Runs a real OCR Job and hands back a dataset:

```
Using uv-scripts, OCR a sample dataset on Hugging Face Jobs. Find my username
with `hf auth whoami`, then run:
  hf jobs uv run --flavor a10g-small --timeout 15m --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    uv-scripts/ocr-demo <my-username>/ocr-demo-results
Then open the output dataset and show me the `markdown` column.
```

**Put it to work.** When you need data for a task:

```
I need a dataset for <my task>. uv-scripts has recipes that create, OCR,
transcribe, classify, deduplicate and embed datasets on Hugging Face. List them:
  hf datasets ls --author uv-scripts
Pick the family that fits, read its README, then read the script's header and docstring: arguments, input and
output types, and the hardware it needs. Start with a small sample if the recipe
has a --max-samples option, choose a --flavor and an explicit --timeout, and run
it with `hf jobs uv run`. Check each step's output before chaining the next.
```

</details>

To teach your agent the rest of the `hf` CLI, install Hugging Face's [`hf` CLI skill](https://huggingface.co/docs/hub/agents-cli) with `hf skills add`.

## Built with uv-scripts

Most recipes tag the dataset card they push with `uv-script`, and many add a *Reproduction* section with the exact command. [386 datasets carry the tag](https://huggingface.co/datasets?other=uv-script) (October 2026), from more than 40 accounts besides this org and its maintainer. Built something? Keep the tag so others can find it, and share the link.

## Browse

Every recipe is in the list below: OCR, detection and segmentation, audio transcription, NER and classification, embeddings and atlas maps, batch LLM/VLM inference, synthetic data and dataset creation. Or browse on **[GitHub](https://github.com/davanstrien/uv-scripts)**. Run `hf jobs hardware` for GPU flavors and prices.
