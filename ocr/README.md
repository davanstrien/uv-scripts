---
viewer: false
tags: [uv-script, ocr, extraction, vision-language-model, document-processing, hf-jobs]
---

# OCR UV Scripts

<a href="https://huggingface.co/uv-scripts"><picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md-dark.svg"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md.svg" alt="Follow uv-scripts on Hugging Face"></picture></a>

> Part of [uv-scripts](https://huggingface.co/uv-scripts): self-contained UV scripts you run on Hugging Face Jobs in one command.

One script per OCR model. Each script runs the model on a GPU with [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs) and writes the text as markdown: as a new column in a Hub dataset, as `.md` files in a Bucket, or as resumable parquet parts (the `-saturate` recipes). A few scripts return JSON from a schema, detect layout regions, or compare the output of two models.

## Quick Start

First, [install the `hf` CLI and sign in](https://huggingface.co/docs/hub/jobs-quickstart). Jobs needs pay-as-you-go credit.

Run [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) on seven scanned pages from [NASA's *Food for Space Flight* booklet](https://huggingface.co/datasets/uv-scripts/ocr-demo). Replace `your-username` with your Hugging Face username:

```bash
hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    uv-scripts/ocr-demo your-username/ocr-demo-results
```

The Job adds a `markdown` column to all seven rows and saves them in `your-username/ocr-demo-results`. Dependency installation and model loading can take a few minutes before OCR starts. The [dataset card](https://huggingface.co/datasets/uv-scripts/ocr-demo) gives the source and licence.

> **Note:** the command needs no flags because the script's [`[tool.hf-jobs]` header](https://huggingface.co/docs/hub/jobs-configuration#define-the-launch-config-in-the-script) sets the GPU, the Docker image and the `HF_TOKEN` secret. The `hf` CLI reads the header from version 1.32. `hf jobs uv run --dry-run <script>` shows the resolved settings. Flags override the header, for example `--timeout 1h` for a larger dataset or `--flavor` for other hardware. **Older CLIs ignore the header without a warning**, and the Job then fails on a CPU without your token. Check with `hf version` and upgrade. If you cannot upgrade, copy the header values from the top of the script (or from `jobs` in [`models.json`](models.json)) as flags. For `glm-ocr.py` that is `--flavor a10g-small --secrets HF_TOKEN --image vllm/vllm-openai:v0.29.0 --python /usr/bin/python3 -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages`.

### Try the same pages as a PDF

The [OCR demo Bucket](https://huggingface.co/buckets/uv-scripts/ocr-demo) holds the original PDF, a seven-page extract that matches the dataset, and the page images. Mount the `demo/` prefix to process only the extract, and create your own Bucket for the results. The header sets the GPU and image, but the `-v` mounts are always flags:

```bash
hf buckets create your-username/ocr-output --private
hf jobs uv run --timeout 15m \
    -v hf://buckets/uv-scripts/ocr-demo/demo:/input:ro \
    -v hf://buckets/your-username/ocr-output/pdf:/output:rw \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr-bucket.py \
    /input /output
```

This writes `food-for-space-flight/page_001.md` through `page_007.md` under the `pdf/` prefix of your output Bucket.

## Use your own documents

**Images in a Hub dataset:** in the Quick Start command, replace the input dataset ID and choose a new output dataset ID. Start with `--max-samples 10`. This limits OCR, but loading the input dataset can still download more rows. The defaults expect a `train` split and an `image` column. Use `--split` and `--image-column` if yours differ. If the input already has a `markdown` column, choose a different `--output-column`, for example `glm_markdown`. Add `--private` for a private output dataset.

**Scans or PDFs on your machine:** put a few images or a short PDF in `./my-scans` for the first run. The recipe processes every supported file in the folder and its subfolders, and every page of each PDF. Create the output folder before you start the Job:

```bash
mkdir -p ./ocr-output
hf jobs uv run --timeout 15m \
    -v ./my-scans:/input -v ./ocr-output:/output:rw \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr-bucket.py \
    /input /output
```

The CLI uploads the local folders to a private Bucket and mounts them in the Job. `:rw` lets the Job write output. The script saves one `.md` file per image, or per PDF page. See [mounting local data](https://huggingface.co/docs/huggingface_hub/en/guides/jobs#mount-local-data).

## Get and check your results

The CLI prints a link to the Job page, which shows status and logs. From the terminal:

```bash
hf jobs inspect JOB_ID
hf jobs logs JOB_ID
```

When the Job completes:

- **Dataset output:** open `https://huggingface.co/datasets/your-username/ocr-demo-results` and compare the images with their `markdown` results.
- **Bucket output:** browse the output Bucket, or download the files with `hf buckets sync hf://buckets/your-username/ocr-output/pdf ./ocr-output`.
- **Local-folder output:** run the `hf buckets sync` command that the CLI printed at launch. The results are not synced back automatically. Images give files such as `page.md`. A PDF gives files such as `report/page_001.md`.

A completed Job does not mean every page worked. Look for empty results and `[OCR ERROR]` markers, and compare a few outputs with their pages before you scale up. Check tables and reading order in particular. Each dataset run also writes a dataset card with the model settings and a command to reproduce the run.

## Pick a model

These are the maintained recipes. Each one has a tested `[tool.hf-jobs]` header, so the Quick Start command works with only the script name changed. The exception is `lift-extract.py`, which also needs `--schema`. The table is sorted by model size, smallest first. Scores are the model authors' own numbers. [OmniDocBench](https://github.com/opendatalab/OmniDocBench) scores document parsing of text, tables and formulas across varied PDF pages. [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench) runs pass/fail unit tests on hard PDF pages.

| Script | Model | Size | Good at | Licence | GPU |
|--------|-------|------|---------|---------|-----|
| [`tesseract-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/tesseract-ocr.py) | [Tesseract 5](https://github.com/tesseract-ocr/tesseract) | classical | Baseline plain text, no GPU, 100+ language packs (`--lang`) | Apache-2.0 | `cpu-upgrade` |
| [`pp-ocrv6.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/pp-ocrv6.py) | [PP-OCRv6](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6) | 1.5M–34.5M | Small detection + recognition pipeline, plain text, 48 languages | Apache-2.0 | `t4-small` |
| [`surya-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/surya-ocr.py) | [Surya OCR 2](https://huggingface.co/datalab-to/surya-ocr-2) | 0.65B | Per-block HTML with boxes and reading order; layout and table tasks; PDFs | modified OpenRAIL-M | `a10g-small` |
| [`lighton-ocr3.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/lighton-ocr3.py) | [LightOnOCR-3](https://huggingface.co/collections/lightonai/lightonocr-3) | 0.8B / 4B | 86.3 olmOCR-Bench (4B), 85.5 (0.8B); `--mode grounding` adds labelled layout boxes, image descriptions and chart tables | Apache-2.0 | `a10g-small` |
| [`glm-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/glm-ocr.py) | [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 0.9B | 94.62 OmniDocBench v1.5 | MIT | `a10g-small` |
| [`paddleocr-vl-1.6.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/paddleocr-vl-1.6.py) | [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | 0.9B | 96.33 OmniDocBench v1.6; six task modes | Apache-2.0 | `a10g-small` |
| [`ovis-ocr2.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/ovis-ocr2.py) | [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) | 0.9B | 96.58 OmniDocBench v1.6; LaTeX and HTML tables | Apache-2.0 | `a10g-small` |
| [`lighton-ocr2.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/lighton-ocr2.py) | [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) | 1B | 83.2 olmOCR-Bench | Apache-2.0 | `a10g-small` |
| [`hunyuan-ocr-1.5.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/hunyuan-ocr-1.5.py) | [HunyuanOCR-1.5](https://huggingface.co/tencent/HunyuanOCR) | 1B | 12 task types, including spotting, charts and translation | [Hunyuan Community](https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE) (excludes EU, UK, South Korea) | `a10g-small` |
| [`dots-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/dots-ocr.py) | [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) | 1.7B | 100+ languages; layout modes | MIT | `a10g-small` |
| [`dots-mocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/dots-mocr.py) | [dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr) | 3B | Eight prompt modes, including SVG from charts | MIT | `a10g-small` |
| [`deepseek-ocr2-vllm.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/deepseek-ocr2-vllm.py) | [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) | 3B | Newer DeepSeek-OCR | Apache-2.0 | `a10g-small` |
| [`unlimited-ocr-vllm.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/unlimited-ocr-vllm.py) | [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) | 3.3B | Markdown with layout boxes (`--strip-grounding` for clean text) | MIT | `a10g-small` |
| [`deepseek-ocr-vllm.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/deepseek-ocr-vllm.py) | [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) | 4B | Five prompt modes, including figure description | MIT | `a10g-small` |
| [`nuextract3.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/nuextract3.py) | [NuExtract3](https://huggingface.co/numind/NuExtract3) | 4B | Markdown, or JSON from a template ([below](#structured-extraction-and-layout)) | Apache-2.0 | `a10g-small` |
| [`qianfan-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/qianfan-ocr.py) | [Qianfan-OCR](https://huggingface.co/baidu/Qianfan-OCR) | 4.7B | 93.12 OmniDocBench v1.5; optional reasoning (`--think`); key-information extraction | Apache-2.0 | `a10g-small` |
| [`olmocr2-vllm.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/olmocr2-vllm.py) | [olmOCR-2-7B](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8) | 7B (FP8) | 82.4 olmOCR-Bench | Apache-2.0 | `a10g-small` |
| [`lift-extract.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/lift-extract.py) | [lift](https://huggingface.co/datalab-to/lift) | 9B | JSON from a schema, from images or multi-page PDFs | modified OpenRAIL-M | `a100-large` |

Start with a model under 2B. Use a larger model only if the output of a small one is not good enough. Check the licence before you use a model: Surya and lift use a modified OpenRAIL-M licence (free for research, personal use and startups under $5M; no competitive use against Datalab's API), and the Hunyuan licence excludes the EU, the UK and South Korea.

**Variants and tools:** `glm-ocr-bucket.py`, `surya-ocr-bucket.py` and `lighton-ocr3-bucket.py` read images and PDFs from a Bucket and write one `.md` per page. `lighton-ocr3-bucket.py` also takes `hf://buckets/...` URLs for input and output directly, so it reads large buckets in batches through the Hub API without mounting them. It reads `.jp2`, skips pages that already have output (so a re-run resumes), and in `--mode grounding` writes a `.json` sidecar with the layout blocks. `surya-ocr-bucket.py` also writes a `.json` sidecar with the layout blocks, resumes from existing JSON, and can mount or copy its input (`--io-mode mount|copy`); the dataset version, `surya-ocr.py`, keeps the blocks in a `surya_blocks` column. `lighton-ocr2-saturate.py` and `ovis-ocr2-saturate.py` are for large runs, and `ocr-vllm-judge.py` compares outputs ([Scaling up](#scaling-up)). `pp-doclayout.py` and `lfm2-extract.py` are described in [Structured extraction and layout](#structured-extraction-and-layout).

Which model is best depends on your documents. The public olmOCR-Bench leaderboard is one command away:

```bash
hf datasets leaderboard allenai/olmOCR-bench
```

To rank models on your own collection, [ocr-bench](https://github.com/davanstrien/ocr-bench) builds a per-collection leaderboard (pairwise VLM judge, optional human checks) with these scripts. [LANGUAGES.md](LANGUAGES.md) lists the language coverage that each model card claims. [`models.json`](models.json) is the machine-readable catalogue: model, size, backend, licence, support level, tested launch config and languages for every script.

### Less supported and unsupported

These scripts stay in the repo but have no header, so copy the complete launch command from each script's docstring, including any `--image`, `--python` and `-e` flags. If the docstring gives no hardware, `--flavor a10g-small --secrets HF_TOKEN` is a reasonable start. Status was checked on HF Jobs on 2026-09-23. The `support_note` field in [`models.json`](models.json) has details.

| Script | Status | Use instead |
|--------|--------|-------------|
| `lighton-ocr.py`, `nanonets-ocr.py`, `paddleocr-vl.py` | works, older model | `lighton-ocr2.py`, `nanonets-ocr2.py`, `paddleocr-vl-1.6.py` |
| `lighton-ocr2-server.py`, `ovis-ocr2-server.py` | works | the matching `-saturate.py` recipe |
| `glm-ocr-v2.py` | works | `glm-ocr.py`, unless you need its resume support |
| `lfm2-vl-extract.py` | works | `nuextract3.py` or `lift-extract.py` |
| `nanonets-ocr2.py` | works on its pinned `vllm/vllm-openai:v0.10.2` image | |
| `falcon-ocr.py`, `falcon-ocr-bucket.py` | works (Falcon-OCR v1; a v1.5 update is in progress) | |
| `abot-ocr.py`, `firered-ocr.py`, `numarkdown-ocr.py` | works, little used | |
| `deepseek-ocr.py` | broken: every row is `None` | `deepseek-ocr-vllm.py` |
| `hunyuan-ocr.py` | broken on current vLLM | `hunyuan-ocr-1.5.py` |
| `paddleocr-vl-1.5.py` | broken: every row is an `[OCR ERROR]` | `paddleocr-vl-1.6.py` |
| `rolm-ocr.py` | broken: no room for the KV cache on a 24 GB GPU | `olmocr2-vllm.py` |
| `jina-ocr-v1.py` | broken on vLLM 0.30 (CC-BY-NC-4.0 model) | |
| `smoldocling-ocr.py` | broken: rows are raw DocTags, not markdown | |

## Common options

Every dataset recipe takes `INPUT_DATASET OUTPUT_DATASET` as positional arguments, so you can usually switch models by changing the script URL. Defaults such as batch size and context length follow each model card. Run `--help`, or read the script source on the Hub, for the full list. Local `uv run <script-url> --help` installs the dependencies first.

| Option | What it does | Notes |
|--------|--------------|-------|
| `--max-samples N` | Process only the first N rows | All recipes. The `-saturate.py` recipes also accept `--limit`. The three `-bucket.py` recipes count input files, not PDF pages |
| `--shuffle`, `--seed` | Shuffle before `--max-samples` for a representative sample (seed default 42) | Not the `-saturate.py` recipes |
| `--split` | Input split (default `train`) | |
| `--image-column` | Input image column (default `image`) | |
| `--output-column` | Output column (default `markdown`; `extraction` for lift and LFM2 extract) | Not the `-saturate.py` recipes or `pp-doclayout.py` (fixed `layout` column) |
| `--overwrite` | Replace the output column if the input already has it. Without it the script stops | Not the `-saturate.py` recipes |
| `--private` | Make the output dataset private | Not the `-saturate.py` recipes |
| `--batch-size` | Images per batch (default 8 or 16 for the OCR recipes) | Not `tesseract-ocr.py`, `pp-ocrv6.py` or the `-saturate.py` recipes |
| `--max-model-len`, `--max-tokens`, `--gpu-memory-utilization` | vLLM engine limits | Most vLLM recipes |
| `--config NAME`, `--create-pr` | Push the output as a named config, as a pull request | Most dataset recipes. Used to compare models in one repo ([Scaling up](#scaling-up)). In `-saturate.py`, `--config` selects the input config |
| `--verbose` | Log resolved package versions | Most recipes |

To compare models on the same pages, run them into one dataset with a separate output column each:

```bash
hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    my-dataset my-dataset --max-samples 100 --output-column glm_markdown
hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr2.py \
    my-dataset my-dataset --max-samples 100 --output-column lighton_markdown
```

Most dataset recipes also record the model and settings in an `inference_info` column. The `-saturate.py` recipes write a `model` column instead and keep run metadata under `data/completions/`.

### Model-specific flags

| Script | Flags |
|--------|-------|
| `tesseract-ocr.py` | `--lang` (for example `eng+fra`), `--psm`, `--oem` |
| `pp-ocrv6.py` | `--model-tier tiny\|small\|medium` |
| `surya-ocr.py` | `--task ocr\|layout\|table`, `--table-mode full\|simple`, `--pdf-column`, `--page-range` |
| `glm-ocr.py` | `--task ocr\|formula\|table` |
| `paddleocr-vl-1.6.py` | `--task-mode ocr\|table\|formula\|chart\|spotting\|seal` |
| `ovis-ocr2.py` | `--keep-image-tags`, `--min-pixels`, `--max-pixels` |
| `lighton-ocr3.py` | `--model lightonai/LightOnOCR-3-4B\|lightonai/LightOnOCR-3-0.8B`, `--mode plain\|grounding`, `--blocks-column`, `--longest-edge` |
| `hunyuan-ocr-1.5.py` | `--task-type` (12 types, default `doc_parse`), `--custom-prompt` |
| `dots-ocr.py` | `--prompt-mode ocr\|layout-all\|layout-only` |
| `dots-mocr.py` | `--prompt-mode` (8 modes). For SVG: `--model rednote-hilab/dots.mocr-svg --prompt-mode svg` |
| `deepseek-ocr-vllm.py` | `--prompt-mode document\|image\|free\|figure\|describe` |
| `deepseek-ocr2-vllm.py` | `--prompt-mode document\|free` |
| `unlimited-ocr-vllm.py` | `--strip-grounding`, `--grounding-column` |
| `qianfan-ocr.py` | `--prompt-mode ocr\|table\|formula\|chart\|scene\|kie`, `--think`. `kie` needs `--custom-prompt` |

For example, key-information extraction with Qianfan-OCR:

```bash
hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/qianfan-ocr.py \
    invoices extracted-fields \
    --prompt-mode kie --custom-prompt "Extract: name, date, total. Output as JSON."
```

## Structured extraction and layout

These recipes return structured data instead of page text.

**[NuExtract3](https://huggingface.co/numind/NuExtract3)** (`nuextract3.py`, 4B, Apache-2.0) does markdown OCR by default. Give it a `--template` or a JSON Schema (`--schema`) and it returns JSON in that shape. Both flags accept inline JSON, a URL or a file path, so you can host a schema once and reuse it. Template field names act as instructions to the model, so name them clearly and check the output on a few examples. `--enable-thinking` helps with hard layouts; it is slower and stores the reasoning in a `<output-column>_reasoning` column.

```bash
hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nuextract3.py \
    receipts extracted \
    --template '{"store": "verbatim-string", "date": "date", "total": "number"}'
```

**[lift](https://huggingface.co/datalab-to/lift)** (`lift-extract.py`, 9B) returns JSON that matches a JSON Schema, which you must pass with `--schema` (inline JSON, a URL or a file path). It also reads multi-page PDFs (`--pdf-column`, `--page-range`) and extracts one result per document. The default Transformers backend (`--method hf`) is the tested path. Its weights use a modified OpenRAIL-M licence, so check the terms.

**[LFM2-1.2B-Extract](https://huggingface.co/LiquidAI/LFM2-1.2B-Extract)** (`lfm2-extract.py`) works on a **text** column, so you can run it after an OCR recipe: OCR turns a page into `markdown`, then this recipe turns the markdown into fields. `--format` selects JSON, XML or YAML.

```bash
hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lfm2-extract.py \
    your-username/ocr-demo-results your-username/ocr-demo-fields \
    --text-column markdown --schema '{"title": "the document title", "date": "any date shown"}'
```

**[PP-DocLayout](https://huggingface.co/PaddlePaddle/PP-DocLayout-L)** (`pp-doclayout.py`, 123M) finds layout regions but does not read text. It writes a `layout` column with a box, a class (text, title, table, figure, formula, header, footer and more) and a score for each region. Use it to filter pages, crop regions for OCR, or prepare training data. `--model-name` selects the L, M, S or plus-L model. The input and output can each be a dataset or an `hf://buckets/...` path. Bucket output is written in resumable parquet shards.

```bash
hf jobs uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/pp-doclayout.py \
    your-dataset your-layout-output --max-samples 10
```

## Scaling up

**Large datasets:** `lighton-ocr2-saturate.py` and `ovis-ocr2-saturate.py` start a vLLM server in the Job and send pages to it with adaptive concurrency. They stream results to the output repo as parquet parts. If a run stops, run the same command again and it skips the rows that are done. A failed page is stored as an error row, and `--retry-errors` tries those rows again. The output layout is different from the other recipes; the script header explains how to read it. [SERVING.md](SERVING.md) compares server mode with offline batches (measured throughput and output parity).

```bash
hf jobs uv run --detach --timeout 4h \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr2-saturate.py \
    your-dataset your-output
```

**A live endpoint for an app or agent:** [Jobs serving](https://huggingface.co/docs/hub/jobs-serving) exposes a model server that stays up until you cancel the Job or it reaches its timeout. The [Unlimited-OCR walkthrough](serving-unlimited-ocr.md) covers setup, requests, and parsing several pages in one request.

**Compare models:** run several recipes into one repo with `--config <name> --create-pr`, then judge the outputs pairwise with `ocr-vllm-judge.py` ([ocr-bench](https://github.com/davanstrien/ocr-bench) automates this):

```bash
hf jobs uv run --timeout 1h \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/ocr-vllm-judge.py \
    your-username/my-bench --from-prs --judge-model Qwen/Qwen3-VL-8B-Instruct --max-samples 50
```

**Python API:** `run_uv_job` does not read the `[tool.hf-jobs]` header, so pass the header values yourself:

```python
from huggingface_hub import get_token, run_uv_job

job = run_uv_job(
    "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py",
    script_args=["input-dataset", "output-dataset", "--max-samples", "10"],
    flavor="a10g-small",
    image="vllm/vllm-openai:v0.29.0",
    python="/usr/bin/python3",
    env={"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages"},
    secrets={"HF_TOKEN": get_token()},
)
```

## Troubleshooting

**The Job runs on a CPU, or cannot push to the Hub.** Your `hf` CLI is older than 1.32 and ignored the header. Upgrade, or pass the header values as flags (see the [Quick Start](#quick-start) note).

**A vLLM recipe crashes at startup with an `nvcc` or `nvrtc` error:**

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
nvrtc: error: failed to open libnvrtc-builtins.so...
```

The Job ran on the default image, which has no CUDA toolkit. This happens with an older CLI, with a legacy recipe, or when you override `--image`. Run it on the `vllm/vllm-openai` image:

```bash
--image vllm/vllm-openai:v0.29.0 --python /usr/bin/python3 -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages
```

Use the tag from the script's header. The Surya recipes use `/usr/local/bin/python3` and `site-packages` instead; copy their header exactly. `unlimited-ocr-vllm.py` needs Baidu's `vllm/vllm-openai:unlimited-ocr` image (`:unlimited-ocr-cu129` on H100 or H200).

**The Job stops before it finishes.** Jobs stop at their timeout. Pass a longer `--timeout`, or use a `-saturate.py` recipe, which can resume.

**Run locally on your own GPU.** Most recipes get vLLM from the Docker image, not from their dependencies. Add the vLLM version from the header tag:

```bash
uv run --with vllm==0.29.0 \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    input-dataset output-dataset --max-samples 10
```

The Surya recipes need `vllm==0.20.1`. `unlimited-ocr-vllm.py` needs an architecture that no stable vLLM wheel has yet, so it runs only inside its image.
