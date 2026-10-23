# OCR Scripts — Development Notes

Dev notes for the `ocr/` recipes: the **conventions** to follow, the **per-script gotchas** (the
"why" behind each script's quirks), and the **internal tooling**. Runnable examples live in each
script's docstring and `README.md`; benchmark result tables live in `OCR-BENCHMARK.md`.

- [Conventions & invariants](#conventions--invariants) — read before adding/changing a recipe
- [Script status](#script-status) · [Per-script gotchas](#per-script-gotchas)
- [Internal tooling](#internal-tooling) · [Deferred / tracked](#deferred--tracked) · [Change log](#change-log)

---

## Conventions & invariants

Read this before adding or changing a recipe. Each rule maps to a failure we've actually hit; the
planned **self-review skill** (see [Deferred](#deferred--tracked)) just enforces this list.

- **Catalog entry.** Adding or changing a recipe means updating [`models.json`](./models.json)
  (script → model, params, backend, image pins, language claim) and, if the language claim
  changed, [LANGUAGES.md](./LANGUAGES.md). Language fields record what the **model card claims**
  (with an evidence level), never inferred coverage. Hand-maintained for now; if drift becomes a
  problem, the follow-up is generating the README table from the JSON.
- **Model-card fidelity.** Default prompts, sampling and processor settings come from the model
  card or official client — cite the exact source in a comment (e.g. `# glmocr/config.py
  PageLoaderConfig`), or say "card gives none" if upstream is silent. Upstream-locked values stay
  locked; expose only the knobs upstream sanctions. Deliberate deviations go in the docstring with
  the why. Benchmark before overriding a card value that looks wrong (e.g. hunyuan-1.5
  `repetition_penalty` on tables, [#85](https://github.com/davanstrien/uv-scripts-for-ai/issues/85))
  — never silently "improve" it. Docstring/card claims (params, scores, languages) must match the
  card too.
- **Self-contained single file.** Each recipe is one PEP 723 UV script runnable from a raw URL
  (`hf jobs uv run <url>`). No shared *importable local* module (the job env only gets the one file).
  Extra pip deps are fine — **pin them**. A heavy/stable/shared subsystem may become an opt-in *package*
  dep (e.g. bucket I/O → `bucketbag`, [#67](https://github.com/davanstrien/uv-scripts-for-ai/issues/67)),
  never a local import. Recipes = inline; internal tooling may share freely.
- **GPU + output.** Check `torch.cuda.is_available()` and exit clearly if absent. Write to the Hub
  (`push_to_hub`) or a bucket (`-v hf://…`), never bare local paths (Jobs disk is ephemeral; local *input*
  dirs mount via `-v ./dir:/mnt`, synced to a bucket automatically).
- **vLLM image + fail-fast preflight.** If the model's arch isn't in a stable vLLM wheel, **omit
  `vllm`/`torch` from deps** and run on the pinned `vllm/vllm-openai` image via
  `--image … --python … -e PYTHONPATH=…`; add a preflight that `sys.exit(1)`s naming the exact flags
  (surya-class — see gotchas). Pinned-image scripts: `surya-ocr` (`:v0.20.1`, **site-packages**),
  `nanonets-ocr2` (`:v0.10.2`), `unlimited-ocr` (`:unlimited-ocr`), `deepseek-ocr2`/`glm-ocr` (nightly).
- **Pins are temporary.** An image/version pin (`:v0.10.2`, `:v0.20.1`, a nightly, `surya-ocr==0.20.0`,
  …) is a workaround for a *current* ecosystem gap — a decode regression, an arch not yet in a stable
  wheel, a resolver backtrack. In the recipe, record **why** the pin exists and **what would loosen it**
  (e.g. "move back to the default image when a newer vLLM ships a Qwen2.5-VL decode fix"; "drop the
  nightly once the arch lands in a stable release"). Re-test periodically and relax when the gap closes —
  that's what the `bump-vllm-pins` skill is for; prefer floors over exact pins once you can.
- **Env guards on the bare image.** Set `VLLM_USE_FLASHINFER_SAMPLER=0` (and `VLLM_USE_DEEP_GEMM=0`
  for nightly vLLM) **before** importing `vllm` — both JIT paths need `nvcc`, which the bare uv image lacks.
- **Context-length invariant.** `--max-tokens` ≤ `--max-model-len` ≤ the model's real max context
  (`config.json` `max_position_embeddings`; VLMs → `text_config`, mind `rope_scaling`). vLLM refuses to
  start if `max_model_len` is over (we don't set `VLLM_ALLOW_LONG_MAX_MODEL_LEN`); output can't fit if
  `max_tokens` > `max_model_len`. Check:
  `curl -s https://huggingface.co/<model>/raw/main/config.json | python -c "import json,sys;c=json.load(sys.stdin);t=c.get('text_config',c);print(t.get('max_position_embeddings'),t.get('rope_scaling'))"`.
- **Output-column collision guard.** Every recipe that adds an output column calls
  `ensure_output_columns_free(dataset, [cols], overwrite)` (or the inline sink guard for pp-*) so it
  fails fast instead of duplicating/clobbering an input column; `--overwrite` opts into replacing it.
  Default `--output-column` is `markdown` (never a bare `text`). ([#66](https://github.com/davanstrien/uv-scripts-for-ai/pull/66))
- **Bound large images.** Full-page recipes cap input pixels / resize, or size `max_model_len` to fit —
  a 7–9 MP page is ~14k image tokens. Prefer bounding the input (deterministic) over a giant context;
  don't auto-size `max_model_len` from images (it's fixed at engine init, before images are seen).
- **Dep sanity.** No stale version *caps* that drag a transitive lib back — e.g. `pyarrow<18` forced an
  old `datasets` lacking the `Json` feature → `load_dataset` crashed (the glm-ocr bug). Use floors, not ceilings.
- **Error signalling (known gap).** Scripts currently write sentinels (`[OCR ERROR]`, `[SURYA GENERATE
  ERROR]`) *into* the output column, so partial failures are silent. A companion `ocr_error` status
  column is the deferred fix (see [Deferred](#deferred--tracked)).

---

## Script status

Legend: ✅ production-ready · ⚠️ works only with a required pinned image · 🧪 experimental/on-hold.
"+image" = needs a `--image vllm/vllm-openai:<tag>` override (not the default uv image).

| Script | | Backend | Flavor | Note |
|--------|--|---------|--------|------|
| `deepseek-ocr-vllm.py` | ✅ | vLLM (stable) | l4x1 | `NGramPerReqLogitsProcessor` anti-repeat |
| `deepseek-ocr2-vllm.py` | ✅ | vLLM (nightly) | l4x1 | arch needs nightly; `addict`+`matplotlib` deps |
| `lighton-ocr2.py` | ✅ | vLLM | a100-large / l4x1 | resize 1540px; `--max-model-len` 16384 |
| `lighton-ocr3-bucket.py` | ✅ | vLLM `:v0.30.0` | a10g-small | `hf://buckets/...` or dir → `.md` (+ `.json` blocks in grounding); bucket I/O via `bucketbag` v0.3.1 (no mount); `.jp2` via imagecodecs; resume-by-skip; failed pages write nothing |
| `lighton-ocr3.py` | ✅ | vLLM `:v0.30.0` | a10g-small | 4B/0.8B Qwen3.5; resize 2048px; `--mode grounding` → `layout_blocks`; pinned upstream `lightonocr` client dep — see gotcha |
| `paddleocr-vl-1.5.py` | ✅ | transformers | l4x1 | not vLLM (server-only upstream); single-image |
| `paddleocr-vl-1.6.py` | ✅ | vLLM | l4x1 | smart-resize ~1M px; SOTA OmniDocBench |
| `paddleocr-vl.py` | ✅ | vLLM | l4x1 | |
| `dots-ocr.py` | ✅ | vLLM (stable) | l4x1 | `--max-model-len` 32768; no internal resize |
| `dots-ocr-1.5.py` | ✅ | vLLM 0.17.1 | l4x1 | see gotcha (`content_format`, mirror, bbox space) |
| `glm-ocr.py` | ✅ | vLLM (nightly) | l4x1 | `VLLM_USE_DEEP_GEMM=0`; no pyarrow cap |
| `ovis-ocr2.py` | ✅ | vLLM (stable ≥0.22.1) | l4x1 / a10g-small | `gdn_prefill_backend="triton"` (card); card-exact prompt via `enable_thinking=False` template + `llm.generate`; `<img>` region tags filtered by default (`--keep-image-tags`) |
| `glm-ocr-v2.py` | 🧪 | vLLM (nightly) | l4x1 | CommitScheduler incremental — on hold (see Deferred) |
| `nanonets-ocr.py` | ✅ | vLLM | a10g-small | `--max-model-len` 32768 (`--max-tokens` 15000) |
| `nanonets-ocr2.py` | ⚠️+image | vLLM `:v0.10.2` | a10g-small | Qwen2.5-VL ≥0.11 regression → pure `!` |
| `unlimited-ocr-vllm.py` | ✅+image | vLLM `:unlimited-ocr` | l4x1 | single-image; multi-page → serve |
| `surya-ocr.py` | ✅+image | vLLM `:v0.20.1` | l4x1 | offline backend inject; site-packages PYTHONPATH |
| `surya-ocr-bucket.py` | ✅+image | vLLM `:v0.20.1` | l4x1 | bucket I/O; pin `surya-ocr==0.20.0` |
| `lift-extract.py` | ✅ | hf / vLLM | a100-large | schema-constrained extraction; naming gotcha |
| `nuextract3.py`, `lfm2-extract.py`, `lfm2-vl-extract.py` | ✅ | vLLM | l4x1 | structured extraction |
| `hunyuan-ocr.py` | ✅ | vLLM | l4x1 | **1.0, revision-pinned** (root repo became 1.5 in-place); `transformers<5.13` cap — see gotcha |
| `hunyuan-ocr-1.5.py` | ✅ | vLLM | l4x1 | tracks repo root (=1.5); task-locked prompts; `transformers<5.13` cap — see gotcha |
| `rolm-ocr.py`, `smoldocling-ocr.py`, `numarkdown-ocr.py`, `qianfan-ocr.py`, `firered-ocr.py`, `abot-ocr.py`, `falcon-ocr.py`, `olmocr2-vllm.py`, `dots-mocr.py` | ✅ | vLLM | varies | see `README.md` for flags |
| `pp-ocrv6.py`, `pp-doclayout.py` | ✅ | PaddleOCR / PaddleX | l4x1 | classical det+rec; dataset **or** bucket I/O |

**License note:** Surya and `lift` ship code as Apache-2.0 but **weights under a modified OpenRAIL-M**
(research/personal/<$5M, no competitive use vs Datalab's API) — surfaced in each docstring + card.
HunyuanOCR (1.0 + 1.5) is under the [Tencent Hunyuan Community License](https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE)
(territory excludes EU/UK/South Korea; standard across Tencent's Hunyuan releases) — surfaced in each docstring + card.

---

## Per-script gotchas

Only the scripts with load-bearing quirks; the rest are unremarkable (`README.md` covers flags).

### `surya-ocr.py` / `surya-ocr-bucket.py` — pinned image + site-packages path
Surya-2 (`datalab-to/surya-ocr-2`, 650M, `qwen3_5`) needs its **known-good** vLLM build, and the
`:v0.20.1` image puts python at `/usr/local/bin/python3` and libs at
**`/usr/local/lib/python3.12/site-packages`** (not the usual `dist-packages`) — wrong path →
`No module named 'vllm'` → 0/5. It can't Docker-in-Docker Surya's normal server, so it **injects an
in-process `OfflineVLLMBackend`** into `SuryaInferenceManager` (subclassing Surya's `Backend` ABC) and
reuses Surya's own prompts/`scale_to_fit`/HTML+bbox parsing so the offline path matches the server.
`mm_processor_kwargs={min_pixels:3136, max_pixels:6291456}`, `max_model_len=18000`, `logprobs=1` →
per-block `confidence`. Writes two columns (`--output-column` markdown + `surya_blocks` JSON). Never
name the file `surya.py` (shadows the package). The recipe now **fails fast** if `vllm` isn't importable,
naming the required flags. **Bucket variant:** pin `surya-ocr==0.20.0` (loosening it, or adding
`huggingface-hub>=1.6.0`, lets uv backtrack to a surya without `surya.inference`); **copy beats mount**
for bucket reads (FUSE `rglob` is ~26× slower on a 38k-file bucket; mount also hit a transient CSI flake);
`.jp2` via an `imagecodecs` fallback (Pillow lacks OpenJPEG); resume-by-skip on the output `.json`.

### `nanonets-ocr2.py` — pinned `:v0.10.2` image
Nanonets-OCR2-3B is **Qwen2.5-VL**, which has a vLLM **≥0.11 decode regression** (outputs pure `!` on
every page) — [vllm#27775](https://github.com/vllm-project/vllm/issues/27775). 0.9.2/0.10.1/**0.10.2**
are known-good. Not context length (still `!` at 32768) and not torch.compile. Pip-pinning `vllm==0.10.2`
clashes with modern `transformers` (old tokenizer API), so run on the **`:v0.10.2` image** (ships a
consistent vLLM 0.10.2 + transformers 4.56.1); `vllm`/`torch` omitted from deps. `--max-model-len` 32768
(the 15000 `--max-tokens` can't fit 8192). Re-test the default image when a newer vLLM ships a Qwen2.5-VL
decode fix.

### `dots-ocr-1.5.py` — `content_format="string"` + resized-bbox space
Must pass `chat_template_content_format="string"` to `llm.chat()` — the model's `tokenizer_config.json`
template expects string content; without it you get ~1 token then EOS (empty output). The v1.5 weights
aren't on HF from the authors — **mirrored to `davanstrien/dots.ocr-1.5`** from ModelScope (MIT-based).
Layout bboxes are in the **resized** image space (`Qwen2VLImageProcessor.smart_resize`,
`max_pixels=11,289,600`, `factor=28`); map back with:
```python
import math
def smart_resize(h, w, factor=28, min_pixels=3136, max_pixels=11289600):
    h_bar, w_bar = max(factor, round(h/factor)*factor), max(factor, round(w/factor)*factor)
    if h_bar*w_bar > max_pixels:
        beta = math.sqrt((h*w)/max_pixels); h_bar, w_bar = math.floor(h/beta/factor)*factor, math.floor(w/beta/factor)*factor
    elif h_bar*w_bar < min_pixels:
        beta = math.sqrt(min_pixels/(h*w)); h_bar, w_bar = math.ceil(h*beta/factor)*factor, math.ceil(w*beta/factor)*factor
    return h_bar, w_bar
# orig_x = bbox_x * (orig_w / w_bar);  orig_y = bbox_y * (orig_h / h_bar)
```
(Same `smart_resize`/`max_pixels=11.29M` applies to `dots-ocr.py` v1's processor cap — but the
`content_format="string"` fix does **not**: v1 works with the auto-detected `openai` chat format.)

### `unlimited-ocr-vllm.py` — dedicated image, single-image batch
Baidu `baidu/Unlimited-OCR` (3.3B, DeepSeek-OCR descendant); arch is in **no stable vLLM wheel**, so it
runs on **`vllm/vllm-openai:unlimited-ocr`** (`:unlimited-ocr-cu129` on Hopper), standard `/usr/bin/python3`
+ `dist-packages`. `NGramPerReqLogitsProcessor` (re-exported via `unlimited_ocr`), prompt
`<image>document parsing.`, `limit_mm_per_prompt={"image":1}`, `--strip-grounding` drops `<|det|>`/`<|ref|>`.
**Batch recipe stays single-image**; multi-page is finicky offline (one `<image>` per page; degrades on
hard scans) and belongs to **serving** — both engines read clean multi-page docs, but **SGLang is more
robust on hard/degraded scans**. Serving setup (SGLang pin `lmsysorg/sglang:v0.5.10.post1`, a100+flashinfer
— HF `h200` nodes fail with `CUDA error 802`) is in `serving-unlimited-ocr.md`.

### `lift-extract.py` — naming + backends
Datalab `lift` (9B, Qwen3.5), schema-constrained image/PDF → JSON; the only recipe ingesting PDFs directly.
**Must not be named `lift.py`** (shadows the installed `lift` package → ImportError). Two in-process backends
via `--method`: `hf` (default image, plain `model.generate`) and `vllm` (needs the `vllm/vllm-openai` image;
reproduces lift's own recipe: `mm_processor_kwargs={min_pixels:3136,max_pixels:861696}`, guided JSON schema,
`temperature=0.0,top_p=0.1,max_tokens=12384`). Pin `--model datalab-to/lift` via the `MODEL_CHECKPOINT` env
(settings read env at import).

### `deepseek-ocr-vllm.py` / `deepseek-ocr2-vllm.py`
v1 uses the official offline pattern (`llm.generate()` + `NGramPerReqLogitsProcessor` for repetition).
**Known bug** (hit on vLLM *nightly*, 2026-02-12; unverified on the stable wheels v1 now resolves):
some aspect ratios trip `images_crop dim[2] expected 1024, got 640` (gundam-mode
default vs a validator expecting 1024²) — hit 2/10 on `ufo-ColPali`, aspect-ratio dependent, no upstream
issue filed ([vllm#28160](https://github.com/vllm-project/vllm/issues/28160) is the related request). v2
needs **nightly** vLLM (`DeepseekOCR2ForCausalLM` not in stable) + `addict`/`matplotlib` (its HF custom
code), plus `limit_mm_per_prompt={"image":1}`.

### `hunyuan-ocr.py` / `hunyuan-ocr-1.5.py` — upstream replaced the repo root in-place
On 2026-07-06 Tencent pushed HunyuanOCR-1.5 **into the same repo** `tencent/HunyuanOCR` (1.5 at root,
1.0 archived under `v1.0/`, DFlash draft under `dflash/`, **no 1.0 tag or branch**). vLLM can't load a
repo subfolder, so `hunyuan-ocr.py` pins the last 1.0 commit by `revision` (`f6af82ee…`) — that pin is
the 1.0 *identity*, never loosen it to `main`; 1.5 is its own recipe (`hunyuan-ocr-1.5.py`, tracks root,
has `--revision` as insurance against the next in-place swap). Separate breakage, both scripts: stable
vLLM ≤0.24.0's `hunyuan_vl_image.py` does a string-key `AutoImageProcessor.register(...)` which
transformers 5.13 rejects (`'str' object has no attribute '__module__'` → "architectures failed to be
inspected" at engine init) — hence the `transformers<5.13` cap in both; drop it when the vllm#47872 fix
ships in a stable wheel. 1.5 prompts are task-locked (12 types, Chinese wording, from the official
client's `hunyuan_tasks.py`) and sampling is card-locked (temp 0.0, rep-penalty 1.08) — don't
"improve" either; upstream observed hand-tweaked prompts silently degrade quality.

### `glm-ocr.py`
Chatty on blank pages / can emit degenerate repeats — that's **model quality, not a crash**; don't
re-debug it as a recipe bug. (The actual historical crash was the `pyarrow<18` cap — see Conventions.)

### `lighton-ocr3.py`
Covers the two Qwen3.5 variants (`LightOnOCR-3-4B` default, `-0.8B`); **not** `LightOnOCR-3-1B`, which keeps
the LightOnOCR-2 Pixtral arch (needs `transformers` ≤5.16 for `PixtralRotaryEmbedding`). Image pinned to
**`vllm/vllm-openai:v0.30.0`** because upstream's `pyproject.toml` says vLLM 0.27.1 makes the models loop and
0.30.0 is tested — loosen after re-checking outputs on a newer tag. Imports parser, `fit` (2048px) and
`fix_escaped_dollars` from LightOn's own client (`lightonocr`, not on PyPI) via a commit-pinned GitHub
archive URL, so post-processing matches upstream byte for byte. Sampling = model card vLLM example
(temp 0.2, **top_p 0.9**, max_tokens 4096); LightOn's client `models.py` uses top_p 1.0 instead. Grounding mode writes `to_markdown(blocks)`
to the output column and the parsed blocks JSON to `--blocks-column`; the card says prompts other than
`""`/`grounding` are out of distribution, so no custom prompt flag. Very small type after the 2048
resize (≲15 px line height) can give empty or looping output; a higher `--longest-edge` helps some pages.
**Bucket sibling** (`lighton-ocr3-bucket.py`): input/output are `hf://buckets/...` URLs (bucketbag: list → batched
download → `put_files`) or directories. Use URLs for big buckets: a `-v` mount of `biglam/britannica` (~1 TB)
sat in RUNNING with no log output for 21 min (2026-10-22), while the same pages via bucketbag ran fine.
Prefixes get a trailing `/` (bucketbag matches string prefixes). A PDF counts as done once its `.complete.<mode>` marker exists (written with its last page; grounding needs a grounding marker, plain takes either); a re-run without it OCRs only the missing pages. `--overwrite` does not delete anything first: a page that fails keeps its old output, and the run lists those inputs. PDF pages render one at a time and go to vLLM in windows of `--batch-size` pages, so a long PDF never sits in memory.

### `lighton-ocr2.py`
The original breakage was **not** vLLM — it was a dead `HF_HUB_ENABLE_HF_TRANSFER=1` (the `hf_transfer`
package is gone), which surfaced as "Can't load image processor". Removed. Pixtral ViT + Qwen3, RLVR-trained,
resize 1540px @200 DPI, `--max-model-len` 16384. `paddleocr-vl-1.5.py` uses the **transformers** backend
(single-image) because PaddleOCR-VL only supports vLLM in server mode.

---

## Internal tooling

Not user recipes — benchmark/eval infra. **Result tables + validation history → `OCR-BENCHMARK.md`.**

### `ocr-bench-run.py` — coordinator
Launches N OCR models on the same dataset, each pushing to a shared repo as a separate config via
`--config/--create-pr`. Eval separately with `ocr-vllm-judge.py` / `ocr-elo-bench.py`. Registry (4 models):
`glm-ocr`, `deepseek-ocr` (auto `--prompt-mode free`), `lighton-ocr-2`, `dots-ocr`; each has a `default_args`.
```bash
uv run ocr-bench-run.py source-dataset --output my-bench --max-samples 50
uv run ocr-bench-run.py --list-models              # registry table
uv run ocr-bench-run.py ... --models glm-ocr dots-ocr --dry-run
uv run ocr-elo-bench.py my-bench --from-prs --mode both   # eval from PRs, no merge
```

### `ocr-vllm-judge.py` — offline vLLM jury judge
Pairwise OCR-quality comparisons via vLLM offline `LLM()`; jury mode (multiple models vote, majority
aggregation) with **0 parse failures** (structured output via the `StructuredOutputsParams` → `GuidedDecodingParams`
→ prompt shim). Prefer over the API judge (`ocr-jury-bench.py`) for batch eval — no rate limits, reproducible.
`--from-prs` loads configs from open PRs without merging; `--save-results REPO` persists comparisons/leaderboard/metadata.
A100 recommended for jury mode; L4 works for a single 7B judge.
```bash
hf jobs uv run --flavor a100-large -s HF_TOKEN ocr-vllm-judge.py davanstrien/ocr-bench-nls-50 --from-prs \
    --judge-model Qwen/Qwen2.5-VL-7B-Instruct --judge-model Qwen/Qwen3-VL-8B-Instruct --max-samples 50
```

### `ocr-human-eval.py` — blind human A/B
Gradio app for blind A/B with ELO + optional agreement-vs-judge analysis (split-jury comparisons shown
first; round-robin image variety). Resume-safe (atomic JSON per vote).
```bash
uv run ocr-human-eval.py davanstrien/ocr-bench-rubenstein --from-prs \
    --judge-results davanstrien/ocr-bench-rubenstein-judge --max-samples 5
```

**Validation headline** (full tables in `OCR-BENCHMARK.md`): DeepSeek-OCR is consistently #1 across
datasets and eval methods; middle-pack rankings are dataset-dependent; a jury of small models gives 0
parse failures; **Kimi K2.5 (170B)** is the only judge matching the human's #1 (small judges overrate
LightOnOCR-2's commentary style).

---

## Deferred / tracked

- **bucketbag adoption** — evaluate adopting `bucketbag` for the bucket recipes (slim recipes / harden
  bucketbag / find other beneficiaries) → [#67](https://github.com/davanstrien/uv-scripts-for-ai/issues/67).
- **Self-review skill** (spark) — a dev-only skill (sibling to `bump-vllm-pins`) that reviews a recipe/diff
  against the [Conventions](#conventions--invariants) block: context-length, collision guard, vLLM-image +
  preflight, env guards, dep sanity, image bounding, optional Jobs smoke. Enforces that list; build after
  the current work.
- **Error-signalling** — companion `ocr_error` status column (null cell + truncated exception) instead of
  sentinels in the output column, so "read nothing" ≠ "run errored". Touches ~all recipes; deferred.
- **OCR smoke-test dataset** — a tiny curated set (~20–30 images across doc-type/quality/language/layout,
  ground truth where possible) for fast CI-style regression checks after dep bumps. Pairs with the skill.
- **Multi-page batch (Unlimited-OCR)** — an SGLang-server-in-job recipe for robust multi-page at scale
  (single-image vLLM stays the batch default). Gated on a real corpus-scale need + the `h200`/`fa3` infra
  fix; see `serving-unlimited-ocr.md`.
- **ALTO XML export** — from `surya_blocks` (block-level bbox→`HPOS/VPOS/…`, label→`TextBlock`/`Illustration`);
  the surya-ocr-bucket test bucket ships CA's own ALTO `.xml` as a diff target.
- **Incremental uploads** — superseded by HF Buckets / `bucketbag` ([#67](https://github.com/davanstrien/uv-scripts-for-ai/issues/67));
  `glm-ocr-v2.py` keeps the older CommitScheduler resume path for very large jobs today (do not port it — on hold).
- **Leaderboard Space** — public ELO/pointwise view fed by the benchmark datasets. Idea only.
- **MonkeyOCRv2 — evaluated, not added** (2026-07-14). `zenosai/MonkeyOCRv2-S-Parsing` (0.6B) needs the
  [GitHub repo's](https://github.com/Yuliang-Liu/MonkeyOCRv2) custom multi-stage pipeline (`parsing/parse.py`,
  structure detection) — not pip-packaged, pins `vllm==0.11.2`, and the card text says "academic research and
  non-commercial use only" while the metadata tag says `apache-2.0` (conflicting license signals). Revisit if
  it gets packaged / vLLM-native support or the license is clarified.

**Watch:** `deepseek-ocr2` / `glm-ocr` stay on nightly vLLM until their arch lands in a stable release.
The nightly index (`https://wheels.vllm.ai/nightly`) occasionally has transient build issues (e.g. only
ARM wheels) — if a nightly-recipe install fails on resolution, wait and retry before debugging the recipe.

---

## Change log

- **2026-07-29** — added the first two **`-saturate.py` companions**: `lighton-ocr2-saturate.py` and
  `ovis-ocr2-saturate.py`. Same model/prompt/sampling/post-processing as their `-server.py` siblings;
  the driver half (concurrency, retries, output, resume) is the `saturate` package (pinned `>=0.1.1`,
  the allowed *package*-dep pattern — no local imports). Each carries a `SERVING` dict at the top
  (serve flags + client sampling + context-math assert) as the machine-readable tuning prior — the
  runtime-consumed successor to the dropped `[tool.serving]` header idea. Output shape differs from
  `-server.py`: a NEW dataset repo of `{id, markdown, …, error}` parquet rows keyed by input id
  (resume = anti-join on id), not input+column push; failed pages become durable error rows
  (`--retry-errors` heals), NOT `[OCR ERROR]` sentinels — the first recipes closing the
  error-signalling gap noted in Conventions.
- **2026-07-14** — added `ovis-ocr2.py` (`ATH-MaaS/OvisOCR2`, 0.9B Qwen3.5, 96.58 OmniDocBench v1.6,
  Apache-2.0; stable vLLM ≥0.22.1, `gdn_prefill_backend="triton"`, card-exact prompt/postprocessing).
  Smoke-tested green on the default uv image, a10g-small, resolved vLLM 0.25.1 (5/5 pages, tags filtered,
  stamp OK). Evaluated MonkeyOCRv2 alongside it — deferred (see Deferred / tracked).
- **2026-07-08** — HunyuanOCR upstream repo swap: pinned `hunyuan-ocr.py` to the last 1.0 revision +
  added `hunyuan-ocr-1.5.py` (12 task types, locked sampling); `transformers<5.13` cap in both for the
  stable-vLLM HunyuanVL register breakage (vllm#47872). See the hunyuan gotcha above.
- **2026-07-01** — large full-page scan fixes ([#65](https://github.com/davanstrien/uv-scripts-for-ai/pull/65)):
  surya vLLM-missing preflight; `dots` 8192→32768 + `--max-pixels`; `lighton-ocr2` 8192→16384; `glm` dropped
  `pyarrow<18` (→ `datasets` `Json` load crash) + `VLLM_USE_DEEP_GEMM=0` + `--max-pixels`; `pp-ocrv6`
  `--output-column` + collision guard. Then the output-column collision-guard + `--overwrite` sweep across
  the recipes ([#66](https://github.com/davanstrien/uv-scripts-for-ai/pull/66)); `nanonets-ocr` 8192→32768.
- **Earlier** — per-script fixes are recorded in git history + the gotchas above; benchmark runs in `OCR-BENCHMARK.md`.
