---
viewer: false
tags:
  - uv-script
  - audio
  - transcription
  - automatic-speech-recognition
  - speaker-diarization
  - multilingual
---

# Transcription

Scripts for transcribing — and diarizing — audio files using HF Buckets and Jobs. English and the major European/Asian languages via Cohere Transcribe; 102 languages via the BuzzASR monolingual Whisper fine-tunes.

## Quick Start

Scripts run directly from their Hub URL — no clone or local checkout needed:

```bash
# 1. Download audio from Internet Archive straight into a bucket
hf jobs uv run \
    -v hf://buckets/user/audio-files:/output \
    https://huggingface.co/datasets/uv-scripts/transcription/raw/main/download-ia.py \
    SUSPENSE /output

# 2. Transcribe — audio bucket in, transcript bucket out
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    -e UV_TORCH_BACKEND=cu128 \
    -v hf://buckets/user/audio-files:/input:ro \
    -v hf://buckets/user/transcripts:/output \
    https://huggingface.co/datasets/uv-scripts/transcription/raw/main/cohere-transcribe.py \
    /input /output --language en --compile

# Or: transcribe + figure out WHO said what (speaker diarization + timestamps)
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    -e UV_TORCH_BACKEND=cu128 \
    -v hf://buckets/user/audio-files:/input:ro \
    -v hf://buckets/user/transcripts:/output \
    https://huggingface.co/datasets/uv-scripts/transcription/raw/main/moss-transcribe-diarize.py \
    /input /output --emit-txt
```

No download/upload step. Buckets are mounted directly as volumes via [hf-mount](https://github.com/huggingface/hf-mount).

Audio already on your machine? Mount the folder directly — `-v ./audio:/input` syncs it to a private bucket and mounts it in the job (re-runs only sync changed files; `huggingface_hub` ≥ 1.22), so you can skip the `download-ia`/bucket step.

> **Local dev**: if you've cloned this repo, swap the URL for the local filename (e.g. `cohere-transcribe.py /input /output ...`).

## Scripts

### Transcription

| Script | Model | Backend | Output | Speed |
|--------|-------|---------|--------|-------|
| `cohere-transcribe.py` | Cohere Transcribe (2B) | transformers | `.txt` | 161x RT (A100) |
| `cohere-transcribe-vllm.py` | Cohere Transcribe (2B) | vLLM nightly | `.txt` | 214x RT (A100) |
| `granite-turboctc-transcribe.py` | [Granite Speech 5.0 TurboCTC](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) (470M, English) | transformers (pinned `main`) | `.txt` (lowercase, no punctuation) | 594x RT wall / ~5000x GPU-only (A100); 504x RT wall (A10G) |
| `buzzasr-transcribe.py` | [BuzzASR](https://huggingface.co/BuzzASR) (102 monolingual Whisper-large-v3 fine-tunes, 1.55B) | transformers | `.txt` (mostly lowercase, no punctuation) | 20x RT (L4, sequential) / 51x RT (L4, chunked) |
| `easytranscriber-transcribe.py` | Cohere Transcribe 2B (default) or Whisper variants | [easytranscriber](https://github.com/kb-labb/easytranscriber) | JSON word timestamps (+ optional `.txt` / `.srt`) | 42.9x RT (L4) |
| `moss-transcribe-diarize.py` | [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) (0.9B) | transformers (remote code) | JSON speaker segments `{start, end, speaker, text}` (+ optional `.txt` / `.srt`) | 3.2x RT (A10G, 74-min file) |
| `moss-transcribe-diarize-server.py` | [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) (0.9B) | in-job sgl-omni server | JSON speaker segments (+ optional `.txt`) | 47.4x RT aggregate (A100, 6 streams) |
| `nemotron-transcribe.py` | [Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) (30B MoE, ~4B active, 40 languages) | in-job vLLM 0.32 + NeMo Speech server | `.txt` (+ JSON word timestamps, speakers and diarization with `--timestamps`) | 116x RT aggregate (RTX PRO 6000, 10 files) + ~8 min fixed setup |

**`cohere-transcribe.py`** (recommended for plain text) — uses `model.transcribe()` with automatic long-form chunking, overlap, and reassembly. Stable dependencies.

**`cohere-transcribe-vllm.py`** — experimental vLLM variant. Faster but requires nightly vLLM and has minor duplication at chunk boundaries.

**`granite-turboctc-transcribe.py`** — when you have **a lot of English audio** and want raw text fast. Encoder-only CTC (one forward pass + argmax, no autoregressive decoding), so it cannot loop or hallucinate and the GPU is essentially idle: 5 hours of 1940s radio ran in 3.2 s of GPU time on an A100 (~30x Cohere), with the job's wall time dominated by mp3 decoding and model load. Output is lowercase without punctuation or casing, and quality is a step below Cohere on hard audio (~8.5% word disagreement with Cohere's transcripts on the same episodes, no ground truth). Files are fed whole by default — block attention makes cost linear in length and a 30-min file needs ~2.5 GB — with `--chunk-seconds 30` for hour-plus files or a 24 GB card. Needs `transformers` from `main` (support landed 2026-08-25, pinned to a commit) until the next release; English only.

**`buzzasr-transcribe.py`** — when the audio is in **a language Cohere doesn't cover** (Armenian, Amharic, Georgian, Khmer, Lao, Sorani Kurdish, Wolof, Yoruba, …): one model per language for 102 FLEURS languages. [BuzzASR](https://lemn-lab.github.io/buzz-asr/) (Findings of EMNLP 2026, MIT) fine-tunes Whisper-large-v3 once per FLEURS language, with a native tokenizer for the languages Whisper's tokenizer fragments; the authors report lower CER than Whisper-large-v3 zero-shot on 89/102 languages and the lowest of any open system on 31. Pass `--language armenian` (or the ISO code `hy`) and the matching `BuzzASR/<language>` checkpoint is loaded: every card tags both the name and the code, so the recipe resolves either with a tag filter on the org at run time (no table to maintain; new languages just work). Finding a model by hand is the same one-liner: `hf models list --author BuzzASR --filter ckb --format quiet` → `BuzzASR/sorani-kurdish`, or `--search kurdish` by name. The language prompt is baked into each model, so the model is never asked to detect the language. Default long-form decoding is Whisper's sequential algorithm (timestamps place each window, degenerate windows are re-decoded at higher temperature), which skipped no windows in our tests but is ~2.5x slower than `--long-form chunked` (the transformers pipeline), whose overlap merge drops text: on 3 h of Dutch audiobook the chunked run produced 14% fewer words than sequential, with whole sentences missing. Generation uses the authors' settings (greedy, no repeated 3-gram, repetition penalty 1.2), which also stop a genuinely repeated phrase from being written twice inside one 30 s window. Output is normalised like the training data (mostly lowercase, no punctuation); for a language Whisper already covers well, `openai/whisper-large-v3` through the same recipe gives punctuated, cased text of the same completeness (see Benchmarks), so reach for BuzzASR where Whisper is weak. The recipe rebuilds each model's `generation_config.json` on load: the native-tokenizer checkpoints ship it flagged `_from_model_config`, which makes transformers drop the language prompt and timestamp ids and breaks long-form generation. `--model openai/whisper-large-v3` runs the zero-shot base through the same loop as a control.

**`easytranscriber-transcribe.py`** — when you need **word-level timestamps** (subtitles, search indexing, forced alignment). Runs VAD → ASR → wav2vec2 emissions → forced alignment. Defaults to the Cohere backend so you get the same model as the other scripts with alignment on top; swap to `--backend ct2` + a Whisper model for languages Cohere doesn't cover (e.g. Swedish via `KBLab/kb-whisper-large`).

**`moss-transcribe-diarize.py`** — when you need **who said what**. Joint transcription + speaker diarization + timestamps in one generation pass (no separate ASR/diarization/alignment stages). 128k context handles up to ~90 min of audio per file internally, so files are never pre-chunked and the anonymous speaker labels (`[S01]`, `[S02]`, ...) stay consistent across the whole recording. English + Chinese; also accepts video containers (mp4, mkv, ...); supports hotword biasing for names/jargon.

**`moss-transcribe-diarize-server.py`** — same model at **~12x the throughput** for many files: serves it behind sglang-omni inside the job and posts files concurrently (continuous batching). Splits long tapes into ≤55-min clips to fit the context window (speaker labels reset between clips, recorded as `part` on each segment) and automatically continues from the last timestamp when the model stops early, reporting `coverage_s` per file. Needs `a100-large` — 24 GB cards OOM on long-tape KV. The `hf jobs run` command lives in the script docstring (server + driver in one job).

**`nemotron-transcribe.py`** — when you want **punctuated text, speaker tags and word timestamps from one model, in 40 languages** (speaker tags in 13, up to 8 speakers), and have enough audio to amortise a fixed ~8 min setup. The model only serves on vLLM 0.32.0 with the NeMo Speech plugin, so the script starts that server inside the job and posts files concurrently: 10 half-hour 1940s radio episodes (4.9 h) took 2.5 min once the server was up, a $0.48 job on `rtx-pro-6000`. The `[tool.hf-jobs]` header sets the flavor and a prebuilt image (`cr.hf.co/davanstrien/nemotron-transcribe`, built in a Job by `nemotron-transcribe-image.sh`) that pins NeMo Speech to one commit and carries FlashInfer kernels already compiled for that GPU. Without them, FlashInfer compiles its kernels at every start (~10 min). `--image vllm/vllm-openai:v0.32.0` makes the script install the stack at runtime instead (~2 min slower). Prompts are the model card's presets (`--style verbatim|clean|speakers`, or `--prompt` with `Context terms: <names>` for jargon), and the card's generation-loop guardrail is applied to every transcript. `--timestamps` adds a JSON per file with word start/end times, speaker tags and diarization segments, from the CTC head bundled in the checkpoint. Files over 55 min are split, and speaker tags reset between parts. On the same episodes its words differ from Cohere's by 9–12%, much of it spelling style (the verbatim preset writes "Mister", "thirty nine"); there is no ground truth for these episodes. The vLLM integration is NVIDIA's development path; the card says it is about 2x slower than their NIM.

#### Options — `nemotron-transcribe.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--style` | `verbatim` | Model-card prompt preset: `verbatim` (fillers kept), `clean` (fillers dropped) or `speakers` (`<spk:N>` tags) |
| `--prompt` | none | Custom user prompt (overrides `--style`), e.g. `Transcribe with punctuation and capitalization. Context terms: <names>.` |
| `--timestamps` | off | Also write JSON with word timestamps, speaker tags and diarization segments (uses the `speakers` prompt) |
| `--concurrency` | 16 | Files in flight against the server |
| `--part-minutes` | 55 | Split longer audio into parts (the model is trained on recordings up to 1 h); speaker tags reset between parts |
| `--server` | `http://127.0.0.1:8000` | Use a server already running there instead of starting one |
| `--nemo-ref` | `main` | NeMo Speech ref to install when the image has no prebuilt stack |
| `--max-files` | all | Limit files to process (for testing) |

#### Options — `cohere-transcribe.py` / `cohere-transcribe-vllm.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--language` | required | en, de, fr, it, es, pt, el, nl, pl, ar, vi, zh, ja, ko |
| `--compile` | off | torch.compile encoder (one-time warmup, faster after) |
| `--batch-size` | 16 | Batch size for inference |
| `--max-files` | all | Limit files to process (for testing) |

#### Options — `granite-turboctc-transcribe.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-seconds` | 0 (whole file) | Cut files into N-second windows; use 30 for hour-plus files or 24 GB cards (2.2 GB peak at batch 32 vs 10.3 GB for four 30-min files) |
| `--batch-size` | 4 | Files (or windows) per forward pass; 32-64 with `--chunk-seconds 30` |
| `--decode-workers` | CPU count | Threads for mp3/wav decoding, overlapped with model load and inference |
| `--max-files` | all | Limit files to process (for testing) |

#### Options — `buzzasr-transcribe.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--language` | required | Language name (`dutch`, `sorani-kurdish`) or ISO code (`nl`, `ckb`), resolved via the org's model tags; `--list-languages` prints all of them live from the Hub |
| `--model` | `BuzzASR/<language>` | Any Whisper checkpoint on the Hub (e.g. `openai/whisper-large-v3` as a zero-shot control) |
| `--long-form` | `sequential` | `sequential` (Whisper's own long-form decoding with timestamps + temperature fallback; skipped no windows in our tests) or `chunked` (transformers pipeline, 30 s windows with 5 s overlap; ~2.5x faster, drops text at bad merges) |
| `--batch-size` | 16 | Files per generate call (sequential) or 30 s windows per forward pass (chunked) |
| `--decode-workers` | CPU count | Threads for mp3/wav decoding, overlapped with model load |
| `--max-files` | all | Limit files to process (for testing) |

#### Options — `easytranscriber-transcribe.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--language` | required | ISO 639-1 code. Cohere supports the same 14 languages as above; ct2/hf support any Whisper language |
| `--backend` | `cohere` | `cohere`, `ct2` (CTranslate2 Whisper, fastest for Whisper), or `hf` (transformers) |
| `--transcription-model` | Cohere 2B / distil-whisper-large-v3.5 | HF model ID; override to use KB-Whisper, Whisper-large-v3, etc. |
| `--emissions-model` | per-language default | wav2vec2 for forced alignment: en→`wav2vec2-base-960h`, sv→`voxrex-swedish`, else→`facebook/mms-1b-all` |
| `--vad` | `silero` | `silero` (no auth) or `pyannote` (requires accepting terms + HF_TOKEN) |
| `--tokenizer-lang` | derived from `--language` | NLTK Punkt language name for sentence tokenization |
| `--emit-txt` | off | Also write `.txt` transcripts alongside the JSON alignments |
| `--emit-srt` | off | Also write `.srt` subtitles derived from alignment segments |
| `--batch-size-features` | 8 | Feature-extraction batch size |
| `--batch-size-transcribe` | 16 | ASR batch size (where backend supports it) |
| `--max-files` | all | Limit files to process (for testing) |

#### Options — `moss-transcribe-diarize-server.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--concurrency` | 4 | Concurrent transcription requests (KV-cache bound; 6 works on A100) |
| `--part-minutes` | 55 | Clip length for splitting long audio (longest that fits the context window). Speaker labels reset between clips |
| `--server` | `http://127.0.0.1:8000` | sgl-omni server URL (in-job localhost by default) |
| `--request-timeout` | 3600 | Per-request timeout in seconds |
| `--emit-txt` | off | Also write `.txt` transcripts |
| `--max-files` | all | Limit files to process (for testing) |

#### Options — `moss-transcribe-diarize.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--max-new-tokens` | 0 (auto) | Token budget per file. Auto scales with audio duration (min 5120, max 65536). Output JSON sets `"truncated": true` if the budget was hit — re-run with a higher value |
| `--hotwords` | none | Comma-separated terms (names, companies, jargon) appended to the prompt to bias recognition |
| `--prompt` | built-in | Full prompt override (replaces the built-in transcribe+diarize prompt) |
| `--emit-txt` | off | Also write `.txt` transcripts (`[start - end] SPEAKER: text` per line) |
| `--emit-srt` | off | Also write `.srt` subtitles with speaker prefixes |
| `--max-files` | all | Limit files to process (for testing) |

#### Benchmarks

CBS Suspense (1940s radio drama), 66 episodes, 33 hours of audio.

**`cohere-transcribe.py`** (plain text):

| GPU | Time | RTFx |
|-----|------|------|
| A100-SXM4-80GB | 12.3 min | 161x realtime |
| L4 | ~64s / 30 min episode | 28x realtime |

**`buzzasr-transcribe.py`** (Dutch LibriVox audiobook, 4 chapters, 178 min; Armenian CC0 news/talk from the Internet Archive, 3 files, 25 min):

| Run | GPU | Time | RTFx | Peak GPU | Words out |
|-----|-----|------|------|----------|-----------|
| `BuzzASR/dutch`, sequential (default) | L4 | 8.9 min | 20x | 4.5 GB | 27,438 |
| `BuzzASR/dutch`, `--long-form chunked` | L4 | 3.5 min | 51x | 7.5 GB | 23,669 (−14%; sentences dropped at merges) |
| `openai/whisper-large-v3` control, sequential | L4 | 13.4 min | 13x | 4.3 GB | 27,688 (punctuated + cased; 14% word disagreement with BuzzASR; single-file batches, before the batching refactor) |
| `openai/whisper-large-v3` control, chunked | L4 | 4.5 min | 40x | 7.6 GB | 19,925 (−28% vs its own sequential run) |
| `BuzzASR/armenian`, sequential (25 min, 3 files) | L4 | 1.9 min | 13x | 4.1 GB | 2,198 words on the 19.5-min programme, coherent Armenian |
| `openai/whisper-large-v3` control, sequential, same files | L4 | 14.8 min | 2x | 4.5 GB | 582 words on the same programme, garbled + mojibake |

Peak host memory for the 178-min sequential batch was 9.0 GB; with the transformers docs' `logprob_threshold` fallback it was 17.6 GB, because that setting keeps every token's full-vocabulary scores on the host, and the output was identical without it, so the recipe leaves it out. No ground truth for either set: word counts and per-file diffs measure completeness, not accuracy. Two things follow. The chunked pipeline's losses are the pipeline's, not the model's (Whisper-large-v3 loses 28% of its words the same way), so `sequential` is the default for both. And on Dutch, a language Whisper already handles well (the authors report a 1.3x CER gain), zero-shot Whisper-large-v3 with the same sequential decoding was as complete and kept punctuation and casing, so the fine-tunes earn their place on the languages Whisper handles badly: on Armenian (6.8x claimed CER gain) the control produced a quarter of the words, garbled, at 2x realtime because its temperature fallback fired on almost every window.

**`easytranscriber-transcribe.py`** (JSON alignments + optional .txt/.srt; VAD → ASR → wav2vec2 → forced alignment):

| GPU | Time | RTFx | Output |
|-----|------|------|--------|
| L4 | 46.2 min | 42.9x realtime | 66 JSON + SRT + TXT (42,633 segments, 295k words) |

**`moss-transcribe-diarize.py`** (Apollo 11 mission audio, one 74-min multi-speaker tape):

| GPU | Time | RTFx | Output |
|-----|------|------|--------|
| A10G | 23.3 min | 3.2x realtime | 743 segments, 7 speakers, 23k tokens, no truncation |

Long files are generation-bound (the whole diarized transcript is decoded in one pass), so RTFx drops as recordings grow; short clips run far faster. `l4x1` also works — same class of GPU.

**`moss-transcribe-diarize-server.py`** (Apollo 11 mission audio from the [Internet Archive](https://archive.org/details/Apollo11Audio) — the full collection in one job):

| GPU | Audio | Time | RTFx | Cost | Output |
|-----|-------|------|------|------|--------|
| A100 | 174.5 h (103 tapes) | 3.8 h | 47.4x realtime aggregate | $9.46 | 45k speaker segments, 97% coverage |

Result published as [`davanstrien/apollo-11-diarized`](https://huggingface.co/datasets/davanstrien/apollo-11-diarized) with a searchable demo at [`davanstrien/apollo-11-search`](https://huggingface.co/spaces/davanstrien/apollo-11-search).

### Data

| Script | Description |
|--------|-------------|
| `download-ia.py` | Download audio from Internet Archive into a mounted bucket |

## Serve as a live endpoint

MOSS-Transcribe-Diarize can be served as an OpenAI-compatible transcription API on Jobs (see [Serving on Jobs](https://huggingface.co/docs/hub/jobs-serving)). Two verified paths; both expose `/v1/audio/transcriptions` at `https://<job-id>--8000.hf.jobs` (requests need an HF token with read access to your namespace).

**vLLM** (simplest; returns the raw `[start][Sxx]text[end]` transcript). Model support was merged into vLLM on 2026-07-08 — after the v0.24.0 release — so use a commit-pinned nightly image until the next release (then plain `vllm/vllm-openai:latest` works). The image needs the audio extras installed first:

```bash
hf jobs run --detach --expose 8000 --flavor l4x1 -s HF_TOKEN --timeout 2h \
    vllm/vllm-openai:nightly-2c17d33f4291a55b447317640c81eb61077b1b00 -- \
    bash -c "pip install librosa soundfile && vllm serve OpenMOSS-Team/MOSS-Transcribe-Diarize --trust-remote-code"
```

Parse the response `text` into segments with `parse_transcript` from the model's [GitHub package](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize).

**sglang-omni** (upstream's recommended production server; adds `response_format=verbose_json` with parsed speaker segments). Its prebuilt image predates this model, so install it at job start over a current [sglang nightly image](https://hub.docker.com/r/lmsysorg/sglang/tags) — the tag below is the tested pin; newer nightlies should also work. The clone + fresh-venv sequence is [upstream's documented install](https://github.com/sgl-project/sglang-omni/blob/main/docs/get_started/installation.md) and adds under a minute:

```bash
hf jobs run --detach --expose 8000 --flavor l4x1 -s HF_TOKEN --timeout 2h \
    lmsysorg/sglang:nightly-dev-cu13-20260709-074bb928 -- \
    bash -c "pip install -q uv 2>/dev/null; git clone --depth 1 https://github.com/sgl-project/sglang-omni.git && cd sglang-omni && uv venv .venv -p 3.12 && . .venv/bin/activate && uv pip install . && sgl-omni serve --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize --host 0.0.0.0 --port 8000 --max-running-requests 16 --mem-fraction-static 0.80"
```

Query either server the same way (`response_format=verbose_json` on sglang-omni only; raise `max_new_tokens`, e.g. `65536`, for long recordings). **Caution — long audio on the serve path**: sglang-omni silently truncates input past ~62 minutes (observed deterministically at ~62.5 min on the same tape that transcribes fully via the transformers recipe, regardless of `max_new_tokens`). Split longer recordings before posting — `moss-transcribe-diarize-server.py` does this automatically:

```bash
curl -X POST https://<job-id>--8000.hf.jobs/v1/audio/transcriptions \
    -H "Authorization: Bearer $HF_TOKEN" \
    -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
    -F file=@meeting.wav \
    -F response_format=verbose_json \
    -F max_new_tokens=65536
```

The job — and its billing — stops at `--timeout` or `hf jobs cancel <job_id>`.

## Notes

- **Gated model**: Accept terms at the [model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) before use.
- **Tokenizer workaround**: `cohere-transcribe.py` applies a one-line patch for a tokenizer compat issue. Will be removed once upstream fixes land ([model discussion](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/discussions/11)).
- **easytranscriber**: the Cohere backend requires `transformers>=5.4.0` (pinned in the script). Pyannote VAD is gated — accept terms at [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) if using `--vad pyannote`. Otherwise stick with the default Silero VAD.
- **moss-transcribe-diarize**: not gated (Apache 2.0). Uses `trust_remote_code=True` (model code lives in the [model repo](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)); inference helpers install from the model's [GitHub repo](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) pinned to a commit (the package isn't on PyPI). Generation time scales with audio length — long multi-speaker recordings emit tens of thousands of tokens, so watch the `truncated` flag in the output JSON.
- **sglang-omni install**: the `lmsysorg/sglang-omni:dev` image predates this model — don't use it. Install from git over a current `lmsysorg/sglang` nightly image as shown above, and follow the clone + fresh-venv sequence exactly: installing outside the repo skips its `[tool.uv]` dependency overrides and fails on a protobuf conflict. Once they publish a fresh image, `sgl-omni serve` alone will do.
