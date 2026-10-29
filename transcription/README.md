---
viewer: false
tags:
  - uv-script
  - hf-jobs
  - audio
  - transcription
  - automatic-speech-recognition
  - speaker-diarization
  - multilingual
---

# Transcription UV Scripts

<a href="https://huggingface.co/uv-scripts"><picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md-dark.svg"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md.svg" alt="Follow uv-scripts on Hugging Face"></picture></a>

> Part of [uv-scripts](https://huggingface.co/uv-scripts): self-contained UV scripts you run on Hugging Face Jobs in one command.

Scripts that transcribe audio files on a GPU with [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs). Audio comes in from a [Bucket](https://huggingface.co/docs/hub/storage-buckets) or a local folder; transcripts go out as `.txt`, JSON (speaker segments or word timestamps), `.srt` or parquet.

## Quick Start

First, [install the `hf` CLI and sign in](https://huggingface.co/docs/hub/jobs-quickstart). Jobs needs pay-as-you-go credit. Use `hf` 1.32 or newer (`hf version`): older CLIs ignore the script's launch header without a warning, and the Job then runs on a CPU.

Transcribe four minutes of NASA's *Houston We Have a Podcast* (a host and a guest, from the [demo Bucket](https://huggingface.co/buckets/uv-scripts/transcription-demo)) with [Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B). Replace `your-username` with your Hugging Face username:

```bash
hf buckets create your-username/transcripts --private
hf jobs uv run --timeout 45m \
    -v hf://buckets/uv-scripts/transcription-demo/podcast:/input:ro \
    -v hf://buckets/your-username/transcripts/podcast:/output \
    https://huggingface.co/datasets/uv-scripts/transcription/raw/main/nemotron-transcribe.py \
    /input /output --timestamps
```

The Job writes a `.txt` with speaker tags and a `.json` with word timestamps to your Bucket:

```
<spk:0> Jim, thank you so much for coming on the podcast today. ... <spk:3> It is, uh, the mission control room, of course, is now open for business ...
```

Our test run took about 7.5 minutes on `rtx-pro-6000` (about $0.34), almost all of it model setup. Read the result with `hf buckets cp hf://buckets/your-username/transcripts/podcast/nasa-podcast-ep100.txt -`.

<details>
<summary>More on this example: JSON output, MOSS on the same clip, harder audio, launch headers</summary>

**JSON output.** Every word has a start and end time and a speaker (`parts[].alignment.words`), plus diarization segments (`parts[].alignment.diarization`):

```json
{"word": "sustainable", "start": 0.02, "end": 0.98, "speaker": "0", "part": 0}
```

**MOSS on the same clip.** [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) on the same clip took under 2 minutes on `l4x1` (about $0.02) and wrote one line per segment (`[61.82 - 67.06] S02: The mission control room, ...`):

```bash
hf jobs uv run --flavor l4x1 -s HF_TOKEN --timeout 30m \
    -e UV_TORCH_BACKEND=cu128 \
    -v hf://buckets/uv-scripts/transcription-demo/podcast:/input:ro \
    -v hf://buckets/your-username/transcripts/podcast-moss:/output \
    https://huggingface.co/datasets/uv-scripts/transcription/raw/main/moss-transcribe-diarize.py \
    /input /output --emit-txt
```

**Harder audio.** Mount `/apollo11` instead of `/podcast`: two clips of 1969 mission tapes with radio noise and crosstalk. The [demo Bucket's README](https://huggingface.co/buckets/uv-scripts/transcription-demo) gives the sources and licences (NASA, public domain).

**Why no flags.** The Nemotron scripts carry a [`[tool.hf-jobs]` header](https://huggingface.co/docs/hub/jobs-configuration#define-the-launch-config-in-the-script) that sets the GPU, image and secrets. The other scripts have no header yet and take flags, as in the MOSS command. `hf jobs uv run --dry-run <script>` shows the resolved settings.

</details>

## Use your own audio

**Audio on your machine:** mount the folder directly. `-v ./audio:/input` syncs it to a private Bucket and mounts it in the Job; re-runs only sync changed files (`huggingface_hub` ≥ 1.22). Start with `--max-files 2`.

**Audio in a Bucket:** replace the demo input with `-v hf://buckets/your-username/your-audio:/input:ro`. The scripts walk the input folder recursively and mirror its layout in the output (easytranscriber puts its JSON under `alignments/`). `nemotron-transcribe-saturate.py` is the exception: it takes a glob (`"hf://buckets/user/audio/**/*.mp3"`) and writes parquet.

**Audio on the Internet Archive:** `download-ia.py` downloads an item's audio straight into a mounted Bucket:

```bash
hf jobs uv run \
    -v hf://buckets/your-username/audio-files:/output \
    https://huggingface.co/datasets/uv-scripts/transcription/raw/main/download-ia.py \
    SUSPENSE /output
```

> **Local dev**: if you've cloned this repo, swap the URL for the local filename (e.g. `cohere-transcribe.py /input /output ...`).

## Pick a model

| You need | Script | Trade-off |
|---|---|---|
| Punctuated text in one of 14 languages (en, de, fr, it, es, pt, el, nl, pl, ar, vi, zh, ja, ko) | `cohere-transcribe.py` | Gated: accept the model terms first |
| A lot of English audio, as fast as possible | `granite-turboctc-transcribe.py` | Lowercase, no punctuation |
| A language Cohere doesn't cover (102 FLEURS languages) | `buzzasr-transcribe.py` | One model per language; mostly lowercase, no punctuation |
| Word-level timestamps (subtitles, search, alignment) | `easytranscriber-transcribe.py` | Four-stage pipeline, slower |
| Who said what, in English or Chinese | `moss-transcribe-diarize.py` (`-server.py` for many files) | Slow on long files; server version needs `a100-large` |
| Word timestamps and speaker tags from one model, in 40 languages (speaker tags in 13) | `nemotron-transcribe.py` | ~8 min fixed setup before the first file |
| Nemotron text for a large batch, resumable | `nemotron-transcribe-saturate.py` | Plain text only |

Only the two Nemotron scripts set their own GPU and image. The others need flags, for example Cohere on an L4:

<details>
<summary>Cohere Transcribe command</summary>

Accept the terms on the [model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) first; the model is gated.

```bash
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    -e UV_TORCH_BACKEND=cu128 \
    -v hf://buckets/your-username/audio-files:/input:ro \
    -v hf://buckets/your-username/transcripts:/output \
    https://huggingface.co/datasets/uv-scripts/transcription/raw/main/cohere-transcribe.py \
    /input /output --language en --compile
```

Each script's docstring has its own `hf jobs uv run` example.

</details>

[MODELS.md](MODELS.md) has the details for each script: speed, quality notes, pins and known issues. [BENCHMARKS.md](BENCHMARKS.md) has the measured runs.

## Serve as a live endpoint

Nemotron and MOSS-Transcribe-Diarize can run as a server in a Job with an [exposed port](https://huggingface.co/docs/hub/jobs-serving), so you pay the setup once and send requests from a laptop or an agent. Start one with `nemotron-transcribe.py --serve`; [SERVING.md](SERVING.md) has the commands, request examples and caveats.

## Options

The folder scripts take `/input /output` and `--max-files N` for a test run (the saturate script takes a glob and an output dataset). `--help` lists all flags.

<details>
<summary><code>nemotron-transcribe.py</code></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--style` | `verbatim` | Model-card prompt preset: `verbatim` (fillers kept), `clean` (fillers dropped) or `speakers` (`<spk:N>` tags) |
| `--prompt` | none | Custom user prompt (overrides `--style`), e.g. `Transcribe with punctuation and capitalization. Context terms: <names>.` |
| `--timestamps` | off | Also write JSON with word timestamps, speaker tags and diarization segments (uses the `speakers` prompt) |
| `--concurrency` | 16 | Files in flight against the server |
| `--part-minutes` | 55 | Split longer audio into equal parts of at most this length, up to 60 (the model is trained on recordings up to 1 h); speaker tags reset between parts |
| `--server` | `http://127.0.0.1:8000` | Use a server already running there instead of starting one; a serving job's `https://<job_id>--8000.hf.jobs` gets your HF token as Bearer auth (no other host does) |
| `--serve` | off | Only start the server and keep it up, for a job started with `--expose 8000` (no input/output dirs) |
| `--nemo-ref` | the image's commit (`7b05f096`) | NeMo Speech ref to install when the image has no prebuilt stack; `main` follows the model card |
| `--workdir` | `/opt/nemotron-transcribe` (prebuilt image), else `/tmp/nemotron-transcribe` | Where the serving stack and checkpoint live (the prebuilt image's stack is used when present) |
| `--request-timeout` | 3600 | Per-request timeout in seconds |
| `--max-files` | all | Limit files to process (for testing) |

</details>

<details>
<summary><code>cohere-transcribe.py</code></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--language` | required | en, de, fr, it, es, pt, el, nl, pl, ar, vi, zh, ja, ko |
| `--compile` | off | torch.compile encoder (one-time warmup, faster after) |
| `--batch-size` | 16 | Batch size for inference |
| `--max-files` | all | Limit files to process (for testing) |

</details>

<details>
<summary><code>cohere-transcribe-vllm.py</code></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--language` | required | Same 14 languages as `cohere-transcribe.py` |
| `--gpu-memory-utilization` | 0.8 | Fraction of GPU memory vLLM may use |
| `--verbose` | off | More logging |
| `--max-files` | all | Limit files to process (for testing) |

</details>

<details>
<summary><code>granite-turboctc-transcribe.py</code></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-seconds` | 0 (whole file) | Cut files into N-second windows; use 30 for hour-plus files or 24 GB cards (2.2 GB peak at batch 32 vs 10.3 GB for four 30-min files) |
| `--batch-size` | 4 | Files (or windows) per forward pass; 32-64 with `--chunk-seconds 30` |
| `--decode-workers` | CPU count | Threads for mp3/wav decoding, overlapped with model load and inference |
| `--max-files` | all | Limit files to process (for testing) |

</details>

<details>
<summary><code>buzzasr-transcribe.py</code></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--language` | required | Language name (`dutch`, `sorani-kurdish`) or ISO code (`nl`, `ckb`), resolved via the org's model tags; `--list-languages` prints all of them live from the Hub |
| `--model` | `BuzzASR/<language>` | Any Whisper checkpoint on the Hub (e.g. `openai/whisper-large-v3` as a zero-shot control) |
| `--long-form` | `sequential` | `sequential` (Whisper's own long-form decoding with timestamps + temperature fallback; skipped no windows in our tests) or `chunked` (transformers pipeline, 30 s windows with 5 s overlap; ~2.5x faster, drops text at bad merges) |
| `--batch-size` | 16 | Files per generate call (sequential) or 30 s windows per forward pass (chunked) |
| `--decode-workers` | CPU count | Threads for mp3/wav decoding, overlapped with model load |
| `--max-files` | all | Limit files to process (for testing) |

</details>

<details>
<summary><code>easytranscriber-transcribe.py</code></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--language` | required | ISO 639-1 code. Cohere supports the same 14 languages as `cohere-transcribe.py`; ct2/hf support any Whisper language |
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

</details>

<details>
<summary><code>moss-transcribe-diarize-server.py</code></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--concurrency` | 4 | Concurrent transcription requests (KV-cache bound; 6 works on A100) |
| `--part-minutes` | 55 | Clip length for splitting long audio (longest that fits the context window). Speaker labels reset between clips |
| `--server` | `http://127.0.0.1:8000` | sgl-omni server URL (in-job localhost by default) |
| `--request-timeout` | 3600 | Per-request timeout in seconds |
| `--emit-txt` | off | Also write `.txt` transcripts |
| `--max-files` | all | Limit files to process (for testing) |

</details>

<details>
<summary><code>moss-transcribe-diarize.py</code></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--max-new-tokens` | 0 (auto) | Token budget per file. Auto scales with audio duration (min 5120, max 65536). Output JSON sets `"truncated": true` if the budget was hit — re-run with a higher value |
| `--hotwords` | none | Comma-separated terms (names, companies, jargon) appended to the prompt to bias recognition |
| `--prompt` | built-in | Full prompt override (replaces the built-in transcribe+diarize prompt) |
| `--emit-txt` | off | Also write `.txt` transcripts (`[start - end] SPEAKER: text` per line) |
| `--emit-srt` | off | Also write `.srt` subtitles with speaker prefixes |
| `--max-files` | all | Limit files to process (for testing) |

</details>

## Notes

- **Gated model**: Accept terms at the [model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) before use.
- **easytranscriber**: the Cohere backend requires `transformers>=5.4.0` (pinned in the script). Pyannote VAD is gated — accept terms at [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) if using `--vad pyannote`. Otherwise stick with the default Silero VAD.
- **MOSS-Transcribe-Diarize**: not gated (Apache 2.0). Runs remote code from the model repo; see [MODELS.md](MODELS.md#notes).
