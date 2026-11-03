# Transcription benchmarks

> Part of [uv-scripts/transcription](README.md).

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

**`nemotron-transcribe.py`** by `--precision` (Nixon White House tapes from the [Internet Archive](https://archive.org/details/rmn_e005a), public domain: 6 tapes, 11.3 h, cut into 17 parts of up to 55 min and re-encoded to 16 kHz mono FLAC; `rtx-pro-6000`, prebuilt image, vLLM 0.32.0, MTP8, all 17 files in flight):

| Precision | Transcription | RTFx | Whole job | Cost | Parts at max_tokens |
|-----------|---------------|------|-----------|------|---------------------|
| BF16 | 5.0 min | 135x | 11.2 min | $0.52 | 0 (1 on an earlier mp3 run) |
| FP8 | 3.0 min | 222x | 7.4 min | $0.34 | 2 |
| NVFP4 | 2.1 min / 2.9 min (two runs) | 322x / 237x | 6.6 / 7.3 min | $0.30 / $0.34 | 0 / 1 |

"Transcription" starts once the server is up; "whole job" includes the checkpoint download (17–30 s) and engine load and warmup (4–5.5 min). A part that loops until `max_tokens` finishes last and sets the batch time: in the slower NVFP4 run one part took 171 s, the others 122 s or less. The cards report error rates about 1.0% (FP8) and 3.6% (NVFP4) higher than BF16, in relative terms. With no ground truth here, word agreement with the BF16 run (difflib ratio, mean over files) was 0.80 for FP8 and 0.82–0.83 for NVFP4, and 0.80 for a second BF16 run on the mp3 input, so on this noisy 1971 audio the quantized checkpoints differ from BF16 about as much as two BF16 runs differ from each other.

Did not work on 2026-11-03: vLLM 0.33.0 (the NeMo Speech plugin fails at startup: `_parse_chat_message_content_parts` is missing); FP8 on `l40sx1` (FlashInfer's JIT build of a Mamba kernel fails at the first request). For comparison, `nemotron-transcribe-saturate.py` (SGLang-Omni main, BF16) took 7.0 min (97x) on the same FLAC files, and its FP8 checkpoint does not load.
