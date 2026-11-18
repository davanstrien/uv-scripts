# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface-hub",
#     "librosa",
#     "numpy",
#     "soundfile",
# ]
#
# [tool.hf-jobs]
# # vllm, torch and transformers come from the image (PYTHONPATH). Local GPU: uv run --with vllm.
# image = "vllm/vllm-openai:v0.32.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
# flavor = "l4x1"
# secrets = ["HF_TOKEN"]
# timeout = "1h"
#
# [tool.uv-scripts]
# maintenance = "active"
# ///

"""
Transcribe a directory of audio files with Reson8/Rumble-2-Flash via vLLM.

Rumble-2-Flash is a 0.5B Qwen3-ASR model (Apache 2.0, 45 languages). On the
Open ASR Leaderboard (v1.16.1) it averages 6.60 WER on the English track.

Designed for HF Buckets mounted as volumes with `hf jobs uv run -v ...`.

Long files are cut into chunks of at most --chunk-seconds. Each cut is placed at
the quietest point in the last few seconds before the limit, so chunks do not
overlap and words are rarely split. All chunks in a batch of files go through
vLLM in one call. Files that already have a transcript in the output directory
are skipped, so a rerun continues where the last run stopped.

Input:                              Output:
  /input/episode1.mp3         ->      /output/episode1.txt
  /input/sub/clip.wav         ->      /output/sub/clip.txt
                                      /output/summary.jsonl

Examples:

  # HF Jobs with bucket volumes (the header sets image, flavor and secrets)
  hf jobs uv run \\
      -v hf://buckets/user/audio-input:/input:ro \\
      -v hf://buckets/user/transcripts:/output \\
      https://huggingface.co/datasets/uv-scripts/transcription/raw/main/rumble-transcribe.py \\
      /input /output --language English

  # Local GPU (Linux + CUDA)
  uv run --with vllm rumble-transcribe.py ./audio ./out --max-files 3

Options from the model card's chat template:
  --language NAME    English name ("Dutch", "English"). Omit to detect per chunk.
  --words-for-digits Write numbers as words (default: digits).
  --keep-fillers     Keep "um" and "uh" (default: dropped).
  --no-verbatim      Collapse repeated words (default: kept).
"""

import argparse
import importlib.metadata
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "Reson8/Rumble-2-Flash"
SAMPLE_RATE = 16000
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus"}
ASR_TEXT_TAG = "<asr_text>"


def discover_audio_files(input_dir: Path) -> list[Path]:
    """Return every audio file under input_dir, sorted."""
    files = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
    return files


def load_audio(file_path: Path) -> np.ndarray:
    """Load a file as 16 kHz mono float32."""
    import librosa

    audio, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def quietest_sample(audio: np.ndarray, start: int, end: int) -> int:
    """Return the sample index at the centre of the quietest 20 ms frame in [start, end)."""
    frame = SAMPLE_RATE // 50
    window = audio[start:end]
    n_frames = len(window) // frame
    if n_frames == 0:
        return end
    frames = window[: n_frames * frame].reshape(n_frames, frame)
    energy = np.sqrt(np.mean(frames**2, axis=1))
    best_frame = int(np.argmin(energy))
    return start + best_frame * frame + frame // 2


def chunk_audio(audio: np.ndarray, chunk_seconds: float, search_seconds: float) -> list[np.ndarray]:
    """Cut audio into chunks of at most chunk_seconds, each cut at a quiet point."""
    max_len = int(chunk_seconds * SAMPLE_RATE)
    search_len = int(search_seconds * SAMPLE_RATE)
    chunks = []
    start = 0
    while len(audio) - start > max_len:
        limit = start + max_len
        cut = quietest_sample(audio, limit - search_len, limit)
        chunks.append(audio[start:cut])
        start = cut
    chunks.append(audio[start:])
    return chunks


def parse_output(raw: str) -> tuple[str | None, str]:
    """Split raw model output into (detected language, text).

    With --language set, the prompt already ends with "language X<asr_text>",
    so the output is plain text. Without it, the output starts with
    "language <Name><asr_text>".
    """
    text = raw.replace("<|im_end|>", "").strip()
    if ASR_TEXT_TAG not in text:
        return None, text
    header, _, body = text.partition(ASR_TEXT_TAG)
    language = header.removeprefix("language").strip() or None
    return language, body.strip()


def build_prompt(tokenizer, args) -> str:
    """Render the model's chat template for one audio input."""
    messages = [{"role": "user", "content": [{"type": "audio"}]}]
    template_kwargs = {
        "digits": not args.words_for_digits,
        "filler": args.keep_fillers,
        "verbatim": not args.no_verbatim,
    }
    if args.language:
        template_kwargs["language"] = args.language
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs
    )


def output_path_for(file_path: Path, input_dir: Path, output_dir: Path) -> Path:
    return output_dir / file_path.relative_to(input_dir).with_suffix(".txt")


def transcribe_batch(llm, sampling_params, prompt, files, args, input_dir, output_dir) -> list[dict]:
    """Load, chunk and transcribe one batch of files; write one .txt per file."""
    chunk_owner = []  # file index for each chunk, in order
    requests = []
    durations = []
    for file_index, file_path in enumerate(files):
        audio = load_audio(file_path)
        durations.append(len(audio) / SAMPLE_RATE)
        for chunk in chunk_audio(audio, args.chunk_seconds, args.search_seconds):
            chunk_owner.append(file_index)
            requests.append({"prompt": prompt, "multi_modal_data": {"audio": (chunk, SAMPLE_RATE)}})

    outputs = llm.generate(requests, sampling_params)

    texts_per_file = [[] for _ in files]
    languages_per_file = [Counter() for _ in files]
    truncated_per_file = [0 for _ in files]
    for file_index, output in zip(chunk_owner, outputs):
        completion = output.outputs[0]
        language, text = parse_output(completion.text)
        if text:
            texts_per_file[file_index].append(text)
        if language:
            languages_per_file[file_index][language] += 1
        if completion.finish_reason == "length":
            truncated_per_file[file_index] += 1

    results = []
    for file_index, file_path in enumerate(files):
        full_text = " ".join(texts_per_file[file_index])
        txt_path = output_path_for(file_path, input_dir, output_dir)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(full_text + "\n", encoding="utf-8")

        top_language = languages_per_file[file_index].most_common(1)
        result = {
            "file": str(file_path.relative_to(input_dir)),
            "duration_s": round(durations[file_index], 1),
            "chunks": chunk_owner.count(file_index),
            "truncated_chunks": truncated_per_file[file_index],
            "language": args.language or (top_language[0][0] if top_language else None),
            "word_count": len(full_text.split()),
        }
        results.append(result)
        logger.info(
            f"  {result['file']}: {result['duration_s']:.0f}s, {result['chunks']} chunks, "
            f"{result['word_count']} words, language={result['language']}"
        )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with Reson8/Rumble-2-Flash via vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", help="Directory containing audio files (searched recursively)")
    parser.add_argument("output_dir", help="Directory for .txt transcripts and summary.jsonl")
    parser.add_argument("--model", default=MODEL, help=f"Model id (default: {MODEL})")
    parser.add_argument("--language", default=None, help='English language name, e.g. "Dutch". Omit to detect.')
    parser.add_argument("--words-for-digits", action="store_true", help="Write numbers as words")
    parser.add_argument("--keep-fillers", action="store_true", help='Keep fillers such as "um" and "uh"')
    parser.add_argument("--no-verbatim", action="store_true", help="Collapse repeated words")
    parser.add_argument("--chunk-seconds", type=float, default=30.0, help="Maximum chunk length (default: 30)")
    parser.add_argument(
        "--search-seconds",
        type=float,
        default=5.0,
        help="Look this far back from the chunk limit for a quiet cut point (default: 5)",
    )
    parser.add_argument("--batch-files", type=int, default=32, help="Files loaded per vLLM call (default: 32)")
    parser.add_argument("--max-files", type=int, default=None, help="Process at most this many files")
    parser.add_argument("--overwrite", action="store_true", help="Redo files that already have a transcript")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens per chunk (default: 1024)")
    parser.add_argument("--max-model-len", type=int, default=4096, help="vLLM max model length (default: 4096)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    if args.search_seconds >= args.chunk_seconds:
        parser.error("--search-seconds must be smaller than --chunk-seconds")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_audio_files(input_dir)
    if not files:
        logger.error(f"No audio files in {input_dir} ({', '.join(sorted(AUDIO_EXTENSIONS))})")
        sys.exit(1)
    if args.max_files:
        files = files[: args.max_files]
    todo = [f for f in files if args.overwrite or not output_path_for(f, input_dir, output_dir).exists()]
    logger.info(f"Found {len(files)} audio file(s); {len(files) - len(todo)} already done, {len(todo)} to do")
    if not todo:
        return

    from vllm import LLM, SamplingParams

    logger.info(f"Loading {args.model} in vLLM {importlib.metadata.version('vllm')}...")
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"audio": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    # skip_special_tokens=False keeps <asr_text>, which separates the detected language from the text.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, skip_special_tokens=False)
    prompt = build_prompt(llm.get_tokenizer(), args)
    logger.info(f"Prompt: {prompt!r}")

    summary_path = output_dir / "summary.jsonl"
    total_audio_s = 0.0
    start_time = time.time()
    for batch_start in range(0, len(todo), args.batch_files):
        batch = todo[batch_start : batch_start + args.batch_files]
        logger.info(f"Batch {batch_start // args.batch_files + 1}: {len(batch)} file(s)")
        results = transcribe_batch(llm, sampling_params, prompt, batch, args, input_dir, output_dir)
        with open(summary_path, "a", encoding="utf-8") as summary_file:
            for result in results:
                summary_file.write(json.dumps(result) + "\n")
        total_audio_s += sum(r["duration_s"] for r in results)

    elapsed = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"Transcribed {len(todo)} file(s), {total_audio_s / 60:.1f} min of audio, in {elapsed:.1f}s")
    logger.info(f"  RTFx (load + chunk + transcribe): {total_audio_s / elapsed:.0f}x realtime")
    logger.info(f"  Output: {output_dir}  Summary: {summary_path}")


if __name__ == "__main__":
    main()
