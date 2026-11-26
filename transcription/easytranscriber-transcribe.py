# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "easytranscriber>=0.2.2",
#     "easyaligner",
#     "transformers>=5.4.0",
#     "torch>=2.7.0,!=2.9.*",
#     "torchaudio>=2.7.0,!=2.9.*",
#     "ctranslate2>=4.4.0",
#     "pyannote.audio>=3.3.1",
#     "silero-vad~=6.0",
#     "nltk>=3.8.2",
#     "msgspec",
#     "soundfile",
#     "librosa",
#     "static-ffmpeg",
#     "huggingface-hub",
# ]
# ///

"""
Transcribe audio with word-level timestamps using kb-labb/easytranscriber.

Runs VAD -> ASR -> emissions -> forced alignment and writes per-file JSON
with `speeches[].alignments[].words[].{text,start,end,score}`. Optionally
also writes plain `.txt` transcripts and `.srt` subtitles.

Designed to work with HF Buckets mounted as volumes via `hf jobs uv run -v ...`.

Layout:
  INPUT                              OUTPUT (default JSON only)
    /input/ep1.mp3          ->         /output/alignments/ep1.json
    /input/sub/clip.wav     ->         /output/alignments/sub/clip.json

With --emit-txt / --emit-srt, side-files land at:
    /output/ep1.txt, /output/ep1.srt   (preserving relative sub-dirs)

Default backend is Cohere Transcribe 2B (same model as cohere-transcribe.py)
but here with word-level alignments on top. Pass --backend ct2 to use a
Whisper variant (e.g. KBLab/kb-whisper-large for Swedish).

Examples:

  # Smoke test
  uv run easytranscriber-transcribe.py ./test-audio ./test-output \\
      --language en --max-files 1 --emit-txt --emit-srt

  # Swedish with KB-Whisper (ct2 backend)
  uv run easytranscriber-transcribe.py ./audio-sv ./output-sv \\
      --language sv --backend ct2 \\
      --transcription-model KBLab/kb-whisper-large \\
      --emit-srt

  # HF Jobs with bucket volumes
  hf jobs uv run --flavor l4x1 -s HF_TOKEN \\
      -e UV_TORCH_BACKEND=cu128 \\
      -v hf://buckets/user/audio-files:/input:ro \\
      -v hf://buckets/user/transcripts-aligned:/output \\
      easytranscriber-transcribe.py /input /output \\
      --language en --emit-txt --emit-srt
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus"}

COHERE_MODEL = "CohereLabs/cohere-transcribe-03-2026"
WHISPER_DEFAULT_MODEL = "distil-whisper/distil-large-v3.5"

# Cohere Transcribe's 14 supported languages.
# Source: easytranscriber/src/easytranscriber/asr/cohere.py
COHERE_LANGUAGES = frozenset(
    {"ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"}
)

# Language -> default wav2vec2 emissions (forced alignment) model.
# Anything not listed falls back to the multilingual MMS model.
LANGUAGE_EMISSIONS_DEFAULTS = {
    "en": "facebook/wav2vec2-base-960h",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
}
FALLBACK_EMISSIONS_MODEL = "facebook/mms-1b-all"

# Language -> NLTK Punkt tokenizer language name.
# easyaligner.text.load_tokenizer wraps nltk.tokenize.punkt.PunktTokenizer.
LANGUAGE_TOKENIZER_MAP = {
    "en": "english", "sv": "swedish", "de": "german", "fr": "french",
    "it": "italian", "es": "spanish", "pt": "portuguese", "el": "greek",
    "nl": "dutch", "pl": "polish", "ru": "russian", "cs": "czech",
    "da": "danish", "fi": "finnish", "no": "norwegian", "tr": "turkish",
    "et": "estonian",
}


def check_cuda_availability():
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")


def discover_audio_files(input_dir: Path) -> list[Path]:
    """Walk input_dir recursively, returning sorted list of audio files."""
    files = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
    return files


def get_audio_duration(file_path: Path) -> float | None:
    """Get audio duration in seconds."""
    try:
        import librosa
        return librosa.get_duration(path=str(file_path))
    except Exception:
        return None


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(segments, out_path: Path) -> None:
    """Write AlignmentSegments to an SRT file."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _format_srt_timestamp(float(seg.start))
        end = _format_srt_timestamp(float(seg.end))
        text = (seg.text or "").strip().replace("\n", " ")
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_txt(segments, out_path: Path) -> None:
    """Write concatenated segment text to a .txt file."""
    text = "\n".join((seg.text or "").strip() for seg in segments if seg.text)
    out_path.write_text(text + ("\n" if text and not text.endswith("\n") else ""), encoding="utf-8")


def resolve_transcription_model(backend: str, override: str | None) -> str:
    if override:
        return override
    if backend == "cohere":
        return COHERE_MODEL
    return WHISPER_DEFAULT_MODEL


def resolve_emissions_model(language: str, override: str | None) -> str:
    if override:
        return override
    return LANGUAGE_EMISSIONS_DEFAULTS.get(language, FALLBACK_EMISSIONS_MODEL)


def resolve_tokenizer_lang(language: str, override: str | None) -> str:
    if override:
        return override
    return LANGUAGE_TOKENIZER_MAP.get(language, "english")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with word-level timestamps via easytranscriber.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Backends: cohere (default, 14 langs), ct2 (Whisper via CTranslate2), hf (transformers).

Examples:
  uv run easytranscriber-transcribe.py ./audio ./out --language en
  uv run easytranscriber-transcribe.py ./audio ./out --language en --emit-srt --emit-txt
  uv run easytranscriber-transcribe.py ./sv ./out-sv --language sv --backend ct2 \\
      --transcription-model KBLab/kb-whisper-large

HF Jobs with bucket volumes:
  hf jobs uv run --flavor l4x1 -s HF_TOKEN -e UV_TORCH_BACKEND=cu128 \\
      -v hf://buckets/user/audio-files:/input:ro \\
      -v hf://buckets/user/transcripts-aligned:/output \\
      easytranscriber-transcribe.py /input /output --language en --emit-txt --emit-srt
        """,
    )
    parser.add_argument("input_dir", help="Directory containing audio files (recursively scanned)")
    parser.add_argument("output_dir", help="Directory to write alignments/JSON (and optional .txt/.srt)")
    parser.add_argument("--language", required=True, help="ISO 639-1 language code (e.g. en, sv, de)")
    parser.add_argument(
        "--backend", default="cohere", choices=["cohere", "ct2", "hf"],
        help="ASR backend (default: cohere)",
    )
    parser.add_argument(
        "--transcription-model", default=None,
        help=f"Transcription model HF ID. Default: {COHERE_MODEL} (cohere) or {WHISPER_DEFAULT_MODEL} (ct2/hf)",
    )
    parser.add_argument(
        "--emissions-model", default=None,
        help="wav2vec2 HF ID for forced alignment. Default picked from --language.",
    )
    parser.add_argument(
        "--vad", default="silero", choices=["silero", "pyannote"],
        help="VAD backend (default: silero). pyannote requires accepting terms + HF_TOKEN.",
    )
    parser.add_argument(
        "--tokenizer-lang", default=None,
        help="NLTK Punkt language name (english, swedish, ...). Default derived from --language.",
    )
    parser.add_argument("--batch-size-features", type=int, default=8, help="Batch size for the pipeline's transcription and emission-extraction steps (default: 8)")
    parser.add_argument("--emit-txt", action="store_true", help="Also write .txt transcript per file")
    parser.add_argument("--emit-srt", action="store_true", help="Also write .srt subtitles per file")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Print resolved package versions")

    args = parser.parse_args()

    check_cuda_availability()

    language = args.language.lower()
    if args.backend == "cohere" and language not in COHERE_LANGUAGES:
        logger.error(
            f"Language '{language}' is not supported by the Cohere backend. "
            f"Supported: {', '.join(sorted(COHERE_LANGUAGES))}. "
            f"Use --backend ct2 with a Whisper model that covers this language."
        )
        sys.exit(1)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    alignments_dir = output_dir / "alignments"
    vad_dir = output_dir / ".work" / "vad"
    transcriptions_dir = output_dir / ".work" / "transcriptions"
    emissions_dir = output_dir / ".work" / "emissions"

    logger.info(f"Scanning {input_dir} for audio files...")
    files = discover_audio_files(input_dir)
    if not files:
        logger.error(f"No audio files found in {input_dir}")
        logger.error(f"Supported extensions: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        sys.exit(1)

    if args.max_files:
        files = files[: args.max_files]
    logger.info(f"Found {len(files)} audio file(s)")

    # Relative paths (strings) — the library joins audio_dir + audio_path internally
    # and reuses the same relative structure (with .json suffix) for all output dirs.
    rel_paths = [str(f.relative_to(input_dir)) for f in files]

    transcription_model = resolve_transcription_model(args.backend, args.transcription_model)
    emissions_model = resolve_emissions_model(language, args.emissions_model)
    tokenizer_lang = resolve_tokenizer_lang(language, args.tokenizer_lang)

    logger.info(f"Backend:            {args.backend}")
    logger.info(f"Transcription model: {transcription_model}")
    logger.info(f"Emissions model:     {emissions_model}")
    logger.info(f"VAD:                 {args.vad}")
    logger.info(f"Language:            {language} (tokenizer={tokenizer_lang})")

    # easyaligner shells out to `ffmpeg` to convert audio to WAV — HF Jobs base
    # images don't ship ffmpeg, so bootstrap a static binary onto PATH before
    # importing the library.
    import static_ffmpeg
    static_ffmpeg.add_paths()

    # Imports that pull in torch/transformers/etc. are deferred so argparse --help stays fast.
    from easyaligner.text import load_tokenizer
    from easytranscriber.pipelines import pipeline
    from easytranscriber.text.normalization import text_normalizer

    tokenizer = load_tokenizer(tokenizer_lang)

    cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or "models"

    logger.info("Starting pipeline (VAD -> ASR -> emissions -> alignment)...")
    start = time.time()
    alignments = pipeline(
        vad_model=args.vad,
        emissions_model=emissions_model,
        transcription_model=transcription_model,
        audio_paths=rel_paths,
        audio_dir=str(input_dir),
        backend=args.backend,
        language=language,
        tokenizer=tokenizer,
        text_normalizer_fn=text_normalizer,
        batch_size_features=args.batch_size_features,
        output_vad_dir=str(vad_dir),
        output_transcriptions_dir=str(transcriptions_dir),
        output_emissions_dir=str(emissions_dir),
        output_alignments_dir=str(alignments_dir),
        cache_dir=cache_dir,
        hf_token=os.environ.get("HF_TOKEN"),
        save_json=True,
        delete_emissions=True,
        return_alignments=True,
    )
    elapsed = time.time() - start

    # Post-process: optional .txt / .srt side-files + summary.jsonl.
    total_audio_duration = 0.0
    results = []

    for file_path, rel, items in zip(files, rel_paths, alignments):
        rel_path = Path(rel)
        # The pipeline may hand back either list[SpeechSegment] (which nests
        # AlignmentSegments under `.alignments`) or a pre-flattened list of
        # AlignmentSegments. Normalise to a flat list either way.
        align_segments = []
        for item in items or []:
            nested = getattr(item, "alignments", None)
            if nested:
                align_segments.extend(nested)
            elif hasattr(item, "words"):
                align_segments.append(item)
        num_words = sum(len(seg.words or []) for seg in align_segments)

        if args.emit_txt:
            txt_path = output_dir / rel_path.with_suffix(".txt")
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            _write_txt(align_segments, txt_path)

        if args.emit_srt:
            srt_path = output_dir / rel_path.with_suffix(".srt")
            srt_path.parent.mkdir(parents=True, exist_ok=True)
            _write_srt(align_segments, srt_path)

        duration = get_audio_duration(file_path)
        if duration:
            total_audio_duration += duration

        results.append({
            "file": rel,
            "duration_s": round(duration, 1) if duration else None,
            "num_segments": len(align_segments),
            "num_words": num_words,
        })
        logger.info(
            f"  {rel}: {len(align_segments)} segment(s), {num_words} word(s)"
            f"{f', {duration:.0f}s audio' if duration else ''}"
        )

    summary_path = output_dir / "summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    elapsed_str = f"{elapsed / 60:.1f} min" if elapsed > 60 else f"{elapsed:.1f}s"
    logger.info("=" * 50)
    logger.info(f"Done! Processed {len(files)} file(s) in {elapsed_str}")
    logger.info(f"  Alignments: {alignments_dir}")
    if args.emit_txt:
        logger.info(f"  Text:       {output_dir}/<rel>.txt")
    if args.emit_srt:
        logger.info(f"  Subtitles:  {output_dir}/<rel>.srt")
    if total_audio_duration > 0:
        rtfx = total_audio_duration / elapsed
        logger.info(f"  Audio:      {total_audio_duration / 60:.1f} min total")
        logger.info(f"  RTFx:       {rtfx:.1f}x realtime")
    logger.info(f"  Summary:    {summary_path}")

    if args.verbose:
        import importlib.metadata
        logger.info("--- Package versions ---")
        for pkg in [
            "easytranscriber", "easyaligner", "transformers", "torch", "torchaudio",
            "ctranslate2", "pyannote.audio", "silero-vad", "nltk", "librosa",
            "soundfile", "huggingface-hub",
        ]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("easytranscriber: audio -> JSON alignments (+ optional .txt/.srt)")
        print("=" * 60)
        print("\nRuns VAD -> ASR -> forced alignment and writes word-level timestamps.")
        print("Default backend: Cohere Transcribe 2B (14 langs). Use --backend ct2")
        print("for Whisper variants (e.g. Swedish via KBLab/kb-whisper-large).")
        print()
        print("Usage:")
        print("  uv run easytranscriber-transcribe.py INPUT_DIR OUTPUT_DIR --language en")
        print()
        print("Examples:")
        print("  uv run easytranscriber-transcribe.py ./audio ./out --language en")
        print("  uv run easytranscriber-transcribe.py ./audio ./out --language en \\")
        print("      --emit-txt --emit-srt")
        print("  uv run easytranscriber-transcribe.py ./sv ./out-sv --language sv \\")
        print("      --backend ct2 --transcription-model KBLab/kb-whisper-large")
        print()
        print("HF Jobs with bucket volumes:")
        print("  hf jobs uv run --flavor l4x1 -s HF_TOKEN -e UV_TORCH_BACKEND=cu128 \\")
        print("      -v hf://buckets/user/audio-files:/input:ro \\")
        print("      -v hf://buckets/user/transcripts-aligned:/output \\")
        print("      easytranscriber-transcribe.py /input /output \\")
        print("      --language en --emit-txt --emit-srt")
        print()
        print("For full help: uv run easytranscriber-transcribe.py --help")
        sys.exit(0)

    main()
