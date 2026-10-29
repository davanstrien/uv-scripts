# Serving transcription models on Jobs

> Part of [uv-scripts/transcription](README.md).


Two models can run as a server in a Job with an [exposed port](https://huggingface.co/docs/hub/jobs-serving), so you pay the setup once and send requests from anywhere with an HF token.

### Nemotron: serve, explore, scale

To pay the ~8 min setup once and reuse a hot model, run the server on its own with an [exposed port](https://huggingface.co/docs/hub/jobs-serving), then send requests from anywhere with an HF token (Bearer auth):

```bash
# Serve: the job logs https://<job_id>--8000.hf.jobs once the model is up
hf jobs uv run --detach --expose 8000 --timeout 2h \
    https://huggingface.co/datasets/uv-scripts/transcription/raw/main/nemotron-transcribe.py \
    --serve

# Scale: the same script as a driver, from a laptop
uv run https://huggingface.co/datasets/uv-scripts/transcription/raw/main/nemotron-transcribe.py \
    ./audio ./transcripts --server https://<job_id>--8000.hf.jobs
```

For exploring, any OpenAI-compatible client works: `base_url="https://<job_id>--8000.hf.jobs/v1"`, `api_key=<HF token>`, with the audio as a `data:audio/flac;base64,...` `audio_url` part (see the model card). Measured from a laptop: a warm 2-minute clip came back in 2.4 s, and the driver did two half-hour episodes with `--timestamps` in 54 s. The first requests after start are slow (~1 min) while vLLM compiles two Triton kernels that its warmup misses. Cancel the job when done (`hf jobs cancel <job_id>`); exposed ports are billed on top of the hardware price, and each half-hour file is uploaded as ~30 MB, so for bulk work the in-job driver is still cheaper.

### MOSS-Transcribe-Diarize as an OpenAI-compatible API

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
