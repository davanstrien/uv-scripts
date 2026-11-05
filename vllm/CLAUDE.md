# vLLM Scripts Development Notes

## Repository Purpose
This repository contains UV scripts for vLLM-based inference tasks. Focus on GPU-accelerated inference using vLLM's optimized engine.

## Key Patterns

### 1. GPU Requirements
All scripts MUST check for GPU availability:
```python
if not torch.cuda.is_available():
    logger.error("CUDA is not available. This script requires a GPU.")
    sys.exit(1)
```

### 2. vLLM Docker Image
Run on the pinned **`vllm/vllm-openai:v0.32.0`** image for HF Jobs (ships vLLM + the CUDA toolkit, so FlashInfer works) with `--python /usr/bin/python3 -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages`. Each script's `[tool.hf-jobs]` header sets these. **Pin the tag (test-then-pin), not `:latest`.** Tested 2026-11-05 on `a10g-small`, `a10g-large` and `a100-large` (vLLM 0.32.0).

**Do not list `vllm`, `torch`, `transformers` or `flashinfer-python` in the PEP 723 dependencies.** uv would install newer PyPI copies next to the image's. `PYTHONPATH` shadows those, but the `flashinfer-python` 0.7.2.post1 wheel also ships a top-level `b12x` module that the image lacks; the image's vLLM imports it and crashes (`FileNotFoundError: .../flashinfer/experimental/b12x/_api.py`). Import these packages inside `main()` so `uv run <script> --help` works without them. Local GPU runs use `uv run --with vllm <script>`. The custom nightly / FlashInfer `[[tool.uv.index]]` blocks (§3) are **no longer needed**.

**APIs (verified):** structured outputs use `StructuredOutputsParams` (`structured_outputs=`, ≥0.11.0 — `GuidedDecodingParams`/`guided_decoding=` removed in 0.12.0); classification auto-detects the pooling runner from the model architecture (the old `LLM(..., task="classify")` arg was removed → use `LLM(model)`).

### 3. Dependencies
List only what the image does not provide (for example `datasets`, `huggingface-hub`, `toolz`). No custom `[[tool.uv.index]]` blocks are needed; see §2.

## Current Scripts

1. **classify-dataset.py**: BERT-style text classification
   - Uses vLLM's classify task
   - Supports batch processing with configurable size
   - Automatically extracts label mappings from model config

## Future Scripts

Potential additions:
- Text generation with vLLM
- Embedding generation using sentence transformers
- Multi-modal inference
- Structured output generation

## Testing

Local testing requires GPU. For scripts without local GPU access:
1. Use HF Jobs with small test datasets
2. Verify script runs without syntax errors: `python -m py_compile script.py`
3. Check dependencies resolve: `uv pip compile`

## Performance Considerations

- Default batch size: 10,000 for local, up to 100,000 for HF Jobs
- L4 GPUs are cost-effective for classification
- Monitor GPU memory usage and adjust batch sizes accordingly