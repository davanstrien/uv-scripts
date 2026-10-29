# cr.hf.co/davanstrien/minicpm-ocr

Runtime image for [`../../minicpm-ocr.py`](../../minicpm-ocr.py), public on the Hugging Face
container registry: https://huggingface.co/containers/davanstrien/minicpm-ocr

| Tag | Digest |
|---|---|
| `1.0-vllm-5d35ce3` (also `latest`) | `sha256:2fb9a09980abecb3102b99b0685b57b4ecde22702b3a441c03fab22e30239aea` |

Built with kaniko on an HF CPU Job (`cpu-upgrade`, about 20 minutes),
pushed to `cr.hf.co`. The published digest was built before `sdk-requirements.lock` existed;
the lock is that image's `/opt/sdk` environment (`uv pip freeze`), so a rebuild reproduces it. Why a custom image: MiniCPM-OCR's architecture is in vLLM main but not in a
vLLM release yet, and the SDK needs PaddlePaddle in its own environment. Drop it for a stock
`vllm/vllm-openai` tag plus a pip-installed SDK once a vLLM release has the architecture.

Build note: pip warns that `transformers 5.21.0.dev0` does not satisfy vLLM's
`transformers<5.21.0` (a pre-release of 5.21.0 is excluded by `<5.21.0`). This is OpenBMB's pin;
vLLM serves the model with it (smoke test, 27 Oct 2026).
