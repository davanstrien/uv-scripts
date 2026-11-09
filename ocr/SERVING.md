# Server-mode OCR: official support per model + the `-server.py` recipes

The recipes in this folder come in two shapes:

- **Offline batch** (`<model>.py`): the script owns the vLLM engine (`LLM.generate`)
  and processes the dataset in fixed-size batches. One `hf jobs uv run` command.
- **Server + driver** (`<model>-server.py`): one job starts `vllm serve` in the
  background, then a lightweight driver (no torch/vllm deps — installs in seconds)
  posts images concurrently to the OpenAI-compatible endpoint on localhost.

Why server mode? Two measured reasons (100-page historical-scan A/B on `l4x1`,
same data, same sampling, driver concurrency 32):

| Model | Offline (inference-only) | Server | Speedup | Output parity |
|---|---|---|---|---|
| OvisOCR2 0.9B | 0.83 img/s | 1.44 img/s | ~1.7× | 94/100 byte-identical, rest ≥0.978 similar |
| LightOnOCR-2 1B | 0.33 img/s | 0.59 img/s | ~1.8× | 64/100 byte-identical at temp 0.2, rest median 0.998 |
| Nanonets-OCR2 3B | 0.31 img/s (steady-state) | 0.36 img/s | ~1.2× | 52/100 byte-identical, median 0.999; 3 hard plate pages fork under greedy (worst case was the *offline* arm degenerating into a 22k-char repetition loop) |

1. **Throughput**: offline `llm.generate` drains at every batch boundary (GPU idles
   while the CPU decodes the next batch of images); a concurrent driver keeps vLLM's
   continuous batching fed. The speedup is a floor, not a ceiling — at concurrency 32
   the server's KV cache was <10% used.
2. **Failure isolation**: offline, one bad image (e.g. a None cell) fails its whole
   batch of 16; server mode fails that one request. On a real dataset where ~half the
   image cells were empty, the offline recipe produced 0 usable outputs and the server
   recipe produced all of the valid ones.

**LightOnOCR-3-0.8B** (`lighton-ocr3.py` against `lighton-ocr3-saturate.py`, 600 pages: 350
book pages from finebooks/bhl-impact-gt and 250 typed industry documents, 2026-11-05, steady
state): `a10g-small` 0.90 → 2.2 pages/s, `a100-large` 1.27 → 7.2 pages/s. Most of the gain is
pages in flight (the offline recipe sends batches of 16 and waits for the slowest page); on
small flavors the window is capped by host memory, because each page in flight holds about
0.1 GiB of image data. On fast GPUs one vLLM API server cannot preprocess images quickly
enough (128-256 pages in flight, 36-60 running in the engine, none waiting); 4 API servers
(`--api-server-count 4`) gave +16% on `a100-large`, +9% on `rtx-pro-6000`, +10% on `a10g-large`. At temperature 0.2 the outputs of two server runs differ on about 40%
of pages; the offline recipe differs from a server run by the same amount. The character
error rate on the 282 book pages with at least 20 characters of ground truth (IMPACT) was
0.077 to 0.083 mean and 0.020 median in every 0.8B run, offline or server.

The job command stays the standard `hf jobs uv run` shape: the driver spawns
`vllm serve` itself as a subprocess when no server is reachable (flags live in the
script's `SERVE_ARGS`, taken from the model's own card where they exist), so the only
thing to get right is `--image vllm/vllm-openai:<tag>` — which provides the `vllm`
binary. Without it the script fails fast, printing the exact correct command. Pass
`--server URL` to use an already-running or remote endpoint instead (nothing is
spawned when the server is reachable, so the moss-style explicit `bash -c` serve
command keeps working too).

For **interactive/agent** use (a live endpoint instead of a batch run), see
[serving-unlimited-ocr.md](serving-unlimited-ocr.md) — `hf jobs run --expose` gives an
OpenAI-compatible URL that outlives a single script.

## The recurring official serve pattern for OCR

Three flags recur across independent vendors' official serve commands (DeepSeek's vLLM
recipe, LightOn's card, Paddle's vLLM recipe, Unlimited-OCR's recipe):

```
--no-enable-prefix-caching --mm-processor-cache-gb 0 --limit-mm-per-prompt '{"image": 1}'
```

OCR workloads never reuse images, so prefix/multimodal caches only cost memory.

**One exception: HPD-Parsing** (`hpd-parsing-saturate.py`). Its vendor entrypoint sets
`--enable-prefix-caching` on purpose. Hierarchical parallel decoding forks a child branch per
page region, and the children reuse the parent branch's prefix KV, so the cache does real work
(hit rate 52.9% on 5 UFO pages in July 2026, 43.9% on 10 in November). Do not change it to
`--no-enable-prefix-caching` in a sweep.

## Version pins carry over — via the image tag

Where an offline recipe pins an engine version, the server variant needs the same pin
as a `vllm/vllm-openai:<tag>` image (e.g. Nanonets-OCR2-3B uses `v0.10.2`, matching its
offline recipe). [`models.json`](models.json) records the required image per script.
When trying a new model/image combination, sanity-check the first few outputs before
scaling: a mismatched combination can produce degenerate output while looking like
normal load from the outside (full GPU utilisation, no errors).

## Which models document server mode themselves? (surveyed 2026-07-16)

Server mode is the officially documented path for most models in this collection —
for several of them it's the *only* documented vLLM path. Summary of each model's own
card/docs (verbatim commands live in the linked sources):

| Model | Official server example | Notes |
|---|---|---|
| lightonai/LightOnOCR-1B / 2-1B | ✅ vLLM | Card ships the serve command + client; ≥0.11.1 for v1; images longest-dim 1540px |
| nanonets/OCR-s / OCR2-3B | ✅ vLLM | Bare `vllm serve` + client in card; **pin v0.10.2 for OCR2** (see above) |
| dots-studio/dots.mocr | ✅ vLLM ≥0.11 | Direct from Hub id on `vllm/vllm-openai:v0.11.0`+ |
| rednote-hilab/dots.ocr | ✅ (GitHub) | HF card shows a legacy pre-0.11 hack; GitHub README: integrated upstream since 0.11 |
| tencent/HunyuanOCR | ✅ vLLM 0.18.1 | serve.sh in repo; nightly adds DFlash speculative decoding |
| zai-org/GLM-OCR | ✅ vLLM + SGLang + Ollama | Only card here with an SGLang serve example; needs vLLM nightly |
| numind/NuExtract3 | ✅ vLLM | Production-grade recipe: MTP speculative decoding, per-request `chat_template_kwargs` |
| numind/NuMarkdown-8B-Thinking | ✅ vLLM | Thinking always on — parse `<think>`/`<answer>` |
| baidu/Unlimited-OCR | ✅ vLLM + SGLang | Custom image / dev wheel; see [serving-unlimited-ocr.md](serving-unlimited-ocr.md) |
| baidu/Qianfan-OCR | ✅ vLLM (minimal) | Needs `--hf-overrides '{"architectures": ["InternVLChatModel"]}'` |
| PaddlePaddle/PaddleOCR-VL 1.x | ✅ paddle `genai_server` + vLLM recipe | Serves the 0.9B VLM only; raw serve skips the layout stage (official quality warning) |
| deepseek-ai/DeepSeek-OCR / -2 | ⚠️ vLLM recipe pages only | docs.vllm.ai recipes; needs the DeepSeek n-gram logits processor flags |
| allenai/olmOCR-2-7B-FP8 | ✅ (GitHub) | olmocr toolkit spawns `vllm serve` itself; YAML-front-matter prompt required |
| reducto/RolmOCR | ✅ vLLM | Card serve + client (client's model string has a typo — pass `--served-model-name`) |
| tiiuae/Falcon-OCR | ✅ vLLM in official Docker | `ghcr.io/tiiuae/falcon-ocr`; task-token prompts |
| datalab-to/surya-ocr-2, lift | ⚠️ wrapper-mediated | Server-native but driven via their own managers (`SURYA_INFERENCE_URL`, `lift_vllm`) |
| LiquidAI LFM2.5-VL-Extract | ⚠️ family docs | vLLM ≥0.23 + SGLang cookbooks, not Extract-specific |
| ATH-MaaS/OvisOCR2 | ❌ offline vLLM only | `ovis-ocr2-server.py` is our translation of the card's offline args (parity-validated, table above) |
| FireRedTeam/FireRed-OCR | ❌ transformers only | |
| acvlab/ABot-OCR | ❌ offline script only | pinned vllm 0.18.0 |

SGLang reality check: only GLM-OCR and Unlimited-OCR ship SGLang serve paths (olmOCR
dropped SGLang for vLLM in v0.1.75). For this collection, "server mode" effectively
means a vLLM OpenAI-compatible endpoint.
