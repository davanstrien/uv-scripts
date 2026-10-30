# Language support

What each model's own card claims about language coverage — compiled from the model cards as of **2026-07-15**. Treat these as **claims, not guarantees**: only one model (Surya) publishes per-language scores, and a "multilingual" tag tells you very little. Machine-readable version (plus params, backend, image pins): [`models.json`](models.json).

The fastest way to find out whether a model handles *your* language is to run it on a few of your own pages — `--max-samples 10` costs a few cents on a T4/L4:

```bash
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/surya-ocr.py \
    your-dataset your-output --max-samples 10
```

_Sorted by strength of evidence, then coverage:_

| Model | Card claim | Evidence |
|-------|-----------|----------|
| [Surya OCR 2](https://huggingface.co/datalab-to/surya-ocr-2) | **91 languages** | The only **per-language benchmark** published ([full table](https://github.com/datalab-to/surya/blob/master/static/docs/multilingual.md)): 38/91 score ≥90%, 76/91 ≥80%. Weakest of the top-15: Arabic 72.7%, Vietnamese 73.2% |
| [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) | 100 languages | Explicit **low-resource** claim, backed by an in-house benchmark (1,493 PDFs across 100 languages) — aggregate scores only, not per-language |
| [Tesseract 5](https://github.com/tesseract-ocr/tesseract) | 125 traineddata packs (~100+ languages + script models) | Fully **enumerable** ([tessdata_best](https://github.com/tesseract-ocr/tessdata_best)) — the broadest *named* coverage here, incl. many low-resource languages; script models (Latin, Cyrillic, Devanagari, …) cover languages without a dedicated pack |
| [Qianfan-OCR](https://huggingface.co/baidu/Qianfan-OCR) | 192 languages | Headline claim; no list or per-language numbers |
| [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) / [1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) / [1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | 109 languages; 1.5 adds Tibetan + Bengali | Names scripts (Cyrillic, Arabic, Devanagari, Thai) and claims handwriting + historical documents; no per-language numbers |
| [PP-OCRv6](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6) | 48 languages | Official card claim |
| [MiniCPM-OCR 1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0) | Chinese and English ("primary languages") | Card names two; no per-language numbers. Examples include historical scans |
| [Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B) | 11 named + "many more" (incl. Arabic + CJK) | Illustrative list, no benchmark — but the only card claiming multilingual **handwriting** |
| [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) | 11 (9 European + zh/ja) | Declared, not benchmarked. [v1](https://huggingface.co/lightonai/LightOnOCR-1B-1025) is explicitly European/Latin-script only (9) |
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 8 (zh en fr es ru de ja ko) | Declared in card metadata only; no per-language evidence in the card body |
| [HunyuanOCR-1.5](https://huggingface.co/tencent/HunyuanOCR) | "Multilingual" | Nothing enumerated, but explicitly targets **low-resource + ancient-script OCR** as a design goal (new in 1.5; the pinned 1.0 revision doesn't claim this) |
| [dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr) · [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) / [-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) · [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) · [NuExtract3](https://huggingface.co/numind/NuExtract3) | "Multilingual", unspecified | A tag or one-liner only — no count, list, or benchmark |
| [olmOCR-2](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8) · [Nanonets-OCR-s](https://huggingface.co/nanonets/Nanonets-OCR-s) · [SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview) · [LFM2.5-VL-Extract](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-Extract) | English only | Stated on the card |
| [Falcon-OCR](https://huggingface.co/tiiuae/Falcon-OCR) · [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) · [Xiaomi-OCR-0](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0) · [LightOnOCR-3](https://huggingface.co/collections/lightonai/lightonocr-3) · [FireRed-OCR](https://huggingface.co/FireRedTeam/FireRed-OCR) · [ABot-OCR](https://huggingface.co/acvlab/ABot-OCR) · [RolmOCR](https://huggingface.co/reducto/RolmOCR) · [NuMarkdown-8B](https://huggingface.co/numind/NuMarkdown-8B-Thinking) · [lift](https://huggingface.co/datalab-to/lift) | Not stated | No language information on the card at all |

Text-only extraction: [LFM2-1.2B-Extract](https://huggingface.co/LiquidAI/LFM2-1.2B-Extract) (chains after OCR) names 9 languages: English, Arabic, Chinese, French, German, Japanese, Korean, Portuguese, Spanish.

## Reading the table for low-resource work

- **Only Surya lets you check your language before running anything** — its [benchmark table](https://github.com/datalab-to/surya/blob/master/static/docs/multilingual.md) has a row per language.
- **Tesseract's coverage is enumerable** — if your language has a [traineddata pack](https://github.com/tesseract-ocr/tessdata_best), it's supported (quality varies; it's the legacy baseline, not the quality leader).
- **dots.ocr and HunyuanOCR-1.5 are the VLMs that talk about low-resource languages at all**; PaddleOCR-VL-1.5 names Tibetan and Bengali specifically.
- A big claimed number (192, 109) without a list or per-language scores means you're testing it yourself either way — which is cheap (see the command above).
