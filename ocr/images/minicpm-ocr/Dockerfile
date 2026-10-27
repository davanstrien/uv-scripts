# MiniCPM-OCR 1.0 runtime for Hugging Face Jobs: cr.hf.co/davanstrien/minicpm-ocr
#
# - vLLM nightly 5d35ce3 (27 Oct 2026). MiniCPMOcrForConditionalGeneration landed in vLLM main on
#   23 Oct (18343393b4, the commit OpenBMB's install.sh builds from source) and is not in a
#   release yet; this nightly is newer and also has DSpark speculative decoding.
# - transformers at OpenBMB's pin (549e5b4, "5.21.0.dev0"), inside the nightly's
#   transformers>=5.16.1,<5.21.0 requirement.
# - The minicpm-ocr SDK (layout, crops, assembly) in its own venv at /opt/sdk, as OpenBMB's docs
#   require (SDK and inference service in separate environments), with PaddlePaddle GPU for the
#   PP-DocLayoutV3 layout model.
# - The layout model is baked in (sha256-checked) at $OCR_ASSETS_DIR, so a run never fetches it
#   from Paddle's CDN. Model weights are not baked in: pull a pinned revision at run time.
FROM vllm/vllm-openai:nightly-5d35ce3934a7804b298559d2d99eec33792e1eb9@sha256:3431ebee919296f7a5f3604db353f639e1474a4c7ef37595e31de6699fd831a7

ARG TRANSFORMERS_COMMIT=549e5b410733fb50e8efbbe4aa585cb89f6478ea
ARG MINICPM_OCR_SDK=1.0.1
ARG LAYOUT_URL=https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayoutV3_infer.tar
ARG LAYOUT_SHA256=98b9bac88c80f6bc0fda7e0bfc2cae180020371c0b2edbb1eb498a70ace751b1

RUN python3 -m pip install --no-cache-dir \
      "transformers @ https://github.com/huggingface/transformers/archive/${TRANSFORMERS_COMMIT}.tar.gz"

RUN python3 -m pip install --no-cache-dir uv \
 && uv venv /opt/sdk --python 3.12 \
 && uv pip install --python /opt/sdk/bin/python --no-cache \
      "minicpm-ocr[all]==${MINICPM_OCR_SDK}" "paddlex==3.4.2" "pypdfium2>=4" \
 && uv pip install --python /opt/sdk/bin/python --no-cache \
      --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ "paddlepaddle-gpu==3.3.0"

ENV OCR_ASSETS_DIR=/opt/minicpm-ocr-assets
RUN mkdir -p ${OCR_ASSETS_DIR}/models \
 && curl -fsSL -o /tmp/layout.tar "${LAYOUT_URL}" \
 && echo "${LAYOUT_SHA256}  /tmp/layout.tar" | sha256sum -c - \
 && tar -xf /tmp/layout.tar -C ${OCR_ASSETS_DIR}/models \
 && rm /tmp/layout.tar

# The SDK CLI is on PATH as `minicpmvocr`; vLLM stays the image's `vllm`.
RUN ln -s /opt/sdk/bin/minicpmvocr /usr/local/bin/minicpmvocr \
 && minicpmvocr --help > /dev/null

ENTRYPOINT []
CMD ["bash"]
