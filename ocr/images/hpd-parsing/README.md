# cr.hf.co/davanstrien/hpd-parsing-vllm

Status: built 2026-11-05, **private**. Jobs pull it only once it is public; until then run the
recipe on the vendor image with the uv bootstrap in its docstring.

Runtime image for [`../../hpd-parsing-saturate.py`](../../hpd-parsing-saturate.py): Paddle's
HPD-Parsing image with `uv` added, so the recipe runs with `hf jobs uv run`.

| Tag | Base (Paddle, pinned) | Digest |
|---|---|---|
| `87496aa5-uv0.13.6` (also `latest`) | `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/hpd-parsing-vllm@sha256:87496aa5…` (`latest-nvidia-gpu`, built 2026-07-16) | `sha256:b7d3c561cc245572b11ec3bce0739fd359c564b890d8e848b53b1a8346a573c5` |

Why a custom image: HPD-Parsing runs on Paddle's vLLM fork (`0.17.1+hpdparsing`, request
forking + P-MTP speculative decoding), which ships in their registry image. That image has no
`uv`, which `hf jobs uv run --image <image>` needs. This image adds one layer, `uv` and `uvx` (0.13.6, checksum-verified)
in `/usr/local/bin`. User, venv, environment and entrypoint are unchanged.

Built with [`build.sh`](build.sh) and `crane append` on an HF CPU Job (`cpu-upgrade`); no Docker
daemon. Drop it when Paddle's image ships `uv`, or when request forking lands in an upstream vLLM release.
