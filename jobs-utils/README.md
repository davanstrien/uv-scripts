---
viewer: false
tags:
  - uv-script
  - huggingface-jobs
  - utilities
  - benchmarking
license: apache-2.0
---

# HF Jobs Utilities

Small utility scripts for working with the [uv-scripts](https://huggingface.co/uv-scripts) collection and Hugging Face Jobs — discovering the available recipes, and testing/benchmarking your Jobs setup.

## Available Scripts

### list-recipes.py

List every recipe (UV script) across the whole `uv-scripts` org, with a runnable URL for each. The zero-setup way to see what's available — no GPU, no token, no account; it only reads public repos.

**Run locally** (nothing to install but [uv](https://docs.astral.sh/uv/getting-started/installation/)):
```bash
uv run https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/list-recipes.py
```

Add `--describe` to also print the first line of each script's docstring (slower — it fetches each file):
```bash
uv run https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/list-recipes.py --describe
```

Each printed URL runs the same way: `uv run <url>` locally, or `hf jobs uv run <url>` on a GPU.

### network-speed-test.py

Test network download speed from within an HF Jobs environment. Useful for:
- Verifying network connectivity
- Comparing download speeds across different GPU flavors
- Benchmarking before running large dataset downloads
- Troubleshooting slow data loading issues

**Features:**
- Pure Python (no dependencies required)
- Fast default test with small file
- Optional `--large` flag for realistic ~100MB test
- Clear metrics (MB/s and Mbps)
- Works locally or on HF Jobs

#### Usage

**Run locally:**
```bash
# Quick test with small file
uv run https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/network-speed-test.py

# More realistic test with ~100MB file
uv run https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/network-speed-test.py --large

# Custom URL
uv run https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/network-speed-test.py \
    --url https://example.com/testfile.bin
```

**Run on HF Jobs:**
```bash
# Test network speed on L4 GPU
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/network-speed-test.py --large

# Compare across different flavors
hf jobs uv run --flavor a10g-small \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/network-speed-test.py --large

hf jobs uv run --flavor h200 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/network-speed-test.py --large
```

#### Example Output

```
Using large file test (~100MB)
============================================================
HF Jobs Network Speed Test
============================================================

Testing download speed from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin
Downloading...

============================================================
Results:
============================================================
Downloaded: 86.68 MB
Time: 2.79 seconds
Speed: 31.03 MB/s (260.31 Mbps)
============================================================
```

## Why These Utilities?

HF Jobs provides different GPU flavors with varying network configurations. These utilities help you:

1. **Make informed decisions** - Compare network speeds before choosing a flavor
2. **Debug issues** - Verify network connectivity when data loading is slow
3. **Optimize workflows** - Understand if network speed is a bottleneck
4. **Benchmark environments** - Document baseline performance for your use case

## About UV Scripts

This is part of the [uv-scripts](https://huggingface.co/uv-scripts) collection - ready-to-run Python scripts that work with a single `uv run` command. No setup, no virtual environments, just instant execution.

Learn more about UV: https://docs.astral.sh/uv/

## License

Apache 2.0
