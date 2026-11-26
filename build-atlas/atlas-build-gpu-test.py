# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "embedding-atlas>=0.18.0",
#     "datasets",
#     "cuml-cu12",
# ]
#
# [[tool.uv.index]]
# url = "https://pypi.nvidia.com/simple"
# ///

"""Test GPU UMAP via cuml.accel with embedding-atlas."""

import os
import subprocess
import time

# Enable cuml acceleration before anything imports umap
os.environ["CUML_ACCEL_ENABLED"] = "1"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Dataset path or mount path (e.g. /input/dataset.parquet)")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--output", default="/output/atlas")
    parser.add_argument("--text", default="text")
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    print(f"Input: {args.input}")
    print(f"Sample: {args.sample}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print(f"CUML_ACCEL_ENABLED={os.environ.get('CUML_ACCEL_ENABLED')}")

    # Verify cuml is available
    try:
        import cuml
        print(f"cuML version: {cuml.__version__}")
    except ImportError as e:
        print(f"WARNING: cuml not available: {e}")

    # Verify GPU
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
    except ImportError:
        print("torch not installed, skipping GPU check")

    start = time.time()

    cmd = [
        "embedding-atlas",
        args.input,
        "--text", args.text,
        "--batch-size", str(args.batch_size),
        "--export-application", args.output,
    ]

    if args.split:
        cmd.extend(["--split", args.split])

    if args.sample:
        cmd.extend(["--sample", str(args.sample)])

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=os.environ)

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s (exit code: {result.returncode})")

    if result.returncode == 0:
        # Check output
        parquet = os.path.join(args.output, "data", "dataset.parquet")
        if os.path.exists(parquet):
            size_mb = os.path.getsize(parquet) / (1024**2)
            print(f"Output parquet: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
