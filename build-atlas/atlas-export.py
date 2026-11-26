# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface-hub",
#     "torch",  # For GPU detection
#     "datasets",  # For dataset sampling
# ]
# ///

"""
Generate static Embedding Atlas visualizations and deploy to HuggingFace Spaces.

This script creates interactive embedding visualizations that run entirely in the browser,
using WebGPU acceleration for smooth performance with millions of points.

Example usage:
    # Basic usage (creates Space from dataset)
    uv run atlas-export.py \
        stanfordnlp/imdb \
        --space-name my-imdb-viz

    # With custom model and configuration
    uv run atlas-export.py \
        beans \
        --space-name bean-disease-atlas \
        --image-column image \
        --model openai/clip-vit-base-patch32 \
        --sample 10000
    
    # With custom batch size for GPU memory management
    uv run atlas-export.py \
        large-text-dataset \
        --space-name large-text-viz \
        --model sentence-transformers/all-mpnet-base-v2 \
        --batch-size 64  # Increase for faster processing on powerful GPUs
    
    # Small batch size for limited GPU memory
    uv run atlas-export.py \
        image-dataset \
        --space-name image-viz \
        --image-column image \
        --batch-size 8  # Reduce to avoid OOM errors

    # Run on HF Jobs with GPU (requires HF token for Space deployment)
    # Get your token: python -c "from huggingface_hub import get_token; print(get_token())"
    hf jobs uv run --flavor t4-small \
        -s HF_TOKEN=your-token-here \
        https://huggingface.co/datasets/uv-scripts/build-atlas/raw/main/atlas-export.py \
        my-dataset \
        --space-name my-atlas \
        --model nomic-ai/nomic-embed-text-v1.5

    # Use pre-computed embeddings
    uv run atlas-export.py \
        my-dataset-with-embeddings \
        --space-name my-viz \
        --no-compute-embeddings \
        --x-column umap_x \
        --y-column umap_y
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo, get_token, login, upload_folder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_gpu_available() -> bool:
    """Check if GPU is available for computation."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def sample_dataset_to_parquet(
    dataset_id: str,
    sample_size: int,
    split: str = "train",
) -> tuple[Path, Path]:
    """Sample dataset and save to local parquet file."""
    from datasets import load_dataset

    logger.info(f"Pre-sampling {sample_size} examples from {dataset_id}...")

    # Load with streaming to avoid loading entire dataset
    ds = load_dataset(dataset_id, streaming=True, split=split)

    # Shuffle and sample
    ds = ds.shuffle(seed=42)
    sampled_ds = ds.take(sample_size)

    # Create temporary directory and parquet file
    temp_dir = Path(tempfile.mkdtemp(prefix="atlas_data_"))
    parquet_path = temp_dir / "data.parquet"

    logger.info("Saving sampled data to temporary file...")
    sampled_ds.to_parquet(str(parquet_path))

    file_size = parquet_path.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"Created {file_size:.1f}MB parquet file with {sample_size} samples")

    return parquet_path, temp_dir


def build_atlas_command(args) -> tuple[list, str, Optional[Path]]:
    """Build the embedding-atlas command with all parameters."""
    temp_data_dir = None

    # If sampling is requested, pre-sample the dataset
    if args.sample:
        parquet_path, temp_data_dir = sample_dataset_to_parquet(
            args.dataset_id, args.sample, args.split
        )
        dataset_input = str(parquet_path)
    else:
        dataset_input = args.dataset_id

    # Use uvx to run embedding-atlas with required dependencies
   
    cmd = [
        "uvx",
        "--with",
        "datasets",
        "--with",
        "hf-transfer",
        "embedding-atlas",
        dataset_input,
    ]
    # Add all optional parameters
    if args.model:
        cmd.extend(["--model", args.model])

    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])

    # Only specify text column if provided or if not an image-only dataset
    if args.text_column:
        cmd.extend(["--text", args.text_column])
    elif not args.image_column:
        # Default to "text" only for text datasets (when no image column is specified)
        cmd.extend(["--text", "text"])

    if args.image_column:
        cmd.extend(["--image", args.image_column])

    if args.split and not args.sample:  # Don't add split if we pre-sampled
        cmd.extend(["--split", args.split])

    # Remove the --sample parameter since we've already sampled
    # if args.sample:
    #     cmd.extend(["--sample", str(args.sample)])

    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    if not args.compute_embeddings:
        cmd.append("--no-compute-embeddings")

        if args.x_column:
            cmd.extend(["--x", args.x_column])

        if args.y_column:
            cmd.extend(["--y", args.y_column])

        if args.neighbors_column:
            cmd.extend(["--neighbors", args.neighbors_column])

    # Add export flag with output path
    export_path = "atlas_export.zip"
    cmd.extend(["--export-application", export_path])

    return cmd, export_path, temp_data_dir


def create_space_readme(args) -> str:
    """Generate README.md content for the Space."""
    title = args.space_name.replace("-", " ").title()

    readme = f"""---
title: {title}
emoji: 🗺️
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
license: mit
datasets:
- {args.dataset_id}
---

# 🗺️ {title}

Interactive embedding visualization of [{args.dataset_id}](https://huggingface.co/datasets/{args.dataset_id}) using [Embedding Atlas](https://github.com/apple/embedding-atlas).

## Features

- Interactive embedding visualization
- Real-time search and filtering
- Automatic clustering with labels
- WebGPU-accelerated rendering
"""

    if args.model:
        readme += f"\n## Model\n\nEmbeddings generated using: `{args.model}`\n"

    if args.sample:
        readme += f"\n## Data\n\nVisualization includes {args.sample:,} samples from the dataset.\n"

    readme += """
## How to Use

- **Click and drag** to navigate
- **Scroll** to zoom in/out
- **Click** on points to see details
- **Search** using the search box
- **Filter** using metadata panels

---

*Generated with [UV Scripts Atlas Export](https://huggingface.co/uv-scripts)*
"""

    return readme


def extract_and_prepare_static_files(zip_path: str, output_dir: Path) -> None:
    """Extract the exported atlas ZIP and prepare for static deployment."""
    logger.info(f"Extracting {zip_path} to {output_dir}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    # The ZIP should contain index.html and associated files
    if not (output_dir / "index.html").exists():
        raise FileNotFoundError("index.html not found in exported atlas")

    logger.info(f"Extracted {len(list(output_dir.iterdir()))} items")


def deploy_to_space(
    output_dir: Path,
    space_name: str,
    organization: Optional[str] = None,
    private: bool = False,
    hf_token: Optional[str] = None,
) -> str:
    """Deploy the static files to a HuggingFace Space."""
    api = HfApi(token=hf_token)

    # Construct full repo ID
    if organization:
        repo_id = f"{organization}/{space_name}"
    else:
        # Get username from API
        user_info = api.whoami()
        username = user_info["name"]
        repo_id = f"{username}/{space_name}"

    logger.info(f"Creating Space: {repo_id}")

    # Create the Space repository
    try:
        create_repo(
            repo_id,
            repo_type="space",
            space_sdk="static",
            private=private,
            token=hf_token,
            exist_ok=True,  # Allow updating existing Space
        )
        logger.info(f"Created new Space: {repo_id}")
    except Exception as e:
        if "already exists" in str(e):
            logger.info(f"Space {repo_id} already exists, updating...")
        else:
            raise

    # Upload all files
    logger.info("Uploading files to Space...")
    upload_folder(
        folder_path=str(output_dir), repo_id=repo_id, repo_type="space", token=hf_token
    )

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    logger.info(f"✅ Space deployed successfully: {space_url}")

    return space_url


def main():
   

    parser = argparse.ArgumentParser(
        description="Generate and deploy static Embedding Atlas visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "dataset_id",
        type=str,
        help="HuggingFace dataset ID to visualize",
    )

    # Space configuration
    parser.add_argument(
        "--space-name",
        type=str,
        required=True,
        help="Name for the HuggingFace Space",
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="HuggingFace organization (default: your username)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Space private",
    )

    # Atlas configuration
    parser.add_argument(
        "--model",
        type=str,
        help="Embedding model to use (e.g., sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        help="Name of text column (default: auto-detect)",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        help="Name of image column for image datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Number of samples to visualize (default: all)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code in dataset/model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for processing embeddings (default: 32 for text, 16 for images). "
        "Larger values use more memory but may be faster on GPUs",
    )

    # Pre-computed embeddings
    parser.add_argument(
        "--no-compute-embeddings",
        action="store_false",
        dest="compute_embeddings",
        help="Use pre-computed embeddings from dataset",
    )
    parser.add_argument(
        "--x-column",
        type=str,
        help="Column with X coordinates (for pre-computed projections)",
    )
    parser.add_argument(
        "--y-column",
        type=str,
        help="Column with Y coordinates (for pre-computed projections)",
    )
    parser.add_argument(
        "--neighbors-column",
        type=str,
        help="Column with neighbor indices (for pre-computed)",
    )

    # Additional options
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only generate locally, don't deploy to Space",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Local directory for output (default: temp directory)",
    )

    args = parser.parse_args()

    # Check GPU availability
    if check_gpu_available():
        logger.info("🚀 GPU detected - may accelerate embedding generation")
    else:
        logger.info("💻 Running on CPU - embedding generation may be slower")

    # Login to HuggingFace if needed
    if not args.local_only:
        # Try to get token from various sources
        hf_token = (
            args.hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or get_token()  # This will check HF CLI login
        )

        if hf_token:
            login(token=hf_token)
            logger.info("✅ Authenticated with Hugging Face")
        elif sys.stdin.isatty():
            logger.warning(
                "No HF token provided. You may not be able to push to the Hub."
            )
            response = input("Continue anyway? (y/n): ")
            if response.lower() != "y":
                sys.exit(0)
        else:
            # In non-interactive environments, fail immediately if no token
            logger.error(
                "No HF token found. Cannot deploy to Space in non-interactive environment."
            )
            logger.error(
                "Please set HF_TOKEN environment variable or use --hf-token argument."
            )
            logger.error("Checked: HF_TOKEN, HUGGING_FACE_HUB_TOKEN, and HF CLI login")
            sys.exit(1)

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = None
    else:
        temp_dir = tempfile.mkdtemp(prefix="atlas_export_")
        output_dir = Path(temp_dir)
        logger.info(f"Using temporary directory: {output_dir}")

    temp_data_dir = None  # Initialize to avoid UnboundLocalError

    try:
        # Build and run embedding-atlas command
        cmd, export_path, temp_data_dir = build_atlas_command(args)
        logger.info(f"Running command: {' '.join(cmd)}")

        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Atlas export failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            sys.exit(1)

        logger.info("✅ Atlas export completed successfully")

        # Extract the exported files
        extract_and_prepare_static_files(export_path, output_dir)

        # Create README for the Space
        readme_content = create_space_readme(args)
        (output_dir / "README.md").write_text(readme_content)

        if args.local_only:
            logger.info(f"✅ Static files prepared in: {output_dir}")
            logger.info(
                "To deploy manually, upload the contents to a HuggingFace Space with sdk: static"
            )
        else:
            # Deploy to HuggingFace Space
            space_url = deploy_to_space(
                output_dir, args.space_name, args.organization, args.private, hf_token
            )

            logger.info(f"\n🎉 Success! Your atlas is live at: {space_url}")
            logger.info("The visualization will be available in a few moments.")

        # Clean up the ZIP file
        if Path(export_path).exists():
            os.remove(export_path)

    finally:
        # Clean up temp directory if used
        if temp_dir and not args.local_only:
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")

        # Also clean up the data sampling directory
        if temp_data_dir and temp_data_dir.exists():
            shutil.rmtree(temp_data_dir)
            logger.info("Cleaned up sampled data")


if __name__ == "__main__":
    # Show example commands if no args provided
    if len(sys.argv) == 1:
        print("Example commands:\n")
        print("# Basic usage:")
        print("uv run atlas-export.py stanfordnlp/imdb --space-name imdb-atlas\n")
        print("# With custom model and sampling:")
        print(
            "uv run atlas-export.py my-dataset --space-name my-viz --model nomic-ai/nomic-embed-text-v1.5 --sample 10000\n"
        )
        print("# For HF Jobs with GPU (experimental UV support):")
        print(
            '# First get your token: python -c "from huggingface_hub import get_token; print(get_token())"'
        )
        print(
            "hf jobs uv run --flavor t4-small -s HF_TOKEN=your-token-here https://huggingface.co/datasets/uv-scripts/build-atlas/raw/main/atlas-export.py dataset --space-name viz --model sentence-transformers/all-mpnet-base-v2\n"
        )
        print("# Local generation only:")
        print(
            "uv run atlas-export.py dataset --space-name test --local-only --output-dir ./atlas-output"
        )
        sys.exit(0)

    main()
