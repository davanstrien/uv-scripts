#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers@git+https://github.com/huggingface/transformers.git@1fba72361e8e0e865d569f7cd15e5aa50b41ac9a",
#     "datasets",
#     "huggingface-hub",
#     "pillow",
#     "tqdm",
#     "torchvision",
#     "accelerate",
# ]
# ///

"""
Detect objects in images using Meta's SAM3 (Segment Anything Model 3).

This script processes images from a HuggingFace dataset and detects a single object
type based on a text prompt, outputting bounding boxes in HuggingFace object detection format.

Examples:
    # Detect photographs in historical newspapers
    uv run detect-objects.py \\
        davanstrien/newspapers-with-images-after-photography \\
        my-username/newspapers-detected \\
        --class-name photograph

    # Detect animals in camera trap images
    uv run detect-objects.py \\
        wildlife-images \\
        wildlife-detected \\
        --class-name animal \\
        --confidence-threshold 0.6

    # Test on small subset
    uv run detect-objects.py input output \\
        --class-name table \\
        --max-samples 10

    # Run on HF Jobs with GPU
    hf jobs uv run --flavor a100-large \\
        -s HF_TOKEN=HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/sam3/raw/main/detect-objects.py \\
        input-dataset output-dataset \\
        --class-name photograph \\
        --confidence-threshold 0.5

Note: To detect multiple object types, run the script multiple times with different
      --class-name values and merge the results.
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List

import torch
from datasets import ClassLabel, Dataset, Sequence, Value, load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
from transformers import Sam3Model, Sam3Processor

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# GPU availability check
if not torch.cuda.is_available():
    logger.error("❌ CUDA is not available. This script requires a GPU.")
    logger.error("For local testing, ensure you have a CUDA-capable GPU.")
    logger.error("For cloud execution, use HF Jobs with --flavor l4x1 or similar.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect objects in images using SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "input_dataset", help="Input HuggingFace dataset ID (e.g., 'username/dataset')"
    )
    parser.add_argument(
        "output_dataset", help="Output HuggingFace dataset ID (e.g., 'username/output')"
    )

    # Object detection configuration
    parser.add_argument(
        "--class-name",
        required=True,
        help="Object class to detect (e.g., 'photograph', 'animal', 'table')",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score for detections (default: 0.5)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold for mask generation (default: 0.5)",
    )

    # Dataset configuration
    parser.add_argument(
        "--image-column",
        default="image",
        help="Name of the column containing images (default: 'image')",
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split to process (default: 'train')"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle dataset before processing"
    )

    # Processing configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)",
    )
    parser.add_argument(
        "--model",
        default="facebook/sam3",
        help="SAM3 model ID (default: 'facebook/sam3')",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model precision (default: 'bfloat16')",
    )

    # Output configuration
    parser.add_argument(
        "--private", action="store_true", help="Make output dataset private"
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token (default: uses HF_TOKEN env var or cached token)",
    )

    return parser.parse_args()


def create_dataset_card(
    source_dataset: str,
    model: str,
    class_name: str,
    num_samples: int,
    total_detections: int,
    images_with_detections: int,
    processing_time: str,
    confidence_threshold: float,
    mask_threshold: float,
    batch_size: int,
    dtype: str,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the object detection process."""
    from datetime import datetime

    model_name = model.split("/")[-1]
    avg_detections = total_detections / num_samples if num_samples > 0 else 0
    detection_rate = (
        (images_with_detections / num_samples * 100) if num_samples > 0 else 0
    )

    return f"""---
tags:
- object-detection
- sam3
- segment-anything
- bounding-boxes
- uv-script
- generated
---

# Object Detection: {class_name.title()} Detection using {model_name}

This dataset contains object detection results (bounding boxes) for **{class_name}** detected in images from [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using Meta's SAM3 (Segment Anything Model 3).

**Generated using**: [uv-scripts/sam3](https://huggingface.co/datasets/uv-scripts/sam3) detection script

## Detection Statistics

- **Objects Detected**: {class_name}
- **Total Detections**: {total_detections:,}
- **Images with Detections**: {images_with_detections:,} / {num_samples:,} ({detection_rate:.1f}%)
- **Average Detections per Image**: {avg_detections:.2f}

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Script Repository**: [uv-scripts/sam3](https://huggingface.co/datasets/uv-scripts/sam3)
- **Number of Samples Processed**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Dataset Split**: `{split}`
- **Class Name**: `{class_name}`
- **Confidence Threshold**: {confidence_threshold}
- **Mask Threshold**: {mask_threshold}
- **Batch Size**: {batch_size}
- **Model Dtype**: {dtype}

## Model Information

SAM3 (Segment Anything Model 3) is Meta's state-of-the-art object detection and segmentation model that excels at:
- 🎯 **Zero-shot detection** - Detect objects using natural language prompts
- 📦 **Bounding boxes** - Accurate object localization
- 🎭 **Instance segmentation** - Pixel-perfect masks (not included in this dataset)
- 🖼️ **Any image domain** - Works on photos, documents, medical images, etc.

This dataset uses SAM3 in text-prompted detection mode to find instances of "{class_name}" in the source images.

## Dataset Structure

The dataset contains all original columns from the source dataset plus an `objects` column with detection results in HuggingFace object detection format (dict-of-lists):

- **bbox**: List of bounding boxes in `[x, y, width, height]` format (pixel coordinates)
- **category**: List of category indices (always `0` for single-class detection)
- **score**: List of confidence scores (0.0 to 1.0)

### Schema

```python
{{
    "objects": {{
        "bbox": [[x, y, w, h], ...],      # List of bounding boxes
        "category": [0, 0, ...],           # All same class
        "score": [0.95, 0.87, ...]        # Confidence scores
    }}
}}
```

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{{{{output_dataset_id}}}}", split="{split}")

# Access detections for an image
example = dataset[0]
detections = example["objects"]

# Iterate through all detected objects in this image
for bbox, category, score in zip(
    detections["bbox"],
    detections["category"],
    detections["score"]
):
    x, y, w, h = bbox
    print(f"Detected {class_name} at ({{x}}, {{y}}) with confidence {{score:.2f}}")

# Filter high-confidence detections
high_conf_examples = [
    ex for ex in dataset
    if any(score > 0.8 for score in ex["objects"]["score"])
]

# Count total detections across dataset
total = sum(len(ex["objects"]["bbox"]) for ex in dataset)
print(f"Total detections: {{total}}")
```

## Visualization

To visualize the detections, you can use the visualization script from the same repository:

```bash
# Visualize first sample with detections
uv run https://huggingface.co/datasets/uv-scripts/sam3/raw/main/visualize-detections.py \\
    {{{{output_dataset_id}}}} \\
    --first-with-detections

# Visualize random samples
uv run https://huggingface.co/datasets/uv-scripts/sam3/raw/main/visualize-detections.py \\
    {{{{output_dataset_id}}}} \\
    --num-samples 5

# Save visualizations to files
uv run https://huggingface.co/datasets/uv-scripts/sam3/raw/main/visualize-detections.py \\
    {{{{output_dataset_id}}}} \\
    --num-samples 3 \\
    --output-dir ./visualizations
```

## Reproduction

This dataset was generated using the [uv-scripts/sam3](https://huggingface.co/datasets/uv-scripts/sam3) object detection script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/sam3/raw/main/detect-objects.py \\
    {source_dataset} \\
    <output-dataset> \\
    --class-name {class_name} \\
    --confidence-threshold {confidence_threshold} \\
    --mask-threshold {mask_threshold} \\
    --batch-size {batch_size} \\
    --dtype {dtype}
```

### Running on HuggingFace Jobs (GPU)

This script requires a GPU. To run on HuggingFace infrastructure:

```bash
hf jobs uv run --flavor a100-large \\
    -s HF_TOKEN=HF_TOKEN \\
    https://huggingface.co/datasets/uv-scripts/sam3/raw/main/detect-objects.py \\
    {source_dataset} \\
    <output-dataset> \\
    --class-name {class_name} \\
    --confidence-threshold {confidence_threshold}
```

## Performance

- **Processing Speed**: ~{num_samples / (float(processing_time.split()[0]) * 60) if processing_time.split()[0].replace(".", "").isdigit() else "N/A":.1f} images/second
- **GPU Configuration**: CUDA with {dtype} precision

---

Generated with 🤖 [UV Scripts](https://huggingface.co/uv-scripts)
"""


def load_and_validate_dataset(
    dataset_id: str,
    split: str,
    image_column: str,
    max_samples: int = None,
    shuffle: bool = False,
    hf_token: str = None,
) -> Dataset:
    """Load dataset and validate it has the required image column."""
    logger.info(f"📂 Loading dataset: {dataset_id} (split: {split})")

    try:
        dataset = load_dataset(dataset_id, split=split, token=hf_token)
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_id}': {e}")
        sys.exit(1)

    # Validate image column exists
    if image_column not in dataset.column_names:
        logger.error(f"Column '{image_column}' not found in dataset")
        logger.error(f"Available columns: {dataset.column_names}")
        sys.exit(1)

    # Shuffle if requested
    if shuffle:
        logger.info("🔀 Shuffling dataset")
        dataset = dataset.shuffle()

    # Limit samples if requested
    if max_samples is not None:
        logger.info(f"🔢 Limiting to {max_samples} samples")
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    logger.info(f"✅ Loaded {len(dataset)} samples")
    return dataset


def process_batch(
    batch: Dict[str, List[Any]],
    image_column: str,
    class_name: str,
    processor: Sam3Processor,
    model: Sam3Model,
    confidence_threshold: float,
    mask_threshold: float,
) -> Dict[str, List[List[Dict[str, Any]]]]:
    """Process a batch of images and return detections for a single class."""
    images = batch[image_column]

    # Convert to PIL Images and ensure RGB
    pil_images = []
    for img in images:
        if isinstance(img, str):
            img = Image.open(img)
        if img.mode == "L" or img.mode != "RGB":
            img = img.convert("RGB")
        pil_images.append(img)

    # Process batch through model
    try:
        inputs = processor(
            images=pil_images,
            text=[class_name] * len(pil_images),  # Same prompt for all images
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process outputs using original_sizes from processor
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=confidence_threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )

    except Exception as e:
        logger.warning(f"⚠️  Failed to process batch: {e}")
        # Return empty detections for all images in batch
        return {
            "objects": [
                {"bbox": [], "category": [], "score": []}
                for _ in range(len(pil_images))
            ]
        }

    # Convert to HuggingFace object detection format (dict-of-lists per image)
    batch_objects = []

    for result in results:
        boxes = result.get("boxes", torch.tensor([]))
        scores = result.get("scores", torch.tensor([]))

        # Handle empty results
        if len(boxes) == 0:
            batch_objects.append({"bbox": [], "category": [], "score": []})
            continue

        # Build lists for this image
        image_bboxes = []
        image_categories = []
        image_scores = []

        # Convert to float32 before numpy (bfloat16 not supported by numpy)
        boxes_np = boxes.cpu().float().numpy()
        scores_np = scores.cpu().float().numpy()

        for box, score in zip(boxes_np, scores_np):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            image_bboxes.append([float(x1), float(y1), float(width), float(height)])
            image_categories.append(0)  # Single class, always index 0
            image_scores.append(float(score))

        batch_objects.append(
            {
                "bbox": image_bboxes,
                "category": image_categories,
                "score": image_scores,
            }
        )

    return {"objects": batch_objects}


def main():
    args = parse_args()

    class_name = args.class_name.strip()
    if not class_name:
        logger.error("❌ Invalid --class-name argument. Provide a class name.")
        sys.exit(1)

    logger.info("🚀 SAM3 Object Detection")
    logger.info(f"   Input: {args.input_dataset}")
    logger.info(f"   Output: {args.output_dataset}")
    logger.info(f"   Class: {class_name}")
    logger.info(f"   Confidence threshold: {args.confidence_threshold}")
    logger.info(f"   Batch size: {args.batch_size}")

    # Authentication
    if args.hf_token:
        login(token=args.hf_token)
    elif os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))

    # Load dataset
    dataset = load_and_validate_dataset(
        args.input_dataset,
        args.split,
        args.image_column,
        args.max_samples,
        args.shuffle,
        args.hf_token,
    )

    # Load model
    logger.info(f"🤖 Loading SAM3 model: {args.model}")
    try:
        processor = Sam3Processor.from_pretrained(args.model)
        model = Sam3Model.from_pretrained(
            args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto"
        )
        logger.info(f"✅ Model loaded on {model.device}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error("Ensure the model exists and you have access permissions")
        sys.exit(1)

    # Define output schema before processing (dict-of-lists format for object detection)
    logger.info("📊 Creating output schema...")
    new_features = dataset.features.copy()
    new_features["objects"] = {
        "bbox": Sequence(Sequence(Value("float32"), length=4)),
        "category": Sequence(ClassLabel(names=[class_name])),
        "score": Sequence(Value("float32")),
    }

    # Process dataset with explicit output features
    logger.info("🔍 Processing images...")
    start_time = time.time()
    processed_dataset = dataset.map(
        lambda batch: process_batch(
            batch,
            args.image_column,
            class_name,
            processor,
            model,
            args.confidence_threshold,
            args.mask_threshold,
        ),
        batched=True,
        batch_size=args.batch_size,
        features=new_features,
        desc="Detecting objects",
    )
    end_time = time.time()
    processing_time_seconds = end_time - start_time
    processing_time_str = f"{processing_time_seconds / 60:.1f} minutes"

    # Calculate statistics
    total_detections = sum(len(objs) for objs in processed_dataset["objects"])
    images_with_detections = sum(len(objs) > 0 for objs in processed_dataset["objects"])

    logger.info("✅ Detection complete!")
    logger.info(f"   Total detections: {total_detections}")
    logger.info(
        f"   Images with detections: {images_with_detections}/{len(processed_dataset)}"
    )
    logger.info(
        f"   Average detections per image: {total_detections / len(processed_dataset):.2f}"
    )

    # Push to hub
    logger.info(f"📤 Pushing to HuggingFace Hub: {args.output_dataset}")
    try:
        processed_dataset.push_to_hub(args.output_dataset, private=args.private)
        logger.info(
            f"✅ Dataset available at: https://huggingface.co/datasets/{args.output_dataset}"
        )
    except Exception as e:
        logger.error(f"❌ Failed to push to hub: {e}")
        logger.info("💾 Saving locally as backup...")
        processed_dataset.save_to_disk("./output_dataset")
        logger.info("✅ Saved to ./output_dataset")
        sys.exit(1)

    # Create and push dataset card
    logger.info("📝 Creating dataset card...")
    card_content = create_dataset_card(
        source_dataset=args.input_dataset,
        model=args.model,
        class_name=class_name,
        num_samples=len(processed_dataset),
        total_detections=total_detections,
        images_with_detections=images_with_detections,
        processing_time=processing_time_str,
        confidence_threshold=args.confidence_threshold,
        mask_threshold=args.mask_threshold,
        batch_size=args.batch_size,
        dtype=args.dtype,
        image_column=args.image_column,
        split=args.split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(args.output_dataset, token=args.hf_token or os.getenv("HF_TOKEN"))
    logger.info("✅ Dataset card created and pushed!")


if __name__ == "__main__":
    main()
