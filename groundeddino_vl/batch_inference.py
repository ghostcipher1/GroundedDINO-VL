#!/usr/bin/env python3
"""Batch inference script for GroundedDINO-VL.

Processes directories of images with automatic class detection and exports results
in multiple formats (COCO JSON, YOLO, Label Studio, CSV).

Usage:
    # Basic usage with default classes
    python -m groundeddino_vl.batch_inference \
        --input /data/datasets/bears \
        --output /data/results/bears \
        --config models/GroundingDINO_SwinB_cfg.py \
        --checkpoint checkpoints/groundingdino_swinb_cogcoor.pth

    # Custom class list
    python -m groundeddino_vl.batch_inference \
        --input /data/datasets/wildlife \
        --output /data/results/wildlife \
        --classes "bear,wolf,cougar,coyote,fox" \
        --box-threshold 0.30 \
        --text-threshold 0.25

    # Export to Label Studio format
    python -m groundeddino_vl.batch_inference \
        --input /data/datasets/predators \
        --output /data/results/predators \
        --format labelstudio \
        --batch-size 32

Environment Variables:
    GDVL_CONFIG: Default config path
    GDVL_CHECKPOINT: Default checkpoint path
    CUDA_VISIBLE_DEVICES: GPU selection (default: 0)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Import GroundedDINO-VL components
try:
    from groundeddino_vl.utils.inference import Model
except ImportError:
    print("ERROR: Failed to import groundeddino_vl. Install with: pip install -e .")
    sys.exit(1)


# Default wildlife classes (same as Label Studio backend)
DEFAULT_WILDLIFE_CLASSES = [
    "bear",
    "bobcat",
    "cougar",
    "coyote",
    "fox",
    "wolf",
    "raccoon",
    "skunk",
    "opossum",
    "snake",
    "horse",
    "cow",
    "sheep",
    "goat",
    "peafowl",
    "duck",
    "rabbit",
    "chicken",
    "pig",
    "deer",
    "automobile",
    "pickup",
    "tractor",
    "person",
]

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def find_images(input_dir: Path) -> List[Path]:
    """Recursively find all images in input directory."""
    images = []
    for ext in SUPPORTED_IMAGE_FORMATS:
        images.extend(input_dir.rglob(f"*{ext}"))
        images.extend(input_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def export_coco_json(results: List[Dict[str, Any]], output_path: Path, classes: List[str]) -> None:
    """Export results to COCO JSON format."""
    coco_output = {
        "info": {
            "description": "GroundedDINO-VL Batch Inference Results",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Build category list
    for idx, class_name in enumerate(classes):
        coco_output["categories"].append({"id": idx + 1, "name": class_name, "supercategory": ""})

    class_to_id = {name: idx + 1 for idx, name in enumerate(classes)}
    annotation_id = 1

    for img_id, result in enumerate(results, start=1):
        # Add image info
        coco_output["images"].append(
            {
                "id": img_id,
                "file_name": result["filename"],
                "width": result["width"],
                "height": result["height"],
            }
        )

        # Add annotations
        for detection in result["detections"]:
            x1, y1, x2, y2 = detection["xyxy"]
            width = x2 - x1
            height = y2 - y1
            area = width * height

            coco_output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": class_to_id.get(detection["label"], 0),
                    "bbox": [x1, y1, width, height],
                    "area": area,
                    "iscrowd": 0,
                    "score": detection["score"],
                }
            )
            annotation_id += 1

    with open(output_path, "w") as f:
        json.dump(coco_output, f, indent=2)


def export_yolo_format(results: List[Dict[str, Any]], output_dir: Path, classes: List[str]) -> None:
    """Export results to YOLO format (one .txt per image)."""
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_to_id = {name: idx for idx, name in enumerate(classes)}

    # Write classes.txt
    with open(output_dir / "classes.txt", "w") as f:
        f.write("\n".join(classes))

    for result in results:
        label_file = labels_dir / f"{Path(result['filename']).stem}.txt"
        with open(label_file, "w") as f:
            for detection in result["detections"]:
                x1, y1, x2, y2 = detection["xyxy"]
                w = result["width"]
                h = result["height"]

                # Convert to YOLO format: class_id x_center y_center width height
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                box_width = (x2 - x1) / w
                box_height = (y2 - y1) / h

                class_id = class_to_id.get(detection["label"], 0)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} ")
                f.write(f"{box_width:.6f} {box_height:.6f}\n")


def export_labelstudio_json(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Export results to Label Studio JSON format."""
    ls_tasks = []

    for result in results:
        annotations = []
        for detection in result["detections"]:
            x1, y1, x2, y2 = detection["xyxy"]
            w = result["width"]
            h = result["height"]

            # Convert to Label Studio percentage format
            x_percent = (x1 / w) * 100
            y_percent = (y1 / h) * 100
            width_percent = ((x2 - x1) / w) * 100
            height_percent = ((y2 - y1) / h) * 100

            annotations.append(
                {
                    "value": {
                        "x": x_percent,
                        "y": y_percent,
                        "width": width_percent,
                        "height": height_percent,
                        "rotation": 0,
                        "rectanglelabels": [detection["label"]],
                    },
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "score": detection["score"],
                }
            )

        ls_tasks.append(
            {
                "data": {"image": f"/data/local-files/?d={result['filepath']}"},
                "predictions": [
                    {
                        "result": annotations,
                        "score": result.get("avg_score", 0.0),
                        "model_version": result.get("model_version", "unknown"),
                    }
                ],
            }
        )

    with open(output_path, "w") as f:
        json.dump(ls_tasks, f, indent=2)


def export_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Export results to CSV format."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "score", "x1", "y1", "x2", "y2", "width", "height"])

        for result in results:
            for detection in result["detections"]:
                x1, y1, x2, y2 = detection["xyxy"]
                writer.writerow(
                    [
                        result["filename"],
                        detection["label"],
                        f"{detection['score']:.4f}",
                        f"{x1:.2f}",
                        f"{y1:.2f}",
                        f"{x2:.2f}",
                        f"{y2:.2f}",
                        result["width"],
                        result["height"],
                    ]
                )


def process_batch(
    model: Model,
    image_paths: List[Path],
    classes: List[str],
    box_threshold: float,
    text_threshold: float,
) -> Tuple[List[Dict[str, Any]], float]:
    """Process a batch of images and return results."""
    import cv2

    results = []
    total_time = 0.0

    for img_path in image_paths:
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"WARNING: Failed to load {img_path}, skipping")
                continue

            h, w = image.shape[:2]

            # Run inference
            start_time = time.time()
            detections = model.predict_with_classes(
                image=image,
                classes=classes,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            inference_time = time.time() - start_time
            total_time += inference_time

            # Extract detections
            detection_list = []
            if hasattr(detections, "xyxy") and detections.xyxy is not None:
                xyxy = np.asarray(detections.xyxy)
                scores = (
                    np.asarray(detections.confidence)
                    if hasattr(detections, "confidence")
                    else np.zeros(len(xyxy))
                )
                class_ids = (
                    np.asarray(detections.class_id)
                    if hasattr(detections, "class_id")
                    else np.zeros(len(xyxy), dtype=int)
                )

                for i in range(len(xyxy)):
                    class_id = int(class_ids[i]) if i < len(class_ids) else 0
                    label = classes[class_id] if class_id < len(classes) else "unknown"

                    detection_list.append(
                        {
                            "xyxy": xyxy[i].tolist(),
                            "score": float(scores[i]) if i < len(scores) else 0.0,
                            "label": label,
                        }
                    )

            results.append(
                {
                    "filename": img_path.name,
                    "filepath": str(img_path),
                    "width": w,
                    "height": h,
                    "detections": detection_list,
                    "inference_time": inference_time,
                    "num_detections": len(detection_list),
                    "avg_score": (
                        float(np.mean([d["score"] for d in detection_list]))
                        if detection_list
                        else 0.0
                    ),
                }
            )

        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
            continue

    return results, total_time


def main() -> None:
    """Main entry point for batch inference."""
    parser = argparse.ArgumentParser(
        description="Batch inference with GroundedDINO-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input directory containing images"
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output directory for results"
    )

    # Model configuration
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("GDVL_CONFIG"),
        help="Path to model config file (.py)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.environ.get("GDVL_CHECKPOINT"),
        help="Path to model checkpoint (.pth)",
    )

    # Detection parameters
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help='Comma-separated class list (e.g., "bear,wolf,cougar"). Default: wildlife classes',
    )
    parser.add_argument(
        "--box-threshold", type=float, default=0.25, help="Box confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Text confidence threshold (default: 0.25)",
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of images to process before saving checkpoint (default: 1)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu, default: cuda)"
    )

    # Export format
    parser.add_argument(
        "--format",
        type=str,
        default="all",
        choices=["coco", "yolo", "labelstudio", "csv", "all"],
        help="Export format (default: all)",
    )

    # Visualization
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save annotated images with bounding boxes",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        print(f"ERROR: Input directory does not exist: {args.input}")
        sys.exit(1)

    if not args.config or not args.checkpoint:
        print("ERROR: --config and --checkpoint are required (or set GDVL_CONFIG/GDVL_CHECKPOINT)")
        sys.exit(1)

    # Parse classes
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        classes = DEFAULT_WILDLIFE_CLASSES

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Find all images
    print(f"Scanning for images in {args.input}...")
    image_paths = find_images(args.input)
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found. Exiting.")
        sys.exit(0)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"Classes: {', '.join(classes)}")

    try:
        model = Model(
            model_config_path=args.config, model_checkpoint_path=args.checkpoint, device=args.device
        )
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    print("Model loaded successfully")

    # Process images
    print(f"\nProcessing {len(image_paths)} images...")
    print(f"Box threshold: {args.box_threshold}")
    print(f"Text threshold: {args.text_threshold}")

    all_results = []
    total_inference_time = 0.0

    # Process in batches with progress bar
    with tqdm(total=len(image_paths), desc="Processing images", unit="img") as pbar:
        for i in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[i : i + args.batch_size]

            batch_results, batch_time = process_batch(
                model, batch_paths, classes, args.box_threshold, args.text_threshold
            )

            all_results.extend(batch_results)
            total_inference_time += batch_time
            pbar.update(len(batch_paths))

    # Calculate statistics
    total_detections = sum(r["num_detections"] for r in all_results)
    avg_time_per_image = total_inference_time / len(all_results) if all_results else 0
    images_per_second = len(all_results) / total_inference_time if total_inference_time > 0 else 0

    print(f"\n{'='*80}")
    print("BATCH INFERENCE COMPLETE")
    print(f"{'='*80}")
    print(f"Total images processed: {len(all_results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections / len(all_results):.2f}")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Average time per image: {avg_time_per_image:.3f}s")
    print(f"Throughput: {images_per_second:.2f} images/second")
    print(f"{'='*80}\n")

    # Export results
    print("Exporting results...")

    # Save summary JSON
    summary = {
        "total_images": len(all_results),
        "total_detections": total_detections,
        "total_inference_time": total_inference_time,
        "avg_time_per_image": avg_time_per_image,
        "throughput": images_per_second,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "classes": classes,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
    }

    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Export in requested formats
    if args.format in ["coco", "all"]:
        print("  - Exporting COCO JSON...")
        export_coco_json(all_results, args.output / "results_coco.json", classes)

    if args.format in ["yolo", "all"]:
        print("  - Exporting YOLO format...")
        export_yolo_format(all_results, args.output, classes)

    if args.format in ["labelstudio", "all"]:
        print("  - Exporting Label Studio JSON...")
        export_labelstudio_json(all_results, args.output / "results_labelstudio.json")

    if args.format in ["csv", "all"]:
        print("  - Exporting CSV...")
        export_csv(all_results, args.output / "results.csv")

    # Save detailed results JSON
    with open(args.output / "results_detailed.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Visualize if requested
    if args.visualize:
        print("Generating visualizations...")
        viz_dir = args.output / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        try:
            import cv2

            for result in tqdm(all_results, desc="Creating visualizations", unit="img"):
                img = cv2.imread(result["filepath"])
                if img is None:
                    continue

                # Draw bounding boxes
                for detection in result["detections"]:
                    x1, y1, x2, y2 = [int(v) for v in detection["xyxy"]]
                    label = f"{detection['label']} {detection['score']:.2f}"

                    # Draw box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)

                    # Draw label text
                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                    )

                # Save visualization
                output_path = viz_dir / result["filename"]
                cv2.imwrite(str(output_path), img)

        except Exception as e:
            print(f"WARNING: Visualization failed: {e}")

    print(f"\nResults saved to: {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
