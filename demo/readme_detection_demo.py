#!/usr/bin/env python3
"""
Simple Inference Demo using GroundedDINO-VL Public API

This demo shows how to use the new simplified GroundedDINO-VL API for object detection.
Compare this with inference_on_a_image.py to see how much simpler the new API is! Please note,
that this demo is for illustration purposes only and may not cover all edge cases. It's recommended
to use the `demo_images` directory and run the script below. If you want to use your own specific
directory, just change the '--image' path accordingly.

Usage:
    python demo/simple_inference.py \\
        --config configs/GroundingDINO_SwinT_OGC.py \\
        --checkpoint weights/groundingdino_swint_ogc.pth \\
        --image path/to/image.jpg \\
        --text "car . person . dog" \\
        --output outputs/

Author: GroundedDINO-VL Team
License: Apache 2.0
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

# Import the new public API
from groundeddino_vl import load_model, predict, load_image, annotate


def main():
    parser = argparse.ArgumentParser(
        description="GroundedDINO-VL Object Detection - Simple API Demo"
    )
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to model config file")
    parser.add_argument(
        "--checkpoint", "-p", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--text", "-t", type=str, required=True, help='Text prompt (e.g., "car . person . dog")'
    )
    parser.add_argument("--output", "-o", type=str, default="outputs", help="Output directory")
    parser.add_argument(
        "--box-threshold", type=float, default=0.35, help="Box confidence threshold"
    )
    parser.add_argument(
        "--text-threshold", type=float, default=0.25, help="Text confidence threshold"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"GroundedDINO-VL Object Detection")
    print(f"=" * 50)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image: {args.image}")
    print(f"Text prompt: {args.text}")
    print(f"Device: {args.device}")
    print(f"=" * 50)

    # Step 1: Load the model (just one line!)
    print("\n[1/3] Loading model...")
    model = load_model(config_path=args.config, checkpoint_path=args.checkpoint, device=args.device)
    print("✓ Model loaded successfully")

    # Step 2: Load and prepare image
    print("\n[2/3] Loading image...")
    image_np, image_tensor = load_image(args.image)
    print(f"✓ Image loaded: {image_np.shape}")

    # Step 3: Run inference (just one line!)
    print("\n[3/3] Running inference...")
    result = predict(
        model=model,
        image=image_tensor,
        text_prompt=args.text,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )
    print(f"✓ Inference complete")

    # Display results
    print(f"\n{'Results':=^50}")
    print(f"Found {len(result)} objects:")
    for i, (label, score) in enumerate(zip(result.labels, result.scores), 1):
        box = result.boxes[i - 1]
        print(f"  {i}. {label:20s} (confidence: {score:.3f})")
        print(f"     Box (cxcywh): [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]")

    # Convert boxes to xyxy format for better readability
    if len(result) > 0:
        boxes_xyxy = result.to_xyxy(denormalize=True)
        print(f"\nBoxes in pixel coordinates (x1, y1, x2, y2):")
        for i, box in enumerate(boxes_xyxy, 1):
            print(
                f"  {i}. {result.labels[i-1]:20s}: "
                f"[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]"
            )

    # Save annotated image
    print(f"\n{'Saving Results':=^50}")
    annotated_image = annotate(image_np, result)

    output_path = output_dir / "annotated_result.jpg"
    cv2.imwrite(str(output_path), annotated_image)
    print(f"✓ Annotated image saved to: {output_path}")

    # Also save the raw image for comparison
    raw_output_path = output_dir / "raw_image.jpg"
    cv2.imwrite(str(raw_output_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    print(f"✓ Original image saved to: {raw_output_path}")

    print(f"\n{'Done!':=^50}")


if __name__ == "__main__":
    main()
