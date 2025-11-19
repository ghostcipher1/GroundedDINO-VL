#!/usr/bin/env python3
import os
from pathlib import Path
import cv2
from groundeddino_vl import load_model, predict, load_image, annotate


def main():
    config_path = "groundeddino_vl/models/configs/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "checkpoints/groundingdino_swint_ogc.pth"
    demo_images_dir = Path("demo/demo_images")
    output_dir = Path("demo/outputs")

    model = load_model(config_path=config_path, checkpoint_path=checkpoint_path, device="cpu")

    images_config = [
        ("dog_01d.jpg", "dog"),
        ("raccoon_01d.jpg", "raccoon"),
        ("apples_01d.jpg", "apple"),
        ("tractor_01d.jpg", "tractor"),
    ]

    for image_name, text_prompt in images_config:
        image_path = demo_images_dir / image_name
        print(f"\nProcessing {image_name} - detecting '{text_prompt}'...")

        image_np, image_tensor = load_image(str(image_path))
        result = predict(
            model=model,
            image=image_tensor,
            text_prompt=text_prompt,
            box_threshold=0.35,
            text_threshold=0.25,
            device="cpu",
        )

        annotated_image = annotate(image_np, result)

        stem = Path(image_name).stem
        output_path = output_dir / f"{stem}_annotated.jpg"
        cv2.imwrite(str(output_path), annotated_image)
        print(f"✓ Saved: {output_path}")

        print(f"  Found {len(result)} {text_prompt}(s)")
        for i, score in enumerate(result.scores, 1):
            print(f"    {i}. Confidence: {score:.3f}")


if __name__ == "__main__":
    main()
