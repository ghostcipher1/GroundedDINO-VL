# ------------------------------------------------------------------------
# GroundedDINO-VL - High-level Public API
# Copyright (c) 2025 GhostCipher. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
GroundedDINO-VL Public API

This module provides a clean, user-friendly API for common vision-language tasks.

Example:
    >>> from groundeddino_vl import load_model, predict
    >>>
    >>> # Load a model
    >>> model = load_model("config.py", "weights.pth", device="cuda")
    >>>
    >>> # Make predictions
    >>> results = predict(
    >>>     model,
    >>>     image="path/to/image.jpg",
    >>>     text_prompt="car . person . dog .",
    >>>     box_threshold=0.35,
    >>>     text_threshold=0.25
    >>> )
    >>>
    >>> # Access results
    >>> print(results.boxes)    # Bounding boxes
    >>> print(results.labels)   # Detected labels
    >>> print(results.scores)   # Confidence scores
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

import groundeddino_vl.data.transforms as T

# Import internal utilities
from groundeddino_vl.models import build_model
from groundeddino_vl.utils.inference import annotate as _annotate_internal
from groundeddino_vl.utils.inference import load_image as _load_image_internal
from groundeddino_vl.utils.inference import predict as _predict_internal
from groundeddino_vl.utils.misc import clean_state_dict
from groundeddino_vl.utils.slconfig import SLConfig


@dataclass
class DetectionResult:
    """
    Container for object detection results.

    Attributes:
        boxes: Bounding boxes in cxcywh format (center_x, center_y, width, height),
               normalized to [0, 1]
        labels: Detected object labels/phrases
        scores: Confidence scores for each detection
        image_size: Original image size (height, width)
    """

    boxes: torch.Tensor
    labels: List[str]
    scores: torch.Tensor
    image_size: Optional[Tuple[int, int]] = None

    def to_xyxy(self, denormalize: bool = True) -> torch.Tensor:
        """
        Convert boxes from cxcywh to xyxy format.

        Args:
            denormalize: If True and image_size is available, convert to pixel coordinates

        Returns:
            Boxes in xyxy format (x1, y1, x2, y2)
        """
        from torchvision.ops import box_convert

        boxes_xyxy = box_convert(self.boxes, in_fmt="cxcywh", out_fmt="xyxy")

        if denormalize and self.image_size is not None:
            h, w = self.image_size
            boxes_xyxy = boxes_xyxy * torch.tensor([w, h, w, h])

        return torch.as_tensor(boxes_xyxy, dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of detections."""
        return len(self.labels)

    def __repr__(self) -> str:
        return (
            f"DetectionResult(detections={len(self)}, "
            f"labels={self.labels[:3]}{'...' if len(self) > 3 else ''})"
        )


def load_model(
    config_path: Union[str, Path], checkpoint_path: Union[str, Path], device: str = "cuda"
) -> torch.nn.Module:
    """
    Load a GroundedDINO-VL model from configuration and checkpoint.

    This is the primary way to load models in GroundedDINO-VL. The function handles
    configuration parsing, model building, and checkpoint loading automatically.

    Args:
        config_path: Path to model configuration file (.py format)
        checkpoint_path: Path to model checkpoint file (.pth format)
        device: Device to load model on ("cuda" or "cpu")

    Returns:
        Loaded model in evaluation mode

    Example:
        >>> model = load_model(
        >>>     "configs/GroundingDINO_SwinT_OGC.py",
        >>>     "weights/groundingdino_swint_ogc.pth",
        >>>     device="cuda"
        >>> )
        >>> model.eval()  # Already in eval mode

    Raises:
        FileNotFoundError: If config or checkpoint file doesn't exist
        RuntimeError: If model loading fails
    """
    config_path = str(config_path)
    checkpoint_path = str(checkpoint_path)

    # Load configuration
    args = SLConfig.fromfile(config_path)
    args.device = device

    # Build model
    model: torch.nn.Module = build_model(args)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Clean state dict and load
    model.load_state_dict(clean_state_dict(state_dict), strict=False, assign=True)

    # Set to eval mode and move to device
    model.eval()
    model.to(device)

    return model


def load_image(image_path: Union[str, Path]) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Load and preprocess an image for model inference.

    This function applies the standard GroundedDINO-VL image preprocessing pipeline:
    - Resize to max 1333 pixels with aspect ratio preserved
    - Convert to tensor
    - Normalize with ImageNet statistics

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (original_image_array, preprocessed_tensor)
        - original_image_array: RGB image as numpy array for visualization
        - preprocessed_tensor: Normalized tensor ready for model input

    Example:
        >>> image_np, image_tensor = load_image("photo.jpg")
        >>> # Use image_tensor for prediction
        >>> # Use image_np for visualization/annotation
    """
    return _load_image_internal(str(image_path))


def preprocess_image(
    image: Union[np.ndarray, Image.Image, torch.Tensor], max_size: int = 1333, size: int = 800
) -> torch.Tensor:
    """
    Preprocess an image array/PIL image for model inference.

    Use this when you already have an image loaded in memory (e.g., from OpenCV,
    PIL, or other sources) and need to prepare it for the model.

    Args:
        image: Input image as numpy array, PIL Image, or tensor
        max_size: Maximum image dimension (default: 1333)
        size: Target size for the shorter edge (default: 800)

    Returns:
        Preprocessed image tensor ready for model input

    Example:
        >>> import cv2
        >>> image = cv2.imread("photo.jpg")
        >>> image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> tensor = preprocess_image(image_rgb)
    """
    transform = T.Compose(
        [
            T.RandomResize([size], max_size=max_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        # Assume it's already in the right format
        return torch.as_tensor(image, dtype=torch.float32)

    image_transformed, _ = transform(image, None)
    return torch.as_tensor(image_transformed, dtype=torch.float32)


def predict(
    model: torch.nn.Module,
    image: Union[str, Path, np.ndarray, Image.Image, torch.Tensor],
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda",
) -> DetectionResult:
    """
    Perform object detection on an image using text prompts.

    This is the main inference function for GroundedDINO-VL. It accepts various image
    formats and returns structured detection results.

    Args:
        model: Loaded GroundedDINO-VL model (from load_model())
        image: Input image as:
            - Path to image file (str or Path)
            - Numpy array (H, W, 3) in RGB format
            - PIL Image
            - Preprocessed tensor (C, H, W)
        text_prompt: Detection prompt, e.g., "car . person . dog ."
            Multiple objects should be separated by " . "
            Prompt will automatically be lowercased and have "." appended if needed
        box_threshold: Minimum confidence for box detection (0.0-1.0)
        text_threshold: Minimum confidence for text matching (0.0-1.0)
        device: Device to run inference on ("cuda" or "cpu")

    Returns:
        DetectionResult containing boxes, labels, scores, and image size

    Example:
        >>> model = load_model("config.py", "weights.pth")
        >>>
        >>> # From file path
        >>> result = predict(model, "photo.jpg", "car . person")
        >>>
        >>> # From numpy array
        >>> import cv2
        >>> img = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)
        >>> result = predict(model, img, "dog . cat")
        >>>
        >>> print(f"Found {len(result)} objects")
        >>> for label, score in zip(result.labels, result.scores):
        >>>     print(f"{label}: {score:.2f}")

    Notes:
        - Text prompts are case-insensitive
        - Multiple objects in prompt should be separated with " . "
        - Returns normalized boxes in cxcywh format (use result.to_xyxy() to convert)
    """
    # Handle image input
    image_np = None
    if isinstance(image, (str, Path)):
        # Load from file
        image_np, image_tensor = load_image(image)
    elif isinstance(image, np.ndarray):
        # Store original for size info
        image_np = image
        image_tensor = preprocess_image(image)
    elif isinstance(image, Image.Image):
        # Convert PIL to numpy, then preprocess
        image_np = np.array(image)
        image_tensor = preprocess_image(image)
    elif isinstance(image, torch.Tensor):
        # Assume already preprocessed
        image_tensor = image
    else:
        raise TypeError(
            f"image must be str, Path, np.ndarray, PIL.Image, or torch.Tensor, "
            f"got {type(image)}"
        )

    # Run inference
    boxes, logits, phrases = _predict_internal(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    # Get image size if available
    image_size = None
    if image_np is not None:
        image_size = (image_np.shape[0], image_np.shape[1])  # (H, W)

    # Return structured result
    return DetectionResult(boxes=boxes, labels=phrases, scores=logits, image_size=image_size)


def annotate(
    image: np.ndarray,
    result: DetectionResult,
    show_labels: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Annotate an image with detection results.

    Draws bounding boxes and labels on the image. This is a convenience wrapper
    around the internal annotation function.

    Args:
        image: Image array in RGB format (H, W, 3)
        result: DetectionResult from predict()
        show_labels: Whether to show labels on boxes
        show_confidence: Whether to show confidence scores

    Returns:
        Annotated image in BGR format (ready for cv2.imshow or saving)

    Example:
        >>> model = load_model("config.py", "weights.pth")
        >>> image_np, _ = load_image("photo.jpg")
        >>> result = predict(model, image_np, "car . person")
        >>> annotated = annotate(image_np, result)
        >>>
        >>> import cv2
        >>> cv2.imwrite("output.jpg", annotated)  # Already in BGR

    Note:
        Returns image in BGR format (not RGB) for OpenCV compatibility
    """
    return _annotate_internal(
        image_source=image, boxes=result.boxes, logits=result.scores, phrases=result.labels
    )


# Public API exports
__all__ = [
    # Core functions
    "load_model",
    "predict",
    "load_image",
    "preprocess_image",
    "annotate",
    # Data structures
    "DetectionResult",
]
