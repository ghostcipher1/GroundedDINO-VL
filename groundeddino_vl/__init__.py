"""
GroundedDINO-VL - Modern Vision-Language Foundation Models

A refactored and modernized framework based on GroundingDINO,
optimized for PyTorch 2.7+ and CUDA 12.x.

Copyright (c) 2025 GhostCipher. All rights reserved.
Based on GroundingDINO by IDEA-Research.
Licensed under the Apache License, Version 2.0.

Quick Start:
    >>> from groundeddino_vl import load_model, predict
    >>>
    >>> # Load a model
    >>> model = load_model("config.py", "weights.pth", device="cuda")
    >>>
    >>> # Run inference
    >>> result = predict(model, "image.jpg", "car . person . dog")
    >>>
    >>> # Access results
    >>> print(f"Found {len(result)} objects")
    >>> for label, score in zip(result.labels, result.scores):
    >>>     print(f"{label}: {score:.2f}")

Public API:
    Core Functions:
        - load_model: Load a model from config and checkpoint
        - predict: Run object detection with text prompts
        - load_image: Load and preprocess an image
        - preprocess_image: Preprocess an in-memory image
        - annotate: Draw detection results on an image

    Data Structures:
        - DetectionResult: Container for detection results

    Advanced:
        - models: Model architectures and builders (for advanced users)
        - utils: Inference utilities and helpers (for advanced users)
        - data: Data loading and transforms (for advanced users)
        - ops: CUDA operations (if available, for advanced users)
"""

from groundeddino_vl.version import __version__

__author__ = "ghostcipher1"

# Import and re-export the public API.
# NOTE: Avoid hard failures (and heavyweight imports like torch) during light-weight
# environments such as minimal test runs that only need ls_backend modules.
# We try to import the API, but if dependencies are missing we provide lazy stubs
# that raise on use while still allowing `import groundeddino_vl` to succeed.
try:  # pragma: no cover - exercised indirectly in environments with full deps
    from groundeddino_vl.api import (
        DetectionResult,
        annotate,
        load_image,
        load_model,
        predict,
        preprocess_image,
    )
except Exception:  # pragma: no cover

    class _MissingDependencyProxy:
        def __init__(self, name: str):
            self._name = name

        def __call__(self, *args, **kwargs):
            raise ImportError(
                f"{self._name} is unavailable because optional heavy dependencies "
                f"(e.g., torch) are not installed in this environment."
            )

    class DetectionResult:  # type: ignore
        pass

    annotate = _MissingDependencyProxy("annotate")  # type: ignore
    load_image = _MissingDependencyProxy("load_image")  # type: ignore
    load_model = _MissingDependencyProxy("load_model")  # type: ignore
    predict = _MissingDependencyProxy("predict")  # type: ignore
    preprocess_image = _MissingDependencyProxy("preprocess_image")  # type: ignore

# Public API exports
__all__ = [
    # Metadata
    "__version__",
    "__author__",
    # Core API functions
    "load_model",
    "predict",
    "load_image",
    "preprocess_image",
    "annotate",
    # Data structures
    "DetectionResult",
]
