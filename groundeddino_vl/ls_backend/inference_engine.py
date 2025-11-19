"""Inference engine for GroundedDINO-VL (Label Studio ML Backend).

This module implements the core prediction logic required by the API:

- run_inference(image_bytes, prompt_text):
    * Accepts raw image bytes and decodes them (PIL preferred, cv2 fallback).
    * Dynamically routes to caption-driven or class-driven prediction.
    * Returns raw detections (xyxy, scores, labels) and metadata.

The InferenceEngine class provides a simple wrapper around this function for
integration with a Label Studio ML Backend service.
"""

from __future__ import annotations

import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from . import model_loader
from .config import DEFAULT_SETTINGS


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR NumPy array.

    Tries PIL first (no OpenCV dependency), then falls back to OpenCV if
    available. The returned image is BGR to match the expected input of the
    high-level Model.preprocess_image (which internally converts BGR->RGB).
    """
    # Try PIL first
    try:
        from PIL import Image  # type: ignore

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        rgb = np.asarray(img)  # HxWx3 RGB uint8
        bgr = rgb[..., ::-1].copy()  # convert to BGR
        return np.asarray(bgr, dtype=np.uint8)  # type: ignore[no-any-return]
    except Exception:
        pass

    # Fallback to cv2
    try:
        import cv2  # type: ignore

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        return np.asarray(bgr, dtype=np.uint8)  # type: ignore[no-any-return]
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Failed to decode image bytes: {e}")


def _parse_task(
    prompt_text: Union[str, List[str], Tuple[str, ...]],
) -> Tuple[str, Optional[List[str]]]:
    """Determine task type and normalize inputs.

    Returns a tuple of (mode, classes) where:
      - mode is either "caption" or "classes".
      - classes is a list of class names when mode=="classes", else None.
    """
    # If already a list/tuple → classes mode
    if isinstance(prompt_text, (list, tuple)):
        classes = [str(x).strip() for x in prompt_text if str(x).strip()]
        return "classes", classes

    # If string starting with explicit prefix: "classes: a, b, c"
    if isinstance(prompt_text, str):
        text = prompt_text.strip()
        lower = text.lower()
        if lower.startswith("classes:"):
            tail = text.split(":", 1)[1]
            classes = [c.strip() for c in tail.split(",") if c.strip()]
            return "classes", classes

        # Conservative heuristic: if comma-separated words and short total length
        # treat as classes; otherwise treat as a free-form caption.
        if "," in text and len(text) < 200:
            parts = [p.strip() for p in text.split(",")]
            # If all parts are single or few words without spaces, consider classes
            if all(p and len(p.split()) <= 3 for p in parts):
                return "classes", parts

    return "caption", None


def run_inference(
    image_bytes: bytes, prompt_text: Union[str, List[str], Tuple[str, ...]]
) -> Dict[str, Any]:
    """Run a single-image inference and return detections and metadata.

    Args:
        image_bytes: Raw image bytes.
        prompt_text: Either a free-form caption (str) or classes (list[str] or
            a string formatted as "classes: cat, dog").

    Returns:
        A dict with keys:
          - detections: {"xyxy": List[List[float]], "scores": List[float], "labels": List[str]}
          - metadata: {"inference_time": float, "model_version": str,
                       "box_threshold": float, "text_threshold": float,
                       "width": int, "height": int}
          - labelstudio: List[Dict[str, Any]]  # LS-compliant results
    """
    # Decode image
    image_bgr = _decode_image_bytes(image_bytes)
    h, w = image_bgr.shape[:2]

    # Load model (singleton) using default settings
    model = model_loader.load_model(
        model_config_path=DEFAULT_SETTINGS.model_config_path
        or "groundeddino_vl/models/configs/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path=DEFAULT_SETTINGS.model_checkpoint_path
        or "checkpoints/groundingdino_swint_ogc.pth",
        device=None,
    )

    box_th = float(getattr(DEFAULT_SETTINGS, "box_threshold", 0.25))
    text_th = float(getattr(DEFAULT_SETTINGS, "text_threshold", 0.25))

    mode, classes = _parse_task(prompt_text)

    start = time.time()
    if mode == "classes" and classes is not None:
        detections = model.predict_with_classes(
            image=image_bgr, classes=classes, box_threshold=box_th, text_threshold=text_th
        )
        labels = classes
    else:
        # Treat as free-form caption
        caption = prompt_text if isinstance(prompt_text, str) else ", ".join(classes or [])
        detections, phrases = model.predict_with_caption(
            image=image_bgr, caption=caption, box_threshold=box_th, text_threshold=text_th
        )
        labels = phrases
    elapsed = time.time() - start

    # Extract raw arrays
    xyxy = getattr(detections, "xyxy", None)
    scores = getattr(detections, "confidence", None)
    if xyxy is None:
        xyxy_list: List[List[float]] = []
    else:
        xyxy_list = np.asarray(xyxy, dtype=float).tolist()
    if scores is None:
        scores_list: List[float] = []
    else:
        scores_list = np.asarray(scores, dtype=float).tolist()

    # Resolve model/version info
    info = model_loader.get_model_info()
    version = str(info.get("version", "unknown"))

    # Build Label Studio-compliant prediction envelope
    # Assemble simple bbox list for the helper that creates LS payload
    bboxes: List[Dict[str, Any]] = []
    xyxy_arr = getattr(detections, "xyxy", None)
    conf_arr = getattr(detections, "confidence", None)
    if xyxy_arr is not None:
        xyxy_np = np.asarray(xyxy_arr, dtype=float)
        n = xyxy_np.shape[0]
        for i in range(n):
            x1, y1, x2, y2 = xyxy_np[i].tolist()
            item: Dict[str, Any] = {
                "xyxy": [x1, y1, x2, y2],
                "width": int(w),
                "height": int(h),
                "label": str(labels[i]) if i < len(labels) else "object",
            }
            if conf_arr is not None and i < len(conf_arr):
                try:
                    item["score"] = float(conf_arr[i])
                except Exception:
                    pass
            bboxes.append(item)

    # Prefer explicit 'model_version' key when available
    model_version = str(info.get("model_version") or info.get("version") or "unknown")

    # Use utils helper to produce LS-format with top-level score + version
    from .utils import build_ls_prediction  # local import to avoid cycles

    ls_prediction = build_ls_prediction(
        bboxes, from_name="bbox", to_name="image", model_version=model_version
    )

    return {
        "detections": {
            "xyxy": xyxy_list,
            "scores": scores_list,
            "labels": list(labels),
        },
        "metadata": {
            "inference_time": elapsed,
            "model_version": version,
            "box_threshold": box_th,
            "text_threshold": text_th,
            "width": w,
            "height": h,
        },
        # For the LS backend, the server returns this object as the prediction
        "labelstudio": ls_prediction,
    }


class InferenceEngine:
    """Placeholder inference engine that will run model predictions."""

    def __init__(self, model: Optional[Any] = None) -> None:
        self.model = model

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run a prediction given a payload containing image bytes and prompt.

        Expected payload keys:
          - image_bytes: raw bytes of the image (required)
          - prompt: str or List[str] describing the task (required)

        Returns LS-ready predictions alongside raw outputs and metadata.
        """
        image_bytes = payload.get("image_bytes")
        prompt = payload.get("prompt")
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise ValueError("payload.image_bytes must be raw bytes")
        if prompt is None:
            raise ValueError("payload.prompt must be provided")

        result = run_inference(image_bytes=image_bytes, prompt_text=prompt)

        # For a Label Studio ML backend, typically we return a list of results for a single task.
        # Here we include both LS-shaped result and raw metadata for transparency.
        return {
            "results": result["labelstudio"],
            "meta": result["metadata"],
            "raw": result["detections"],
        }
