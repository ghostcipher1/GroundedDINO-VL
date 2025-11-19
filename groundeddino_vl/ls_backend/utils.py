"""Shared utilities for Label Studio integration.

Provides helpers for image loading, conversion to Label Studio result format,
and other utilities used by the LS backend.
"""

from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.request import Request, urlopen

import numpy as np
import requests

try:  # optional, used for size detection
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:  # optional fallback for size detection
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

# (no additional imports below; keep groups separated by blank lines)

# Module logger for ls_backend utilities
logger = logging.getLogger(__name__)


def load_image(input_source: Any) -> Tuple[np.ndarray, int, int]:
    """Load an image from various input formats and return a NumPy array (BGR),
    along with width and height.

    Supported input formats:
      - URL (http/https)
      - Local file path
      - Base64 data URL ("data:image/...;base64,...")
      - Raw bytes (already-loaded bytes object)

    Returns:
      (image_array, width, height)

    Raises:
      ValueError: If the input cannot be decoded into an image.
    """

    # Ensure OpenCV is available
    if cv2 is None:
        logger.error("OpenCV (cv2) is required to decode images but is not available")
        raise ValueError("OpenCV (cv2) is required to decode images but is not available")

    def _decode(img_bytes: bytes) -> Tuple[np.ndarray, int, int]:
        try:
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            logger.exception("Exception while decoding image bytes with OpenCV")
            raise ValueError(f"Failed to decode image bytes: {exc}") from exc
        if img is None:
            logger.error("cv2.imdecode returned None for provided image bytes")
            raise ValueError("Failed to decode image bytes: unsupported or corrupted data")
        h, w = img.shape[:2]
        return img, int(w), int(h)

    # 1) If already bytes/bytearray -> decode directly
    if isinstance(input_source, (bytes, bytearray)):
        return _decode(bytes(input_source))

    # Expect string for the remaining branches
    if not isinstance(input_source, str):
        logger.error("Unsupported input type %s, expected str or bytes", type(input_source))
        raise ValueError(f"Unsupported input type {type(input_source)}, expected str or bytes.")

    s = input_source.strip()

    # 2) Base64 Data URL: data:image/...;base64,XXXX
    if s.startswith("data:image"):
        try:
            header, encoded = s.split(",", 1)
            img_bytes = base64.b64decode(encoded)
        except Exception as exc:
            logger.exception("Invalid base64 image data URL")
            raise ValueError(f"Invalid base64 image data: {exc}") from exc
        return _decode(img_bytes)

    # 3) URL (http/https)
    if s.startswith("http://") or s.startswith("https://"):
        try:
            resp = requests.get(s, timeout=20)
            resp.raise_for_status()
            img_bytes = resp.content
        except Exception as exc:
            logger.exception("Failed to download image from URL: %s", s)
            raise ValueError(f"Failed to download image from URL: {exc}") from exc
        return _decode(img_bytes)

    # 4) Local file path
    try:
        with open(s, "rb") as f:
            img_bytes = f.read()
    except Exception as exc:
        logger.exception("Failed to read image from local path: %s", s)
        raise ValueError(f"Failed to read image from local path '{s}': {exc}") from exc

    return _decode(img_bytes)


def _maybe_extract_image_ref(task: Dict[str, Any]) -> Union[str, bytes, None]:
    """Extract a reference to an image from a Label Studio-like task JSON.

    Looks for common fields both at the root and under task["data"].
    Returns either bytes, a URL/path/base64 string, or None.
    """
    # Common fields at root
    for key in ("image_bytes", "image", "image_url", "imageUrl", "img", "url"):
        if key in task:
            val = task[key]
            if isinstance(val, (str, bytes)):
                return val

    data = task.get("data")
    if isinstance(data, dict):
        for key in ("image", "image_url", "imageUrl", "img", "url", "image_bytes"):
            if key in data:
                val = data[key]
                if isinstance(val, (str, bytes)):
                    return val

    return None


def _read_bytes_from_url(url: str, timeout: int = 20) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as resp:
        return bytes(resp.read())


def _to_image_bytes(ref: Union[str, bytes]) -> bytes:
    """Resolve various image references to raw bytes.

    Supports:
      - bytes/bytearray
      - data URI: data:image/...;base64,XXXXX
      - naked base64 string
      - http(s) URL
      - local filesystem path
    """
    if isinstance(ref, (bytes, bytearray)):
        return bytes(ref)
    if not isinstance(ref, str):
        raise ValueError("Unsupported image reference type")

    s = ref.strip()
    # Data URI
    if s.startswith("data:image/") and ";base64," in s:
        b64 = s.split(",", 1)[1]
        return base64.b64decode(b64)

    # Heuristic: plain base64
    if len(s) > 256 and all(c.isalnum() or c in "+/=_-" for c in s[:300]):
        try:
            return base64.b64decode(s)
        except Exception:
            pass

    # URL
    if s.startswith("http://") or s.startswith("https://"):
        return _read_bytes_from_url(s)

    # Local file
    if os.path.isfile(s):
        with open(s, "rb") as f:
            return f.read()

    raise ValueError("Unrecognized image reference; expected URL, base64, or bytes")


def _image_size_from_bytes(image_bytes: bytes) -> Tuple[int, int]:
    """Return (width, height) from raw image bytes using PIL or OpenCV.

    Raises ValueError if the image cannot be decoded.
    """
    # Prefer PIL if available
    if Image is not None:
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                w, h = img.size
                return int(w), int(h)
        except Exception:
            pass

    # Fallback to OpenCV
    if cv2 is not None:
        try:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if decoded is None:
                raise ValueError("cv2.imdecode returned None")
            h, w = decoded.shape[:2]
            return int(w), int(h)
        except Exception:
            pass

    raise ValueError("Failed to decode image to determine size")


def load_image_from_ls_task(task_json: Dict[str, Any]) -> Tuple[bytes, int, int]:
    """Load image referenced by a Label Studio task.

    Supports references as:
      - direct URL (http/https)
      - local file path
      - base64 data:image/...;base64, ... (or naked base64)

    Returns:
      (image_bytes, width, height)
    """
    ref = _maybe_extract_image_ref(task_json)
    if ref is None:
        raise ValueError("No image reference found in task JSON")

    image_bytes = _to_image_bytes(ref)
    width, height = _image_size_from_bytes(image_bytes)
    return image_bytes, width, height


def convert_xyxy_to_lspct(
    xyxy: Union[List[float], Tuple[float, float, float, float]], width: int, height: int
) -> Dict[str, float]:
    """Convert absolute xyxy bbox to Label Studio percentage rectangle value.

    Args:
      xyxy: [x1, y1, x2, y2] in pixels.
      width, height: image size in pixels.

    Returns:
      dict with keys x, y, width, height, rotation (percentages 0..100).
    """
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w_px = max(0.0, x2 - x1)
    h_px = max(0.0, y2 - y1)

    if width <= 0 or height <= 0:
        left = top = w_pct = h_pct = 0.0
    else:
        left = (x1 / float(width)) * 100.0
        top = (y1 / float(height)) * 100.0
        w_pct = (w_px / float(width)) * 100.0
        h_pct = (h_px / float(height)) * 100.0

    # Clamp to [0, 100] to avoid LS errors on slight overflow
    def _clamp(v: float) -> float:
        return float(max(0.0, min(100.0, v)))

    return {
        "x": _clamp(left),
        "y": _clamp(top),
        "width": _clamp(w_pct),
        "height": _clamp(h_pct),
        "rotation": 0.0,
    }


def build_ls_prediction(
    bboxes: List[Dict[str, Any]],
    *,
    from_name: str = "bbox",
    to_name: str = "image",
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Label Studio-compliant prediction dict from bboxes.

    Expected bbox format per item (dict):
      - xyxy: [x1, y1, x2, y2] absolute pixel coords
      - width: image width in pixels
      - height: image height in pixels
      - label: str label name (optional; defaults to "object")
      - score: float confidence (optional)

    Returns dict:
      {"result": [...], "model_version": str, "score": float}
    """
    results: List[Dict[str, Any]] = []
    scores: List[float] = []

    for b in bboxes or []:
        xyxy = b.get("xyxy")
        w = int(b.get("width", 0))
        h = int(b.get("height", 0))
        if xyxy is None:
            continue

        value = convert_xyxy_to_lspct(xyxy, w, h)
        label = str(b.get("label", "object"))
        score = b.get("score")
        if isinstance(score, (int, float)):
            value["score"] = float(score)
            scores.append(float(score))

        results.append(
            {
                "from_name": from_name,
                "to_name": to_name,
                "type": "rectanglelabels",
                "value": {**value, "rectanglelabels": [label]},
            }
        )

    overall_score: Optional[float] = None
    if scores:
        overall_score = float(sum(scores) / max(1, len(scores)))

    return {
        "result": results,
        "model_version": model_version or "groundeddino-vl",
        "score": overall_score,
    }


def convert_to_labelstudio_format(
    detections: Any, labels: List[str], width: int, height: int
) -> List[Dict[str, Any]]:
    """Convert detections into Label Studio percentage-based rectangles.

    Inputs:
      - detections: supervision.Detections-like object exposing .xyxy and .confidence
      - labels: list of label strings corresponding to detections
      - width/height: original image dimensions in pixels

    Returns a list of LS-compliant result dicts using RectangleLabels shape with
    percentage-based coordinates.
    """
    xyxy = getattr(detections, "xyxy", None)
    scores = getattr(detections, "confidence", None)
    if xyxy is None:
        return []

    xyxy = np.asarray(xyxy, dtype=float)
    n = xyxy.shape[0]
    labels = list(labels)

    results: List[Dict[str, Any]] = []
    for i in range(n):
        x1, y1, x2, y2 = xyxy[i].tolist()
        value: Dict[str, Any] = convert_xyxy_to_lspct([x1, y1, x2, y2], width, height)

        lbl = labels[i] if i < len(labels) else "object"
        score = float(scores[i]) if scores is not None and i < len(scores) else None

        # Attach the label inside rectanglelabels, ensure single entry
        value["rectanglelabels"] = [str(lbl)]
        if score is not None:
            value["score"] = score

        # LS result entry with required names
        res: Dict[str, Any] = {
            "type": "rectanglelabels",
            "from_name": "bbox",
            "to_name": "image",
            "value": value,
        }
        results.append(res)

    return results
