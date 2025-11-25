# API Reference

Complete reference documentation for GroundedDINO-VL's public API.

---

## Table of Contents

- [High-Level API](#high-level-api)
  - [load_model](#load_model)
  - [predict](#predict)
  - [annotate](#annotate)
  - [load_image](#load_image)
- [Result Objects](#result-objects)
  - [PredictionResult](#predictionresult)
- [Low-Level API](#low-level-api)
  - [Model Class](#model-class)
- [Utility Functions](#utility-functions)
- [Type Definitions](#type-definitions)

---

## High-Level API

The high-level API provides simple, clean interfaces for common tasks.

### load_model

Load a GroundingDINO model from configuration and checkpoint files.

```python
def load_model(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda"
) -> torch.nn.Module
```

**Parameters:**
- `config_path` (str): Path to model configuration file (`.py` extension)
- `checkpoint_path` (str): Path to model checkpoint file (`.pth` extension)
- `device` (str, optional): Device to load model on. Options: `"cuda"`, `"cpu"`, `"mps"`. Default: `"cuda"`

**Returns:**
- `torch.nn.Module`: Loaded model in evaluation mode

**Raises:**
- `FileNotFoundError`: If config or checkpoint file not found
- `RuntimeError`: If model loading fails
- `ValueError`: If config is invalid

**Example:**
```python
from groundeddino_vl import load_model

# Load on GPU
model = load_model(
    config_path="GroundingDINO_SwinT_OGC.py",
    checkpoint_path="groundingdino_swint_ogc.pth",
    device="cuda"
)

# Load on CPU
model = load_model(
    config_path="config.py",
    checkpoint_path="weights.pth",
    device="cpu"
)
```

**Notes:**
- Model weights are automatically downloaded on first use
- The model is set to evaluation mode (`model.eval()`)
- Weights are cached in `~/.cache/groundeddino-vl/`

---

### predict

Run object detection on an image with text prompts.

```python
def predict(
    model: torch.nn.Module,
    image: Union[str, np.ndarray, PIL.Image.Image, torch.Tensor],
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> PredictionResult
```

**Parameters:**
- `model` (torch.nn.Module): Loaded GroundingDINO model from `load_model()`
- `image` (str | np.ndarray | PIL.Image | torch.Tensor): Input image
  - **str**: Path to image file
  - **np.ndarray**: RGB image array with shape `(H, W, 3)`, dtype `uint8`
  - **PIL.Image**: PIL Image object
  - **torch.Tensor**: Tensor with shape `(3, H, W)`, dtype `float32`
- `text_prompt` (str): Text description of objects to detect, separated by ` . `
  - Example: `"person . car . dog"`
  - Example: `"red car . blue truck . yellow bicycle"`
- `box_threshold` (float, optional): Confidence threshold for bounding boxes (0.0 - 1.0). Default: `0.35`
- `text_threshold` (float, optional): Confidence threshold for text-image matching (0.0 - 1.0). Default: `0.25`

**Returns:**
- `PredictionResult`: Object containing detection results

**Raises:**
- `ValueError`: If image format is invalid
- `RuntimeError`: If prediction fails

**Example:**
```python
from groundeddino_vl import load_model, predict

model = load_model("config.py", "weights.pth")

# From image path
result = predict(
    model=model,
    image="photo.jpg",
    text_prompt="person . car . bicycle",
)

# From numpy array
import cv2
image = cv2.imread("photo.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = predict(model, image_rgb, "cat . dog")

# With custom thresholds
result = predict(
    model=model,
    image="photo.jpg",
    text_prompt="person",
    box_threshold=0.50,  # Higher confidence
    text_threshold=0.30,
)
```

**Notes:**
- Images are automatically preprocessed (resized, normalized)
- Original aspect ratio is preserved
- Bounding boxes are returned in normalized coordinates `[0, 1]`

---

### annotate

Visualize detection results by drawing bounding boxes and labels on images.

```python
def annotate(
    image: Union[str, np.ndarray],
    result: PredictionResult,
    show_labels: bool = True,
    show_confidence: bool = True,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray
```

**Parameters:**
- `image` (str | np.ndarray): Original image (path or RGB numpy array)
- `result` (PredictionResult): Detection result from `predict()`
- `show_labels` (bool, optional): Whether to show label text. Default: `True`
- `show_confidence` (bool, optional): Whether to show confidence scores. Default: `True`
- `box_color` (Tuple[int, int, int], optional): RGB color for bounding boxes. Default: `(0, 255, 0)` (green)
- `text_color` (Tuple[int, int, int], optional): RGB color for text. Default: `(255, 255, 255)` (white)
- `thickness` (int, optional): Line thickness for boxes. Default: `2`

**Returns:**
- `np.ndarray`: Annotated image in BGR format (for OpenCV compatibility)

**Example:**
```python
from groundeddino_vl import load_model, predict, annotate
import cv2

model = load_model("config.py", "weights.pth")
result = predict(model, "photo.jpg", "person . car")

# Basic annotation
annotated = annotate("photo.jpg", result)
cv2.imwrite("output.jpg", annotated)

# Custom styling
annotated = annotate(
    image="photo.jpg",
    result=result,
    show_labels=True,
    show_confidence=True,
    box_color=(255, 0, 0),  # Red boxes
    text_color=(0, 255, 255),  # Cyan text
    thickness=3,
)
```

---

### load_image

Load and preprocess an image for inference.

```python
def load_image(
    image_path: str
) -> Tuple[np.ndarray, torch.Tensor]
```

**Parameters:**
- `image_path` (str): Path to image file

**Returns:**
- `Tuple[np.ndarray, torch.Tensor]`:
  - `np.ndarray`: Original image as RGB numpy array
  - `torch.Tensor`: Preprocessed image tensor for model input

**Example:**
```python
from groundeddino_vl import load_image

# Load and preprocess
image_np, image_tensor = load_image("photo.jpg")

print(f"Original shape: {image_np.shape}")  # (H, W, 3)
print(f"Tensor shape: {image_tensor.shape}")  # (3, H', W')
```

---

## Result Objects

### PredictionResult

Container for detection results returned by `predict()`.

```python
@dataclass
class PredictionResult:
    labels: List[str]          # Detected object labels
    scores: List[float]        # Confidence scores (0-1)
    boxes: List[List[float]]   # Bounding boxes [cx, cy, w, h] (normalized)
    image_size: Tuple[int, int]  # (height, width) of original image
```

**Attributes:**
- `labels` (List[str]): List of detected object class names
- `scores` (List[float]): Confidence scores for each detection (0.0 to 1.0)
- `boxes` (List[List[float]]): Bounding boxes in `[cx, cy, w, h]` format (normalized to [0, 1])
  - `cx`: Center X coordinate
  - `cy`: Center Y coordinate
  - `w`: Box width
  - `h`: Box height
- `image_size` (Tuple[int, int]): Original image dimensions `(height, width)`

**Methods:**

#### to_xyxy

Convert boxes to `[x1, y1, x2, y2]` format.

```python
def to_xyxy(self, denormalize: bool = False) -> List[List[float]]
```

**Parameters:**
- `denormalize` (bool, optional): If True, convert to pixel coordinates. Default: `False`

**Returns:**
- `List[List[float]]`: Boxes in `[x1, y1, x2, y2]` format

**Example:**
```python
result = predict(model, "photo.jpg", "person . car")

# Normalized coordinates [0, 1]
boxes_norm = result.to_xyxy(denormalize=False)
print(boxes_norm)  # [[0.1, 0.2, 0.5, 0.8], ...]

# Pixel coordinates
boxes_pixel = result.to_xyxy(denormalize=True)
print(boxes_pixel)  # [[64, 128, 320, 512], ...]
```

#### to_xywh

Convert boxes to `[x, y, w, h]` format (top-left corner).

```python
def to_xywh(self, denormalize: bool = False) -> List[List[float]]
```

**Parameters:**
- `denormalize` (bool, optional): If True, convert to pixel coordinates. Default: `False`

**Returns:**
- `List[List[float]]`: Boxes in `[x, y, w, h]` format

**Example:**
```python
result = predict(model, "photo.jpg", "person . car")

# Pixel coordinates in xywh format
boxes = result.to_xywh(denormalize=True)
for box in boxes:
    x, y, w, h = box
    print(f"Box at ({x}, {y}) with size {w}x{h}")
```

#### to_coco_format

Convert results to COCO annotation format.

```python
def to_coco_format(self, image_id: int, category_mapping: Dict[str, int]) -> List[Dict]
```

**Parameters:**
- `image_id` (int): COCO image ID
- `category_mapping` (Dict[str, int]): Mapping from label names to category IDs

**Returns:**
- `List[Dict]`: List of COCO annotations

**Example:**
```python
result = predict(model, "photo.jpg", "person . car . dog")

category_mapping = {"person": 1, "car": 2, "dog": 3}
annotations = result.to_coco_format(
    image_id=1,
    category_mapping=category_mapping
)

# Output format:
# [
#   {
#     "image_id": 1,
#     "category_id": 1,
#     "bbox": [x, y, w, h],
#     "score": 0.95,
#     "area": 12345.0
#   },
#   ...
# ]
```

#### __len__

Get number of detections.

```python
def __len__(self) -> int
```

**Example:**
```python
result = predict(model, "photo.jpg", "person . car")
print(f"Found {len(result)} objects")
```

---

## Low-Level API

For advanced users who need fine-grained control.

### Model Class

Low-level model interface with Supervision integration.

```python
from groundeddino_vl.utils.inference import Model

class Model:
    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str = "cuda"
    )
```

**Parameters:**
- `model_config_path` (str): Path to model configuration file
- `model_checkpoint_path` (str): Path to model checkpoint file
- `device` (str, optional): Device to load model on. Default: `"cuda"`

#### predict_with_caption

Run detection and return Supervision detections.

```python
def predict_with_caption(
    self,
    image: np.ndarray,
    caption: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> Tuple[sv.Detections, List[str]]
```

**Parameters:**
- `image` (np.ndarray): Input image in BGR format (OpenCV convention)
- `caption` (str): Text prompt with objects separated by ` . `
- `box_threshold` (float, optional): Box confidence threshold. Default: `0.35`
- `text_threshold` (float, optional): Text matching threshold. Default: `0.25`

**Returns:**
- `Tuple[sv.Detections, List[str]]`: Supervision detections object and label list

**Example:**
```python
import cv2
from groundeddino_vl.utils.inference import Model
import supervision as sv

# Load model
model = Model("config.py", "weights.pth")

# Load image (BGR format)
image = cv2.imread("photo.jpg")

# Predict
detections, labels = model.predict_with_caption(
    image=image,
    caption="person . car . dog",
    box_threshold=0.35,
)

# Access detections
print(f"Boxes: {detections.xyxy}")
print(f"Confidence: {detections.confidence}")
print(f"Labels: {labels}")

# Visualize with Supervision
annotator = sv.BoxAnnotator()
annotated = annotator.annotate(scene=image, detections=detections)
```

#### predict_with_classes

Run detection with predefined class list.

```python
def predict_with_classes(
    self,
    image: Union[str, np.ndarray],
    classes: List[str],
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> sv.Detections
```

**Parameters:**
- `image` (str | np.ndarray): Image path or BGR numpy array
- `classes` (List[str]): List of class names to detect
- `box_threshold` (float, optional): Box confidence threshold. Default: `0.35`
- `text_threshold` (float, optional): Text matching threshold. Default: `0.25`

**Returns:**
- `sv.Detections`: Supervision detections with populated `class_id` field

**Example:**
```python
from groundeddino_vl.utils.inference import Model

model = Model("config.py", "weights.pth")

# Define classes
classes = ["person", "car", "bicycle", "dog", "cat"]

# Detect
detections = model.predict_with_classes(
    image="photo.jpg",
    classes=classes,
    box_threshold=0.35,
)

# Access class IDs
for i in range(len(detections)):
    class_id = detections.class_id[i]
    class_name = classes[class_id]
    confidence = detections.confidence[i]
    print(f"{class_name}: {confidence:.2f}")
```

---

## Utility Functions

### download_model_weights

Explicitly download and validate model weights.

```python
from groundeddino_vl.weights_manager import download_model_weights

def download_model_weights(
    model_name: str = "groundingdino_swint_ogc",
    cache_dir: Optional[str] = None,
    force: bool = False
) -> str
```

**Parameters:**
- `model_name` (str, optional): Name of model to download. Default: `"groundingdino_swint_ogc"`
- `cache_dir` (str, optional): Custom cache directory. Default: `~/.cache/groundeddino-vl/`
- `force` (bool, optional): Force re-download even if cached. Default: `False`

**Returns:**
- `str`: Path to downloaded weights file

**Example:**
```python
from groundeddino_vl.weights_manager import download_model_weights

# Download default model
weights_path = download_model_weights()
print(f"Weights at: {weights_path}")

# Download to custom location
weights_path = download_model_weights(
    model_name="groundingdino_swint_ogc",
    cache_dir="/custom/cache/dir/",
    force=True  # Re-download
)
```

---

## Type Definitions

```python
from typing import Union, List, Tuple, Dict, Optional
import numpy as np
import torch
from PIL import Image

# Image types
ImageInput = Union[str, np.ndarray, Image.Image, torch.Tensor]

# Box formats
BoxCXCYWH = List[float]  # [center_x, center_y, width, height]
BoxXYXY = List[float]    # [x1, y1, x2, y2]
BoxXYWH = List[float]    # [x, y, width, height]

# Detection output
Detection = Tuple[str, float, BoxCXCYWH]  # (label, score, box)
```

---

## Backward Compatibility

**GroundedDINO-VL is now fully independent** from the legacy GroundingDINO implementation. The `groundingdino/` namespace provides a lightweight compatibility shim (24KB, reduced from 520KB).

### What Still Works

High-level API and module-level access via the shim (with deprecation warnings):

```python
# High-level API (works via shim, shows deprecation warning)
from groundingdino import load_model, predict, annotate

# Module-level access (works via shim)
from groundingdino import models, util, datasets
```

### What No Longer Works

Deep imports into internal modules are no longer supported:

```python
# ✗ No longer works (use groundeddino_vl instead)
from groundingdino.util.misc import NestedTensor
from groundingdino.datasets.transforms import Compose

# ✓ Use groundeddino_vl for internal modules
from groundeddino_vl.utils.misc import NestedTensor
from groundeddino_vl.data.transforms import Compose
```

**Note**: `util` → `utils` and `datasets` → `data` in the new namespace.

**Migration Guide**: See [MIGRATION_FROM_GROUNDINGDINO.md](MIGRATION_FROM_GROUNDINGDINO.md) for detailed migration instructions

---

## See Also

- [Quick Start Guide](QUICKSTART.md) - Practical examples and common use cases
- [Installation Guide](INSTALLATION.md) - Setup and requirements
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions

---

**Need Help?** Open an issue on [GitHub](https://github.com/ghostcipher1/GroundedDINO-VL/issues).
