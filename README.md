# GroundedDINO-VL

**Modern Vision-Language Foundation Models for PyTorch 2.7 + CUDA 12.8**

[![CI](https://github.com/ghostcipher1/GroundedDINO-VL/actions/workflows/ci.yml/badge.svg)](https://github.com/ghostcipher1/GroundedDINO-VL/actions/workflows/ci.yml)
[![GPU CI](https://github.com/ghostcipher1/GroundedDINO-VL/actions/workflows/gpu-ci.yml/badge.svg)](https://github.com/ghostcipher1/GroundedDINO-VL/actions/workflows/gpu-ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/PyPI-groundeddino_vl-blue.svg)](https://pypi.org/project/groundeddino_vl/)
[![Downloads](https://img.shields.io/pypi/dm/groundeddino_vl.svg)](https://pypi.org/project/groundeddino_vl/)

---

## Overview

**GroundedDINO-VL** is a modern vision-language framework derived from [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), refactored and maintained for current GPU infrastructure with PyTorch 2.7 and CUDA 12.8 support.

This project provides a clean, modernized implementation while maintaining compatibility with the original GroundingDINO research and models.

### Key Features

- ✅ **Modern Stack**: PyTorch 2.7 + CUDA 12.8 support
- ✅ **Zero-Shot Detection**: Detect objects using natural language descriptions
- ✅ **High Performance**: Based on GroundingDINO's COCO zero-shot 52.5 AP
- ✅ **Backward Compatible**: Existing GroundingDINO code continues to work
- ✅ **Clean Architecture**: Refactored package structure with better organization

---

## Example Images Using GroundedDINO-VL
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/f83fc105-00ff-4eca-a3bf-6711a5b47fdb" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/5ce29e8c-f3a3-4511-ad9a-ce022a1eadf9" width="400"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/5314ad18-b34c-472a-aaa8-2f3a42465da3" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/a91752f2-b659-486e-b141-e13d6b80937f" width="400"></td>
  </tr>
</table>



---

## Installation

### Requirements

- **Python**: 3.9, 3.10, 3.11, or 3.12
- **PyTorch**: 2.7.0+ (comes with CUDA support)
- **C++17 Compiler**: GCC 7+, Clang 5+, or MSVC 2019+
- **CUDA Toolkit** (optional): 12.6 or 12.8 for GPU acceleration

### Quick Install (PyPI)

```bash
pip install groundeddino_vl
```

### Install with GPU Support (CUDA 12.8)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install groundeddino_vl
```

### Install from Source (Development)

```bash
# Clone repository
git clone https://github.com/ghostcipher1/GroundedDINO-VL.git
cd GroundedDINO-VL

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install in development mode
pip install -e .
```


### Verify Installation

```bash
python -c "import groundeddino_vl; print(f'GroundedDINO-VL {groundeddino_vl.__version__}')"
```

### Building from Source (Advanced)

For detailed build instructions, including troubleshooting and custom compiler flags, see [BUILD_GUIDE.md](BUILD_GUIDE.md).

<img alt="LabelStudio logo" src="https://user-images.githubusercontent.com/12534576/192582340-4c9e4401-1fe6-4dbb-95bb-fdbba5493f61.png" />

## Label Studio ML Backend (GroundedDINO-VL v2.0.0)

GroundedDINO-VL v2.0.0 introduces an optional Label Studio ML Backend (`ls_backend`) that allows GroundedDINO-VL to act as a real-time auto-annotation service inside Label Studio.

This backend runs as a standalone FastAPI service (default port 9090) and provides:
- On-demand inference
- Auto-labeling ("magic wand")
- Batch annotation assistance
- Optional PostgreSQL/SQLite history logging

### Documentation

To keep this README focused, full documentation has been moved into dedicated files:

- **Overview**  
  [docs/ls_backend/overview.md](docs/ls_backend/overview.md)

- **Installation & Environment Setup**  
  [docs/ls_backend/installation.md](docs/ls_backend/installation.md)

- **Using GroundedDINO-VL with Label Studio**  
  [docs/ls_backend/using_with_labelstudio.md](docs/ls_backend/using_with_labelstudio.md)

- **Database Support (PostgreSQL or SQLite)**  
  [docs/ls_backend/database.md](docs/ls_backend/database.md)

- **Troubleshooting**  
  [docs/ls_backend/troubleshooting.md](docs/ls_backend/troubleshooting.md)

---

## Quick Start

### Modern High-Level API (Recommended)

The recommended way to use GroundedDINO-VL is through the clean public API that abstracts away preprocessing, postprocessing, and model management:

#### Basic Detection with Text Prompts
```python
from groundeddino_vl import load_model, predict

# Load model once
model = load_model(
    config_path="path/to/config.py",
    checkpoint_path="path/to/weights.pth",
    device="cuda"
)

# Run detection with text prompt
result = predict(
    model=model,
    image="path/to/image.jpg",
    text_prompt="car . person . dog",  # Objects separated by " . "
    box_threshold=0.35,
    text_threshold=0.25,
)

# Access results
print(f"Found {len(result)} objects")
for label, score in zip(result.labels, result.scores):
    print(f"{label}: {score:.2f}")

# Convert boxes to pixel coordinates (xyxy format)
boxes_xyxy = result.to_xyxy(denormalize=True)
print(f"Boxes: {boxes_xyxy}")
```

#### Detection from Image Arrays
```python
import cv2
from groundeddino_vl import load_model, predict

# Load image with OpenCV (BGR format)
image_bgr = cv2.imread("photo.jpg")

# Convert BGR to RGB for the API
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

model = load_model("config.py", "weights.pth")
result = predict(model, image_rgb, "cat . dog . bird")

print(f"Detections: {result}")
```

#### Annotate and Visualize Results
```python
from groundeddino_vl import load_model, predict, annotate, load_image
import cv2

model = load_model("config.py", "weights.pth")

# load_image returns (original_array, preprocessed_tensor)
image_np, _ = load_image("photo.jpg")

result = predict(model, image_np, "car . truck . bus")

# Annotate image (returns BGR format for OpenCV)
annotated = annotate(image_np, result, show_labels=True, show_confidence=True)

# Save or display result
cv2.imwrite("output.jpg", annotated)
cv2.imshow("Result", annotated)
cv2.waitKey(0)
```

### Advanced API: Low-Level Control with Supervision

For advanced users who need fine-grained control or want to use Supervision detections directly:

```python
import cv2
from groundeddino_vl.utils.inference import Model
import supervision as sv

# Load image with OpenCV (BGR format)
image_bgr = cv2.imread("photo.jpg")

# Use the Model class for lower-level access
model = Model(
    model_config_path="config.py",
    model_checkpoint_path="weights.pth"
)

# Predict with caption (returns sv.Detections + labels)
detections, labels = model.predict_with_caption(
    image=image_bgr,
    caption="person . car . bicycle",  # Objects separated by " . "
    box_threshold=0.35,
    text_threshold=0.25,
)

# Visualize with Supervision
box_annotator = sv.BoxAnnotator()
annotated = box_annotator.annotate(scene=image_bgr, detections=detections)

# Add labels with confidence scores
label_annotator = sv.LabelAnnotator()
labels_with_conf = [
    f"{label} {conf:.2f}"
    for label, conf in zip(labels, detections.confidence)
]
annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections,
    labels=labels_with_conf
)

cv2.imshow("Result", annotated)
cv2.waitKey(0)
```

#### Class-Based Detection
```python
# Detect specific classes instead of generic captions
detections = model.predict_with_classes(
    image=image_bgr,
    classes=["cat", "dog", "bird"],
    box_threshold=0.35,
    text_threshold=0.25,
)

# class_id field is automatically populated
print(f"Class IDs: {detections.class_id}")
```

---
### Backward Compatibility

Existing GroundingDINO code continues to work (with deprecation warnings):

```python
# Old imports (still supported)
import groundingdino
from groundingdino.util import inference  # Note: util not utils

# New recommended imports
import groundeddino_vl
from groundeddino_vl.utils import inference  # Note: utils (plural)
```

---

## Package Structure

```
groundeddino_vl/
├── models/           # Model architectures
│   ├── grounding_dino/
│   └── configs/
├── ops/              # CUDA operations
│   └── csrc/
├── utils/            # Utilities (inference, config, visualization)
├── data/             # Data loading and transforms
├── api/              # High-level API (future)
└── exporters/        # Model export (ONNX, TensorRT - future)
```

---

## Migration Guide

### From groundingdino-cu128

The package has been renamed from `groundingdino-cu128` to `groundeddino_vl`:

| Old | New | Status |
|-----|-----|--------|
| `pip install groundingdino-cu128` | `pip install groundeddino_vl` | ✅ Both work |
| `import groundingdino` | `import groundeddino_vl` | ✅ Both work (old shows warning) |
| `groundingdino.util` | `groundeddino_vl.utils` | ⚠️ Note: plural |
| `groundingdino.datasets` | `groundeddino_vl.data` | ⚠️ Renamed |

See [MIGRATION_GUIDE_PHASE1.md](MIGRATION_GUIDE_PHASE1.md) for detailed migration instructions.

---

## Original GroundingDINO Research

This project is based on the groundbreaking work:

**Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**

```bibtex
@article{liu2023grounding,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```

**Original Project**: [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
**Paper**: [arXiv:2303.05499](https://arxiv.org/abs/2303.05499)

### Original Highlights

- **Open-Set Detection**: Detect everything with language
- **High Performance**: COCO zero-shot **52.5 AP** (training without COCO data)
- **COCO Fine-tune**: **63.0 AP**
- **Flexible**: Works with Stable Diffusion, Segment Anything, etc.

### Research Benchmarks

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/zero-shot-object-detection-on-mscoco)](https://paperswithcode.com/sota/zero-shot-object-detection-on-mscoco?p=grounding-dino-marrying-dino-with-grounded)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/zero-shot-object-detection-on-odinw)](https://paperswithcode.com/sota/zero-shot-object-detection-on-odinw?p=grounding-dino-marrying-dino-with-grounded)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=grounding-dino-marrying-dino-with-grounded)

---

## Development

### Version Format

GroundedDINO-VL uses semantic versioning:

- Example: `v2.0.0`

### Setup Development Environment

```bash
# Clone and enter directory
git clone https://github.com/ghostcipher1/GroundedDINO-VL.git
cd GroundedDINO-VL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=groundeddino_vl --cov=groundingdino

# Run specific test file
pytest tests/test_import.py -v

# Run with coverage report
pytest tests/ --cov=groundeddino_vl --cov-report=html
```

### Code Quality

```bash
# Format code
black groundeddino_vl groundingdino tests
isort groundeddino_vl groundingdino tests

# Lint code
flake8 groundeddino_vl groundingdino tests --max-line-length=100

# Type checking
mypy groundeddino_vl --ignore-missing-imports

# Security check
bandit -r groundeddino_vl groundingdino
```

### Building Distributions

```bash
# Build source distribution and wheel
python -m build --no-isolation

# Check distribution integrity
twine check dist/*

# Artifacts in dist/
ls -lh dist/
```

---

## Project Structure & Versioning

### Why GroundedDINO-VL?

This project evolved from the `groundingdino-cu128` fork to:

1. **Modernize Architecture**: Clean package structure optimized for current workflows
2. **Future Features**: Foundation for ONNX export, TensorRT optimization, and high-level APIs
3. **Clear Identity**: Distinct branding while honoring GroundingDINO's research contributions

### Relationship to GroundingDINO

- **Based on**: GroundingDINO research and original implementation
- **Legacy Package**: `groundingdino-cu128` (deprecated, redirects to GroundedDINO-VL)
- **Compatibility**: Full backward compatibility with original GroundingDINO imports

---

## Contributing

Contributions are welcome! Please see:

- **Issues**: [github.com/ghostcipher1/GroundedDINO-VL/issues](https://github.com/ghostcipher1/GroundedDINO-VL/issues)
- **Pull Requests**: Follow the existing code style and add tests
- **Documentation**: Help improve docs and examples

---

## License

Copyright (c) 2025 GhostCipher. All rights reserved.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

### Original GroundingDINO License

```
Copyright (c) 2023 IDEA. All Rights Reserved.
Licensed under the Apache License, Version 2.0
```

This project maintains the original Apache 2.0 license and properly attributes the original GroundingDINO research team.

---

## Acknowledgments

- **GroundingDINO Team** at IDEA Research for the original research and implementation
- **Deformable DETR** for the multi-scale deformable attention mechanism
- **DINO** for the transformer-based detection architecture
- **PyTorch Team** for the excellent deep learning framework

---

## Links

- **Homepage**: [github.com/ghostcipher1/GroundedDINO-VL](https://github.com/ghostcipher1/GroundedDINO-VL)
- **PyPI**: [pypi.org/project/groundeddino_vl](https://pypi.org/project/groundeddino_vl/)
- **Original GroundingDINO**: [github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- **Legacy Fork**: [github.com/ghostcipher1/groundingdino-cu128](https://github.com/ghostcipher1/groundingdino-cu128)

---

**Built with ❤️ for the computer vision community**
