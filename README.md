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

---

## Quick Start

### High-Level API 

The new API returns Supervision Detections objects and abstracts preprocessing, postprocessing,
and caption handling.

#### Caption-Based Detection
```
import cv2
from groundeddino_vl.utils.inference import Model

image = cv2.imread("path/to/image.jpg")

model = Model(
    model_config_path="path/to/config.py",
    model_checkpoint_path="path/to/weights.pth"
)

detections, labels = model.predict_with_caption(
    image=image,
    caption="a dog, a cat",
    box_threshold=0.35,
    text_threshold=0.25,
)

print(detections)
print(labels)

```
### Class-Based Detection
```
classes = ["cat", "dog"]

detections = model.predict_with_classes(
    image=image,
    classes=classes,
    box_threshold=0.35,
    text_threshold=0.25,
)

print("Detections:", detections)
print("Class IDs:", detections.class_id)

```

### Visualize with Supervision
```
import supervision as sv

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [f"{phrase} {conf:.2f}" for phrase, conf in zip(labels, detections.confidence)]

annotated_image = box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)

sv.plot_image(annotated_image)

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

## Docker Usage

### Pull from GitHub Container Registry

```bash
docker pull ghcr.io/ghostcipher1/groundeddino_vl:latest
```

### Run with GPU Support

```bash
docker run --gpus all -it ghcr.io/ghostcipher1/groundeddino_vl:latest python
```

### Build Locally

```bash
docker build -t groundeddino_vl:local .
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

GroundedDINO-VL uses **year-based versioning**:

- Format: `YYYY.MAJOR.MINOR` (e.g., `2025.11.0`)
- **2025**: Release year
- **11**: Major version within the year
- **0**: Minor version/patch

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
