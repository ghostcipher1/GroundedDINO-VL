# GroundedDINO-VL Repository Information

**Version**: v2.0.0  
**License**: Apache License 2.0  
**Python Support**: 3.9, 3.10, 3.11, 3.12  
**PyTorch**: 2.7.0+  
**CUDA**: 12.6 or 12.8 (optional)

---

## Project Overview

**GroundedDINO-VL** is a modern vision-language framework derived from [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), refactored and maintained for current GPU infrastructure with PyTorch 2.7 and CUDA 12.8 support.

This project provides zero-shot object detection using natural language descriptions, achieving high performance on COCO and other benchmarks with a clean, modernized architecture while maintaining full backward compatibility with the original GroundingDINO implementation.

---

## Key Features

- ✅ **Modern Stack**: PyTorch 2.7 + CUDA 12.8 support
- ✅ **Zero-Shot Detection**: Detect objects using natural language descriptions
- ✅ **High Performance**: Based on GroundingDINO's COCO zero-shot 52.5 AP
- ✅ **Backward Compatible**: Existing GroundingDINO code continues to work
- ✅ **Clean Architecture**: Refactored package structure with better organization
- ✅ **Label Studio Integration**: Optional ML backend for real-time auto-annotation (v2.0.0+)
- ✅ **Multiple Export Formats**: Support for ONNX and TensorRT (future)

---

## Repository Structure

```
groundeddino-vl/
├── groundeddino_vl/          # Main package (primary namespace)
│   ├── api/                  # High-level API
│   ├── data/                 # Data loading and transforms
│   ├── datasets/             # Dataset utilities
│   ├── exporters/            # Model export utilities
│   ├── ls_backend/           # Label Studio ML backend
│   ├── models/               # Model architectures
│   │   └── grounding_dino/
│   ├── ops/                  # CUDA operations
│   │   └── csrc/             # C++ and CUDA source
│   ├── utils/                # Utilities (inference, config, visualization)
│   └── version.py
├── groundingdino/            # Legacy package (for backward compatibility)
│   ├── config/
│   ├── datasets/
│   ├── models/
│   │   └── GroundingDINO/
│   │       ├── backbone/
│   │       ├── csrc/
│   │       └── models/
│   └── util/
├── docs/                     # Documentation
│   ├── ls_backend/           # Label Studio backend docs
│   ├── CHANGELOG.md
│   ├── CONTRIBUTING.md
│   ├── MIGRATION_TO_API.md
│   ├── PROJECT_STRUCTURE.md
│   ├── SECURITY.md
│   └── CLEANUP_SUMMARY.md
├── tests/                    # Test suite
│   ├── ls_backend/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_import.py
│   ├── test_import_names.py
│   └── test_predict.py
├── demo/                     # Example scripts
│   ├── demo_images/
│   └── readme_detection_demo.py
├── scripts/                  # Utility scripts
├── checkpoints/              # Model checkpoints
├── .github/                  # GitHub workflows and templates
│   └── workflows/            # CI/CD pipelines
├── .zencoder/                # Zencoder configuration
├── pyproject.toml            # Python project metadata
├── setup.py                  # Build configuration (C++17 extensions)
├── README.md                 # Main documentation
├── BUILD_GUIDE.md            # Build instructions
├── LICENSE                   # Apache 2.0 license
└── requirements.txt          # Python dependencies
```

---

## System Requirements

### Prerequisites

- **Python**: 3.9 or higher
- **C++ Compiler**: GCC 7+, Clang 5+, or MSVC 2019+
- **CUDA Toolkit** (optional): 12.6 or 12.8 for GPU acceleration
- **PyTorch**: 2.7.0+ (with CUDA support for GPU inference)

### Optional Dependencies

| Feature | Package | Purpose |
|---------|---------|---------|
| ONNX Export | `onnx`, `onnxruntime` | Model export to ONNX format |
| TensorRT | TensorRT SDK | GPU-optimized inference |
| JetPack | JetPack SDK | NVIDIA Jetson compatibility |
| Development | `pytest`, `black`, `mypy`, `flake8` | Development and testing tools |

---

## Installation

### PyPI Installation (Recommended)

```bash
pip install groundeddino_vl
```

### With GPU Support (CUDA 12.8)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install groundeddino_vl
```

### Development Installation

```bash
git clone https://github.com/ghostcipher1/GroundedDINO-VL.git
cd GroundedDINO-VL
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

### Development Installation with All Tools

```bash
pip install -e ".[dev]"
```

### Verify Installation

```bash
python -c "import groundeddino_vl; print(f'GroundedDINO-VL {groundeddino_vl.__version__}')"
```

---

## Quick Start

### Caption-Based Detection

```python
import cv2
from groundeddino_vl.utils.inference import Model

# Load image
image = cv2.imread("path/to/image.jpg")

# Initialize model
model = Model(
    model_config_path="path/to/config.py",
    model_checkpoint_path="path/to/weights.pth"
)

# Detect with caption
detections, labels = model.predict_with_caption(
    image=image,
    caption="a dog, a cat",
    box_threshold=0.35,
    text_threshold=0.25,
)

print("Detections:", detections)
print("Labels:", labels)
```

### Class-Based Detection

```python
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

### Visualization

```python
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

### Backward Compatibility

Old imports continue to work with deprecation warnings:

```python
# Old imports (still supported)
import groundingdino
from groundingdino.util import inference

# New recommended imports
import groundeddino_vl
from groundeddino_vl.utils import inference
```

---

## Development Setup

### Create Development Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Running Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=groundeddino_vl --cov=groundingdino

# Specific test file
pytest tests/test_import.py -v

# Coverage report
pytest tests/ --cov=groundeddino_vl --cov-report=html
```

### Code Quality Tools

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
# Build wheel and source distribution
python -m build --no-isolation

# Check integrity
twine check dist/*

# View artifacts
ls -lh dist/
```

---

## Label Studio Integration (v2.0.0+)

GroundedDINO-VL includes an optional Label Studio ML backend for real-time auto-annotation.

### Key Capabilities

- On-demand inference
- Auto-labeling ("magic wand")
- Batch annotation assistance
- Optional PostgreSQL/SQLite history logging

### Documentation

Complete documentation for the Label Studio backend is available in:
- [docs/ls_backend/overview.md](docs/ls_backend/overview.md) - Backend overview
- [docs/ls_backend/installation.md](docs/ls_backend/installation.md) - Setup instructions
- [docs/ls_backend/using_with_labelstudio.md](docs/ls_backend/using_with_labelstudio.md) - Integration guide
- [docs/ls_backend/database.md](docs/ls_backend/database.md) - Database configuration
- [docs/ls_backend/troubleshooting.md](docs/ls_backend/troubleshooting.md) - Troubleshooting

---

## Migration from groundingdino-cu128

| Feature | Old | New | Status |
|---------|-----|-----|--------|
| Package Installation | `pip install groundingdino-cu128` | `pip install groundeddino_vl` | ✅ Both work |
| Import Namespace | `import groundingdino` | `import groundeddino_vl` | ✅ Both work |
| Utilities Module | `groundingdino.util` | `groundeddino_vl.utils` | ⚠️ Note: plural |
| Datasets Module | `groundingdino.datasets` | `groundeddino_vl.data` | ⚠️ Renamed |

See [docs/MIGRATION_TO_API.md](docs/MIGRATION_TO_API.md) for detailed migration instructions.

---

## Research Background

**GroundedDINO-VL** is based on the groundbreaking GroundingDINO research:

**Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**

- **Paper**: [arXiv:2303.05499](https://arxiv.org/abs/2303.05499)
- **Original Repository**: [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

### Performance Benchmarks

- **COCO Zero-Shot**: 52.5 AP (without COCO training data)
- **COCO Fine-tuned**: 63.0 AP
- **ODINW Zero-Shot**: State-of-the-art results

---

## Dependencies Summary

### Core Runtime Dependencies

- **torch** >= 2.7.0, < 2.8
- **torchvision** >= 0.18.0
- **transformers** >= 4.0
- **opencv-python** >= 4.5
- **supervision** >= 0.26.1
- **numpy** >= 1.19
- **timm** >= 0.6 (PyTorch Image Models)
- **addict** >= 2.0 (Dictionary utilities)
- **yapf** >= 0.30 (Code formatting)
- **pycocotools** >= 2.0 (COCO dataset utilities)

### Development Dependencies

- **pytest** >= 7.0 (Testing framework)
- **pytest-cov** >= 4.0 (Coverage reporting)
- **black** == 25.11.0 (Code formatter)
- **isort** >= 5.12 (Import sorting)
- **flake8** >= 6.0 (Linting)
- **mypy** >= 1.0 (Type checking)
- **pylint** >= 2.17 (Advanced linting)
- **bandit** >= 1.7 (Security checking)
- **build** >= 1.0 (Distribution building)
- **twine** >= 4.0 (Package publishing)

---

## Contributing

Contributions are welcome! For guidelines and procedures, see:

- **Contribution Guidelines**: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
- **Issue Tracker**: [github.com/ghostcipher1/GroundedDINO-VL/issues](https://github.com/ghostcipher1/GroundedDINO-VL/issues)
- **Pull Requests**: Follow existing code style and add tests

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make changes and add tests
4. Format code with `black` and `isort`
5. Run tests: `pytest tests/`
6. Submit a pull request

---

## Security

For security policy, reporting vulnerabilities, and security updates, see [docs/SECURITY.md](docs/SECURITY.md).

---

## License

Copyright (c) 2025 GhostCipher.

Licensed under the **Apache License 2.0** - see [LICENSE](LICENSE) for details.

This project includes modifications of the original [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) project, also licensed under Apache 2.0.

See [NOTICE](NOTICE) for full attribution.

---

## Additional Resources

- **Changelog**: [docs/CHANGELOG.md](docs/CHANGELOG.md)
- **Build Guide**: [BUILD_GUIDE.md](BUILD_GUIDE.md)
- **Project Structure**: [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)
- **Main README**: [README.md](README.md)
- **PyPI Package**: [pypi.org/project/groundeddino_vl](https://pypi.org/project/groundeddino_vl/)
- **GitHub Repository**: [github.com/ghostcipher1/GroundedDINO-VL](https://github.com/ghostcipher1/GroundedDINO-VL)

---

## Version History

**v2.0.0** (Current)
- Label Studio ML backend integration
- Modernized API with high-level inference utilities
- PyTorch 2.7 + CUDA 12.8 support
- Clean package structure (groundeddino_vl namespace)
- Full backward compatibility with original GroundingDINO

For detailed version history, see [docs/CHANGELOG.md](docs/CHANGELOG.md).
