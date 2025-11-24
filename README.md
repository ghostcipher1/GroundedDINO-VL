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

- **Modern Stack**: PyTorch 2.7 + CUDA 12.8 support
- **Zero-Shot Detection**: Detect objects using natural language descriptions
- **High Performance**: Based on GroundingDINO's COCO zero-shot 52.5 AP
- **Backward Compatible**: Existing GroundingDINO code continues to work
- **Clean Architecture**: Refactored package structure with better organization
- **Label Studio Integration**: Real-time ML backend for auto-annotation workflows

---

## Example Results

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

## Documentation Index

### Getting Started
- [Installation Guide](docs/INSTALLATION.md) - System requirements, installation methods, and verification
- [Quick Start Guide](docs/QUICKSTART.md) - Basic usage examples and common workflows
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation with examples
- **[Batch Inference](BATCH_INFERENCE_QUICKSTART.txt)** - Process thousands of images with COCO/YOLO/Label Studio export

### Advanced Topics
- [Label Studio Integration](docs/LABEL_STUDIO.md) - Auto-annotation and ML backend setup
- [Building from Source](BUILD_GUIDE.md) - Detailed compilation and build instructions
- [Project Structure](docs/PROJECT_STRUCTURE.md) - Codebase organization and architecture

### Integration & Deployment
- [Testing & Validation](docs/TESTING.md) - Test suite, CI/CD, and quality assurance
- [Migration Guide](docs/MIGRATION_TO_API.md) - Upgrading from previous versions
- [Security Best Practices](docs/SECURITY.md) - Security guidelines and considerations

### Contributing & Support
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute to the project
- [Changelog](docs/CHANGELOG.md) - Version history and release notes
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

---

## Quick Installation

### Via PyPI (Recommended)

```bash
pip install groundeddino_vl
```

### With GPU Support (CUDA 12.8)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install groundeddino_vl
```

### From Source

```bash
git clone https://github.com/ghostcipher1/GroundedDINO-VL.git
cd GroundedDINO-VL
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

**See [Installation Guide](docs/INSTALLATION.md) for detailed instructions and system requirements.**

---

## Quick Start Example

```python
from groundeddino_vl import load_model, predict, annotate

# Load model (auto-downloads weights on first run)
model = load_model(
    config_path="path/to/config.py",
    checkpoint_path="path/to/weights.pth",
    device="cuda"
)

# Run detection with text prompt
result = predict(
    model=model,
    image="path/to/image.jpg",
    text_prompt="car . person . dog",
    box_threshold=0.35,
    text_threshold=0.25,
)

# Visualize results
annotated_image = annotate(image, result, show_labels=True)
```

**See [Quick Start Guide](docs/QUICKSTART.md) for more examples and usage patterns.**

---

## Label Studio Integration

<img alt="LabelStudio logo" src="https://user-images.githubusercontent.com/12534576/192582340-4c9e4401-1fe6-4dbb-95bb-fdbba5493f61.png" width="200" />

GroundedDINO-VL includes an optional Label Studio ML Backend for real-time auto-annotation:

- On-demand inference via FastAPI service
- Auto-labeling with the "magic wand" feature
- Batch annotation assistance
- PostgreSQL/SQLite history logging

**Complete setup guide**: [Label Studio Integration Documentation](docs/LABEL_STUDIO.md)

---

## Research Foundation

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

### Research Benchmarks

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/zero-shot-object-detection-on-mscoco)](https://paperswithcode.com/sota/zero-shot-object-detection-on-mscoco?p=grounding-dino-marrying-dino-with-grounded)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/zero-shot-object-detection-on-odinw)](https://paperswithcode.com/sota/zero-shot-object-detection-on-odinw?p=grounding-dino-marrying-dino-with-grounded)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=grounding-dino-marrying-dino-with-grounded)

---

## System Requirements

| Component | Requirement |
|-----------|------------|
| **Python** | 3.9, 3.10, 3.11, or 3.12 |
| **PyTorch** | 2.7.0+ |
| **CUDA** (optional) | 12.6 or 12.8 |
| **C++ Compiler** | GCC 7+, Clang 5+, or MSVC 2019+ |
| **GPU** (optional) | NVIDIA with Compute Capability 6.0+ |

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
- **Issues**: [github.com/ghostcipher1/GroundedDINO-VL/issues](https://github.com/ghostcipher1/GroundedDINO-VL/issues)

---

**Built with ❤️ for the computer vision community**
