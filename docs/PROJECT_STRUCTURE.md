# Project Structure

This document describes the organization of the GroundedDINO-VL project.

## Directory Structure

```
GroundedDINO-VL/
├── .github/                   # GitHub configuration
│   ├── workflows/            # CI/CD pipelines
│   │   ├── ci.yml           # CPU testing
│   │   ├── gpu-ci.yml       # GPU testing
│   │   └── publish.yml      # Release publishing
│   ├── ISSUE_TEMPLATE/       # Issue templates
│   └── *.md                  # GitHub documentation
├── demo/                     # Example scripts and notebooks
│   ├── *.py                 # Python demo scripts
│   └── *.ipynb              # Jupyter notebooks
├── docs/                     # Project documentation
│   ├── CHANGELOG.md         # Version history
│   ├── CONTRIBUTING.md      # Contribution guidelines
│   ├── SECURITY.md          # Security policy
│   ├── PROJECT_STRUCTURE.md # This file
│   ├── ls_backend/          # Label Studio backend docs
│   └── *.md                 # Additional guides
├── groundeddino_vl/         # Main package (primary namespace)
│   ├── __init__.py          # Package entry point
│   ├── version.py           # Version management
│   ├── api/                 # High-level API
│   │   └── __init__.py      # Public API exports
│   ├── models/              # Model implementations
│   │   ├── __init__.py
│   │   ├── configs/         # Model configuration files
│   │   └── grounding_dino/  # GroundingDINO model
│   │       ├── __init__.py
│   │       ├── backbone/    # Backbone architectures
│   │       ├── transformer.py
│   │       ├── ms_deform_attn.py
│   │       └── model.py
│   ├── ops/                 # CUDA operations
│   │   ├── __init__.py
│   │   ├── _C.*.so          # Compiled C++ extension
│   │   └── csrc/            # C++ and CUDA source code
│   │       ├── deform_attn/ # Deformable attention kernels
│   │       ├── vision.cpp   # Main extension module
│   │       └── *.cu         # CUDA kernels
│   ├── utils/               # Utility modules
│   │   ├── __init__.py
│   │   ├── inference.py     # High-level inference API
│   │   ├── box_ops.py      # Bounding box operations
│   │   ├── visualizer.py   # Visualization utilities
│   │   ├── slconfig.py     # Configuration utilities
│   │   ├── logger.py       # Logging utilities
│   │   └── *.py            # Other utilities
│   ├── data/                # Data loading and transforms
│   │   ├── __init__.py
│   │   └── *.py            # Data loading modules
│   ├── datasets/            # Dataset implementations
│   │   ├── __init__.py
│   │   └── *.py            # Dataset modules
│   ├── ls_backend/          # Label Studio ML backend
│   │   ├── __init__.py
│   │   ├── server.py       # FastAPI server
│   │   ├── config.py       # Backend configuration
│   │   ├── model_loader.py # Model loading
│   │   ├── inference_engine.py # Inference logic
│   │   ├── database.py     # Database support
│   │   ├── schemas.py      # Request/response schemas
│   │   └── utils.py        # Backend utilities
│   ├── exporters/           # Model export framework
│   │   └── __init__.py      # (ONNX, TensorRT - future)
│   └── summaries/           # Analysis and summaries
│       └── *.md             # Project summaries
├── groundingdino/           # Legacy namespace (backward compatibility)
│   ├── __init__.py
│   ├── version.py
│   ├── config/              # Configuration files
│   ├── models/              # Model implementations
│   ├── util/                # Utilities
│   └── datasets/            # Dataset modules
├── tests/                   # Test suite
│   ├── conftest.py
│   ├── test_*.py           # Test modules
│   └── ls_backend/         # Backend tests
├── .editorconfig            # Editor configuration
├── .gitattributes           # Git attributes
├── .gitignore              # Git ignore rules
├── LICENSE                 # Apache 2.0 license
├── NOTICE                  # Attribution notice
├── MANIFEST.in             # Package data manifest
├── pyproject.toml          # Python project configuration
├── setup.py                # Build configuration (C++17 extensions)
├── requirements.txt        # Python dependencies (reference)
├── README.md               # Main documentation
├── BUILD_GUIDE.md          # Build instructions
├── FIXES_SUMMARY.md        # Summary of recent fixes
├── repo.md                 # Repository information
└── CHANGELOG.md            # Legacy changelog location
```

## Key Directories

### `groundeddino_vl/` (Primary Package)
The main package containing the modernized GroundedDINO-VL implementation.

- **`api/`**: High-level API for simplified inference
  - `load_model()`: Load models from config/checkpoint
  - `predict()`: Run object detection
  - `annotate()`: Draw results on images
  - `DetectionResult`: Unified result container

- **`models/`**: Model architectures
  - `grounding_dino/`: GroundingDINO implementation
  - `configs/`: Model configuration files (.py format)
  - `backbone/`: Vision transformers (Swin, etc.)
  - `transformer.py`: Transformer modules

- **`ops/`**: CUDA operations
  - `_C.*.so`: Compiled C++ extension (if CUDA available)
  - `csrc/`: C++17 source code
    - `deform_attn/`: Multi-scale deformable attention kernels
    - GPU/CPU implementations

- **`utils/`**: Utilities
  - `inference.py`: Low-level inference API
  - `visualizer.py`: Draw bounding boxes
  - `box_ops.py`: Bounding box operations
  - `slconfig.py`: Configuration utilities
  - `logger.py`: Logging setup

- **`data/`**: Data loading
  - DataLoader implementations
  - Transform utilities
  - Preprocessing functions

- **`ls_backend/`**: Label Studio integration
  - FastAPI server (port 9090)
  - Auto-annotation service
  - Database support (PostgreSQL/SQLite)

- **`exporters/`**: Model export (future)
  - ONNX export (planned)
  - TensorRT optimization (planned)

### `groundingdino/` (Legacy Namespace)
Backward compatibility layer for existing code.
- Provides same imports as original GroundingDINO
- Uses `groundeddino_vl` under the hood
- Shows deprecation warnings

### `docs/`
Complete project documentation:
- **CHANGELOG.md**: Version history and features
- **CONTRIBUTING.md**: Development guidelines
- **SECURITY.md**: Vulnerability reporting
- **PROJECT_STRUCTURE.md**: This file
- **ls_backend/**: Label Studio backend documentation

### `demo/`
Example usage scripts and Jupyter notebooks demonstrating:
- Basic inference
- Zero-shot detection
- Visualization
- Integration with other tools

### `tests/`
Comprehensive test suite:
- `test_import.py`: Package import tests
- `test_api.py`: High-level API tests
- `test_predict.py`: Inference tests
- `ls_backend/`: Backend-specific tests

## Build Artifacts

The following are generated during build and should not be committed:
- `build/` - Build output
- `dist/` - Distribution packages (.whl, .tar.gz)
- `*.egg-info/` - Package metadata
- `*.so` - Compiled C++ extensions
- `*.o` - Object files
- `__pycache__/` - Python bytecode
- `*.pyc` - Compiled Python files

These are ignored via `.gitignore`.

## Configuration Files

### `.editorconfig`
Coding style preferences (line length, indentation, charset).

### `.gitattributes`
Git behavior for different file types (line endings, diffs).

### `.gitignore`
Version control exclusions (build artifacts, caches, virtual environments).

### `pyproject.toml`
Modern Python project configuration:
- Package metadata (name, version, authors)
- Dependencies (runtime and optional)
- Build system (setuptools)
- Tool configurations (pytest, black, isort, mypy, etc.)

### `setup.py`
C++17 extension building:
- CUDA detection
- Compiler checks
- Extension module compilation
- Prerequisite verification

### `requirements.txt`
Runtime and development dependencies (reference copy).

## Documentation Files

- **README.md**: Main project documentation (GitHub home page)
- **BUILD_GUIDE.md**: Detailed build instructions
- **FIXES_SUMMARY.md**: Recent fixes and improvements
- **repo.md**: Repository information
- **docs/CHANGELOG.md**: Version history
- **docs/ls_backend/**: Label Studio backend guides
- **LICENSE**: Apache License 2.0
- **NOTICE**: Attribution and licensing

## Package Namespaces

The project supports two import namespaces:

### Primary (Recommended)
```python
import groundeddino_vl
from groundeddino_vl import load_model, predict
from groundeddino_vl.utils import inference
from groundeddino_vl.models import grounding_dino
```

### Legacy (Backward Compatibility)
```python
import groundingdino  # Works but shows deprecation warning
from groundingdino.util import inference  # Use groundeddino_vl.utils instead
```

Both provide identical functionality.

## Python Support

- **Minimum**: Python 3.9
- **Tested**: Python 3.9, 3.10, 3.11, 3.12
- **Recommended**: Python 3.11+ (performance improvements)

## Build Requirements

- **C++ Compiler**: GCC 7+, Clang 5+, or MSVC 2019+
- **C++ Standard**: C++17 (mandatory for extensions)
- **CUDA Toolkit** (optional): 12.6 or 12.8 for GPU support
- **PyTorch**: 2.7.0+ (with matching CUDA version)

## Version Management

- Version defined in `pyproject.toml`
- Accessible via: `import groundeddino_vl; groundeddino_vl.__version__`
- Uses `importlib.metadata` for installed packages
- Fallback: "0.0.0" for development installations

## Dependency Strategy

### Core Dependencies (automatically installed)
- torch, torchvision
- transformers
- timm, numpy, opencv-python
- supervision, pycocotools
- fastapi, uvicorn, SQLAlchemy

### Optional Dependencies
- **dev**: Testing and linting tools
- **onnx**: ONNX export support
- **trt**: TensorRT optimization
- **jetson**: JetPack compatibility

Install with: `pip install groundeddino-vl[dev]`

## CI/CD Pipeline

### Workflows
- **ci.yml**: CPU testing (Python 3.10, 3.11, 3.12)
- **gpu-ci.yml**: GPU testing (CUDA 12.8, self-hosted)
- **publish.yml**: Release publishing to PyPI

### Test Matrix
- Multiple Python versions
- CPU and GPU configurations
- Windows, macOS, Linux support

## Development Workflow

1. **Setup**: `pip install -e ".[dev]"`
2. **Code**: Follow style in `.editorconfig`
3. **Test**: `pytest tests/`
4. **Format**: `black groundeddino_vl groundingdino`
5. **Check**: `flake8`, `mypy`, `bandit`
6. **Commit**: Push to feature branch
7. **Release**: Tag version, GitHub release, PyPI

---

For detailed build instructions, see [BUILD_GUIDE.md](../BUILD_GUIDE.md).
For version history, see [CHANGELOG.md](./CHANGELOG.md).
