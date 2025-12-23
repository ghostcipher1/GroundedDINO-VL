# Installation Guide

This guide provides comprehensive instructions for installing GroundedDINO-VL across different platforms and configurations.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
  - [Method 1: PyPI Installation (Recommended)](#method-1-pypi-installation-recommended)
  - [Method 2: Installation from Source](#method-2-installation-from-source)
  - [Method 3: Development Installation](#method-3-development-installation)
- [GPU Support Setup](#gpu-support-setup)
- [Verification](#verification)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|--------------|
| **Operating System** | Linux, Windows, macOS |
| **Python** | 3.9, 3.10, 3.11, or 3.12 |
| **RAM** | 8 GB minimum (16 GB recommended) |
| **Disk Space** | 5 GB for installation and model weights |

### C++ Compiler Requirements

GroundedDINO-VL includes C++17 CUDA extensions that require a compatible compiler:

| Platform | Compiler | Version | C++17 Support |
|----------|----------|---------|---------------|
| **Linux** | GCC | 7.0+ | Full |
| **Linux** | Clang | 5.0+ | Full |
| **Windows** | MSVC | Visual Studio 2019+ | Full |
| **macOS** | Apple Clang | Xcode 10+ | Full |

### GPU Support (Optional)

For CUDA acceleration:

| Component | Requirement                             |
|-----------|-----------------------------------------|
| **GPU** | NVIDIA GPU with Compute Capability 6.0+ |
| **CUDA Toolkit** | 12.6 or 12.9                            |
| **PyTorch** | 2.7.0+ with CUDA 12.6/12.9 support      |
| **NVIDIA Driver** | 525.60.13+ (Linux) or 527.41+ (Windows) |

**Supported NVIDIA GPUs:**
- RTX 50 Series (Blackwell): RTX 5060, 5060 Ti, 5070, 5070 Ti, 5080, 5090
- RTX 40 Series (Ada Lovelace): RTX 4090, 4080, 4070, etc.
- RTX 30 Series (Ampere): RTX 3090, 3080, 3070, etc.
- RTX 20 Series (Turing): RTX 2080 Ti, 2070, etc.
- GTX 16 Series: GTX 1660 Ti, 1650, etc.
- Tesla/Quadro: V100, P100, A100, etc.

---

## Installation Methods

### Method 1: PyPI Installation (Recommended)

The simplest way to install GroundedDINO-VL is via PyPI.

#### CPU-Only Installation

```bash
# Install latest version
pip install groundeddino_vl

# Install specific version
pip install groundeddino_vl==2.0.3
```

#### GPU Installation (CUDA 12.8)

```bash
# Install PyTorch with CUDA 12.8 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install GroundedDINO-VL
pip install groundeddino_vl
```

#### GPU Installation (CUDA 12.6)

```bash
# Install PyTorch with CUDA 12.6 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install GroundedDINO-VL
pip install groundeddino_vl
```

---

### Method 2: Installation from Source

Install directly from the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/ghostcipher1/GroundedDINO-VL.git
cd GroundedDINO-VL

# Install PyTorch with CUDA support (if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install GroundedDINO-VL
pip install .
```

---

### Method 3: Development Installation

For contributors or those who want to modify the code:

```bash
# Clone the repository
git clone https://github.com/ghostcipher1/GroundedDINO-VL.git
cd GroundedDINO-VL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install runtime dependencies
pip install -r requirements.txt

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

**Development dependencies include:**
- pytest (testing)
- black (code formatting)
- isort (import sorting)
- flake8 (linting)
- mypy (type checking)
- bandit (security scanning)

---

## GPU Support Setup

### Linux

#### Install CUDA Toolkit

```bash
# Download CUDA 12.8 installer
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_550.54.15_linux.run

# Run installer
sudo sh cuda_12.8.0_550.54.15_linux.run

# Set environment variables (add to ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Reload environment
source ~/.bashrc
```

#### Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Check NVIDIA driver
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.8     |
# +-----------------------------------------------------------------------------------------+
```

### Windows

#### Install CUDA Toolkit

1. Download CUDA 12.8 installer from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
2. Run the installer (typically `cuda_12.8.0_windows.exe`)
3. Select "Custom Installation" and ensure these components are selected:
   - CUDA Toolkit
   - CUDA Development
   - CUDA Runtime
   - CUDA Documentation (optional)

#### Set Environment Variables

```powershell
# Add to system PATH (PowerShell as Administrator)
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:PATH = "$env:CUDA_HOME\bin;$env:PATH"

# Make permanent via System Properties
# Settings > System > About > Advanced system settings > Environment Variables
```

#### Verify Installation

```powershell
# Check CUDA
nvcc --version

# Check NVIDIA driver
nvidia-smi
```

### macOS

macOS does not support CUDA. Use CPU-only installation:

```bash
pip install torch torchvision
pip install groundeddino_vl
```

For Apple Silicon (M1/M2/M3), PyTorch uses Metal Performance Shaders (MPS):

```python
import torch

# Check MPS availability
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

---

## Verification

### Basic Import Test

```bash
# Verify installation
python -c "import groundeddino_vl; print(f'GroundedDINO-VL {groundeddino_vl.__version__}')"

# Expected output:
# GroundedDINO-VL 2.0.3
```

### CUDA Extension Verification

```bash
# Check if CUDA extensions are available
python -c "import groundeddino_vl; print('CUDA available:', groundeddino_vl.__cuda_available__)"

# Expected output (GPU build):
# CUDA available: True
```

### PyTorch CUDA Verification

```bash
# Check PyTorch CUDA support
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Expected output (GPU installation):
# PyTorch version: 2.7.0+cu128
# CUDA available: True
```

### Complete Verification Script

Create a file `verify_installation.py`:

```python
#!/usr/bin/env python3
"""Verify GroundedDINO-VL installation."""

import sys

def verify_installation():
    print("=== GroundedDINO-VL Installation Verification ===\n")

    # Check Python version
    print(f"Python version: {sys.version}")

    # Check GroundedDINO-VL
    try:
        import groundeddino_vl
        print(f"✓ GroundedDINO-VL: {groundeddino_vl.__version__}")
    except ImportError as e:
        print(f"✗ GroundedDINO-VL import failed: {e}")
        return False

    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch not found")
        return False

    # Check CUDA extensions
    try:
        print(f"  CUDA extensions: {groundeddino_vl.__cuda_available__}")
    except AttributeError:
        print("  CUDA extensions: Not available")

    # Check key modules
    modules = [
        "groundeddino_vl.models",
        "groundeddino_vl.utils",
        "groundeddino_vl.api",
    ]

    print("\nModule imports:")
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            return False

    print("\n✓ Installation verified successfully!")
    return True

if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)
```

Run the verification:

```bash
python verify_installation.py
```

---

## Platform-Specific Instructions

### Ubuntu/Debian Linux

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    g++-11 \
    python3-dev \
    python3-pip \
    git \
    wget

# Install CUDA (if needed)
# See CUDA Toolkit installation above

# Install GroundedDINO-VL
pip install groundeddino_vl
```

### CentOS/RHEL/Fedora

```bash
# Install system dependencies
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel python3-pip git wget

# Install CUDA (if needed)
# Download appropriate RPM from NVIDIA

# Install GroundedDINO-VL
pip install groundeddino_vl
```

### Windows

```powershell
# Ensure Visual Studio 2019+ is installed with C++ support
# Install from: https://visualstudio.microsoft.com/downloads/

# Install CUDA Toolkit (if needed)
# See Windows CUDA installation above

# Install GroundedDINO-VL
pip install groundeddino_vl
```

### Docker Container

```dockerfile
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install GroundedDINO-VL
RUN pip install groundeddino_vl

# Set working directory
WORKDIR /workspace

CMD ["/bin/bash"]
```

Build and run:

```bash
docker build -t groundeddino-vl .
docker run --gpus all -it groundeddino-vl
```

---

## Troubleshooting

### Issue: "No module named 'groundeddino_vl'"

**Cause:** Package not installed or wrong Python environment

**Solution:**
```bash
# Check which Python is being used
which python
python --version

# Reinstall in correct environment
pip install groundeddino_vl
```

### Issue: "ImportError: cannot import name '_C'"

**Cause:** CUDA extensions not built or missing dependencies

**Solution:**
```bash
# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with verbose output
pip install --force-reinstall --no-cache-dir groundeddino_vl -v
```

### Issue: "OSError: CUDA_HOME environment variable is not set"

**Cause:** CUDA Toolkit not found

**Solution:**
```bash
# Linux/macOS
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# Windows (PowerShell)
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
```

### Issue: "RuntimeError: CUDA out of memory"

**Cause:** GPU memory exhausted

**Solution:**
- Reduce batch size
- Use smaller images
- Enable gradient checkpointing
- Use mixed precision (FP16)

```python
# Enable mixed precision
import torch
with torch.cuda.amp.autocast():
    result = predict(model, image, text_prompt)
```

### Issue: Slow installation or download

**Cause:** PyPI mirror or network issues

**Solution:**
```bash
# Use a faster mirror (example: Alibaba mirror for China)
pip install groundeddino_vl -i https://mirrors.aliyun.com/pypi/simple/

# Or increase timeout
pip install groundeddino_vl --timeout=120
```

---

## Next Steps

After successful installation:

1. **Quick Start**: See [QUICKSTART.md](QUICKSTART.md) for usage examples
2. **API Reference**: Check [API_REFERENCE.md](API_REFERENCE.md) for detailed API documentation
3. **Model Weights**: Download pre-trained models (handled automatically on first use)

For build-from-source instructions, see [BUILD_GUIDE.md](../BUILD_GUIDE.md).

---

**Need Help?** Check the [Troubleshooting Guide](TROUBLESHOOTING.md) or open an issue on [GitHub](https://github.com/ghostcipher1/GroundedDINO-VL/issues).
