# Building groundingdino-cu128 with C++17 Extensions

This guide explains how to build groundingdino-cu128 v0.2.1 from source with C++17 CUDA extensions.

## Overview

groundingdino-cu128 uses **C++17** for its CUDA extensions, providing:
- Modern C++ features for better performance
- Improved compatibility with PyTorch 2.7+
- Better compiler optimizations
- Enhanced template support

## Build System Architecture

### pyproject.toml Configuration

The build system is configured in [pyproject.toml](pyproject.toml):

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
```

**Key points:**
- ✅ `torch` is **NOT** in `build-system.requires` (prevents build-time CUDA installation failures)
- ✅ PyTorch is a **runtime dependency** (installed automatically via `dependencies`)
- ✅ Uses `--no-isolation` flag in CI/CD to use pre-installed PyTorch

### Extension Modules

Extensions are defined in [setup.py](setup.py):

**Extension:** `groundingdino._C`

**Source files:**
- `groundingdino/models/GroundingDINO/csrc/**/*.cpp` (CPU/CUDA)
- `groundingdino/models/GroundingDINO/csrc/**/*.cu` (CUDA only)

**Compiler flags:**
```python
# C++ compiler
extra_compile_args = {"cxx": ["-std=c++17"]}

# NVIDIA CUDA compiler (nvcc)
extra_compile_args["nvcc"] = [
    "-std=c++17",  # C++17 support
    "-DCUDA_HAS_FP16=1",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]
```

## Requirements

### Compiler Requirements

| Platform | Minimum Compiler | C++17 Support |
|----------|-----------------|---------------|
| **Linux** | GCC 7+ or Clang 5+ | ✅ Full |
| **Windows** | Visual Studio 2019+ | ✅ Full |
| **macOS** | Xcode 10+ (Clang 5+) | ✅ Full |

### CUDA Requirements (Optional)

| Component | Version |
|-----------|---------|
| **CUDA Toolkit** | 12.6 or 12.8 |
| **PyTorch** | 2.7.0+ with CUDA 12.6/12.8 |
| **GPU** | NVIDIA GPU with Compute Capability 6.0+ |

### Python Requirements

- Python 3.9, 3.10, 3.11, or 3.12
- pip 20.0 or newer
- setuptools 68+
- wheel

## Build Instructions

### 1. Install Prerequisites

#### Linux (Ubuntu/Debian)

```bash
# Install C++17 compiler
sudo apt-get update
sudo apt-get install -y build-essential g++-11

# Install Python development headers
sudo apt-get install -y python3-dev

# (Optional) Install CUDA Toolkit
# Download from: https://developer.nvidia.com/cuda-downloads
```

#### Windows

```powershell
# Install Visual Studio 2019 or newer
# Download from: https://visualstudio.microsoft.com/downloads/
#
# During installation, select:
#   - Desktop development with C++
#   - MSVC v142+ (or newer) build tools
#   - Windows 10/11 SDK

# (Optional) Install CUDA Toolkit
# Download from: https://developer.nvidia.com/cuda-downloads
```

#### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify compiler version
clang --version  # Should be 5.0 or newer
```

### 2. Install PyTorch with CUDA

```bash
# CUDA 12.8 (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# OR CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# OR CPU-only
pip install torch torchvision
```

### 3. Install Build Dependencies

```bash
pip install setuptools>=68 wheel build
```

### 4. Build from Source

#### Option A: Build and Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/ghostcipher1/groundingdino-cu128.git
cd groundingdino-cu128

# Install in development mode (editable)
pip install -e .

# OR install normally
pip install .
```

#### Option B: Build Distribution

```bash
# Build source distribution (.tar.gz)
python -m build --sdist --no-isolation

# Build wheel (.whl)
python -m build --wheel --no-isolation

# Distributions will be in dist/
ls -lh dist/
```

### 5. Verify Installation

```bash
python -c "import groundingdino; print(f'Version: {groundingdino.__version__}')"
python -c "import shadow_dino; print(f'CUDA available: {shadow_dino.__cuda_available__}')"
python -c "from groundingdino import datasets; print('✓ datasets module works')"
```

## Build Modes

### 1. With CUDA Extensions (GPU)

**Requirements:**
- CUDA Toolkit installed
- PyTorch with CUDA support
- NVIDIA GPU available

**Build behavior:**
```python
# setup.py detects:
torch.cuda.is_available() == True
CUDA_HOME is not None

# Builds:
groundingdino._C extension with CUDA support
```

**Verification:**
```python
import groundingdino
print(hasattr(groundingdino, '_C'))  # True
print(dir(groundingdino._C))  # ['ms_deform_attn_forward', 'ms_deform_attn_backward', ...]
```

### 2. CPU-Only Mode

**Requirements:**
- C++17 compiler
- PyTorch (CPU version)

**Build behavior:**
```python
# setup.py detects:
torch.cuda.is_available() == False OR CUDA_HOME is None

# Returns: empty extension list
# Package installs without _C extension
```

**Verification:**
```python
import groundingdino
print(hasattr(groundingdino, '_C'))  # False
# Package still works for inference on CPU
```

### 3. Without PyTorch (Build-time)

**Requirements:**
- setuptools>=68, wheel

**Build behavior:**
```python
# setup.py detects:
TORCH_AVAILABLE == False

# Skips extension build
# PyTorch will be installed as runtime dependency
```

**Use case:** Creating source distributions for PyPI

## Troubleshooting

### Issue: "error: invalid argument '-std=c++17' not allowed with 'C'"

**Cause:** Trying to compile .c files with C++ flags

**Solution:** Ensure all source files use `.cpp` extension, not `.c`

### Issue: "torch not found during build"

**Cause:** Building with isolated environment (`--no-isolation` flag missing)

**Solution:** Use `--no-isolation` flag:
```bash
python -m build --no-isolation
```

### Issue: "CUDA_HOME is not set"

**Cause:** CUDA Toolkit not found

**Solution:**
```bash
# Linux
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Windows (PowerShell)
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:PATH = "$env:CUDA_HOME\bin;$env:PATH"
```

### Issue: "error: identifier is undefined" (CUDA compilation)

**Cause:** CUDA SDK version mismatch or missing includes

**Solution:** Verify CUDA Toolkit version matches PyTorch:
```bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

### Issue: "undefined reference to `THCudaCheck_XXXXX`"

**Cause:** PyTorch API mismatch (PyTorch 1.x code with PyTorch 2.x)

**Solution:** This fork already fixes these issues for PyTorch 2.7+. Ensure you're using the correct version.

### Issue: Compiler not found (Windows)

**Cause:** Visual Studio C++ build tools not installed

**Solution:**
1. Open Visual Studio Installer
2. Modify your installation
3. Select "Desktop development with C++"
4. Install MSVC v142 or newer build tools

### Issue: Slow compilation

**Cause:** NVCC compiling large CUDA files

**Solution:** Use ninja build system for parallel compilation:
```bash
pip install ninja
# ninja is automatically detected by torch.utils.cpp_extension
```

## Advanced: Custom Build Configuration

### Environment Variables

```bash
# Set custom compiler
export CC=gcc-11
export CXX=g++-11

# Set CUDA architecture (optional, for specific GPU)
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

# Enable verbose build output
export VERBOSE=1
```

### Debug Build

```bash
# Build with debug symbols
CFLAGS="-g -O0" python -m build --no-isolation
```

### Custom Compiler Flags

Edit `setup.py` and modify `extra_compile_args`:

```python
extra_compile_args = {
    "cxx": [
        "-std=c++17",
        "-O3",  # Optimization level
        "-march=native",  # CPU-specific optimizations
        "-DDEBUG",  # Custom defines
    ]
}
```

## CI/CD Build Configuration

GitHub Actions workflows ([.github/workflows/](.github/workflows/)):

### CPU Build (ci.yml)
```yaml
- name: Install build dependencies
  run: |
    python -m pip install --upgrade pip
    pip install setuptools>=68 wheel build
    pip install -r requirements.txt

- name: Build source distribution
  run: python -m build --sdist --no-isolation
```

### CUDA Build (build-wheels.yml)
```yaml
- name: Install PyTorch with CUDA
  run: |
    python -m pip install torch torchvision \
      --index-url https://download.pytorch.org/whl/cu128

- name: Install build dependencies
  run: |
    python -m pip install setuptools>=68 wheel build ninja
    python -m pip install -r requirements.txt

- name: Build wheel
  run: python -m build --wheel --no-isolation
```

## Testing the Build

### Quick Test (verify-build-local.py)

```bash
python verify-build-local.py
```

## Performance Considerations

### C++17 Benefits

- **Constexpr improvements:** Better compile-time optimizations
- **Structured bindings:** Cleaner code, better inlining
- **Fold expressions:** Faster template instantiation
- **Inline variables:** Reduced symbol overhead

### CUDA Optimization Flags

The build uses these NVCC flags for optimal performance:

```bash
-DCUDA_HAS_FP16=1                    # Enable FP16 operations
-D__CUDA_NO_HALF_OPERATORS__         # Avoid deprecated half operators
-D__CUDA_NO_HALF_CONVERSIONS__       # Explicit half conversions
-D__CUDA_NO_HALF2_OPERATORS__        # Avoid deprecated half2 operators
```

### Build Time Optimization

```bash
# Use ninja for parallel compilation (faster)
pip install ninja

# Use ccache for incremental builds (development)
sudo apt-get install ccache
export CC="ccache gcc"
export CXX="ccache g++"
```

## References

- **PyTorch C++ Extension Guide:** https://pytorch.org/tutorials/advanced/cpp_extension.html
- **setuptools documentation:** https://setuptools.pypa.io/
- **PEP 517 (pyproject.toml):** https://peps.python.org/pep-0517/
- **C++17 compiler support:** https://en.cppreference.com/w/cpp/compiler_support/17
- **CUDA Toolkit docs:** https://docs.nvidia.com/cuda/

## Support

- **Issues:** https://github.com/ghostcipher1/groundingdino-cu128/issues
- **Discussions:** https://github.com/ghostcipher1/groundingdino-cu128/discussions
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Testing:** [TESTING.md](TESTING.md)
