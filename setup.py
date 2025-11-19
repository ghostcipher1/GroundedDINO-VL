# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Modified from
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/setup.py
# https://github.com/facebookresearch/detectron2/blob/main/setup.py
# https://github.com/open-mmlab/mmdetection/blob/master/setup.py
# https://github.com/Oneflow-Inc/libai/blob/main/setup.py
# ------------------------------------------------------------------------------------------------
#
# IMPORTANT: Installation for CUDA Extension Building
# ========================================================
# To ensure the CUDA extension (_C module) is properly built with your environment's
# PyTorch and CUDA versions, use:
#
#   python -m pip install -e . --no-build-isolation
#
# The --no-build-isolation flag is critical because:
# - It uses your pre-installed PyTorch and CUDA toolkit
# - Avoids downloading/installing CUDA again during build
# - Ensures ABI compatibility between extension and runtime
# - Prevents "Undefined backend" errors in multi-scale deformable attention kernel
#
# Without --no-build-isolation, the extension may be compiled against different
# PyTorch/CUDA versions than your runtime environment, causing device type resolution
# failures when running inference on GPU.
# ------------------------------------------------------------------------------------------------

import glob
import os
import platform
import re
import subprocess
import sys

from setuptools import find_packages, setup

try:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_HOME = None
    CppExtension = None
    CUDAExtension = None

# GroundedDINO-VL package info
package_name = "groundeddino-vl"
cwd = os.path.dirname(os.path.abspath(__file__))


sha = "Unknown"
try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
except Exception:
    pass
requirements = ["torch", "torchvision"]

# Only check torch version if torch is available
if TORCH_AVAILABLE:
    torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
else:
    torch_ver = None


def check_cuda_toolkit():
    """Check if NVIDIA CUDA Toolkit is installed and accessible."""
    # Check CUDA_HOME environment variable
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    # Check if nvcc is in PATH
    nvcc_found = False
    nvcc_version = None
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            nvcc_found = True
            # Extract version from output
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    nvcc_version = line.strip()
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check common CUDA installation paths
    cuda_paths = []
    if cuda_home:
        cuda_paths.append(cuda_home)

    if platform.system() == "Windows":
        default_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            r"C:\CUDA",
        ]
    else:
        default_paths = [
            "/usr/local/cuda",
            "/usr/local/cuda-12.8",
            "/usr/local/cuda-12.6",
            "/opt/cuda",
        ]

    for path in default_paths:
        if os.path.exists(path) and os.path.isdir(path):
            cuda_paths.append(path)

    return {
        "found": nvcc_found or len(cuda_paths) > 0,
        "nvcc_available": nvcc_found,
        "nvcc_version": nvcc_version,
        "cuda_home": cuda_home,
        "cuda_paths": cuda_paths,
    }


def check_cpp17_compiler():
    """Check if a C++17 compatible compiler is available."""
    is_windows = platform.system() == "Windows"
    compiler_found = False
    compiler_info = None

    if is_windows:
        # Check for MSVC (Visual Studio)
        try:
            # Try to find cl.exe (MSVC compiler)
            result = subprocess.run(["cl"], capture_output=True, text=True, timeout=5)
            if result.returncode != 9009:  # 9009 means command not found
                compiler_found = True
                compiler_info = "MSVC (Visual Studio)"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Also check for clang-cl
        try:
            result = subprocess.run(
                ["clang-cl", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                compiler_found = True
                compiler_info = "Clang (clang-cl)"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    else:
        # Check for GCC
        try:
            result = subprocess.run(["gcc", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Check GCC version (need 7+ for C++17)
                version_line = result.stdout.split("\n")[0]
                try:
                    version_str = version_line.split()[-1]
                    major_version = int(version_str.split(".")[0])
                    if major_version >= 7:
                        compiler_found = True
                        compiler_info = f"GCC {version_str}"
                except (ValueError, IndexError):
                    compiler_found = True  # Assume compatible if we can't parse
                    compiler_info = "GCC (version unknown)"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check for Clang
        try:
            result = subprocess.run(
                ["clang", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Check Clang version (need 5+ for C++17)
                version_line = result.stdout.split("\n")[0]
                try:
                    # Extract version number
                    match = re.search(r"version\s+(\d+)", version_line)
                    if match:
                        major_version = int(match.group(1))
                        if major_version >= 5:
                            compiler_found = True
                            compiler_info = f"Clang {match.group(1)}"
                    else:
                        compiler_found = True  # Assume compatible if we can't parse
                        compiler_info = "Clang (version unknown)"
                except (ValueError, IndexError):
                    compiler_found = True
                    compiler_info = "Clang (version unknown)"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return {
        "found": compiler_found,
        "info": compiler_info,
    }


def prompt_user(prompt_text, default="yes"):
    """Prompt user for yes/no input. Returns True for yes, False for no."""
    # Check if we're in a non-interactive environment
    if not sys.stdin.isatty():
        # Non-interactive (CI/CD, pip install from wheel, etc.)
        # Default to 'yes' to allow installation to proceed
        print(f"\nNon-interactive environment detected. Defaulting to 'yes'.")
        return True

    valid_responses = {"yes": True, "y": True, "no": False, "n": False}
    default_response = default.lower()

    while True:
        try:
            response = input(f"{prompt_text} [yes/no] (default: {default}): ").strip().lower()
            if not response:
                response = default_response
            if response in valid_responses:
                return valid_responses[response]
            else:
                print("Please enter 'yes' or 'no' (or 'y'/'n').")
        except (EOFError, KeyboardInterrupt):
            print("\n\nInstallation cancelled by user.")
            return False


def check_prerequisites():
    """Check for required prerequisites and prompt user if missing."""
    # Skip during build phase - only check during actual installation
    import os

    # Check if we're in a build context (PEP 517 build isolation or building distributions)
    build_commands = ["sdist", "bdist", "bdist_wheel", "bdist_egg", "build"]
    if any(cmd in sys.argv for cmd in build_commands):
        return  # Skip during build - prerequisites checked during installation
    if os.environ.get("PEP517_BUILD_BACKEND") or os.environ.get("BUILDING_WHEEL"):
        return  # Skip during build

    # Use ASCII-safe characters for Windows compatibility
    print("\n" + "=" * 70)
    print("Checking prerequisites for GroundedDINO-VL installation...")
    print("=" * 70)

    cuda_check = check_cuda_toolkit()
    compiler_check = check_cpp17_compiler()

    missing_items = []

    # Check CUDA Toolkit
    if not cuda_check["found"]:
        missing_items.append("NVIDIA CUDA Toolkit")
        print("\n[X] NVIDIA CUDA Toolkit NOT FOUND")
        print("   - nvcc not found in PATH")
        if not cuda_check["cuda_home"]:
            print("   - CUDA_HOME environment variable not set")
        print("   - No CUDA installation detected in common paths")
    else:
        print("\n[OK] NVIDIA CUDA Toolkit FOUND")
        if cuda_check["nvcc_available"]:
            print(f"   - nvcc available: {cuda_check['nvcc_version'] or 'version detected'}")
        if cuda_check["cuda_home"]:
            print(f"   - CUDA_HOME: {cuda_check['cuda_home']}")
        if cuda_check["cuda_paths"]:
            print(f"   - CUDA paths found: {', '.join(cuda_check['cuda_paths'])}")

    # Check C++17 Compiler
    if not compiler_check["found"]:
        missing_items.append("C++17 compatible compiler")
        print("\n[X] C++17 Compatible Compiler NOT FOUND")
        print("   - No compatible compiler detected in PATH")
    else:
        print(f"\n[OK] C++17 Compatible Compiler FOUND")
        print(f"   - Compiler: {compiler_check['info']}")

    print("=" * 70)

    if missing_items:
        print("\n[WARNING] MISSING PREREQUISITES DETECTED")
        print(f"\nThe following required components are missing:")
        for item in missing_items:
            print(f"  - {item}")

        print("\nInstallation instructions:")
        if "NVIDIA CUDA Toolkit" in missing_items:
            print("\n  For CUDA Toolkit:")
            print("  - Download from: https://developer.nvidia.com/cuda-downloads")
            print("  - Install CUDA 12.6 or 12.8")
            print("  - Set CUDA_HOME environment variable")
            print("  - Add CUDA bin directory to PATH")
            print("  - Verify: nvcc --version")

        if "C++17 compatible compiler" in missing_items:
            print("\n  For C++17 Compiler:")
            is_windows = platform.system() == "Windows"
            if is_windows:
                print(
                    "  - Install Visual Studio 2019+ with 'Desktop development with C++' workload"
                )
                print("  - Or install Build Tools for Visual Studio")
            else:
                print("  - Linux: sudo apt-get install build-essential (GCC 7+)")
                print("  - macOS: xcode-select --install (Clang 5+)")

        print("\n" + "=" * 70)

        # Prompt user
        proceed = prompt_user(
            "\nDo you want to proceed with installation anyway? "
            "(This will likely fail or result in 'NameError: name '_C' is not defined' errors)",
            default="yes",
        )

        if not proceed:
            print("\n" + "=" * 70)
            print("Installation cancelled.")
            print("Please install the missing prerequisites and try again.")
            print("=" * 70 + "\n")
            sys.exit(1)
        else:
            print("\n⚠️  Proceeding with installation despite missing prerequisites...")
            print("   Installation may fail or the package may not work correctly.\n")
    else:
        print("\n[OK] All prerequisites satisfied. Proceeding with installation...\n")


# Global flag to ensure prerequisites are only checked once
_PREREQUISITES_CHECKED = False


def get_extensions():
    # Check prerequisites on first call (handles cases where setup() is called directly)
    global _PREREQUISITES_CHECKED
    if not _PREREQUISITES_CHECKED:
        check_prerequisites()
        _PREREQUISITES_CHECKED = True

    # If torch is not available, skip building extensions
    # Extensions will be built when torch is installed
    if not TORCH_AVAILABLE:
        print("\n" + "=" * 70)
        print("PyTorch not found. Skipping C++ extension compilation.")
        print("Extensions will be compiled on first import if CUDA is available.")
        print("=" * 70 + "\n")
        return []

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "groundeddino_vl", "ops", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources

    extension = CppExtension

    # C++17 compiler flags for better compatibility and modern features
    # Windows (MSVC) uses different flag syntax than Unix-like systems
    is_windows = platform.system() == "Windows"
    if is_windows:
        # MSVC: /std:c++17 (C++17 standard)
        # Note: PyTorch's BuildExtension typically handles MSVC flags automatically,
        # but we explicitly set it for clarity and to ensure C++17 is used
        cxx_flags = ["/std:c++17"]
    else:
        # Unix-like (Linux, macOS): -std=c++17
        cxx_flags = ["-std=c++17"]

    extra_compile_args = {"cxx": cxx_flags}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        print("\n" + "=" * 70)
        print("Compiling with CUDA support")
        print(f"CUDA_HOME: {CUDA_HOME}")
        print("Using C++17 standard (required for CUDA extensions)")
        print(f"Platform: {platform.system()}")
        print(f"C++ compiler flags: {cxx_flags}")
        print("=" * 70 + "\n")
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]

        # CUDA compiler (nvcc) flags - C++17 is required for CUDA compilation
        # Note: nvcc uses -std=c++17 regardless of platform
        nvcc_flags = [
            "-std=c++17",  # C++17 support for CUDA compiler (required)
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # Add host compiler C++17 flag for nvcc to pass to host compiler
        # This ensures the host compiler (gcc/clang/msvc) also uses C++17
        if not is_windows:
            # For GCC/Clang on Linux/macOS, explicitly pass C++17 to host compiler
            # On Windows with MSVC, PyTorch's BuildExtension handles this automatically
            # through the cxx flags we set above
            nvcc_flags.append("-Xcompiler=-std=c++17")

        extra_compile_args["nvcc"] = nvcc_flags
    else:
        print("\n" + "=" * 70)
        print("CUDA not available - building in CPU-only mode")
        if not torch.cuda.is_available():
            print("Reason: PyTorch CUDA not detected")
        if CUDA_HOME is None:
            print("Reason: CUDA_HOME not set")
        print("\nTo enable CUDA support, ensure:")
        print("  - CUDA Toolkit 12.6 or 12.8 is installed")
        print("  - PyTorch with CUDA support is installed")
        print("  - C++17 compatible compiler is available (REQUIRED):")
        print("    * Windows: Visual Studio 2019+ with C++ build tools")
        print("      (MSVC 19.20+ with /std:c++17 support)")
        print("    * Linux: GCC 7+ or Clang 5+ (with -std=c++17 support)")
        print("    * macOS: Xcode Command Line Tools (Clang 5+ with -std=c++17)")
        print("\n  NOTE: C++17 is mandatory for CUDA extension compilation.")
        print("  Without C++17, the _C module will fail to compile, resulting in")
        print("  'NameError: name '_C' is not defined' errors at runtime.")
        print("=" * 70 + "\n")
        define_macros += [("WITH_HIP", None)]
        extra_compile_args["nvcc"] = []
        return []

    sources = [os.path.relpath(s, cwd) for s in sources]
    include_dirs = [os.path.relpath(extensions_dir, cwd)]

    ext_modules = [
        extension(
            "groundeddino_vl._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


def parse_requirements(fname="requirements.txt", with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists

    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            elif "@git+" in line:
                info["package"] = line
            else:
                # Remove versioning from the package
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest  # NOQA
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == "__main__":
    print(f"Building wheel for {package_name}")

    # Check prerequisites before proceeding (if not already checked)
    if not _PREREQUISITES_CHECKED:
        check_prerequisites()
        _PREREQUISITES_CHECKED = True

    # Note: Most metadata is now in pyproject.toml
    # This setup.py only handles the C++ extension compilation
    ext_modules = get_extensions()

    # Only use BuildExtension if torch is available and we have extensions
    if TORCH_AVAILABLE and ext_modules:
        setup(
            ext_modules=ext_modules,
            cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
            version=None,
        )
    else:
        setup(version=None)
