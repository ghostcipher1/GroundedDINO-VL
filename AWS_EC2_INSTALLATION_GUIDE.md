# GroundedDINO-VL Installation Guide for AWS EC2 (NVIDIA A10G GPU)

**Target Environment:**
- AWS EC2 Instance with NVIDIA A10G GPU
- Ubuntu 22.04 Deep Learning AMI
- Pre-installed CUDA environment

This guide provides step-by-step instructions for installing and testing GroundedDINO-VL with full GPU acceleration.

---

## Table of Contents

1. [Prerequisites Verification](#prerequisites-verification)
2. [Environment Setup](#environment-setup)
3. [Installation](#installation)
4. [Verification](#verification)
5. [Testing with Sample Images](#testing-with-sample-images)
6. [Performance Benchmarking](#performance-benchmarking)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites Verification

### Step 1: Verify NVIDIA GPU

First, confirm your NVIDIA A10G GPU is detected and functioning:

```bash
# Check GPU status
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx              Driver Version: 535.xx.xx      CUDA Version: 12.x     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A10G         Off  | 00000000:00:1E.0 Off |                    0 |
|  0%   30C    P0    54W / 300W |      0MiB / 23028MiB |      0%      Default |
+-----------------------------------------------------------------------------------------+
```

### Step 2: Check CUDA Installation

The Deep Learning AMI should have CUDA pre-installed. Verify:

```bash
# Check CUDA version
nvcc --version
```

**Expected output:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 12.x, Vxx.x.xxx
```

### Step 3: Check CUDA Environment Variables

Verify CUDA paths are properly set:

```bash
# Check CUDA_HOME
echo $CUDA_HOME

# Check if nvcc is in PATH
which nvcc

# Check library path
echo $LD_LIBRARY_PATH
```

If `CUDA_HOME` is not set, add to your `~/.bashrc`:

```bash
# Determine CUDA location
ls -la /usr/local/cuda*

# Set CUDA_HOME (adjust version as needed)
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Reload environment
source ~/.bashrc
```

### Step 4: Check C++ Compiler

GroundedDINO-VL requires a C++17 compatible compiler:

```bash
# Check GCC version (should be 7.0+)
gcc --version

# Check g++ version
g++ --version
```

**Expected output:**
```
gcc (Ubuntu 11.x.x-1ubuntu1~22.04) 11.x.x
```

If GCC is not installed or outdated:

```bash
sudo apt-get update
sudo apt-get install -y build-essential g++-11
```

---

## Environment Setup

### Step 5: Create Python Virtual Environment

**IMPORTANT:** Always use a virtual environment to avoid conflicts:

```bash
# Create a new virtual environment
python3 -m venv ~/groundeddino-venv

# Activate the virtual environment
source ~/groundeddino-venv/bin/activate

# Verify you're in the virtual environment
which python
# Should show: /home/ubuntu/groundeddino-venv/bin/python
```

**Add to your `~/.bashrc` for convenience:**

```bash
echo 'alias activate-gdino="source ~/groundeddino-venv/bin/activate"' >> ~/.bashrc
source ~/.bashrc
```

Now you can activate with: `activate-gdino`

### Step 6: Upgrade pip and Build Tools

```bash
# Ensure you're in the virtual environment
source ~/groundeddino-venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Verify pip version
pip --version
# Should show pip 23.x or later
```

---

## Installation

### Step 7: Install PyTorch with CUDA Support

**For CUDA 12.8:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**For CUDA 12.6 (if that's what's installed):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Alternative - Check your exact CUDA version first:**

```bash
nvcc --version | grep "release"
```

Then install the matching PyTorch version.

**Verify PyTorch installation:**

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda)"
```

**Expected output:**
```
PyTorch: 2.7.0+cu128
CUDA Available: True
CUDA Version: 12.8
```

### Step 8: Install GroundedDINO-VL

**Option A: Install from PyPI (Recommended)**

```bash
pip install groundeddino_vl
```

**Option B: Install from Source**

```bash
# Clone the repository
cd ~
git clone https://github.com/ghostcipher1/GroundedDINO-VL.git
cd GroundedDINO-VL

# Install with no build isolation (ensures CUDA extensions build correctly)
pip install -e . --no-build-isolation
```

**Note:** Using `--no-build-isolation` ensures that the CUDA extensions are built using your pre-installed PyTorch and CUDA versions, preventing ABI compatibility issues.

### Step 9: Installation Progress

During installation, you should see:

```
======================================================================
Checking prerequisites for GroundedDINO-VL installation...
======================================================================

[OK] NVIDIA CUDA Toolkit FOUND
   - nvcc available: Cuda compilation tools, release 12.8, V12.8.xxx
   - CUDA_HOME: /usr/local/cuda-12.8

[OK] C++17 Compatible Compiler FOUND
   - Compiler: GCC 11.x.x

======================================================================
Compiling with CUDA support
CUDA_HOME: /usr/local/cuda-12.8
Using C++17 standard (required for CUDA extensions)
Platform: Linux
C++ compiler flags: ['-std=c++17']
======================================================================
```

Wait for compilation to complete (this may take 5-10 minutes).

---

## Verification

### Step 10: Verify Installation

Run these verification commands:

```bash
# Test basic import
python -c "import groundeddino_vl; print('GroundedDINO-VL version:', groundeddino_vl.__version__)"

# Verify CUDA extensions
python -c "import groundeddino_vl; print('CUDA available:', groundeddino_vl.__cuda_available__)"

# Check all components
python -c "from groundeddino_vl import load_model, predict, annotate; print('✓ All imports successful')"
```

**Expected output:**
```
GroundedDINO-VL version: 1.1.0
CUDA available: True
✓ All imports successful
```

### Step 11: Comprehensive Verification Script

Create a verification script:

```bash
cat > ~/verify_installation.py << 'EOF'
#!/usr/bin/env python3
"""Comprehensive installation verification for GroundedDINO-VL"""

import sys

def main():
    print("=" * 70)
    print("GroundedDINO-VL Installation Verification")
    print("=" * 70)

    # Check Python version
    print(f"\n1. Python Version: {sys.version.split()[0]}")

    # Check PyTorch
    try:
        import torch
        print(f"\n2. PyTorch:")
        print(f"   ✓ Version: {torch.__version__}")
        print(f"   ✓ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ CUDA Version: {torch.version.cuda}")
            print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError as e:
        print(f"\n2. PyTorch: ✗ FAILED - {e}")
        return False

    # Check GroundedDINO-VL
    try:
        import groundeddino_vl
        print(f"\n3. GroundedDINO-VL:")
        print(f"   ✓ Version: {groundeddino_vl.__version__}")
        print(f"   ✓ CUDA Extensions: {groundeddino_vl.__cuda_available__}")
    except ImportError as e:
        print(f"\n3. GroundedDINO-VL: ✗ FAILED - {e}")
        return False

    # Check API imports
    print(f"\n4. API Components:")
    try:
        from groundeddino_vl import load_model, predict, annotate, load_image
        print(f"   ✓ load_model")
        print(f"   ✓ predict")
        print(f"   ✓ annotate")
        print(f"   ✓ load_image")
    except ImportError as e:
        print(f"   ✗ FAILED - {e}")
        return False

    # Check CUDA extension
    print(f"\n5. CUDA Extension (_C module):")
    try:
        from groundeddino_vl import _C
        print(f"   ✓ _C module loaded successfully")
    except ImportError as e:
        print(f"   ✗ WARNING - {e}")
        print(f"   Note: This is expected if CUDA is not available")

    print("\n" + "=" * 70)
    print("✓ Installation verified successfully!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

chmod +x ~/verify_installation.py
```

Run the verification:

```bash
python ~/verify_installation.py
```

---

## Testing with Sample Images

### Step 12: Download Sample Images and Model Weights

```bash
# Create working directory
mkdir -p ~/groundeddino-test
cd ~/groundeddino-test

# Download a sample image
wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/.asset/demo9.jpg -O test_image.jpg

# Download model config and weights (if not auto-downloaded)
mkdir -p configs weights

# Download config
wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py -O configs/GroundingDINO_SwinB_cfg.py

# Download weights (auto-downloaded on first use, but you can pre-download)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth -O weights/groundingdino_swinb_cogcoor.pth
```

### Step 13: Create Simple Test Script

```bash
cat > ~/groundeddino-test/simple_test.py << 'EOF'
#!/usr/bin/env python3
"""Simple test of GroundedDINO-VL with GPU"""

import torch
from groundeddino_vl import load_model, predict, load_image, annotate
import cv2
import time

def main():
    print("=" * 70)
    print("GroundedDINO-VL GPU Test")
    print("=" * 70)

    # Check GPU
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Configuration
    config_path = "configs/GroundingDINO_SwinB_cfg.py"
    checkpoint_path = "weights/groundingdino_swinb_cogcoor.pth"
    image_path = "test_image.jpg"
    text_prompt = "car . person . dog . cat . truck . bus . bicycle"

    print(f"\nConfig: {config_path}")
    print(f"Weights: {checkpoint_path}")
    print(f"Image: {image_path}")
    print(f"Text Prompt: {text_prompt}")

    # Load model
    print("\n[1/4] Loading model on GPU...")
    start_time = time.time()
    model = load_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device="cuda"
    )
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f} seconds")

    # Load image
    print("\n[2/4] Loading image...")
    image_np, image_tensor = load_image(image_path)
    print(f"✓ Image shape: {image_np.shape}")

    # Warm-up run (CUDA initialization)
    print("\n[3/4] Warm-up run...")
    _ = predict(
        model=model,
        image=image_tensor,
        text_prompt=text_prompt,
        box_threshold=0.35,
        text_threshold=0.25,
        device="cuda"
    )
    print("✓ Warm-up complete")

    # Actual inference
    print("\n[4/4] Running inference...")
    start_time = time.time()
    result = predict(
        model=model,
        image=image_tensor,
        text_prompt=text_prompt,
        box_threshold=0.35,
        text_threshold=0.25,
        device="cuda"
    )
    inference_time = time.time() - start_time
    print(f"✓ Inference completed in {inference_time:.3f} seconds")

    # Results
    print("\n" + "=" * 70)
    print(f"Detections: {len(result)} objects found")
    print("=" * 70)
    for i, (label, score) in enumerate(zip(result.labels, result.scores), 1):
        print(f"{i}. {label:15s} - confidence: {score:.3f}")

    # Save annotated image
    print("\nSaving annotated image...")
    annotated_image = annotate(image_np, result, show_labels=True)
    cv2.imwrite("output_annotated.jpg", annotated_image)
    print("✓ Saved to: output_annotated.jpg")

    # GPU memory usage
    print(f"\nGPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    print("\n" + "=" * 70)
    print("✓ Test completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
EOF

chmod +x ~/groundeddino-test/simple_test.py
```

### Step 14: Run the Test

```bash
cd ~/groundeddino-test
python simple_test.py
```

**Expected output:**
```
======================================================================
GroundedDINO-VL GPU Test
======================================================================

GPU: NVIDIA A10G
GPU Memory: 22.2 GB

Config: configs/GroundingDINO_SwinB_cfg.py
Weights: weights/groundingdino_swinb_cogcoor.pth
Image: test_image.jpg
Text Prompt: car . person . dog . cat . truck . bus . bicycle

[1/4] Loading model on GPU...
✓ Model loaded in 3.45 seconds

[2/4] Loading image...
✓ Image shape: (800, 1200, 3)

[3/4] Warm-up run...
✓ Warm-up complete

[4/4] Running inference...
✓ Inference completed in 0.127 seconds

======================================================================
Detections: 5 objects found
======================================================================
1. car             - confidence: 0.847
2. person          - confidence: 0.912
3. dog             - confidence: 0.756
4. car             - confidence: 0.689
5. person          - confidence: 0.834

✓ Saved to: output_annotated.jpg

GPU Memory Used: 2.34 GB
GPU Memory Cached: 2.50 GB

======================================================================
✓ Test completed successfully!
======================================================================
```

### Step 15: View Results

To view the annotated image on your local machine:

```bash
# On your EC2 instance - copy to a web-accessible location or download via SCP
# Option 1: Using SCP from your local machine
scp -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com:~/groundeddino-test/output_annotated.jpg ./

# Option 2: View using base64 encoding (if you have terminal that supports images)
cat output_annotated.jpg | base64
```

---

## Performance Benchmarking

### Step 16: Run Performance Benchmark

```bash
cat > ~/groundeddino-test/benchmark.py << 'EOF'
#!/usr/bin/env python3
"""Benchmark GroundedDINO-VL performance on NVIDIA A10G"""

import torch
import time
import numpy as np
from groundeddino_vl import load_model, predict, load_image

def benchmark(model, image_tensor, text_prompt, num_runs=50):
    """Run benchmark"""
    times = []

    # Warm-up
    for _ in range(5):
        _ = predict(model, image_tensor, text_prompt, device="cuda")

    # Benchmark
    torch.cuda.synchronize()
    for i in range(num_runs):
        start = time.time()
        _ = predict(model, image_tensor, text_prompt, device="cuda")
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Run {i+1}/{num_runs}: {elapsed*1000:.2f} ms", end='\r')

    print()  # New line
    return times

def main():
    print("=" * 70)
    print("GroundedDINO-VL Performance Benchmark (NVIDIA A10G)")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = load_model(
        config_path="configs/GroundingDINO_SwinB_cfg.py",
        checkpoint_path="weights/groundingdino_swinb_cogcoor.pth",
        device="cuda"
    )

    # Load image
    print("Loading image...")
    _, image_tensor = load_image("test_image.jpg")

    # Benchmark
    print("\nRunning benchmark (50 iterations)...")
    times = benchmark(model, image_tensor, "car . person . dog", num_runs=50)

    # Statistics
    times_ms = np.array(times) * 1000
    print("\n" + "=" * 70)
    print("Benchmark Results:")
    print("=" * 70)
    print(f"Mean:   {times_ms.mean():.2f} ms")
    print(f"Median: {np.median(times_ms):.2f} ms")
    print(f"Min:    {times_ms.min():.2f} ms")
    print(f"Max:    {times_ms.max():.2f} ms")
    print(f"Std:    {times_ms.std():.2f} ms")
    print(f"\nThroughput: {1000/times_ms.mean():.2f} FPS")
    print("=" * 70)

if __name__ == "__main__":
    main()
EOF

chmod +x ~/groundeddino-test/benchmark.py
python ~/groundeddino-test/benchmark.py
```

**Expected benchmark results on NVIDIA A10G:**
- Mean inference time: ~120-150ms per image
- Throughput: ~7-8 FPS
- GPU memory usage: ~2-3GB

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**

```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Monitor GPU memory
nvidia-smi -l 1  # Updates every second

# Reduce batch size or use smaller images in your code
```

### Issue 2: CUDA Extension Not Found

**Error:** `ImportError: cannot import name '_C'`

**Solutions:**

```bash
# Reinstall with verbose output
pip uninstall groundeddino_vl -y
pip install groundeddino_vl --no-cache-dir -v

# Or rebuild from source
cd ~/GroundedDINO-VL
pip install -e . --no-build-isolation --force-reinstall
```

### Issue 3: CUDA Version Mismatch

**Error:** `AssertionError: Torch not compiled with CUDA enabled`

**Solution:**

```bash
# Check CUDA version
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Issue 4: Slow Inference

**Possible causes:**
1. CPU inference instead of GPU
2. Not using warm-up run
3. Large images

**Solution:**

```python
# Ensure GPU is being used
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")

# Always do warm-up run first
model = load_model(..., device="cuda")
_ = predict(model, image, "warm-up", device="cuda")  # Warm-up
result = predict(model, image, "actual prompt", device="cuda")  # Fast
```

### Issue 5: Connection Issues (SSH Timeout)

**For long installations:**

```bash
# Use screen or tmux to maintain session
sudo apt-get install screen

# Start screen session
screen -S groundeddino

# Run installation
# ...

# Detach: Ctrl+A, then D
# Reattach: screen -r groundeddino
```

---

## Additional Resources

- **Full Documentation**: [github.com/ghostcipher1/GroundedDINO-VL](https://github.com/ghostcipher1/GroundedDINO-VL)
- **API Reference**: See [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **Quick Start**: See [docs/QUICKSTART.md](docs/QUICKSTART.md)
- **Batch Inference**: See [BATCH_INFERENCE_QUICKSTART.txt](BATCH_INFERENCE_QUICKSTART.txt)

---

## Summary Checklist

- [ ] Verified NVIDIA A10G GPU with `nvidia-smi`
- [ ] Confirmed CUDA installation with `nvcc --version`
- [ ] Set CUDA environment variables
- [ ] Created and activated Python virtual environment
- [ ] Installed PyTorch with CUDA support
- [ ] Installed GroundedDINO-VL
- [ ] Ran verification script successfully
- [ ] Tested with sample image
- [ ] Confirmed GPU acceleration is working
- [ ] Benchmarked performance

---

**Installation Complete!**

Your GroundedDINO-VL setup with GPU acceleration is ready to use. The NVIDIA A10G should provide excellent performance for object detection tasks.

For any issues, check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md) or open an issue on GitHub.
