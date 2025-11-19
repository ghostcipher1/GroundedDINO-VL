# Label Studio ML Backend - Installation & Setup

This guide walks you through installing and running the GroundedDINO-VL Label Studio ML Backend. We've made it **dead simple**—just follow these steps.

## Quick Start (5 minutes)

### 1. Install the Package
```bash
pip install groundeddino-vl
```

### 2. Download Model Weights (First Time Only)
```bash
# Auto-downloads 662MB to ~/.cache/groundeddino-vl/
# Shows progress bar, validates integrity
python -m groundeddino_vl download-weights
```

### 3. Start the Server
```bash
groundeddino-vl-server --host 0.0.0.0 --port 9090
```

✓ **Done!** Server is running at `http://0.0.0.0:9090`

---

## What Happens

| Step | What's Happening |
|------|------------------|
| Install | Package installed to your Python environment |
| Download | Model weights cached locally (~662MB, one-time) |
| Start | Server loads model and listens for requests |

**First run takes ~10 seconds** (includes download). **Subsequent runs take ~1 second** (from cache).

---

## Table of Contents

- [Requirements](#requirements)
- [Installation Steps](#installation-steps)
- [Start the Server](#start-the-server)
- [Connect to Label Studio](#connect-to-label-studio)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Requirements

- **Python**: 3.9+ (tested on 3.9, 3.10, 3.11, 3.12)
- **System**: Linux, macOS, or Windows
- **Disk**: ~1GB free (for weights + dependencies)
- **Internet**: Required for first-run weight download (after that, offline mode OK)
- **GPU** (optional): NVIDIA GPU with CUDA 12.6+ for faster inference

### Check Python Version
```bash
python --version  # Should be 3.9+
```

---

## Installation Steps

### Step 1: Create Virtual Environment (Recommended)

**Using venv (built-in):**
```bash
python -m venv ~/groundeddino-env
source ~/groundeddino-env/bin/activate  # On Windows: ~\groundeddino-env\Scripts\activate
```

**Using conda:**
```bash
conda create -n groundeddino python=3.11
conda activate groundeddino
```

### Step 2: Install Package

```bash
pip install groundeddino-vl
```

This installs:
- GroundedDINO-VL model code
- PyTorch 2.7+ with CUDA 12.8 support
- All dependencies (transformers, FastAPI, uvicorn, etc.)

### Step 3: Download Model Weights (One-Time)

First time only, download the model weights:

```bash
python -m groundeddino_vl download-weights
```

**What happens:**
- Downloads 662MB checkpoint from HuggingFace
- Shows progress bar: `groundingdino_swint_ogc.pth: 100%|██████████| 662M/662M`
- Validates SHA256 checksum
- Caches at `~/.cache/groundeddino-vl/`

**Verify download:**
```bash
ls ~/.cache/groundeddino-vl/
# Should show: groundingdino_swint_ogc.pth (662M)
```

---

## Start the Server

### Option 1: Simple Start (Default Settings)
```bash
groundeddino-vl-server
```

Output:
```
[ls_backend] Loading model at startup...
[weights_manager] Using cached: ~/.cache/groundeddino-vl/groundingdino_swint_ogc.pth
[ls_backend] Model loaded successfully on device: cuda
INFO:     Uvicorn running on http://0.0.0.0:9090 (Press CTRL+C to quit)
```

Server running at: **`http://0.0.0.0:9090`**

### Option 2: Custom Host & Port
```bash
groundeddino-vl-server --host 127.0.0.1 --port 8000
```

### Option 3: Custom Model Paths
```bash
groundeddino-vl-server \
  --config /path/to/config.py \
  --checkpoint /path/to/weights.pth
```

### Option 4: Use Environment Variables
```bash
export GDVL_CONFIG=/path/to/config.py
export GDVL_CHECKPOINT=/path/to/weights.pth
groundeddino-vl-server --host 0.0.0.0 --port 9090
```

### Option 5: Python Script
```python
# Start server programmatically
import subprocess
subprocess.run(["groundeddino-vl-server", "--port", "9090"])
```

---

## Connect to Label Studio

### What You Need
- Label Studio running (on same machine or accessible network)
- Server URL: `http://localhost:9090` (or your server address)
- Task with images

### Steps

**1. Open Label Studio**
```bash
label-studio start
```
(or navigate to http://localhost:8080)

**2. Create/Open Project**
- Go to your project with object detection task
- Make sure images are loaded in tasks

**3. Add ML Backend**
- Click ⚙️ **Settings** (top right)
- Go to **ML-Assist** tab
- Click **Add Model**
- Enter URL: `http://localhost:9090`
- Click **Validate and Save**

**4. Use Model**
- In labeling view, click **ML-Assist** button
- Select detection model and click **Submit**
- Model detects objects, Label Studio shows predictions
- Adjust/save predictions as ground truth

---

## Environment Variables

Customize behavior via environment variables:

```bash
# Use custom cache directory
export GDVL_CACHE_DIR=/data/ml-models

# Disable auto-download (use explicit paths only)
export GDVL_AUTO_DOWNLOAD=0

# Skip checksum validation (NOT recommended)
export GDVL_SKIP_CHECKSUM=1

# Then start server
groundeddino-vl-server
```

---

## Troubleshooting

### "Connection refused" or "Cannot connect to server"

**Problem:** Label Studio can't reach the backend.

**Solution:**
- Check server is running: `ps aux | grep groundeddino-vl-server`
- Check port is correct in Label Studio settings (default: 9090)
- If running on different machine, use actual IP: `http://192.168.1.100:9090`
- Check firewall allows port 9090

### Model loading fails

**Problem:** `FileNotFoundError: Config file not found`

**Solution:**
```bash
# Download weights if not already done
python -m groundeddino_vl download-weights

# Or provide explicit paths
groundeddino-vl-server \
  --config ~/.cache/groundeddino-vl/GroundingDINO_SwinT_OGC.py \
  --checkpoint ~/.cache/groundeddino-vl/groundingdino_swint_ogc.pth
```

### "Checksum validation failed"

**Problem:** Downloaded weights are corrupted.

**Solution:**
```bash
# Delete corrupted file and re-download
rm ~/.cache/groundeddino-vl/groundingdino_swint_ogc.pth
python -m groundeddino_vl download-weights
```

### CUDA out of memory

**Problem:** "CUDA out of memory" error during inference.

**Solution:**
- Run on CPU (slower but works): No CUDA setup needed
- Use smaller batch size in Label Studio
- Reduce image resolution

### Slow inference

**Problem:** Predictions are slow.

**Solution:**
- Check device: Model should use CUDA (GPU) not CPU
- First inference includes model load (slow), subsequent inferences are fast
- If on CPU, consider using GPU machine

### Port already in use

**Problem:** `Address already in use: ('0.0.0.0', 9090)`

**Solution:**
```bash
# Use different port
groundeddino-vl-server --port 9091

# Or kill existing process
pkill -f groundeddino-vl-server
```

---

## Next Steps

- See [using_with_labelstudio.md](using_with_labelstudio.md) for advanced Label Studio integration
- See [overview.md](overview.md) for backend architecture
- See [../../README.md](../../README.md) for general project info
