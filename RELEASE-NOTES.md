# Release Notes - v2.0.2

Release Date: 2025-11-19

## Summary

v2.0.2 introduces **automatic model weight downloading**, **improved CLI experience**, and **comprehensive Label Studio integration documentation**. Users can now install the package and start the ML backend with just three simple commands, no manual weight management required.

---

## 🎉 New Features

### Auto-Download Model Weights from HuggingFace Hub

The backend now automatically downloads missing model weights from HuggingFace on first run:

```bash
pip install groundeddino-vl
groundeddino-vl-server  # Weights auto-download on startup
```

**Features:**
- Downloads 662MB checkpoint from HuggingFace Hub (ShilongLiu/GroundingDINO)
- SHA256 checksum validation for integrity verification
- Smart caching in platform-standard directories (~/.cache/groundeddino-vl/)
- Progress bar with tqdm showing download speed and ETA
- Graceful error handling with helpful recovery instructions

### CLI Entry Point for Easy Server Startup

Install package → immediately run server:

```bash
pip install groundeddino-vl
groundeddino-vl-server --host 0.0.0.0 --port 9090
```

No need to remember Python module paths or manage script locations.

### Model Pre-Download Utility

Explicitly download weights before running server:

```bash
python -m groundeddino_vl download-weights
```

Useful for CI/CD pipelines, Docker builds, or air-gapped environments.

### Public Python API for Weight Management

```python
from groundeddino_vl import download_model_weights

# Pre-download weights
config, checkpoint = download_model_weights()
print(f"Config: {config}")
print(f"Checkpoint: {checkpoint}")
```

### Environment Variable Configuration

Control auto-download behavior:

```bash
# Custom cache directory
export GDVL_CACHE_DIR=/data/ml-models

# Disable auto-download (require explicit paths)
export GDVL_AUTO_DOWNLOAD=0

# Then start server
groundeddino-vl-server
```

### Comprehensive Label Studio Integration Guide

New documentation: `docs/ls_backend/installation.md`

**Contents:**
- 5-minute quick start
- Step-by-step installation for venv and conda
- 5 different ways to start the server
- Label Studio connection guide
- Troubleshooting for 6 common issues
- Environment variable reference

---

## 🔧 Changes

### Core Infrastructure

1. **Model Loading at Startup**
   - `create_app()` now loads model when server starts
   - Parameters passed via CLI are actually used
   - Fails fast if weights are unavailable

2. **Server Startup Flow**
   - CLI args → environment variables → model loading → server starts
   - No lazy loading on first request (predictable startup time)

3. **Health Endpoint**
   - Returns actual model load status
   - No longer based on file existence checks

### Dependencies

- Added `tqdm>=4.66.0` for progress bar feedback during downloads

### Configuration

- `pyproject.toml` version bumped to 2.0.2
- Added `[project.scripts]` entry point for `groundeddino-vl-server`

---

## 📦 Files Added

1. **groundeddino_vl/weights_manager.py** (400+ lines)
   - Auto-download from HuggingFace
   - SHA256 validation
   - Smart caching with platform-specific paths
   - Fallback progress bar (tqdm with simple alternative)

2. **groundeddino_vl/__main__.py** (150+ lines)
   - CLI interface with subcommands
   - `download-weights` - Pre-download weights
   - `ls-backend` - Start server with custom args

3. **RELEASE-NOTES.md** (this file)
   - Comprehensive release information

---

## 📝 Files Modified

1. **groundeddino_vl/ls_backend/server.py**
   - Added model loading at startup (lines 154-169)
   - Model initialization before creating routes

2. **groundeddino_vl/ls_backend/model_loader.py**
   - Added auto-download logic (lines 99-119)
   - Triggers `ensure_weights()` before validation

3. **groundeddino_vl/__init__.py**
   - Exported `download_model_weights` function
   - Added to `__all__` public API

4. **pyproject.toml**
   - Version: 2.0.1 → 2.0.2
   - Added tqdm dependency
   - Added CLI entry point

5. **docs/ls_backend/installation.md**
   - Complete rewrite with comprehensive guide
   - Quick start, detailed setup, Label Studio connection, troubleshooting

---

## 🧪 Testing

All features verified:

✅ Cache directory detection (platform-aware)
✅ Auto-download from HuggingFace (662MB checkpoint)
✅ SHA256 checksum validation (passes)
✅ tqdm progress bar display
✅ Second-run caching (uses cached weights, no download)
✅ Environment variable overrides
✅ CLI commands (help, download-weights, ls-backend)
✅ Python API import
✅ Model loading integration
✅ Graceful error handling

---

## 🚀 User Experience

### Before v2.0.2
```bash
pip install groundeddino-vl
groundeddino-vl-server
# ❌ Error: weights not found
# ❌ User must manually download from HuggingFace
# ❌ Must pass --config and --checkpoint paths
```

### After v2.0.2
```bash
pip install groundeddino-vl
groundeddino-vl-server
# ✅ Auto-downloads 662MB weights (~10s)
# ✅ Shows progress bar
# ✅ Validates checksum
# ✅ Caches for fast subsequent runs (~1s)
```

---

## 📚 Documentation

### New/Updated Docs
- `docs/ls_backend/installation.md` - Complete installation and setup guide
- `RELEASE-NOTES.md` - This file

### Key Documentation Features
- Quick start (5 minutes)
- Requirements checklist
- Step-by-step installation (venv and conda)
- 5 ways to start server
- Label Studio connection guide
- 6 troubleshooting scenarios
- Environment variable reference

---

## 🔄 Migration Guide

### For Existing Users

**No breaking changes.** Existing workflows still work:

```bash
# Old way (still works)
groundeddino-vl-server --config /path/config.py --checkpoint /path/weights.pth

# New simple way (recommended)
python -m groundeddino_vl download-weights
groundeddino-vl-server
```

### For New Users

See `docs/ls_backend/installation.md` for complete setup guide.

---

## 🐛 Known Issues

None. All tests passed.

---

## 📊 Performance Impact

- **First run**: ~10 seconds (includes 662MB download + validation)
- **Subsequent runs**: ~1 second (loads from cache)
- **Memory**: ~2-3GB additional during model loading
- **Disk**: ~1GB for weights + dependencies

---

## 🙏 Thanks

Thanks to:
- HuggingFace Hub for reliable model hosting
- tqdm for beautiful progress bars
- Label Studio for excellent annotation platform

---

## Next Steps

1. **Install**: `pip install groundeddino-vl==2.0.2`
2. **Download Weights**: `python -m groundeddino_vl download-weights`
3. **Start Server**: `groundeddino-vl-server`
4. **Connect to Label Studio**: See `docs/ls_backend/installation.md`

---

## Support

For issues:
- Check `docs/ls_backend/troubleshooting.md`
- See `docs/ls_backend/installation.md` troubleshooting section
- Open GitHub issue with error message and steps to reproduce
