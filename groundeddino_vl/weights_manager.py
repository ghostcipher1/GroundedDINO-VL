"""Weight file management and auto-download functionality for GroundedDINO-VL.

This module handles:
- Detecting if model weights exist locally
- Automatically downloading missing weights from HuggingFace Hub
- Caching weights in a platform-appropriate cache directory
- Validating downloaded files via checksums
- Providing helpful error messages for network issues
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import Request, urlopen

# HuggingFace Hub repository containing the default model weights
HUGGINGFACE_REPO = "ShilongLiu/GroundingDINO"
DEFAULT_CONFIG_FILE = "GroundingDINO_SwinT_OGC.py"
DEFAULT_CHECKPOINT_FILE = "groundingdino_swint_ogc.pth"

# Known checksums for validation (SHA256)
KNOWN_CHECKSUMS = {
    DEFAULT_CHECKPOINT_FILE: "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
}


def _get_cache_dir() -> Path:
    """Get the cache directory for model weights.

    Uses XDG_CACHE_HOME on Linux/macOS, %LOCALAPPDATA% on Windows,
    or falls back to ~/.cache/groundeddino-vl

    Can be overridden via GDVL_CACHE_DIR environment variable.

    Returns:
        Path to the cache directory (will be created if it doesn't exist).
    """
    # Allow explicit override
    if cache_dir_env := os.environ.get("GDVL_CACHE_DIR"):
        cache_dir = Path(cache_dir_env)
    else:
        # Platform-specific defaults
        if sys.platform == "win32":
            base = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
        else:
            base = Path(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")))

        cache_dir = base / "groundeddino-vl"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _calculate_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA256 checksum of a file.

    Args:
        file_path: Path to the file to checksum.
        chunk_size: Size of chunks to read at a time.

    Returns:
        Hexadecimal SHA256 hash string.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def _download_file(url: str, output_path: Path, timeout: int = 300) -> None:
    """Download a file from a URL with progress feedback using tqdm.

    Args:
        url: URL to download from.
        output_path: Path where to save the file.
        timeout: Timeout in seconds for the download.

    Raises:
        IOError: If download fails.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback without tqdm
        _download_file_simple(url, output_path, timeout)
        return

    print(f"[weights_manager] Downloading: {output_path.name}")
    print(f"[weights_manager] URL: {url}")

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as response:
            total_size = int(response.headers.get("Content-Length", 0))

            with open(output_path, "wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=output_path.name,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

            print(f"[weights_manager] Download complete: {output_path.name}")

    except Exception as e:
        # Clean up partial file
        if output_path.exists():
            output_path.unlink()
        raise IOError(f"Failed to download {url}: {e}") from e


def _download_file_simple(url: str, output_path: Path, timeout: int = 300) -> None:
    """Simple download without tqdm progress bar (fallback).

    Args:
        url: URL to download from.
        output_path: Path where to save the file.
        timeout: Timeout in seconds for the download.

    Raises:
        IOError: If download fails.
    """
    print(f"[weights_manager] Downloading: {output_path.name}")
    print(f"[weights_manager] URL: {url}")

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Simple progress indicator
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_down = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r[weights_manager] Downloaded: {mb_down:.1f}MB / {mb_total:.1f}MB ({percent:.1f}%)", end="", flush=True)

            print()  # Newline after progress
            print(f"[weights_manager] Download complete: {output_path.name}")

    except Exception as e:
        # Clean up partial file
        if output_path.exists():
            output_path.unlink()
        raise IOError(f"Failed to download {url}: {e}") from e


def _ensure_huggingface_file(filename: str, cache_dir: Path) -> Path:
    """Ensure a file from HuggingFace Hub is available locally.

    Downloads from HuggingFace if not already cached. Validates checksums
    if known checksums are available.

    Args:
        filename: Name of the file to fetch (e.g., "groundingdino_swint_ogc.pth").
        cache_dir: Directory to cache the file in.

    Returns:
        Path to the local file.

    Raises:
        FileNotFoundError: If download fails or file validation fails.
    """
    cache_path = cache_dir / filename

    # If file already exists and checksum is known, validate it
    if cache_path.exists():
        if filename in KNOWN_CHECKSUMS:
            print(f"[weights_manager] Validating cached: {filename}")
            actual_sha256 = _calculate_sha256(cache_path)
            expected_sha256 = KNOWN_CHECKSUMS[filename]
            if actual_sha256 == expected_sha256:
                print(f"[weights_manager] Checksum valid, using cached: {cache_path}")
                return cache_path
            else:
                print("[weights_manager] Checksum mismatch, re-downloading...")
                cache_path.unlink()
        else:
            print(f"[weights_manager] Using cached: {cache_path}")
            return cache_path

    # Download from HuggingFace
    url = f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main/{filename}"
    print(f"[weights_manager] Fetching: {filename} from {HUGGINGFACE_REPO}")

    _download_file(url, cache_path)

    # Validate checksum if known
    if filename in KNOWN_CHECKSUMS:
        print("[weights_manager] Validating checksum...")
        actual_sha256 = _calculate_sha256(cache_path)
        expected_sha256 = KNOWN_CHECKSUMS[filename]
        if actual_sha256 != expected_sha256:
            cache_path.unlink()
            raise FileNotFoundError(
                f"Checksum validation failed for {filename}. "
                f"Expected: {expected_sha256}, got: {actual_sha256}. "
                f"File has been deleted. Please try again."
            )
        print("[weights_manager] Checksum validated successfully")

    return cache_path


def ensure_weights(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Ensure model config and checkpoint are available, auto-downloading if needed.

    If paths don't exist and auto-download is enabled (default), attempts to
    download from HuggingFace Hub. Falls back to default paths if not specified.

    Args:
        config_path: Path to model config file. If None or doesn't exist,
            uses default or downloads.
        checkpoint_path: Path to model checkpoint. If None or doesn't exist,
            uses default or downloads.

    Returns:
        Tuple of (resolved_config_path, resolved_checkpoint_path) as strings.

    Raises:
        FileNotFoundError: If files cannot be found or downloaded.
        IOError: If download fails.
    """
    # Check if auto-download is disabled
    if os.environ.get("GDVL_AUTO_DOWNLOAD", "").lower() == "0":
        if not config_path or not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                f"Auto-download is disabled (GDVL_AUTO_DOWNLOAD=0). "
                f"Please provide a valid config path or enable auto-download."
            )
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}. "
                f"Auto-download is disabled (GDVL_AUTO_DOWNLOAD=0). "
                f"Please provide a valid checkpoint path or enable auto-download."
            )
        return config_path, checkpoint_path

    cache_dir = _get_cache_dir()
    resolved_config = config_path
    resolved_checkpoint = checkpoint_path

    # Handle checkpoint
    if not resolved_checkpoint or not os.path.isfile(resolved_checkpoint):
        print(f"[weights_manager] Checkpoint not found: {resolved_checkpoint}")
        try:
            resolved_checkpoint = str(
                _ensure_huggingface_file(DEFAULT_CHECKPOINT_FILE, cache_dir)
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to auto-download checkpoint. "
                f"Download manually from: https://huggingface.co/{HUGGINGFACE_REPO} "
                f"Error: {e}"
            ) from e

    # Handle config
    if not resolved_config or not os.path.isfile(resolved_config):
        print(f"[weights_manager] Config not found: {resolved_config}")
        # Try to find config in the package
        try:
            pkg_root = Path(__file__).parent
            package_config = pkg_root / "models" / "configs" / DEFAULT_CONFIG_FILE
            if package_config.exists():
                resolved_config = str(package_config)
                print(f"[weights_manager] Using packaged config: {resolved_config}")
            else:
                # Try to download config (if available on HF)
                try:
                    resolved_config = str(
                        _ensure_huggingface_file(DEFAULT_CONFIG_FILE, cache_dir)
                    )
                except Exception:
                    # Config is optional if found in package
                    print(
                        f"[weights_manager] Could not find or download config. "
                        f"Falling back to package default: {package_config}"
                    )
                    if package_config.exists():
                        resolved_config = str(package_config)
                    else:
                        raise FileNotFoundError(
                            f"Could not find config file: {DEFAULT_CONFIG_FILE}. "
                            f"Checked: {package_config}"
                        )
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to locate config file. Error: {e}"
            ) from e

    return resolved_config, resolved_checkpoint


def download_model_weights(
    output_dir: Optional[str] = None,
    force: bool = False,
) -> Tuple[str, str]:
    """Explicitly download and cache model weights.

    Useful for pre-downloading weights before running inference or server.

    Args:
        output_dir: Directory to save weights. If None, uses cache directory.
        force: If True, re-download even if cached files exist.

    Returns:
        Tuple of (config_path, checkpoint_path).

    Example:
        >>> config, checkpoint = download_model_weights()
        >>> print(f"Config: {config}")
        >>> print(f"Checkpoint: {checkpoint}")
    """
    if output_dir:
        save_dir = Path(output_dir)
    else:
        save_dir = _get_cache_dir()

    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / DEFAULT_CHECKPOINT_FILE
    config_path = save_dir / DEFAULT_CONFIG_FILE

    # Re-download if forced
    if force:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if config_path.exists():
            config_path.unlink()

    # Download checkpoint
    if not checkpoint_path.exists():
        print("[weights_manager] Downloading checkpoint...")
        url = f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main/{DEFAULT_CHECKPOINT_FILE}"
        _download_file(url, checkpoint_path)

        # Validate
        if DEFAULT_CHECKPOINT_FILE in KNOWN_CHECKSUMS:
            actual_sha256 = _calculate_sha256(checkpoint_path)
            expected_sha256 = KNOWN_CHECKSUMS[DEFAULT_CHECKPOINT_FILE]
            if actual_sha256 != expected_sha256:
                checkpoint_path.unlink()
                raise FileNotFoundError(
                    f"Checksum validation failed for checkpoint. "
                    f"Expected: {expected_sha256}, got: {actual_sha256}."
                )
            print("[weights_manager] Checkpoint validated")
    else:
        print(f"[weights_manager] Checkpoint already cached: {checkpoint_path}")

    # Try to download config (optional)
    if not config_path.exists():
        print("[weights_manager] Attempting to download config...")
        try:
            url = f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main/{DEFAULT_CONFIG_FILE}"
            _download_file(url, config_path)
            print("[weights_manager] Config downloaded")
        except Exception as e:
            print(f"[weights_manager] Config download failed (this may be okay): {e}")
            # Config is optional, try package default
            pkg_config = Path(__file__).parent / "models" / "configs" / DEFAULT_CONFIG_FILE
            if pkg_config.exists():
                print(f"[weights_manager] Using packaged config: {pkg_config}")
                config_path = pkg_config
            else:
                raise
    else:
        print(f"[weights_manager] Config already cached: {config_path}")

    return str(config_path), str(checkpoint_path)
