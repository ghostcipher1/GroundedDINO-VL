"""Weight file management and auto-download functionality for GroundedDINO-VL.

This module handles:
- Detecting if model weights exist locally
- Automatically downloading missing weights from GitHub releases
- Caching weights in the models directory
- Validating downloaded files via checksums
- Providing helpful error messages for network issues
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import Request, urlopen

# GitHub release URL for the model weights
GITHUB_RELEASE_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/"
    "download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
)
DEFAULT_CONFIG_FILE = "GroundingDINO_SwinB_cfg.py"
DEFAULT_CHECKPOINT_FILE = "groundingdino_swinb_cogcoor.pth"

# Known checksums for validation (SHA256)
KNOWN_CHECKSUMS = {
    DEFAULT_CHECKPOINT_FILE: "46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7",
}


def _get_models_dir() -> Path:
    """Get the models directory for model weights.

    Returns the 'models' directory in the project root.
    Creates the directory if it doesn't exist.

    Returns:
        Path to the models directory.
    """
    # Get the project root (parent of groundeddino_vl package)
    pkg_root = Path(__file__).parent
    project_root = pkg_root.parent
    models_dir = project_root / "models"

    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


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


def _download_file(url: str, output_path: Path, timeout: int = 600) -> None:
    """Download a file from a URL with progress feedback using tqdm.

    Args:
        url: URL to download from.
        output_path: Path where to save the file.
        timeout: Timeout in seconds for the download (default 10 minutes).

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
        with urlopen(req, timeout=timeout) as response:  # nosec B310
            total_size = int(response.headers.get("Content-Length", 0))

            with (
                open(output_path, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=output_path.name,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                ) as pbar,
            ):
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


def _download_file_simple(url: str, output_path: Path, timeout: int = 600) -> None:
    """Simple download without tqdm progress bar (fallback).

    Args:
        url: URL to download from.
        output_path: Path where to save the file.
        timeout: Timeout in seconds for the download (default 10 minutes).

    Raises:
        IOError: If download fails.
    """
    print(f"[weights_manager] Downloading: {output_path.name}")
    print(f"[weights_manager] URL: {url}")

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as response:  # nosec B310
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
                        print(  # noqa: E231, E501
                            f"\r[weights_manager] Downloaded: {mb_down:.1f}MB / "
                            f"{mb_total:.1f}MB ({percent:.1f}%)",
                            end="",
                            flush=True,
                        )

            print()  # Newline after progress
            print(f"[weights_manager] Download complete: {output_path.name}")

    except Exception as e:
        # Clean up partial file
        if output_path.exists():
            output_path.unlink()
        raise IOError(f"Failed to download {url}: {e}") from e


def _ensure_checkpoint_file(models_dir: Path) -> Path:
    """Ensure the checkpoint file is available in the models directory.

    Downloads from GitHub releases if not already present.

    Args:
        models_dir: Directory where model checkpoints are stored.

    Returns:
        Path to the local checkpoint file.

    Raises:
        IOError: If download fails or checksum validation fails.
    """
    checkpoint_path = models_dir / DEFAULT_CHECKPOINT_FILE

    # If file already exists, validate it
    if checkpoint_path.exists():
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"[weights_manager] Using existing checkpoint: {checkpoint_path}")
        print(f"[weights_manager] File size: {file_size_mb:.1f}MB")  # noqa: E231

        # Optionally validate checksum if known
        if KNOWN_CHECKSUMS[DEFAULT_CHECKPOINT_FILE]:
            print("[weights_manager] Validating checksum...")
            actual_sha256 = _calculate_sha256(checkpoint_path)
            expected_sha256 = KNOWN_CHECKSUMS[DEFAULT_CHECKPOINT_FILE]
            if actual_sha256 != expected_sha256:
                print("[weights_manager] Checksum mismatch! Re-downloading...")
                checkpoint_path.unlink()
            else:
                print("[weights_manager] Checksum validated successfully")
                return checkpoint_path
        else:
            return checkpoint_path

    # Download from GitHub releases
    print("[weights_manager] Checkpoint not found, downloading from GitHub releases...")
    _download_file(GITHUB_RELEASE_URL, checkpoint_path)

    # Validate checksum after download
    if KNOWN_CHECKSUMS[DEFAULT_CHECKPOINT_FILE]:
        print("[weights_manager] Validating downloaded file...")
        actual_sha256 = _calculate_sha256(checkpoint_path)
        expected_sha256 = KNOWN_CHECKSUMS[DEFAULT_CHECKPOINT_FILE]
        if actual_sha256 != expected_sha256:
            checkpoint_path.unlink()
            raise IOError(
                f"Checksum validation failed for {DEFAULT_CHECKPOINT_FILE}. "
                f"Expected: {expected_sha256}, got: {actual_sha256}. "
                f"File has been deleted. Please try again."
            )
        print("[weights_manager] Checksum validated successfully")
    else:
        actual_sha256 = _calculate_sha256(checkpoint_path)
        print(f"[weights_manager] Checksum: {actual_sha256}")

    return checkpoint_path


def ensure_weights(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Ensure model config and checkpoint are available, auto-downloading if needed.

    If checkpoint_path is not provided or doesn't exist, automatically downloads
    the model weights from GitHub releases to the models directory.

    Args:
        config_path: Path to model config file. If None, uses packaged default.
        checkpoint_path: Path to model checkpoint. If None or doesn't exist,
            downloads automatically to models directory.

    Returns:
        Tuple of (resolved_config_path, resolved_checkpoint_path) as strings.

    Raises:
        FileNotFoundError: If config cannot be found.
        IOError: If checkpoint download fails.
    """
    models_dir = _get_models_dir()
    resolved_config = config_path
    resolved_checkpoint = checkpoint_path

    # Handle checkpoint
    if not resolved_checkpoint or not os.path.isfile(resolved_checkpoint):
        print(f"[weights_manager] Checkpoint not provided or not found: {resolved_checkpoint}")
        try:
            resolved_checkpoint = str(_ensure_checkpoint_file(models_dir))
        except Exception as e:
            raise IOError(
                f"Failed to download checkpoint from GitHub releases. "
                f"Please check your internet connection or download manually from: "
                f"{GITHUB_RELEASE_URL} "
                f"Error: {e}"
            ) from e

    # Handle config
    if not resolved_config or not os.path.isfile(resolved_config):
        print(f"[weights_manager] Config not provided or not found: {resolved_config}")
        # Try to find config in the package
        try:
            pkg_root = Path(__file__).parent
            # Try multiple possible config locations
            possible_configs = [
                pkg_root / "models" / "configs" / DEFAULT_CONFIG_FILE,
                pkg_root / "models" / "configs" / "GroundingDINO_SwinT_OGC.py",  # Fallback
                pkg_root / "config" / DEFAULT_CONFIG_FILE,
                pkg_root / DEFAULT_CONFIG_FILE,
            ]

            for package_config in possible_configs:
                if package_config.exists():
                    resolved_config = str(package_config)
                    print(f"[weights_manager] Using packaged config: {resolved_config}")
                    break
            else:
                # No config found
                raise FileNotFoundError(
                    f"Could not find config file. Checked locations: "
                    f"{[str(p) for p in possible_configs]}"
                )
        except Exception as e:
            raise FileNotFoundError(f"Failed to locate config file. Error: {e}") from e

    return resolved_config, resolved_checkpoint


def download_model_weights(
    output_dir: Optional[str] = None,
    force: bool = False,
) -> Tuple[str, str]:
    """Explicitly download and cache model weights.

    Useful for pre-downloading weights before running inference or server.

    Args:
        output_dir: Directory to save weights. If None, uses models directory.
        force: If True, re-download even if file exists.

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
        save_dir = _get_models_dir()

    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / DEFAULT_CHECKPOINT_FILE

    # Re-download if forced
    if force and checkpoint_path.exists():
        print("[weights_manager] Force flag set, removing existing file...")
        checkpoint_path.unlink()

    # Download checkpoint
    if not checkpoint_path.exists():
        print("[weights_manager] Downloading checkpoint...")
        _download_file(GITHUB_RELEASE_URL, checkpoint_path)

        # Calculate checksum
        actual_sha256 = _calculate_sha256(checkpoint_path)
        print(f"[weights_manager] Checksum: {actual_sha256}")
        print("[weights_manager] Checkpoint downloaded successfully")
    else:
        print(f"[weights_manager] Checkpoint already exists: {checkpoint_path}")

    # Find config
    pkg_root = Path(__file__).parent
    possible_configs = [
        pkg_root / "models" / "configs" / DEFAULT_CONFIG_FILE,
        pkg_root / "models" / "configs" / "GroundingDINO_SwinT_OGC.py",
        pkg_root / "config" / DEFAULT_CONFIG_FILE,
        pkg_root / DEFAULT_CONFIG_FILE,
    ]

    config_path = None
    for package_config in possible_configs:
        if package_config.exists():
            config_path = package_config
            print(f"[weights_manager] Using packaged config: {config_path}")
            break

    if config_path is None:
        print("[weights_manager] Warning: Config file not found in package")
        config_path = save_dir / DEFAULT_CONFIG_FILE

    return str(config_path), str(checkpoint_path)


def setup_weights() -> Tuple[str, str]:
    """Setup function called during package installation.

    Downloads the model weights to the models directory if not present.
    This is called automatically during setup.py installation.

    Returns:
        Tuple of (config_path, checkpoint_path).
    """
    print("\n" + "=" * 70)
    print("Setting up GroundedDINO-VL model weights...")
    print("=" * 70)

    try:
        config, checkpoint = download_model_weights()
        print("\n[weights_manager] Model weights setup complete!")
        print(f"[weights_manager] Checkpoint: {checkpoint}")
        print(f"[weights_manager] Config: {config}")
        print("=" * 70 + "\n")
        return config, checkpoint
    except Exception as e:
        print(f"\n[weights_manager] Warning: Failed to download weights during setup: {e}")
        print("[weights_manager] Weights will be downloaded on first use.")
        print("=" * 70 + "\n")
        return "", ""
