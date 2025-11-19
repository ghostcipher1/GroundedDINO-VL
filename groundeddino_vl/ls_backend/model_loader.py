"""Model loader for GroundedDINO-VL (Label Studio ML Backend).

This module provides a singleton-like loader around the high-level
``groundeddino_vl.utils.inference.Model`` API. It ensures the model is
constructed once and reused across requests. It also exposes basic model
metadata for health and debug endpoints.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    # High-level wrapper around GroundedDINO-VL
    from groundeddino_vl.utils.inference import Model as GDVLModel
except Exception as _import_err:  # pragma: no cover - log and re-raise on actual load
    GDVLModel = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ModelConfig:
    model_config_path: Optional[str]
    model_checkpoint_path: Optional[str]
    device: Optional[str]


# Module-level singleton state
_MODEL_INSTANCE: Optional[GDVLModel] = None  # type: ignore[name-defined]
_MODEL_CFG: _ModelConfig = _ModelConfig(None, None, None)


def _detect_device(preferred: Optional[str] = None) -> str:
    """Return a device string, auto-detecting CUDA when available.

    If ``preferred`` is provided, it's returned unchanged. Otherwise, choose
    ``"cuda"`` when a CUDA device is available, else fall back to ``"cpu"``.
    """
    if preferred:
        return preferred
    try:
        # Import torch lazily to avoid hard runtime dependency at import-time
        import torch  # type: ignore

        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover
        # If torch is unavailable or misconfigured, be conservative and use CPU
        pass
    return "cpu"


def _validate_paths(model_config_path: Optional[str], model_checkpoint_path: Optional[str]) -> None:
    if not model_config_path:
        raise ValueError("model_config_path must be provided")
    if not model_checkpoint_path:
        raise ValueError("model_checkpoint_path must be provided")
    if not os.path.isfile(model_config_path):
        raise FileNotFoundError(f"Config file not found: {model_config_path}")
    if not os.path.isfile(model_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_checkpoint_path}")


def load_model(
    model_config_path: str,
    model_checkpoint_path: str,
    device: Optional[str] = None,
) -> Any:
    """Load and return a singleton GroundedDINO-VL model instance.

    The model will be initialized once with the given paths and device and
    cached for subsequent calls. If called again with the same parameters, the
    existing instance is returned. If different parameters are supplied, a new
    model is created and replaces the cached one.

    Args:
        model_config_path: Path to the model config (.py/.yaml depending on setup).
        model_checkpoint_path: Path to the model checkpoint (.pth).
        device: Preferred device string (e.g., "cuda" or "cpu"). If None, CUDA is
            auto-detected and used when available, otherwise CPU is used.

    Returns:
        The high-level GroundedDINO-VL model instance.
    """
    global _MODEL_INSTANCE, _MODEL_CFG

    resolved_device = _detect_device(device)

    # Ensure we can import the high-level Model class
    if GDVLModel is None:  # type: ignore[truthy-function]
        logger.exception("Failed to import GroundedDINO-VL high-level Model class")
        raise RuntimeError("GroundedDINO-VL Model class could not be imported")

    # Validate inputs early for clearer errors
    try:
        _validate_paths(model_config_path, model_checkpoint_path)
    except Exception:
        logger.exception(
            "Invalid model paths (config=%s, checkpoint=%s)",
            model_config_path,
            model_checkpoint_path,
        )
        raise

    requested_cfg = _ModelConfig(model_config_path, model_checkpoint_path, resolved_device)

    # Return existing instance if configuration matches
    if _MODEL_INSTANCE is not None and requested_cfg == _MODEL_CFG:
        return _MODEL_INSTANCE

    # Otherwise, (re)load the model
    try:
        logger.info(
            "Loading GroundedDINO-VL model: config=%s, checkpoint=%s, device=%s",
            model_config_path,
            model_checkpoint_path,
            resolved_device,
        )
        instance = GDVLModel(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=resolved_device,
        )
        _MODEL_INSTANCE = instance
        _MODEL_CFG = requested_cfg
        logger.info("Model loaded successfully on device: %s", resolved_device)
        return instance
    except Exception as e:
        logger.exception(
            "Failed to load GroundedDINO-VL model (config=%s, checkpoint=%s, device=%s)",
            model_config_path,
            model_checkpoint_path,
            resolved_device,
        )
        raise RuntimeError(f"Model load failed: {e}") from e


def _read_version_from_pyproject() -> Optional[str]:
    """Attempt to read version from pyproject.toml in the repository root."""
    try:
        import tomllib  # Python 3.11+

        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(here, "..", ".."))
        candidate = os.path.join(project_root, "pyproject.toml")
        if os.path.isfile(candidate):
            with open(candidate, "rb") as f:
                data = tomllib.load(f)
            version = data.get("project", {}).get("version")
            return str(version) if version else None
    except Exception:  # pragma: no cover
        return None
    return None


def _get_version() -> str:
    # Prefer the packaged version constant if available
    try:
        from groundeddino_vl.version import __version__  # type: ignore

        if __version__:
            return str(__version__)
    except Exception:
        pass
    # Fallback to reading pyproject.toml
    v = _read_version_from_pyproject()
    return v or "unknown"


def get_model_info() -> Dict[str, Any]:
    """Return basic information about the loaded model and environment.

    Keys:
      - model_name: Human-readable model name.
      - version: Package version from groundeddino_vl.version or pyproject.toml.
      - device: Device currently used (or auto-detected default if not loaded).
      - config_path: Path used to load the config (if loaded).
      - checkpoint_path: Path used to load the checkpoint (if loaded).
    """
    device = (
        getattr(_MODEL_INSTANCE, "device", None)
        if _MODEL_INSTANCE is not None
        else _detect_device(_MODEL_CFG.device)
    )

    # Keep backward compatibility: expose both 'version' and the
    # requested 'model_version' key expected by clients/tests.
    _version = _get_version()
    info: Dict[str, Any] = {
        "model_name": "GroundedDINO-VL",
        "model_version": _version,
        "version": _version,
        "device": device,
        "config_path": _MODEL_CFG.model_config_path,
        "checkpoint_path": _MODEL_CFG.model_checkpoint_path,
    }
    return info
