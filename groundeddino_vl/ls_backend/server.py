"""FastAPI server for GroundedDINO-VL Label Studio ML Backend.

Endpoints:
  - GET /health: basic health with model-loaded flag
  - POST /predict: accept Label Studio task JSON (image URL or base64),
                   run inference, return LS-formatted predictions
  - GET /model-info: return model metadata

Notes:
  - CORS is enabled for all origins.
  - Default port is 9090 (configurable via DEFAULT_SETTINGS.server_port or PORT env).
"""

from __future__ import annotations

import argparse
import base64
import os
from typing import Any, Dict, List, Union, cast
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import inference_engine, model_loader
from .config import DEFAULT_SETTINGS


def _read_bytes_from_url(url: str, timeout: int = 20) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as resp:
        return bytes(resp.read())


def _maybe_extract_image_ref(task: Dict[str, Any]) -> Union[str, bytes, None]:
    """Extract an image reference from a Label Studio-like task.

    Returns one of:
      - bytes (if already provided)
      - str URL or data URI
      - None if not found
    """
    # Common fields: direct at root
    for key in ("image_bytes", "image", "image_url", "imageUrl", "img", "url"):
        if key in task:
            val = task[key]
            if isinstance(val, (str, bytes)):
                return val

    # Under "data" (Label Studio style)
    data = task.get("data")
    if isinstance(data, dict):
        for key in ("image", "image_url", "imageUrl", "img", "url", "image_bytes"):
            if key in data:
                val = data[key]
                if isinstance(val, (str, bytes)):
                    return val

    return None


def _extract_prompt(task: Dict[str, Any]) -> Union[str, List[str]]:
    """Extract labeling instructions/caption/classes from task.

    Heuristics looking at several common fields. Falls back to a generic caption.
    """
    # Prefer explicit prompt at root
    for key in ("prompt", "caption", "instruction", "instructions"):
        if key in task:
            val = task[key]
            if val and isinstance(val, (str, list)):
                return cast(Union[str, List[str]], val)

    data = task.get("data")
    if isinstance(data, dict):
        for key in ("prompt", "caption", "text", "instruction", "classes"):
            if key in data:
                val = data[key]
                if val and isinstance(val, (str, list)):
                    return cast(Union[str, List[str]], val)

    # Label list
    for key in ("labels", "classes"):
        if key in task:
            val = task[key]
            if val and isinstance(val, (str, list)):
                return cast(Union[str, List[str]], val)

    return "a photo"


def _to_image_bytes(ref: Union[str, bytes]) -> bytes:
    if isinstance(ref, (bytes, bytearray)):
        return bytes(ref)
    if not isinstance(ref, str):
        raise ValueError("Unsupported image reference type")

    s = ref.strip()
    # Data URI: data:image/...;base64,XXXXX
    if s.startswith("data:image/") and ";base64," in s:
        b64 = s.split(",", 1)[1]
        return base64.b64decode(b64)

    # Plain base64 without data URI (heuristic: long and only base64 chars)
    if len(s) > 256 and all(c.isalnum() or c in "+/=_-" for c in s[:300]):
        try:
            return base64.b64decode(s)
        except Exception:
            pass

    # Otherwise treat as URL
    if s.startswith("http://") or s.startswith("https://"):
        return _read_bytes_from_url(s)

    # Treat as filesystem path as a last resort
    if os.path.isfile(s):
        with open(s, "rb") as f:
            return f.read()

    raise ValueError("Unrecognized image reference; expected URL, base64, or bytes")


def create_app() -> Any:
    """Create and return the FastAPI web application instance."""
    app = FastAPI(
        title="GroundedDINO-VL LS Backend",
        version=str(model_loader.get_model_info().get("version", "unknown")),
    )

    # CORS: allow all
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Optional database initialization (no side effects unless explicitly enabled)
    try:
        use_pg = os.environ.get("USE_POSTGRESQL", "").lower() == "true"
        sqlite_path = os.environ.get("SQLITE_PATH")
        if use_pg or sqlite_path:
            # Lazy import so environments without SQLAlchemy can still run server/tests
            from . import database  # type: ignore

            # Will select the proper URL based on env vars
            database.init_db(echo=False)
    except Exception as e:
        # Do not prevent server from running if DB init fails; keep it optional.
        # Basic visibility only; avoid introducing logging dependencies here.
        print(f"[ls_backend] Optional DB initialization skipped/failed: {e}")

    # Load model at startup if paths are provided
    try:
        config_path = os.environ.get("GDVL_CONFIG")
        checkpoint_path = os.environ.get("GDVL_CHECKPOINT")

        if config_path and checkpoint_path:
            print(f"[ls_backend] Loading model at startup: config={config_path}, checkpoint={checkpoint_path}")
            model_loader.load_model(
                model_config_path=config_path,
                model_checkpoint_path=checkpoint_path,
                device=None,  # Auto-detect CUDA
            )
            print("[ls_backend] Model loaded successfully at startup")
    except Exception as e:
        print(f"[ls_backend] Model loading failed at startup: {e}")
        raise

    @app.get("/health")
    def health() -> Dict[str, Any]:
        info = model_loader.get_model_info()
        # Check if model is actually loaded (both config_path and checkpoint_path are set from load_model call)
        model_loaded = bool(info.get("config_path") and info.get("checkpoint_path"))
        return {"status": "ok", "model_loaded": model_loaded}

    @app.get("/model-info")
    def model_info() -> Dict[str, Any]:
        return model_loader.get_model_info()

    @app.post("/predict")
    def predict(task: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Any]:
        tasks: List[Dict[str, Any]]
        if isinstance(task, list):
            tasks = task
        elif isinstance(task, dict):
            tasks = [task]
        else:
            raise HTTPException(
                status_code=400, detail="Invalid JSON body; expected object or list"
            )

        predictions: List[Any] = []
        for t in tasks:
            img_ref = _maybe_extract_image_ref(t)
            if img_ref is None:
                raise HTTPException(status_code=400, detail="No image reference found in task")
            try:
                image_bytes = _to_image_bytes(img_ref)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to obtain image bytes: {e}")

            prompt = _extract_prompt(t)
            try:
                result = inference_engine.run_inference(image_bytes=image_bytes, prompt_text=prompt)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

            # Append LS-formatted prediction list for this task
            predictions.append(result["labelstudio"])

        return predictions if len(tasks) > 1 else predictions[0]

    return app


def main() -> None:
    """CLI entry point to start the GroundedDINO-VL LS backend server.

    Supports:
      --host
      --port
      --config       Path to model config (.py)
      --checkpoint   Path to model weights (.pth)
    """
    try:
        import uvicorn  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "uvicorn is required to run the server. Install with `pip install uvicorn`."
        ) from e

    parser = argparse.ArgumentParser(description="GroundedDINO-VL LS Backend Server")
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Host interface to bind",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", str(getattr(DEFAULT_SETTINGS, "server_port", 9090)))),
        help="TCP port to listen on",
    )

    # NEW arguments for model loading
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("GDVL_CONFIG"),
        help="Path to GroundedDINO-VL model config file (.py)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.environ.get("GDVL_CHECKPOINT"),
        help="Path to model checkpoint (.pth)",
    )
    args = parser.parse_args()

    # Make accessible globally inside create_app()
    if args.config:
        os.environ["GDVL_CONFIG"] = args.config
    if args.checkpoint:
        os.environ["GDVL_CHECKPOINT"] = args.checkpoint

    uvicorn.run(
        "groundeddino_vl.ls_backend.server:create_app",
        host=args.host,
        port=args.port,
        factory=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
