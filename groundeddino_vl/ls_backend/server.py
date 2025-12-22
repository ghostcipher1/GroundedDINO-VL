"""FastAPI server for GroundedDINO-VL Label Studio ML Backend.

Endpoints:
  - GET /health: basic health with model-loaded flag
  - POST /setup: Label Studio ML backend setup endpoint
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
from typing import Any, Dict, List, Union

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import inference_engine, model_loader
from .config import DEFAULT_SETTINGS
from .utils import (
    _extract_prompt,
    _maybe_extract_image_ref,
    _normalize_image_url,
)


async def _read_bytes_from_url(url: str, timeout: int = 20) -> bytes:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.content


async def _to_image_bytes(ref: Union[str, bytes]) -> bytes:
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

    # Handle local file server URLs (convert to direct file access)
    if "/data/local-files/?d=" in s:
        # Extract the file path from the URL parameter
        import urllib.parse

        parsed = urllib.parse.urlparse(s)
        query_params = urllib.parse.parse_qs(parsed.query)
        if "d" in query_params:
            file_path_param = query_params["d"][0]
            # Handle both /data/datasets/... and datasets/... formats
            if file_path_param.startswith("/data/"):
                file_path = file_path_param
            else:
                file_path = os.path.join("/data", file_path_param)

            # Security check: ensure the resolved path is within allowed directories
            resolved_path = os.path.abspath(file_path)
            allowed_dirs = ["/data/datasets", "/data/groundeddino-vl"]
            if not any(
                resolved_path.startswith(os.path.abspath(allowed_dir))
                for allowed_dir in allowed_dirs
            ):
                raise ValueError(f"Access denied: {resolved_path}")

            if os.path.isfile(resolved_path):
                with open(resolved_path, "rb") as f:
                    return f.read()

    # Otherwise treat as URL
    if s.startswith("http://") or s.startswith("https://"):
        return await _read_bytes_from_url(s)

    # Treat as filesystem path as a last resort
    if os.path.isfile(s):
        # Security check: ensure the resolved path is within allowed directories
        resolved_path = os.path.abspath(s)
        allowed_dirs = ["/data/datasets", "/data/groundeddino-vl"]
        if not any(
            resolved_path.startswith(os.path.abspath(allowed_dir)) for allowed_dir in allowed_dirs
        ):
            raise ValueError(f"Access denied: {resolved_path}")

        with open(s, "rb") as f:
            return f.read()

    raise ValueError("Unrecognized image reference; expected URL, base64, or bytes")


def create_app() -> Any:  # noqa: C901
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
            print(
                f"[ls_backend] Loading model at startup: config={config_path}, "
                f"checkpoint={checkpoint_path}"
            )
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
        # Check if model is actually loaded (both config_path and
        # checkpoint_path are set from load_model call)
        model_loaded = bool(info.get("config_path") and info.get("checkpoint_path"))
        return {"status": "ok", "model_loaded": model_loaded}

    @app.post("/setup")
    def setup(request: Dict[str, Any] = None) -> Dict[str, Any]:
        """Label Studio ML backend setup endpoint.

        Called when connecting the ML backend to Label Studio.
        Returns the labeling configuration schema.

        Args:
            request: Optional Label Studio project configuration
        """
        # Return success response with model info
        return {
            "model_version": model_loader.get_model_info().get("version", "1.0.0"),
            "model_name": "GroundedDINO-VL",
            "model_description": "Zero-shot object detection with vision-language model",
        }

    @app.get("/model-info")
    def model_info() -> Dict[str, Any]:
        return model_loader.get_model_info()

    @app.get("/data/local-files/")
    def serve_local_file(d: str) -> FileResponse:
        """Serve local files for Label Studio image display.

        Args:
            d: Relative path to the file (e.g., 'datasets/automobile/automobile_001.jpg')
        """
        # Support both /data/datasets and project-relative paths
        # First try /data/datasets (absolute path)
        file_path = os.path.join("/data", d)

        # If not found, try relative to project root
        if not os.path.isfile(file_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            file_path = os.path.join(base_dir, d)

        # Resolve to absolute path
        resolved_path = os.path.abspath(file_path)

        # Security check: ensure the resolved path is within allowed directories
        allowed_dirs = ["/data/datasets", "/data/groundeddino-vl"]
        if not any(
            resolved_path.startswith(os.path.abspath(allowed_dir)) for allowed_dir in allowed_dirs
        ):
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if file exists
        if not os.path.isfile(resolved_path):
            raise HTTPException(status_code=404, detail=f"File not found: {d}")

        return FileResponse(resolved_path)

    @app.post("/predict")
    async def predict(task: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        tasks: List[Dict[str, Any]]
        is_single_task = False

        if isinstance(task, list):
            tasks = task
        elif isinstance(task, dict):
            # Check if Label Studio wrapped tasks in a 'tasks' key
            if "tasks" in task:
                tasks = task["tasks"]
            else:
                # Single task sent directly
                tasks = [task]
                is_single_task = True
        else:
            raise HTTPException(
                status_code=400, detail="Invalid JSON body; expected object or list"
            )

        # Log batch size for debugging
        print(f"[predict] Processing {len(tasks)} task(s), is_single_task={is_single_task}")

        # For batch requests, return all predictions to match task count
        # Note: Label Studio has a 100s timeout by default
        # With optimized model loading, each image takes ~1-2 seconds
        # So we can safely process up to ~50-80 images in a batch
        # For larger batches, you need to either:
        # 1. Select smaller chunks (recommended: 50-100 tasks at a time)
        # 2. Increase Label Studio's ML_TIMEOUT_PREDICT env var
        if not is_single_task and len(tasks) > 1000:
            print(
                f"[WARNING] Large batch of {len(tasks)} tasks may timeout. "
                f"Consider processing in smaller chunks."
            )

        predictions: List[Any] = []
        for idx, t in enumerate(tasks):
            # Log task ID for batch debugging
            task_id = t.get("id", "unknown")
            if idx < 3 or idx % 10 == 0:  # Log first 3 and every 10th task
                print(f"[predict] Processing task {idx+1}/{len(tasks)}, id={task_id}")

            # Normalize image URL in task data before processing
            img_ref = _maybe_extract_image_ref(t)
            if img_ref is None:
                print(f"[ERROR] No image reference found in task: {t}")
                raise HTTPException(status_code=400, detail="No image reference found in task")

            # Normalize the image URL if it's a string (for Label Studio compatibility)
            if isinstance(img_ref, str):
                normalized_url = _normalize_image_url(img_ref)
                # Update the task data with normalized URL
                if "data" in t and "image" in t["data"]:
                    t["data"]["image"] = normalized_url

            try:
                image_bytes = await _to_image_bytes(img_ref)
            except Exception as e:
                print(f"[ERROR] Failed to obtain image bytes from {img_ref}: {e}")
                import traceback

                traceback.print_exc()
                raise HTTPException(status_code=400, detail=f"Failed to obtain image bytes: {e}")

            prompt = _extract_prompt(t)
            try:
                result = inference_engine.run_inference(image_bytes=image_bytes, prompt_text=prompt)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

            # Append LS-formatted prediction - extract 'result' field from labelstudio dict
            ls_output = result["labelstudio"]

            # For batch requests, include task ID in prediction
            if not is_single_task and "id" in t:
                if isinstance(ls_output, dict):
                    ls_output["task"] = t["id"]

            if isinstance(ls_output, dict) and "result" in ls_output:
                predictions.append(ls_output)
            else:
                predictions.append(ls_output)

        # Return format depends on whether it was a single task or multiple
        # Single task: {"result": [...], "score": 0.x, "model_version": "..."}
        # Multiple tasks: {"results": [{...}, {...}]}
        if is_single_task and len(predictions) == 1:
            return predictions[0]
        else:
            return {"results": predictions}

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
