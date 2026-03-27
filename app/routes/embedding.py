"""API routes for embedding model configuration and CPU affinity."""

import asyncio
import logging
import multiprocessing
import os

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from embedding import load_embedding_model, apply_cpu_config, get_active_model_info
from models_registry import EMBEDDING_MODELS
from routes.episodes import _run_reembed, _reembed_state
from database import save_embedding_config

logger = logging.getLogger("anamnesis.routes.embedding")

router = APIRouter(prefix="/api/embedding", tags=["embedding"])

_ENV_PATH = os.path.join(os.path.dirname(__file__), "../../.env")


def _sync_env_file(model_id: str, dimensions: int) -> None:
    """Write EMBEDDING_MODEL and EMBEDDING_DIMENSIONS back to .env so the
    next deploy starts with the correct values without a manual edit."""
    env_path = os.path.abspath(_ENV_PATH)
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r") as f:
            lines = f.readlines()
        updated = []
        found_model = found_dims = False
        for line in lines:
            if line.startswith("EMBEDDING_MODEL="):
                updated.append(f"EMBEDDING_MODEL={model_id}\n")
                found_model = True
            elif line.startswith("EMBEDDING_DIMENSIONS="):
                updated.append(f"EMBEDDING_DIMENSIONS={dimensions}\n")
                found_dims = True
            else:
                updated.append(line)
        if not found_model:
            updated.append(f"EMBEDDING_MODEL={model_id}\n")
        if not found_dims:
            updated.append(f"EMBEDDING_DIMENSIONS={dimensions}\n")
        with open(env_path, "w") as f:
            f.writelines(updated)
        logger.info(f".env updated: EMBEDDING_MODEL={model_id}, EMBEDDING_DIMENSIONS={dimensions}")
    except Exception as e:
        logger.warning(f"Could not update .env: {e}")

_TOTAL_CORES = multiprocessing.cpu_count()


class EmbeddingConfigUpdate(BaseModel):
    model_id: Optional[str] = None        # change model (triggers hot-reload)
    cpu_cores: Optional[list[int]] = None  # explicit core list (strict affinity)
    cpu_pct: Optional[int] = None          # % shorthand (ignored if cpu_cores set)


@router.get("/config")
async def get_embedding_config():
    """Return current embedding model info + available models + CPU state."""
    info = get_active_model_info()
    return {
        **info,
        "total_cores": _TOTAL_CORES,
        "available_models": EMBEDDING_MODELS,
    }


@router.put("/config")
async def update_embedding_config(update: EmbeddingConfigUpdate):
    """Update embedding model and/or CPU core affinity.

    - Changing model_id triggers a hot-reload (blocks until model is loaded).
      After this, run POST /api/episodes/reembed to re-embed existing episodes.
    - Changing cpu_cores/cpu_pct rebuilds the thread pool with new affinity.
    """
    if update.model_id is not None:
        # Find dimensions for this model
        dims = None
        for m in EMBEDDING_MODELS:
            if m["model_id"] == update.model_id:
                dims = m["dimensions"]
                break

        logger.info(f"Hot-reloading embedding model: {update.model_id} ({dims} dims)")

        # Run in thread — model loading is CPU-bound and blocks the event loop
        await asyncio.to_thread(
            load_embedding_model,
            update.model_id,
            update.cpu_pct,
            update.cpu_cores,
        )

        result = get_active_model_info()

        # Persist model selection to MongoDB so it survives restarts
        await save_embedding_config(update.model_id, update.cpu_pct, update.cpu_cores)

        # Keep .env in sync so next deploy starts with the correct dims
        _sync_env_file(result["model_id"], result["dimensions"])

        # Auto-trigger reembed — model changed, existing vectors are stale
        _reembed_state["stale"] = True
        if not _reembed_state["running"]:
            logger.info("Model changed — auto-triggering reembed of all episodes")
            asyncio.create_task(_run_reembed())
            result["reembed"] = "started"
        else:
            result["reembed"] = "already_running"

        return result

    # CPU-only update — no model reload needed
    if update.cpu_cores is not None or update.cpu_pct is not None:
        apply_cpu_config(cpu_pct=update.cpu_pct, cpu_cores=update.cpu_cores)
        info = get_active_model_info()
        await save_embedding_config(info["model_id"], update.cpu_pct, update.cpu_cores)
        _sync_env_file(info["model_id"], info["dimensions"])

    return get_active_model_info()
