"""API routes for JSONL conversation log ingestion."""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from jsonl_ingester import (
    get_ingester_status,
    get_jsonl_settings,
    run_jsonl_ingestion,
    stop_jsonl_ingestion,
    update_jsonl_settings,
)
from scheduler import (
    SCHEDULE_PRESETS,
    get_schedule_settings,
    update_schedule_settings,
)
from models_registry import get_all_models, is_free_backend

logger = logging.getLogger("anamnesis.routes.jsonl")

router = APIRouter(prefix="/api/jsonl", tags=["jsonl"])


class JsonlIngestionRequest(BaseModel):
    """Optional parameters for triggering JSONL ingestion."""
    max_exchanges: Optional[int] = None
    source_roots: Optional[dict[str, str]] = None


class JsonlSettingsUpdate(BaseModel):
    """Updatable JSONL ingester settings."""
    summarization_backend: Optional[str] = None   # "ollama" | "claude"
    ollama_model: Optional[str] = None
    max_exchanges_per_run: Optional[int] = None    # 0 = unlimited
    cpu_core_pct: Optional[int] = None             # 1-100


class ScheduleUpdate(BaseModel):
    """Update schedule for crawler and/or JSONL."""
    crawler_schedule: Optional[str] = None
    jsonl_schedule: Optional[str] = None


@router.get("/status")
async def jsonl_status():
    """Return current JSONL ingester state."""
    return get_ingester_status()


@router.post("/ingest")
async def trigger_jsonl_ingestion(request: Optional[JsonlIngestionRequest] = None):
    """Trigger a JSONL ingestion run in the background.

    Returns immediately. Poll /api/jsonl/status for progress.
    """
    status = get_ingester_status()
    if status["running"]:
        return {"status": "already_running", "message": "JSONL ingestion already in progress."}

    logger.info("JSONL ingestion triggered via API")

    kwargs = {}
    if request:
        if request.max_exchanges is not None:
            kwargs["max_exchanges"] = request.max_exchanges
        if request.source_roots is not None:
            kwargs["source_roots"] = request.source_roots

    asyncio.create_task(run_jsonl_ingestion(**kwargs))
    return {"status": "started", "message": "JSONL ingestion started. Poll /api/jsonl/status for progress."}


@router.post("/stop")
async def stop_ingestion():
    """Request the current JSONL ingestion run to stop after its current exchange."""
    return stop_jsonl_ingestion()


@router.get("/settings")
async def get_settings():
    """Return current JSONL ingester settings + schedule."""
    settings = await get_jsonl_settings()
    schedule = await get_schedule_settings()
    settings["schedule"] = schedule.get("jsonl_schedule", "nightly")
    settings["crawler_schedule"] = schedule.get("crawler_schedule", "every_30m")
    settings["schedule_presets"] = list(SCHEDULE_PRESETS.keys())
    return settings


@router.put("/settings")
async def put_settings(update: JsonlSettingsUpdate):
    """Update JSONL ingester settings. Only provided fields are changed."""
    updates = {k: v for k, v in update.model_dump().items() if v is not None}

    if "cpu_core_pct" in updates:
        updates["cpu_core_pct"] = max(1, min(100, updates["cpu_core_pct"]))
    if "max_exchanges_per_run" in updates:
        updates["max_exchanges_per_run"] = max(0, updates["max_exchanges_per_run"])
    if "summarization_backend" in updates:
        if updates["summarization_backend"] not in ("ollama", "claude"):
            return {"error": "summarization_backend must be 'ollama' or 'claude'"}

    settings = await update_jsonl_settings(updates)

    # Auto-disable JSONL schedule when switching to paid backend
    if "summarization_backend" in updates:
        is_free = await is_free_backend(
            updates["summarization_backend"],
            updates.get("ollama_model", ""),
        )
        if not is_free:
            schedule = await get_schedule_settings()
            if schedule.get("jsonl_schedule", "disabled") != "disabled":
                await update_schedule_settings({"jsonl_schedule": "disabled"})
                settings["_schedule_disabled"] = True
                settings["_schedule_warning"] = (
                    "Schedule auto-disabled: paid backend selected. "
                    "JSONL ingestion will only run when manually triggered."
                )

    return settings


@router.put("/schedule")
async def put_schedule(update: ScheduleUpdate):
    """Update schedule for crawler and/or JSONL ingester."""
    updates = {k: v for k, v in update.model_dump().items() if v is not None}

    for k, v in updates.items():
        if v not in SCHEDULE_PRESETS:
            return {"error": f"Invalid schedule preset: {v}. Valid: {list(SCHEDULE_PRESETS.keys())}"}

    settings = await update_schedule_settings(updates)
    return settings


@router.get("/schedule")
async def get_schedule():
    """Return current schedule settings + available presets."""
    settings = await get_schedule_settings()
    return {
        **settings,
        "presets": SCHEDULE_PRESETS,
    }


@router.get("/models")
async def list_models():
    """Return all known models with metadata (free/paid, size, etc)."""
    models = await get_all_models()
    return {"models": models}
