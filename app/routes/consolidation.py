"""Routes for the episode consolidation subsystem.

GET  /api/consolidation/status         — recent run stats + currently-flagged counts
POST /api/consolidation/run            — manual trigger; body: {"dry_run": bool}
POST /api/consolidation/unsupersede/{episode_id} — reverse a supersession
GET  /api/consolidation/schedule       — current schedule preset
PUT  /api/consolidation/schedule       — change preset; body: {"preset": "nightly"|...}

Schedule presets live in the unified `settings._id=schedules` doc under the
key `consolidation_schedule` (see app/scheduler.py).
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from consolidation import (
    count_currently_consolidated,
    count_currently_superseded,
    get_last_run_stats,
    run_regime_1_pass,
    unsupersede,
)
from scheduler import SCHEDULE_PRESETS, get_schedule_settings, update_schedule_settings

router = APIRouter(prefix="/api/consolidation", tags=["consolidation"])


@router.get("/status")
async def consolidation_status():
    """Recent runs + currently-flagged counts. Powers the dashboard panel."""
    return {
        "currently_superseded": await count_currently_superseded(),
        "currently_consolidated": await count_currently_consolidated(),
        "recent_runs": await get_last_run_stats(limit=10),
    }


@router.post("/run")
async def consolidation_run(body: dict | None = None):
    """Manual trigger. Body: {"dry_run": bool} (default False)."""
    body = body or {}
    dry_run = bool(body.get("dry_run", False))
    stats = await run_regime_1_pass(dry_run=dry_run)
    # JSONResponse so the datetime fields serialize predictably
    serializable = {
        **stats,
        "ran_at": stats["ran_at"].isoformat() if stats.get("ran_at") else None,
        "finished_at": stats["finished_at"].isoformat() if stats.get("finished_at") else None,
    }
    return JSONResponse(serializable)


@router.post("/unsupersede/{episode_id}")
async def consolidation_unsupersede(episode_id: str):
    """Reverse a supersession — episode becomes default-searchable again."""
    ok = await unsupersede(episode_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not flagged superseded.")
    return {"ok": True, "restored": episode_id}


@router.get("/schedule")
async def consolidation_schedule_get():
    settings = await get_schedule_settings()
    return {
        "preset": settings.get("consolidation_schedule", "nightly"),
        "available_presets": list(SCHEDULE_PRESETS.keys()),
    }


@router.put("/schedule")
async def consolidation_schedule_set(body: dict):
    preset = (body or {}).get("preset")
    if preset not in SCHEDULE_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid preset; choose from {list(SCHEDULE_PRESETS.keys())}",
        )
    settings = await update_schedule_settings({"consolidation_schedule": preset})
    return {"preset": settings["consolidation_schedule"]}
