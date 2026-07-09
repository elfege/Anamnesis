"""HTTP surface for the RunPod lifecycle — implementation of pin #7.

  GET  /api/avatar/runpod/lifecycle/tiers           — GPU catalog (dropdown source)
  POST /api/avatar/runpod/lifecycle/start           — provision a pod
  POST /api/avatar/runpod/lifecycle/stop/{pod_id}   — soft stop (billing pause)
  POST /api/avatar/runpod/lifecycle/terminate/{pod_id} — hard destroy
  GET  /api/avatar/runpod/lifecycle/status          — live status + cost-so-far
  GET  /api/avatar/runpod/lifecycle/status/{pod_id} — single pod status
  GET  /api/avatar/runpod/lifecycle/log             — recent actions

The Start endpoint requires a `confirm_cost` field so a click-jack or
mispressed button on the UI can't accidentally spend money.
"""

from fastapi import APIRouter, HTTPException

from avatar.runpod_lifecycle import (
    COST_HR_HARD_CONFIRM_THRESHOLD,
    GPU_TIERS,
    RunPodError,
    get_pod_status,
    list_tiers,
    recent_log,
    start_pod,
    status_all,
    stop_pod,
    terminate_pod,
)

router = APIRouter(prefix="/api/avatar/runpod/lifecycle", tags=["runpod-lifecycle"])


@router.get("/tiers")
async def tiers():
    """GPU tier catalog + cost estimates for the confirmation modal."""
    return {
        "tiers": list_tiers(),
        "hard_confirm_threshold_hr": COST_HR_HARD_CONFIRM_THRESHOLD,
    }


@router.post("/start")
async def lifecycle_start(body: dict):
    """Create + register a pod.

    Body:
      {
        "gpu_tier":     "rtx3090"|"rtx4090"|"a100"|"h100"  (required),
        "image":        str  (optional, defaults to avatar-worker image),
        "port":         int  (optional, defaults to 3013),
        "label":        str  (optional, defaults to "runpod · <tier label>"),
        "confirm_cost": bool (REQUIRED — set true after user OKs cost modal)
      }

    Refuses to start unless confirm_cost is truthy. If gpu_tier is above the
    hard_confirm_threshold, also requires cost_ack_string == gpu_tier
    (mirrors deploy_runpod.sh:151-157 — type the alias to confirm).
    """
    body = body or {}
    gpu_tier = (body.get("gpu_tier") or "").strip().lower()
    if gpu_tier not in GPU_TIERS:
        raise HTTPException(status_code=400, detail=f"gpu_tier must be one of {list(GPU_TIERS)}")
    if not body.get("confirm_cost"):
        raise HTTPException(
            status_code=402,
            detail="confirm_cost=true is required — cost modal must be OK'd before spending money",
        )
    tier = GPU_TIERS[gpu_tier]
    if tier["hourly_est"] >= COST_HR_HARD_CONFIRM_THRESHOLD:
        # High-tier gate: require explicit alias typed back to us.
        if (body.get("cost_ack_string") or "").strip().lower() != gpu_tier:
            raise HTTPException(
                status_code=402,
                detail=(
                    f"{gpu_tier} exceeds ${COST_HR_HARD_CONFIRM_THRESHOLD:.2f}/hr — "
                    f"send cost_ack_string='{gpu_tier}' to confirm"
                ),
            )
    try:
        return await start_pod(
            gpu_tier=gpu_tier,
            image=body.get("image") or None,
            port=int(body.get("port") or 3013),
            label=body.get("label") or None,
        )
    except RunPodError as exc:
        raise HTTPException(status_code=502, detail=f"RunPod: {exc}")


@router.post("/stop/{pod_id}")
async def lifecycle_stop(pod_id: str):
    """Graceful stop — billing pauses, volume preserved, restartable via
    the RunPod console. Does NOT remove from the registry (URL still
    valid pointer to the paused pod)."""
    try:
        return await stop_pod(pod_id)
    except RunPodError as exc:
        raise HTTPException(status_code=502, detail=f"RunPod: {exc}")


@router.post("/terminate/{pod_id}")
async def lifecycle_terminate(pod_id: str):
    """Hard destroy — volume freed, billing over. Removes from the registry."""
    try:
        return await terminate_pod(pod_id)
    except RunPodError as exc:
        raise HTTPException(status_code=502, detail=f"RunPod: {exc}")


@router.get("/status")
async def lifecycle_status_all():
    """Live status for every registered pod — cost-so-far computed from
    uptime × cost_hr."""
    try:
        pods = await status_all()
    except RunPodError as exc:
        raise HTTPException(status_code=502, detail=f"RunPod: {exc}")
    total_cost = round(sum(float(p.get("cost_so_far") or 0) for p in pods), 4)
    active = [p for p in pods if p.get("desired_status") == "RUNNING"]
    return {
        "pods":       pods,
        "count":      len(pods),
        "running":    len(active),
        "total_cost_so_far": total_cost,
    }


@router.get("/status/{pod_id}")
async def lifecycle_status_one(pod_id: str):
    try:
        return await get_pod_status(pod_id)
    except RunPodError as exc:
        raise HTTPException(status_code=502, detail=f"RunPod: {exc}")


@router.get("/log")
async def lifecycle_log(limit: int = 50):
    """Recent lifecycle actions (newest first). Feeds the UI action-history panel."""
    return {"actions": await recent_log(limit=limit)}
