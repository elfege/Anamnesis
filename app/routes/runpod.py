"""
runpod.py — RunPod pod lifecycle controlled from the chat UI.

WHAT THIS FILE IS, FOR THE DUMMIES:
====================================

The user wants to spin up / shut down a RunPod GPU pod from the chat UI
(Anamnesis), not from a terminal. This module exposes the four endpoints
the UI needs:

    POST  /api/runpod/start       — create a pod (SSE-streamed progress)
    GET   /api/runpod/status      — current pod state + cost so far
    POST  /api/runpod/stop        — terminate the pod
    GET   /api/runpod/cost-meter  — lightweight live cost meter (poll every 5s)

DESIGN NOTE — why we don't shell out to deploy_runpod.sh:
----------------------------------------------------------
The original task spec asked us to spawn `./deploy_runpod.sh start`
via asyncio. That doesn't fly inside our container because:
  1. The repo root is NOT mounted into the container (only ./app is).
  2. The script depends on `jq`, which isn't installed in the image.
  3. The script writes .env on the host, which the container can't see.

So we reimplement the GraphQL calls in Python, here. This keeps the
container self-sufficient and avoids cross-cutting changes (Dockerfile
rebuild, docker-compose volume mounts, host filesystem writes).

The shell script remains the source-of-truth for terminal use; this
module mirrors its behavior 1:1 (same GraphQL mutations, same polling
loop, same worker_registry registration).

Pod state is tracked in MongoDB (collection: `runpod_state`):
    { _id: "active_pod", pod_id, gpu, profile, image, port,
      created_at, endpoint_url }

We use Mongo instead of a local file (.runpod_pod_id) because the
container is ephemeral — restarting the app would lose the file but
not the Mongo doc.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import RUNPOD_API_KEY, RUNPOD_REGISTRY_AUTH_ID
from database import get_db

logger = logging.getLogger("anamnesis.routes.runpod")

router = APIRouter(prefix="/api/runpod", tags=["runpod"])


# ============================================================================
# Configuration — mirrors deploy_runpod.sh
# ============================================================================

RUNPOD_API = "https://api.runpod.io/graphql"

# Profile -> (default image, default exposed port).
# We expose only "d2" from the UI for now (the trainer profile is unused
# from the chat UI; if needed later, just add a profile selector).
PROFILES: dict[str, dict] = {
    "d2": {
        "image": "ghcr.io/elfege/anamnesis-d2:cuda-runpod",
        "port": 3015,
    },
    "trainer": {
        "image": "elfege/anamnesis-trainer:cuda-latest",
        "port": 3011,
    },
}

GPU_TYPE_IDS: dict[str, str] = {
    "rtx3090": "NVIDIA GeForce RTX 3090",
    "rtx4090": "NVIDIA GeForce RTX 4090",
    "a100":    "NVIDIA A100 80GB PCIe",
    "h100":    "NVIDIA H100 80GB HBM3",
}

# Polling cadence for "is pod RUNNING yet?" loop. The bash script uses
# 10s × 60 = 10 min; we use the same upper bound but emit progress more often.
POD_POLL_INTERVAL_S = 10
POD_POLL_MAX_ATTEMPTS = 60


# ============================================================================
# State helpers (Mongo-backed)
# ============================================================================

def _state_col():
    db = get_db()
    if db is None:
        raise RuntimeError("MongoDB not connected")
    return db["runpod_state"]


def _registry_col():
    db = get_db()
    if db is None:
        raise RuntimeError("MongoDB not connected")
    return db["worker_registry"]


async def _get_active_pod() -> Optional[dict]:
    """Return the currently-active pod doc, or None."""
    return await _state_col().find_one({"_id": "active_pod"})


async def _set_active_pod(doc: dict) -> None:
    """Upsert the active-pod state."""
    await _state_col().update_one(
        {"_id": "active_pod"},
        {"$set": doc},
        upsert=True,
    )


async def _clear_active_pod() -> None:
    await _state_col().delete_one({"_id": "active_pod"})


# ============================================================================
# GraphQL helper
# ============================================================================

async def _gql(query: str, timeout: float = 15.0) -> dict:
    """Send a GraphQL query/mutation to RunPod. Returns parsed JSON."""
    if not RUNPOD_API_KEY:
        raise RuntimeError("RUNPOD_API_KEY not configured")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(RUNPOD_API, headers=headers, json={"query": query})
        r.raise_for_status()
        return r.json()


# ============================================================================
# Pydantic schemas
# ============================================================================

class StartRequest(BaseModel):
    gpu: str = "rtx3090"
    profile: str = "d2"


class StopResponse(BaseModel):
    ok: bool
    stopped_pod_id: Optional[str] = None


# ============================================================================
# SSE helpers
# ============================================================================

def _sse(event: dict) -> str:
    """Encode a dict as one SSE `data:` frame."""
    return f"data: {json.dumps(event)}\n\n"


# ============================================================================
# Lifecycle: start
# ============================================================================

async def _start_stream(gpu: str, profile: str) -> AsyncIterator[str]:
    """
    Create a pod, poll until RUNNING + reachable, register the URL.
    Yields SSE frames the UI can render in its terminal panel.
    """
    yield _sse({"stage": "init", "msg": f"Starting RunPod pod (gpu={gpu}, profile={profile})…"})

    if profile not in PROFILES:
        yield _sse({"error": f"Unknown profile: {profile}. Use one of {list(PROFILES)}"})
        return
    if gpu not in GPU_TYPE_IDS:
        yield _sse({"error": f"Unknown GPU alias: {gpu}. Use one of {list(GPU_TYPE_IDS)}"})
        return
    if not RUNPOD_API_KEY:
        yield _sse({"error": "RUNPOD_API_KEY not configured in container .env"})
        return

    existing = await _get_active_pod()
    if existing:
        yield _sse({
            "error": (
                f"A pod is already tracked: {existing.get('pod_id')}. "
                f"Stop it first via POST /api/runpod/stop."
            )
        })
        return

    cfg = PROFILES[profile]
    image = cfg["image"]
    port = cfg["port"]
    gpu_type_id = GPU_TYPE_IDS[gpu]
    pod_name = f"anamnesis-{profile}"
    ports_spec = f"{port}/http"

    # If image is on a private registry (e.g. ghcr.io/elfege/anamnesis-d2),
    # RunPod needs containerRegistryAuthId to pull it. See app/config.py for
    # how RUNPOD_REGISTRY_AUTH_ID is provisioned.
    auth_clause = ""
    if RUNPOD_REGISTRY_AUTH_ID:
        auth_clause = f', containerRegistryAuthId: "{RUNPOD_REGISTRY_AUTH_ID}"'
        yield _sse({"stage": "creating", "msg": f"Submitting podFindAndDeployOnDemand mutation (image={image}, private-pull auth: {RUNPOD_REGISTRY_AUTH_ID})…"})
    else:
        yield _sse({"stage": "creating", "msg": f"Submitting podFindAndDeployOnDemand mutation (image={image}, no registry auth — public images only)…"})

    # GraphQL mutation — same shape as deploy_runpod.sh
    # Inline-string GraphQL because the RunPod public API expects a single
    # `query` field (no variables on this mutation per their docs).
    mutation = (
        'mutation { podFindAndDeployOnDemand(input: { '
        'cloudType: COMMUNITY, gpuCount: 1, volumeInGb: 50, '
        'containerDiskInGb: 20, '
        f'gpuTypeId: "{gpu_type_id}", '
        f'name: "{pod_name}", '
        f'imageName: "{image}", '
        f'ports: "{ports_spec}", '
        'env: [{ key: "AUTO_LOAD_MODEL", value: "true" }]'
        f'{auth_clause} '
        ') { id desiredStatus runtime { ports { ip publicPort privatePort isIpPublic } } } }'
    )

    try:
        resp = await _gql(mutation, timeout=30.0)
    except Exception as exc:
        yield _sse({"error": f"GraphQL create-pod call failed: {exc}"})
        return

    pod_id = (resp.get("data") or {}).get("podFindAndDeployOnDemand", {}).get("id")
    if not pod_id:
        yield _sse({"error": f"Pod creation rejected: {json.dumps(resp)[:400]}"})
        return

    yield _sse({"stage": "created", "msg": f"Pod created: {pod_id}"})

    # Persist state immediately — even if polling fails we know the pod ID
    # so the user can stop it from the UI later.
    await _set_active_pod({
        "pod_id": pod_id,
        "gpu": gpu,
        "profile": profile,
        "image": image,
        "port": port,
        "created_at": datetime.now(timezone.utc),
        "endpoint_url": None,
    })

    # ── Poll for RUNNING + public port ────────────────────────────
    public_url: Optional[str] = None
    public_ip: Optional[str] = None
    public_port: Optional[int] = None

    yield _sse({"stage": "polling", "msg": "Waiting for pod to become RUNNING (1-3 min typical)…"})

    for attempt in range(1, POD_POLL_MAX_ATTEMPTS + 1):
        await asyncio.sleep(POD_POLL_INTERVAL_S)
        try:
            status_q = (
                'query { pod(input: { podId: "' + pod_id + '" }) { '
                'desiredStatus runtime { ports { ip publicPort privatePort isIpPublic } } '
                '} }'
            )
            sresp = await _gql(status_q, timeout=10.0)
        except Exception as exc:
            yield _sse({
                "stage": "polling",
                "msg": f"poll {attempt}/{POD_POLL_MAX_ATTEMPTS} — query failed: {exc}",
            })
            continue

        pod_data = ((sresp.get("data") or {}).get("pod") or {})
        status = pod_data.get("desiredStatus") or "UNKNOWN"
        ports = (pod_data.get("runtime") or {}).get("ports") or []
        first_port = ports[0] if ports else {}
        public_ip = first_port.get("ip")
        public_port = first_port.get("publicPort")

        yield _sse({
            "stage": "polling",
            "msg": f"poll {attempt}/{POD_POLL_MAX_ATTEMPTS} — status={status} ip={public_ip or '?'} port={public_port or '?'}",
        })

        if status == "RUNNING" and public_ip and public_port:
            public_url = f"http://{public_ip}:{public_port}"
            break

    if not public_url:
        yield _sse({
            "error": (
                f"Pod {pod_id} did not become reachable in "
                f"{POD_POLL_INTERVAL_S * POD_POLL_MAX_ATTEMPTS}s. "
                f"It may still be starting — check RunPod console. "
                f"Stop it via POST /api/runpod/stop if abandoned."
            )
        })
        return

    yield _sse({"stage": "running", "msg": f"Pod is RUNNING at {public_url}"})

    # ── Health-check the service ──────────────────────────────────
    yield _sse({"stage": "health", "msg": "Pinging /health endpoint…"})
    healthy = False
    for hattempt in range(1, 13):  # 60s total
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                hr = await client.get(f"{public_url}/health")
                if hr.status_code == 200:
                    healthy = True
                    break
        except Exception:
            pass
        await asyncio.sleep(5)

    if healthy:
        yield _sse({"stage": "health", "msg": "Service responsive."})
    else:
        yield _sse({
            "stage": "health",
            "msg": "Service not yet responsive after 60s — pod is up but image may still be initializing.",
        })

    # ── Register in worker_registry + persist endpoint URL ────────
    worker_label = f"runpod-{gpu}-{pod_id}"
    try:
        now = datetime.now(timezone.utc)
        await _registry_col().update_one(
            {"worker_id": worker_label},
            {"$set": {
                "worker_id": worker_label,
                "url": public_url,
                "kind": "runpod",
                "registered_at": now,
                "last_seen": now,
            }},
            upsert=True,
        )
    except Exception as exc:
        logger.warning(f"Could not register worker: {exc}")

    await _set_active_pod({
        "pod_id": pod_id,
        "gpu": gpu,
        "profile": profile,
        "image": image,
        "port": port,
        "created_at": datetime.now(timezone.utc),
        "endpoint_url": public_url,
        "worker_label": worker_label,
    })

    yield _sse({
        "stage": "complete",
        "msg": f"RunPod is live: {public_url}",
        "pod_id": pod_id,
        "endpoint_url": public_url,
        "reachable": healthy,
    })


@router.post("/start")
async def start_pod(req: StartRequest):
    """
    Spin up a RunPod pod. Returns an SSE stream of progress events.
    Total time: 5-10 minutes typical (pod create + image pull + service start).

    The client should keep the stream open and render `stage`/`msg` events
    in the UI's terminal panel. The final event has `stage: "complete"` (or
    an `error`).
    """
    return StreamingResponse(
        _start_stream(req.gpu, req.profile),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================================
# Lifecycle: status
# ============================================================================

async def _query_pod_runtime(pod_id: str) -> dict:
    """Fetch live status + cost from RunPod GraphQL."""
    q = (
        'query { pod(input: { podId: "' + pod_id + '" }) { '
        'desiredStatus costPerHr '
        'runtime { uptimeInSeconds ports { ip publicPort privatePort isIpPublic } } '
        '} }'
    )
    return await _gql(q, timeout=8.0)


async def _probe_endpoint(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{url.rstrip('/')}/health")
            return r.status_code == 200
    except Exception:
        return False


@router.get("/status")
async def pod_status():
    """
    Current state of the tracked pod. Returns:
        { pod_id, gpu, profile, status, hourly_cost,
          uptime_seconds, total_cost_so_far, endpoint_url, reachable }
    or
        { pod_id: null, status: "stopped" }
    if no pod is currently tracked.
    """
    active = await _get_active_pod()
    if not active:
        return {"pod_id": None, "status": "stopped"}

    pod_id = active["pod_id"]

    try:
        resp = await _query_pod_runtime(pod_id)
    except Exception as exc:
        return {
            "pod_id": pod_id,
            "gpu": active.get("gpu"),
            "profile": active.get("profile"),
            "endpoint_url": active.get("endpoint_url"),
            "status": "unknown",
            "error": str(exc),
        }

    pod_data = ((resp.get("data") or {}).get("pod") or {})
    if not pod_data:
        # Pod no longer exists at RunPod (terminated externally) — clean up.
        await _clear_active_pod()
        worker_label = active.get("worker_label")
        if worker_label:
            try:
                await _registry_col().delete_one({"worker_id": worker_label})
            except Exception:
                pass
        return {"pod_id": None, "status": "stopped", "msg": "pod no longer exists at RunPod"}

    status = pod_data.get("desiredStatus") or "UNKNOWN"
    hourly = float(pod_data.get("costPerHr") or 0.0)
    runtime = pod_data.get("runtime") or {}
    uptime_s = int(runtime.get("uptimeInSeconds") or 0)
    total_cost = round((hourly * uptime_s) / 3600.0, 4)

    endpoint = active.get("endpoint_url")
    if not endpoint:
        # Late discovery: the pod started but our previous polling didn't
        # catch the URL; try to grab it now.
        ports = runtime.get("ports") or []
        if ports:
            ip = ports[0].get("ip")
            pp = ports[0].get("publicPort")
            if ip and pp:
                endpoint = f"http://{ip}:{pp}"
                await _state_col().update_one(
                    {"_id": "active_pod"},
                    {"$set": {"endpoint_url": endpoint}},
                )

    reachable = await _probe_endpoint(endpoint) if endpoint else False

    return {
        "pod_id": pod_id,
        "gpu": active.get("gpu"),
        "profile": active.get("profile"),
        "status": status,
        "hourly_cost": hourly,
        "uptime_seconds": uptime_s,
        "total_cost_so_far": total_cost,
        "endpoint_url": endpoint,
        "reachable": reachable,
    }


# ============================================================================
# Lifecycle: stop
# ============================================================================

@router.post("/stop", response_model=StopResponse)
async def stop_pod():
    """Terminate the active pod, unregister it, clear local state."""
    active = await _get_active_pod()
    if not active:
        return StopResponse(ok=True, stopped_pod_id=None)

    pod_id = active["pod_id"]
    worker_label = active.get("worker_label")

    mutation = 'mutation { podTerminate(input: { podId: "' + pod_id + '" }) }'
    try:
        await _gql(mutation, timeout=15.0)
    except Exception as exc:
        # Don't leave a dangling state doc — the pod is probably already gone.
        logger.warning(f"podTerminate raised {exc}; clearing local state anyway")

    if worker_label:
        try:
            await _registry_col().delete_one({"worker_id": worker_label})
        except Exception as exc:
            logger.warning(f"Could not unregister worker {worker_label}: {exc}")

    await _clear_active_pod()
    return StopResponse(ok=True, stopped_pod_id=pod_id)


# ============================================================================
# Lifecycle: cost-meter (lightweight, polled every 5s by the UI)
# ============================================================================

# Simple in-process cache so we don't hammer RunPod's GraphQL every 5s
# from every browser tab. Cost data only changes by ~$0.0003/5s anyway.
_cost_cache: dict = {"at": 0.0, "data": None}
_COST_CACHE_TTL = 4.0  # seconds


@router.get("/cost-meter")
async def cost_meter():
    """
    Cheap polling endpoint for the live cost meter widget.
    Returns: { active, hourly, uptime_seconds, total_so_far_usd, status }
    """
    active = await _get_active_pod()
    if not active:
        return {"active": False, "hourly": 0.0, "uptime_seconds": 0, "total_so_far_usd": 0.0}

    now = time.time()
    if _cost_cache["data"] and (now - _cost_cache["at"]) < _COST_CACHE_TTL:
        cached = dict(_cost_cache["data"])
        cached["cached"] = True
        return cached

    pod_id = active["pod_id"]
    try:
        resp = await _query_pod_runtime(pod_id)
    except Exception as exc:
        return {
            "active": True,
            "pod_id": pod_id,
            "hourly": 0.0,
            "uptime_seconds": 0,
            "total_so_far_usd": 0.0,
            "status": "unknown",
            "error": str(exc),
        }

    pod_data = ((resp.get("data") or {}).get("pod") or {})
    if not pod_data:
        return {
            "active": False, "hourly": 0.0, "uptime_seconds": 0,
            "total_so_far_usd": 0.0, "status": "stopped",
        }

    hourly = float(pod_data.get("costPerHr") or 0.0)
    runtime = pod_data.get("runtime") or {}
    uptime_s = int(runtime.get("uptimeInSeconds") or 0)
    total_cost = round((hourly * uptime_s) / 3600.0, 4)
    status = pod_data.get("desiredStatus") or "UNKNOWN"

    out = {
        "active": True,
        "pod_id": pod_id,
        "hourly": hourly,
        "uptime_seconds": uptime_s,
        "total_so_far_usd": total_cost,
        "status": status,
    }
    _cost_cache["at"] = now
    _cost_cache["data"] = out
    return out


# ============================================================================
# GPU catalog (for the start dropdown)
# ============================================================================

@router.get("/gpu-aliases")
async def list_gpu_aliases():
    """List the GPU aliases the UI can offer in its 'Spin' dropdown."""
    return {
        "aliases": [
            {"id": "rtx3090", "label": "RTX 3090 (24 GB) — ~$0.22/hr"},
            {"id": "rtx4090", "label": "RTX 4090 (24 GB) — ~$0.40/hr"},
            {"id": "a100",    "label": "A100 80GB — ~$1.50/hr"},
            {"id": "h100",    "label": "H100 80GB — ~$2.50/hr"},
        ],
        "default": "rtx3090",
        "default_profile": "d2",
    }
