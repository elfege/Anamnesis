"""Full RunPod pod lifecycle — implementation of pin #7 (Priority 1).

Extends the existing runpod_pods.py (URL registry only) with the actual
GraphQL calls to CREATE, STOP, TERMINATE, and OBSERVE pods. Cost tracked
per-pod using RunPod's own `costPerHr` field (accurate, not estimated).

Action history persisted to Mongo `avatar_runpod_lifecycle_log` — feeds
the "recent actions" panel + supports post-hoc cost auditing.

Pod state model:
  RUNNING  — provisioned + billed. Start-response cost_hr is authoritative.
  STOPPED  — RunPod-side stop; billing paused, volume preserved. Restartable.
  TERMINATED — RunPod-side terminate; volume freed, billing over. Not restartable.

Uses RUNPOD_API_KEY from env (already provisioned via AWS Secrets Manager).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from database import get_db
from avatar import runpod_pods

logger = logging.getLogger("anamnesis.runpod_lifecycle")

RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"
LIFECYCLE_LOG = "avatar_runpod_lifecycle_log"
DEFAULT_IMAGE = "ghcr.io/elfege/anamnesis-avatar-worker:cuda-runpod"
DEFAULT_PORT = 3013

# GPU tier catalog — mirrors deploy_runpod.sh:100-104 + cost estimates from
# deploy_runpod.sh:149-155. The `id` field is what RunPod's GraphQL API
# accepts as gpuTypeId; the operator picks by alias.
GPU_TIERS = {
    "rtx3090": {
        "id":         "NVIDIA GeForce RTX 3090",
        "label":      "RTX 3090 · 24 GB · community",
        "hourly_est": 0.30,
        "vram_gb":    24,
    },
    "rtx4090": {
        "id":         "NVIDIA GeForce RTX 4090",
        "label":      "RTX 4090 · 24 GB · community",
        "hourly_est": 0.40,
        "vram_gb":    24,
    },
    "a100": {
        "id":         "NVIDIA A100 80GB PCIe",
        "label":      "A100 · 80 GB · secure",
        "hourly_est": 1.50,
        "vram_gb":    80,
    },
    "h100": {
        "id":         "NVIDIA H100 80GB HBM3",
        "label":      "H100 · 80 GB · secure",
        "hourly_est": 2.50,
        "vram_gb":    80,
    },
}

# Cost sanity threshold — force explicit confirmation above this rate.
COST_HR_HARD_CONFIRM_THRESHOLD = 1.00


class RunPodError(Exception):
    """Wraps GraphQL / HTTP failures with a caller-friendly message."""


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise RunPodError("RUNPOD_API_KEY not configured (env or AWS secret)")
    return key


def _registry_auth_id() -> Optional[str]:
    """Opaque RunPod-side id for pulling the private ghcr.io image; None
    means public-image fallback."""
    return (os.environ.get("RUNPOD_REGISTRY_AUTH_ID") or "").strip() or None


async def _graphql(query: str, timeout: float = 30.0) -> dict:
    """POST a raw GraphQL query. Returns the `data` field, raises on error."""
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(RUNPOD_GRAPHQL, headers=headers, json={"query": query})
    try:
        payload = r.json()
    except Exception:
        raise RunPodError(f"RunPod HTTP {r.status_code}: non-JSON body {r.text[:200]}")
    if r.status_code >= 400 or "errors" in payload:
        errs = payload.get("errors") or [{"message": f"HTTP {r.status_code}"}]
        raise RunPodError("; ".join(e.get("message", "?") for e in errs))
    return payload.get("data", {}) or {}


async def _log(action: str, pod_id: Optional[str], outcome: str, **extra) -> None:
    """Append to the lifecycle audit log."""
    doc = {
        "action":     action,     # "start" | "stop" | "terminate" | "status"
        "pod_id":     pod_id,
        "outcome":    outcome,    # "ok" | "error" | "not_found"
        "ts":         datetime.now(timezone.utc),
        **extra,
    }
    try:
        await get_db()[LIFECYCLE_LOG].insert_one(doc)
    except Exception as exc:
        logger.warning(f"Lifecycle log write failed (non-fatal): {exc}")


# ─── Public API ─────────────────────────────────────────────────

def list_tiers() -> list[dict]:
    """Return the GPU catalog as a serializable list (for the UI dropdown)."""
    return [
        {
            "alias":            alias,
            "label":            info["label"],
            "hourly_estimate":  info["hourly_est"],
            "vram_gb":          info["vram_gb"],
            "hard_confirm":     info["hourly_est"] >= COST_HR_HARD_CONFIRM_THRESHOLD,
        }
        for alias, info in GPU_TIERS.items()
    ]


async def start_pod(
    gpu_tier: str,
    image: Optional[str] = None,
    port: int = DEFAULT_PORT,
    label: Optional[str] = None,
) -> dict:
    """Create + register a pod. Returns {pod_id, url, cost_hr, gpu_type, started_at}."""
    if gpu_tier not in GPU_TIERS:
        raise RunPodError(f"Unknown gpu_tier {gpu_tier!r}; choose from {list(GPU_TIERS)}")
    tier = GPU_TIERS[gpu_tier]
    image = image or DEFAULT_IMAGE

    # Compose the create-mutation. Follows deploy_runpod.sh:222-232 shape
    # but omits containerRegistryAuthId when RUNPOD_REGISTRY_AUTH_ID is
    # unset (public image path). volumeMountPath is REQUIRED — RunPod
    # rejects the mount without it (a bug that surfaced 2026-06-12).
    auth_id = _registry_auth_id()
    auth_clause = f', containerRegistryAuthId: "{auth_id}"' if auth_id else ""
    env_clause = 'env: [{ key: "GPU_TYPE", value: "cuda" }, { key: "MACHINE_NAME", value: "runpod" }]'
    ports_spec = f"{port}/http"
    name = f"anamnesis-avatar-{gpu_tier}"

    mutation = f'''
      mutation {{
        podFindAndDeployOnDemand(input: {{
          cloudType:         COMMUNITY,
          gpuCount:          1,
          volumeInGb:        50,
          volumeMountPath:   "/workspace",
          containerDiskInGb: 40,
          gpuTypeId:         "{tier["id"]}",
          name:              "{name}",
          imageName:         "{image}",
          ports:             "{ports_spec}",
          {env_clause}{auth_clause}
        }}) {{
          id
          costPerHr
          desiredStatus
          machineId
          imageName
        }}
      }}
    '''
    data = await _graphql(mutation, timeout=60.0)
    pod = (data or {}).get("podFindAndDeployOnDemand") or {}
    pod_id = pod.get("id")
    if not pod_id:
        await _log("start", None, "error", gpu_tier=gpu_tier, response=json.dumps(data)[:500])
        raise RunPodError(f"RunPod did not return a pod id: {json.dumps(data)[:200]}")

    cost_hr = float(pod.get("costPerHr") or tier["hourly_est"])
    started_at = datetime.now(timezone.utc)
    url = runpod_pods._url(pod_id, port)  # https://<pod_id>-<port>.proxy.runpod.net
    display_label = label or f"runpod · {tier['label']}"

    # Register in the URL registry (same collection used by chat routing).
    try:
        await runpod_pods.add_pod(pod_id=pod_id, port=port, label=display_label, gpu_type="cuda")
    except Exception as exc:
        # Non-fatal — pod is up on RunPod even if registry write fails.
        logger.error(f"Registry write failed after successful pod create: {exc}")

    await _log(
        "start", pod_id, "ok",
        gpu_tier=gpu_tier,
        cost_hr=cost_hr,
        image=image,
        started_at=started_at,
    )
    return {
        "pod_id":     pod_id,
        "url":        url,
        "gpu_tier":   gpu_tier,
        "gpu_type":   tier["label"],
        "cost_hr":    cost_hr,
        "started_at": started_at.isoformat(),
    }


async def stop_pod(pod_id: str) -> dict:
    """Graceful stop — billing pauses, volume preserved, restartable."""
    mutation = f'''
      mutation {{
        podStop(input: {{ podId: "{pod_id}" }}) {{ id desiredStatus }}
      }}
    '''
    data = await _graphql(mutation, timeout=30.0)
    pod = (data or {}).get("podStop") or {}
    outcome = "ok" if pod.get("id") == pod_id else "error"
    await _log("stop", pod_id, outcome, response=json.dumps(data)[:300])
    return {"pod_id": pod_id, "desired_status": pod.get("desiredStatus"), "outcome": outcome}


async def terminate_pod(pod_id: str) -> dict:
    """Destroy the pod. Volume freed. NOT restartable."""
    mutation = f'''
      mutation {{
        podTerminate(input: {{ podId: "{pod_id}" }})
      }}
    '''
    try:
        data = await _graphql(mutation, timeout=30.0)
        outcome = "ok"
    except RunPodError as exc:
        data = {"error": str(exc)}
        outcome = "error"
    # Also remove from the URL registry — a terminated pod's URL is dead.
    try:
        await runpod_pods.delete_pod(pod_id)
    except Exception as exc:
        logger.warning(f"Registry delete after terminate failed (non-fatal): {exc}")
    await _log("terminate", pod_id, outcome, response=json.dumps(data)[:300])
    return {"pod_id": pod_id, "outcome": outcome}


async def get_pod_status(pod_id: str) -> dict:
    """Live query: desired vs current status, uptime, cost/hr from RunPod."""
    query = f'''
      query {{
        pod(input: {{ podId: "{pod_id}" }}) {{
          id
          desiredStatus
          costPerHr
          imageName
          machineId
          lastStartedAt
          runtime {{ uptimeInSeconds ports {{ ip publicPort privatePort isIpPublic }} }}
        }}
      }}
    '''
    data = await _graphql(query, timeout=15.0)
    pod = (data or {}).get("pod") or {}
    if not pod.get("id"):
        return {"pod_id": pod_id, "found": False}
    rt = pod.get("runtime") or {}
    ports = rt.get("ports") or []
    public = next((p for p in ports if p.get("isIpPublic")), (ports[0] if ports else {}))
    return {
        "pod_id":            pod_id,
        "found":             True,
        "desired_status":    pod.get("desiredStatus"),
        "cost_hr":           float(pod.get("costPerHr") or 0),
        "image":             pod.get("imageName"),
        "machine_id":        pod.get("machineId"),
        "uptime_seconds":    rt.get("uptimeInSeconds") or 0,
        "last_started_at":   pod.get("lastStartedAt"),
        "public_ip":         public.get("ip"),
        "public_port":       public.get("publicPort"),
    }


async def status_all() -> list[dict]:
    """Poll RunPod for every pod currently in our registry. Returns list of
    status dicts, one per registered pod, with cost-so-far computed from
    the last start log entry."""
    out = []
    log_coll = get_db()[LIFECYCLE_LOG]
    for pod in runpod_pods.list_pods_sync():
        pod_id = pod["pod_id"]
        try:
            status = await get_pod_status(pod_id)
        except RunPodError as exc:
            status = {"pod_id": pod_id, "found": False, "error": str(exc)}
        # Cost-so-far — pull the most recent "start" log for this pod.
        start_doc = await log_coll.find_one(
            {"pod_id": pod_id, "action": "start", "outcome": "ok"},
            sort=[("ts", -1)],
        )
        if start_doc and status.get("found"):
            hours = (status.get("uptime_seconds") or 0) / 3600.0
            status["cost_so_far"] = round(hours * float(status.get("cost_hr") or 0), 4)
        status["label"] = pod.get("label")
        out.append(status)
    return out


async def recent_log(limit: int = 50) -> list[dict]:
    """Recent lifecycle actions (newest first)."""
    coll = get_db()[LIFECYCLE_LOG]
    cursor = coll.find({}, {"_id": 0}).sort("ts", -1).limit(limit)
    out = []
    async for doc in cursor:
        # ISO-ify datetimes for JSON transport
        for k, v in list(doc.items()):
            if isinstance(v, datetime):
                doc[k] = v.isoformat()
        out.append(doc)
    return out
