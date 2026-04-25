"""
workers.py — Dynamic GPU worker registry.

WHAT THIS FILE IS, FOR THE DUMMIES:
====================================

The Anamnesis app has a hard-coded list of GPU worker URLs in its .env:

    NANOGPT_URLS=http://192.168.10.15:3011,http://192.168.10.110:3011
    AVATAR_WORKER_URL_1=http://192.168.10.15:3013

Static URLs are fine for local boxes (your office and server) because their
IPs don't change. But for a RunPod cloud pod, the URL is different EVERY
TIME you start a new pod. Hard-coding them in .env + restarting the app
each time would be terrible UX.

This file gives us a third option: a MongoDB collection called
`worker_registry` where ephemeral workers (RunPod, EC2, etc.) can
register themselves on startup and unregister on shutdown. The
Anamnesis app polls this collection alongside the static .env list.

So a typical RunPod session looks like:

    1. ./deploy_runpod.sh start
       └→ creates pod
       └→ POSTs to /api/workers/register with the new URL
       └→ this file inserts a doc into worker_registry

    2. Anamnesis app's failover chain (in routes/anamnesis_gpt.py and
       avatar workers list) periodically reads worker_registry and
       includes the registered URLs as additional endpoints.

    3. ./deploy_runpod.sh stop
       └→ terminates pod
       └→ DELETEs from /api/workers/register/<id>
       └→ this file removes the doc

    4. Anamnesis app stops sending requests to that URL.

NO RESTART OF THE MAIN APP REQUIRED. This is the difference between a
deploy that needs SSH + manual config edit + container reboot vs. a
deploy that just runs a script.


SCHEMA:
========

worker_registry document:
    {
        "worker_id":  "runpod-rtx3090-abc123",   # unique label
        "url":        "http://12.34.56.78:54321",
        "kind":       "runpod" | "ec2" | "manual",
        "registered_at": ISODate,
        "last_seen":  ISODate,                   # updated by health pings
    }
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database import get_db

logger = logging.getLogger("anamnesis.routes.workers")

router = APIRouter(prefix="/api/workers", tags=["workers"])


# ============================================================================
# Pydantic schemas
# ============================================================================

class WorkerRegistration(BaseModel):
    """Body for POST /api/workers/register."""
    url: str           # e.g., "http://12.34.56.78:54321"
    label: str         # e.g., "runpod-rtx3090-abc123" — used as worker_id
    kind: str = "manual"  # "runpod" | "ec2" | "manual"


class WorkerInfo(BaseModel):
    """Response shape for worker listings."""
    worker_id: str
    url: str
    kind: str
    registered_at: datetime
    last_seen: datetime | None = None


# ============================================================================
# Database helper
# ============================================================================

def _registry():
    """Get the worker_registry collection (created lazily by Mongo)."""
    db = get_db()
    if db is None:
        raise RuntimeError("MongoDB not connected")
    return db["worker_registry"]


# ============================================================================
# Routes
# ============================================================================

@router.post("/register", response_model=WorkerInfo)
async def register_worker(reg: WorkerRegistration):
    """
    Register a new GPU worker.

    Called by deploy scripts (`deploy_runpod.sh start`) right after a pod
    becomes reachable. Idempotent: re-registering with the same `label`
    just refreshes the `registered_at` timestamp.
    """
    col = _registry()
    now = datetime.now(timezone.utc)

    doc = {
        "worker_id": reg.label,
        "url": reg.url,
        "kind": reg.kind,
        "registered_at": now,
        "last_seen": now,
    }

    # Upsert by worker_id so re-registering doesn't create duplicates
    await col.update_one(
        {"worker_id": reg.label},
        {"$set": doc},
        upsert=True,
    )
    logger.info(f"Worker registered: {reg.label} → {reg.url} (kind={reg.kind})")
    return WorkerInfo(**doc)


@router.delete("/register/{worker_id}")
async def unregister_worker(worker_id: str):
    """
    Remove a worker from the registry.

    Called by deploy scripts (`deploy_runpod.sh stop`) before the pod
    is terminated. Returns 404 if the worker_id wasn't registered.
    """
    col = _registry()
    result = await col.delete_one({"worker_id": worker_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Worker not found: {worker_id}")
    logger.info(f"Worker unregistered: {worker_id}")
    return {"ok": True, "worker_id": worker_id}


@router.get("/list", response_model=list[WorkerInfo])
async def list_workers(kind: str | None = None):
    """
    List all currently registered ephemeral workers.

    Optional `kind` filter (e.g. ?kind=runpod).

    NOTE: This only returns workers from the dynamic registry. It does
    NOT include the static .env-configured workers (NANOGPT_URLS,
    AVATAR_WORKER_URL_*). For a unified view, the app's failover chain
    merges both sources at request time.
    """
    col = _registry()
    query = {"kind": kind} if kind else {}
    cursor = col.find(query).sort("registered_at", -1)
    out = []
    async for doc in cursor:
        out.append(WorkerInfo(
            worker_id=doc["worker_id"],
            url=doc["url"],
            kind=doc.get("kind", "manual"),
            registered_at=doc["registered_at"],
            last_seen=doc.get("last_seen"),
        ))
    return out


@router.post("/{worker_id}/heartbeat")
async def heartbeat(worker_id: str):
    """
    Optional heartbeat endpoint. A worker can call this periodically to
    update its `last_seen`. The app can then prune workers that haven't
    pinged in a while (not implemented yet — manual cleanup for now).
    """
    col = _registry()
    result = await col.update_one(
        {"worker_id": worker_id},
        {"$set": {"last_seen": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"Worker not registered: {worker_id}")
    return {"ok": True, "worker_id": worker_id}
