"""RunPod avatar-worker pod registry — MongoDB-backed.

Why this exists: a RunPod pod_id changes every time we create a new pod.
Putting that URL in AWS Secrets (a long-lived credential store) was the wrong
shape — every pod cycle required AWS edit + pull_env.sh + container restart.
Per canonical data-store split: AWS=bootstrap, .env=intermediate (only for
truly static workers), MongoDB=runtime SOT for ephemeral state.

Static workers (office, server, dellserver) stay in .env via AVATAR_WORKER_URL_N.
RunPod pods live here.

Public API:
  - hydrate_from_db()        : call once at startup (async)
  - list_endpoints_sync()    : list[(url, label)] in static-shape for config merging
  - list_pods_sync()         : list[dict] for UI display
  - add_pod(...)             : upsert (async)
  - delete_pod(pod_id)       : remove (async)

Cache: module-level list mirrors the Mongo collection. Sync getters are safe
because every mutator updates the cache atomically while doing the DB write,
and the read path is called from many sync-iteration sites
(e.g. config.AVATAR_WORKER_ENDPOINTS in workers.py / emergency.py).
"""
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from database import get_db

COLLECTION = "avatar_runpod_pods"

# In-memory mirror — hydrated on startup, mutated on add/delete.
_cache: List[dict] = []


def _url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def list_endpoints_sync() -> List[Tuple[str, str]]:
    """(url, label) pairs — same shape as config._STATIC_AVATAR_WORKER_ENDPOINTS
    so config.AVATAR_WORKER_ENDPOINTS can concatenate the two."""
    return [(_url(d["pod_id"], d["port"]), d["label"]) for d in _cache]


def list_pods_sync() -> List[dict]:
    """Pod docs for UI display."""
    out = []
    for d in _cache:
        out.append({
            "pod_id":     d["pod_id"],
            "port":       d["port"],
            "label":      d["label"],
            "gpu_type":   d.get("gpu_type"),
            "url":        _url(d["pod_id"], d["port"]),
            "created_at": d["created_at"].isoformat() if d.get("created_at") else None,
        })
    return out


async def hydrate_from_db() -> int:
    """Read everything from Mongo into the cache. Call once at startup."""
    global _cache
    coll = get_db()[COLLECTION]
    docs = []
    async for d in coll.find({}):
        d.pop("_id", None)
        docs.append(d)
    _cache = docs
    return len(_cache)


async def add_pod(pod_id: str, port: int, label: str, gpu_type: Optional[str] = None) -> dict:
    if not pod_id or not isinstance(pod_id, str):
        raise ValueError("pod_id required (string)")
    try:
        port = int(port)
    except (TypeError, ValueError):
        raise ValueError("port required (integer)")
    if not label or not isinstance(label, str):
        raise ValueError("label required (string)")

    now = datetime.now(timezone.utc)
    doc = {
        "pod_id":     pod_id.strip(),
        "port":       port,
        "label":      label.strip(),
        "gpu_type":   (gpu_type or "").strip() or None,
        "created_at": now,
    }
    coll = get_db()[COLLECTION]
    await coll.update_one({"pod_id": doc["pod_id"]}, {"$set": doc}, upsert=True)

    # Mirror in cache
    _cache[:] = [d for d in _cache if d["pod_id"] != doc["pod_id"]]
    _cache.append(doc)

    return {
        "pod_id":   doc["pod_id"],
        "port":     doc["port"],
        "label":    doc["label"],
        "gpu_type": doc["gpu_type"],
        "url":      _url(doc["pod_id"], doc["port"]),
    }


async def delete_pod(pod_id: str) -> bool:
    coll = get_db()[COLLECTION]
    res = await coll.delete_one({"pod_id": pod_id})
    before = len(_cache)
    _cache[:] = [d for d in _cache if d["pod_id"] != pod_id]
    return res.deleted_count > 0 or len(_cache) != before
