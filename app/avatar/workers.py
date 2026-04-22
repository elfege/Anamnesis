"""Worker selection helper shared by XTTS / SadTalker / Demucs clients.

Centralizes the logic that decides which worker to try first given a
user-preferred worker_id (from the UI dropdown) and a no_fallback toggle.
"""
import logging
from typing import Optional

import httpx

import config

logger = logging.getLogger("anamnesis.avatar.workers")


def order_endpoints(
    endpoints: list[tuple[str, str]],
    preferred_worker: Optional[str] = None,
    no_fallback: bool = False,
) -> list[tuple[str, str]]:
    """Return endpoints in the order they should be tried.

    endpoints: list of (url, label) tuples from config.AVATAR_WORKER_ENDPOINTS
    preferred_worker: label (or substring of label) the user picked in the UI.
                     Match is case-insensitive substring for flexibility.
    no_fallback: if True and preferred_worker is set, return ONLY the matching
                 endpoint; otherwise return full list in default order.

    If preferred_worker is None, returns endpoints unchanged (fallback chain).
    If preferred_worker matches: preferred first, then others (unless no_fallback).
    If preferred_worker is set but nothing matches: log warning, fall back to full list.
    """
    if not preferred_worker:
        return list(endpoints)

    needle = preferred_worker.lower()
    matched = [(u, l) for (u, l) in endpoints if needle in l.lower()]
    others = [(u, l) for (u, l) in endpoints if needle not in l.lower()]

    if not matched:
        logger.warning(
            f"preferred_worker={preferred_worker!r} did not match any endpoint label; "
            f"falling back to full chain"
        )
        return list(endpoints)

    if no_fallback:
        return matched
    return matched + others


async def probe_worker(url: str, label: str, timeout: float = 3.0) -> dict:
    """Ping a worker's /health endpoint. Returns a dict with reachability and metadata.

    Shape:
      {
        "url": "...",
        "label": "...",
        "reachable": bool,
        "worker_id": "server-cuda-1660" | None,
        "gpu_type": "cuda" | "rocm" | "cpu" | None,
        "capabilities": ["xtts", "sadtalker", "demucs"],
        "error": None or str
      }
    """
    result = {
        "url": url,
        "label": label,
        "reachable": False,
        "worker_id": None,
        "gpu_type": None,
        "capabilities": [],
        "error": None,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{url}/health")
            if r.status_code != 200:
                result["error"] = f"HTTP {r.status_code}"
                return result
            data = r.json()
            result["reachable"] = True
            result["worker_id"] = data.get("worker_id") or label
            result["gpu_type"] = data.get("gpu_type")
            result["capabilities"] = data.get("capabilities", [])
    except Exception as exc:
        result["error"] = str(exc)
    return result


async def probe_all_workers() -> list[dict]:
    """Probe every worker in config.AVATAR_WORKER_ENDPOINTS in parallel."""
    import asyncio
    endpoints = config.AVATAR_WORKER_ENDPOINTS
    if not endpoints:
        return []
    tasks = [probe_worker(url, label) for url, label in endpoints]
    return list(await asyncio.gather(*tasks))
