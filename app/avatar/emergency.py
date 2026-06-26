"""Emergency stop — the panic button.

This module is INTENTIONALLY isolated from the normal pipeline cancellation
logic. The regular Stop button goes through the WebSocket and is subject to
all the failure modes of the live session (WS dead, sampling collapse mid-
stream, cancel logic refactored, etc.). When that's the problem, you can't
use the same channel to fix it.

This is the channel that ALWAYS works:
  - Plain HTTP POST, not WebSocket
  - No session state, no in-flight task tracking
  - Walks the static config list of workers + ollama endpoints
  - Fires the same kill set every time, regardless of what's "supposed" to
    be running

# DO NOT REFACTOR THIS INTO THE PIPELINE.
# DO NOT ADD CONDITIONAL LOGIC HERE.
# DO NOT REMOVE FALLBACK PATHS.
# The whole point is that this code path is the LAST resort. If you "improve"
# it by coupling it to the rest of the system, you've broken its only job.

What it kills:
  1. Every SadTalker subprocess on every avatar worker (POST /sadtalker/cancel)
  2. Every MuseTalk subprocess on every avatar worker (POST /musetalk/cancel)
  3. Every loaded ollama model (POST /api/generate with keep_alive=0) — this
     frees VRAM and unsticks zombie runners that the rest of the system
     might be waiting on.

What it explicitly does NOT do:
  - Cancel WebSocket sessions (those will see their upstream calls fail and
    naturally cleanup)
  - Cancel XTTS (no worker endpoint exists; XTTS calls are short anyway)
  - Restart any services (operator decision; this only kills GPU jobs)
"""
import asyncio
import logging
from typing import Optional

import httpx

import config

logger = logging.getLogger("anamnesis.avatar.emergency")

# Per-call timeout. Short on purpose — we'd rather report "no response" and
# move on than block the panic button itself.
KILL_TIMEOUT_S = 4.0


async def _post(url: str, json: Optional[dict] = None) -> str:
    try:
        async with httpx.AsyncClient(timeout=KILL_TIMEOUT_S) as c:
            r = await c.post(url, json=json or {})
        return f"HTTP {r.status_code}"
    except httpx.TimeoutException:
        return "timeout"
    except Exception as e:
        return f"err: {type(e).__name__}"


async def panic() -> dict:
    """Fire every kill in parallel; return a flat report.

    Returns shape:
      {
        "sadtalker": {"office (...)": "HTTP 200", ...},
        "musetalk":  {...},
        "ollama_unload": {...},
        "started_at_monotonic": float,
        "wall_ms": int,
      }
    """
    import time as _time
    t0 = _time.monotonic()

    sadtalker_jobs = []
    musetalk_jobs = []
    for url, label in config.AVATAR_WORKER_ENDPOINTS:
        sadtalker_jobs.append((label, _post(f"{url}/sadtalker/cancel")))
        musetalk_jobs.append((label, _post(f"{url}/musetalk/cancel")))

    # Ollama "unload" = generate request with keep_alive=0. Walks ALL configured
    # ollama endpoints, not just the one the pipeline is currently using.
    ollama_jobs = []
    if config.OLLAMA_URL:
        ollama_jobs.append((config.OLLAMA_URL, _post(
            f"{config.OLLAMA_URL}/api/generate",
            {"model": config.OLLAMA_DEFAULT_MODEL, "keep_alive": 0},
        )))
    for url, label, _gpu in config.OLLAMA_ENDPOINTS:
        ollama_jobs.append((label, _post(
            f"{url}/api/generate",
            {"model": config.OLLAMA_DEFAULT_MODEL, "keep_alive": 0},
        )))

    # Run them all concurrently and collect.
    async def gather(jobs):
        results = await asyncio.gather(*[j for _, j in jobs], return_exceptions=True)
        return {label: (r if not isinstance(r, Exception) else f"err: {type(r).__name__}")
                for (label, _), r in zip(jobs, results)}

    sadtalker_res, musetalk_res, ollama_res = await asyncio.gather(
        gather(sadtalker_jobs),
        gather(musetalk_jobs),
        gather(ollama_jobs),
    )

    wall_ms = int((_time.monotonic() - t0) * 1000)
    logger.warning(
        f"EMERGENCY STOP fired — sadtalker={sadtalker_res} "
        f"musetalk={musetalk_res} ollama_unload={ollama_res} "
        f"wall_ms={wall_ms}"
    )
    return {
        "sadtalker": sadtalker_res,
        "musetalk": musetalk_res,
        "ollama_unload": ollama_res,
        "wall_ms": wall_ms,
    }
