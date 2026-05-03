"""
resources.py — unified status probe across all inference + training backends.

WHAT THIS FILE IS, FOR THE DUMMIES:
====================================

The dashboard shows a small "Resource Status" mini-panel summarizing every
backend the platform might talk to:

  - Three Ollama endpoints (OLLAMA_URL_1/2/3)
  - Together.ai (key set?)
  - Anthropic API (key set?)
  - RunPod (key set + active pod info)
  - δ² engine (probe /health)
  - Server / office / dellserver reachability + GPU memory

Each row is one bullet: green dot if all is well, amber if degraded
(reachable but missing model / no GPU), red if down or unconfigured.

WHY ITS OWN FILE:
=================

This is the only consumer of every backend at once. Putting the unified
probe in a single endpoint avoids forcing the dashboard JS to fan out
to N routes and reason about partial failures.

NETWORK BUDGET:
===============

Each probe uses a 1.5–3 s timeout. The whole panel must return in under
~5 s even if multiple endpoints are dead, so probes run concurrently
via asyncio.gather. The dashboard polls every 30 s.

PRIVACY:
========

No personal data leaves this machine. All probes are GET /health–style
liveness checks against URLs already in env vars. Together.ai and Anthropic
checks are local-only (look at the env var) — we never call out to those
APIs from this endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter

from config import (
    ANTHROPIC_API_KEY,
    D2_ENDPOINT_URL,
    OLLAMA_ENDPOINTS,
    RUNPOD_API_KEY,
    RUNPOD_ENDPOINT_URL,
    RUNPOD_POD_ID,
    TOGETHER_API_KEY,
)

logger = logging.getLogger("anamnesis.routes.resources")

router = APIRouter(prefix="/api/resources", tags=["resources"])


# ============================================================================
# Tiny in-process cache of last successful probe times (per-URL)
# ============================================================================
#
# The dashboard wants to show "last successful probe at HH:MM:SS" next to each
# Ollama dot. Stash the timestamps in process memory; reset on restart.

_LAST_OK: dict[str, float] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ts_iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ============================================================================
# Per-resource probes — each returns a dict with at minimum {ok, label}.
# ============================================================================


async def _probe_ollama_one(url: str, label: str, has_gpu: bool) -> dict[str, Any]:
    """Hit /api/version on the Ollama endpoint. ~1.5s timeout."""
    out: dict[str, Any] = {
        "url": url,
        "label": label,
        "has_gpu": has_gpu,
        "ok": False,
        "version": None,
        "last_ok": _ts_iso(_LAST_OK.get(url)),
        "error": None,
    }
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            r = await client.get(f"{url.rstrip('/')}/api/version")
            if r.status_code == 200:
                out["ok"] = True
                try:
                    out["version"] = r.json().get("version")
                except Exception:
                    pass
                _LAST_OK[url] = time.time()
                out["last_ok"] = _ts_iso(_LAST_OK[url])
            else:
                out["error"] = f"HTTP {r.status_code}"
    except httpx.RequestError as e:
        out["error"] = str(e)[:120]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:100]}"
    return out


def _probe_key(name: str, value: str | None) -> dict[str, Any]:
    """Boolean probe — is an API key present in the env?"""
    return {
        "name": name,
        "configured": bool(value),
        # Surface masked tail of the key so the user can confirm which one is loaded
        # without exposing the secret. e.g. "tgp_v1_…YFPXU"
        "masked": (value[:7] + "…" + value[-5:]) if value and len(value) > 16 else (
            "<set>" if value else None
        ),
    }


async def _probe_runpod() -> dict[str, Any]:
    """RunPod state — both env-config + active pod doc from MongoDB."""
    out: dict[str, Any] = {
        "name": "RunPod",
        "configured": bool(RUNPOD_API_KEY),
        "endpoint_url": RUNPOD_ENDPOINT_URL or None,
        "pod_id_env": RUNPOD_POD_ID or None,
        "active_pod": None,
    }
    # Best-effort active-pod read — if Mongo is unavailable, just skip it
    try:
        from routes.runpod import _get_active_pod  # avoid circular import at module load
        active = await _get_active_pod()
        if active:
            # Strip any sensitive fields before returning. Safe fields only.
            out["active_pod"] = {
                "pod_id": active.get("pod_id"),
                "gpu": active.get("gpu"),
                "profile": active.get("profile"),
                "image": active.get("image"),
                "port": active.get("port"),
                "endpoint_url": active.get("endpoint_url"),
                "created_at": active.get("created_at").isoformat()
                    if hasattr(active.get("created_at"), "isoformat") else active.get("created_at"),
            }
    except Exception as e:
        logger.debug(f"RunPod active-pod read skipped: {e}")
    return out


async def _probe_d2() -> dict[str, Any]:
    """δ² engine /health probe."""
    out: dict[str, Any] = {
        "name": "δ² engine",
        "configured": bool(D2_ENDPOINT_URL),
        "endpoint": D2_ENDPOINT_URL or None,
        "ok": False,
        "model_loaded": None,
        "training_status": None,
        "bassin_size": None,
        "error": None,
    }
    if not D2_ENDPOINT_URL:
        out["error"] = "D2_ENDPOINT_URL not set"
        return out
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{D2_ENDPOINT_URL.rstrip('/')}/health")
            if r.status_code == 200:
                out["ok"] = True
                data = r.json()
                out["model_loaded"] = data.get("model_loaded")
                out["training_status"] = data.get("training_status")
                out["bassin_size"] = data.get("bassin_size")
                _LAST_OK[D2_ENDPOINT_URL] = time.time()
                out["last_ok"] = _ts_iso(_LAST_OK[D2_ENDPOINT_URL])
            else:
                out["error"] = f"HTTP {r.status_code}"
    except Exception as e:
        out["error"] = str(e)[:120]
    return out


async def _probe_host(label: str, host: str, port: int = 22, timeout: float = 1.5) -> dict[str, Any]:
    """TCP-reach a host:port (default ssh:22). Returns ok/error and rtt_ms."""
    out: dict[str, Any] = {"label": label, "host": host, "port": port, "ok": False, "rtt_ms": None, "error": None}
    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, lambda: socket.create_connection((host, port), timeout=timeout).close()),
            timeout=timeout + 0.5,
        )
        out["ok"] = True
        out["rtt_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


async def _probe_gpu_remote(label: str, ssh_host: str) -> dict[str, Any]:
    """Run nvidia-smi over SSH on a remote host. Returns free MiB on first GPU.

    Best-effort: many things can fail (no SSH keys propagated, no nvidia-smi,
    GPU absent). Surface whatever signal we get without raising.
    """
    out: dict[str, Any] = {"label": label, "host": ssh_host, "ok": False, "free_mib": None, "total_mib": None, "error": None}
    if not shutil.which("ssh"):
        out["error"] = "ssh binary not in PATH"
        return out
    cmd = [
        "ssh", "-o", "ConnectTimeout=2", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no", ssh_host,
        "nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=4.0)
        except asyncio.TimeoutError:
            proc.kill()
            out["error"] = "ssh timeout"
            return out
        if proc.returncode != 0:
            out["error"] = (stderr.decode(errors="replace").strip() or f"rc={proc.returncode}")[:120]
            return out
        first = stdout.decode().strip().splitlines()[0] if stdout else ""
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 2:
            out["ok"] = True
            try:
                out["free_mib"] = int(parts[0])
                out["total_mib"] = int(parts[1])
            except ValueError:
                out["error"] = f"unparsable: {first[:80]}"
                out["ok"] = False
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:100]}"
    return out


# ============================================================================
# Aggregator — single endpoint the dashboard polls
# ============================================================================


@router.get("/status")
async def resources_status() -> dict[str, Any]:
    """
    Single-shot status of every inference + training backend the platform
    knows about. Designed to be polled every 30 s by the dashboard sidebar.

    Concurrent probes; total wall time bounded by the slowest individual
    probe timeout (~4 s).

    Shape:
      {
        "checked_at": "2026-05-04T...Z",
        "ollama": [{url,label,has_gpu,ok,version,last_ok,error}, ...],
        "together_ai": {name,configured,masked},
        "anthropic":   {name,configured,masked},
        "runpod":      {name,configured,endpoint_url,pod_id_env,active_pod},
        "d2_engine":   {name,configured,endpoint,ok,model_loaded,training_status,bassin_size,...},
        "hosts": [{label,host,port,ok,rtt_ms,error}, ...],
        "gpus":  [{label,host,ok,free_mib,total_mib,error}, ...],
      }
    """
    # Build the ollama probe set from config (NEVER more than the configured ones)
    ollama_probes = [
        _probe_ollama_one(url, label, has_gpu)
        for (url, label, has_gpu) in OLLAMA_ENDPOINTS
    ]

    # Hosts to TCP-probe — keep this list intentionally short, only the two
    # boxes the user actually owns and operates today (server + office aliases
    # exist in ~/.ssh/config; dellserver is THIS box).
    # Probe SSH (22) on each box to confirm reachability without depending on
    # the dashboard being up. This host (dellserver) is itself probed by
    # asking host.docker.internal:22 — works on Docker Desktop and on Linux
    # via the extra_hosts entry in docker-compose.yml.
    host_probes = [
        _probe_host("server (192.168.10.15)", "192.168.10.15", 22),
        _probe_host("dellserver (this host)", "host.docker.internal", 22),
    ]

    # GPU memory only for hosts where SSH actually works.
    gpu_probes = [
        _probe_gpu_remote("server (GPU host)", "server"),
    ]

    # Run everything concurrently
    (
        ollama_results,
        runpod_result,
        d2_result,
        host_results,
        gpu_results,
    ) = await asyncio.gather(
        asyncio.gather(*ollama_probes) if ollama_probes else asyncio.sleep(0, result=[]),
        _probe_runpod(),
        _probe_d2(),
        asyncio.gather(*host_probes),
        asyncio.gather(*gpu_probes),
        return_exceptions=False,
    )

    return {
        "checked_at": _now_iso(),
        "ollama": list(ollama_results),
        "together_ai": _probe_key("Together.ai", TOGETHER_API_KEY),
        "anthropic":   _probe_key("Anthropic API", ANTHROPIC_API_KEY),
        "runpod":      runpod_result,
        "d2_engine":   d2_result,
        "hosts":       list(host_results),
        "gpus":        list(gpu_results),
    }
