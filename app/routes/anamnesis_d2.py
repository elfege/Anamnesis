"""
anamnesis_d2.py — Anamnesis app routes for the δ² engine.

WHAT THIS FILE IS, FOR THE DUMMIES:
====================================

The δ² code itself (optimizer, bassin, controller, training loop) lives
in the d2/ directory of this repo. That code RUNS inside a trainer
container — a separate GPU-attached container, NOT the Anamnesis app.

The Anamnesis app talks to that trainer container over HTTP. This file
holds the routes that:

  1. Forward inference requests to the trainer's /generate endpoint
     (when the user picks "δ²" as backend in the avatar UI)
  2. Forward training-control requests (start, stop, status)
  3. Query bassin statistics for the dashboard

It's a thin proxy layer. All the heavy lifting (forward passes,
gradient updates, bassin storage) happens on the GPU box. The
Anamnesis app just routes requests and handles the UI.

WHY SEPARATE THE ROUTES FROM anamnesis_gpt.py?
================================================

`anamnesis_gpt.py` already proxies to the existing trainer (the
QLoRA-on-Qwen-1.5B "demo, low quality" fine-tune). That code targets
a single endpoint (`/generate`) on the existing trainer container.

δ² is structurally different:
- Different model architecture (custom transformer, not Qwen)
- Different optimizer (DeltaSquaredOptimizer, AdamW, or DialecticalController)
- Different bassin query API (the bassin is a first-class entity)
- Different training-launch API (continual-learning benchmarks, not just SFT)

Could we shoehorn δ² into anamnesis_gpt.py? Yes. Would it be a mess?
Also yes. Separate route file = clean separation of concerns.


CURRENT STATE:
===============

This file is SCAFFOLDED but not yet wired to a running trainer. The δ²
trainer container itself doesn't exist yet — we have the code in d2/
but no Dockerfile that runs it as a service. That's the next step
(make `trainers/Dockerfile` run d2/inference.py as a FastAPI service).

For now, this file exists so:
  - The route surface is defined and reviewable
  - The avatar UI's `δ²` backend option has a target endpoint
  - We can mock responses while the trainer is being built
"""

import logging
import os
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("anamnesis.routes.d2")

router = APIRouter(prefix="/api/d2", tags=["d2"])


# ============================================================================
# Configuration
# ============================================================================

# δ² inference endpoint (a trainer container that loads the d2/ model
# and serves /generate, /bassin/*, /train/*). Set via env when deployed.
D2_ENDPOINT_URL = os.environ.get("D2_ENDPOINT_URL", "")

# How long to wait for the trainer's responses
D2_TIMEOUT_SECONDS = float(os.environ.get("D2_TIMEOUT_SECONDS", "120"))


# ============================================================================
# Pydantic schemas
# ============================================================================

class GenerateRequest(BaseModel):
    """Request body for δ² text generation."""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 200
    # If True and the model is uncertain at any token, recall from the bassin
    # and re-generate with the recalled context as contrastive guidance.
    enable_bassin_recall: bool = True
    # Threshold for "uncertain" — entropy of next-token distribution above this
    # triggers bassin recall. 1.5 is a reasonable default for vocab ~50k.
    uncertainty_threshold: float = 1.5


class TrainStartRequest(BaseModel):
    """Request body for launching a training run."""
    optimizer: str = "controller"  # "adam" | "delta2" | "controller"
    benchmark: str = "permuted_mnist"  # for the standard CL benchmarks
    tasks: int = 5
    epochs: int = 1
    seed: int = 0
    notes: Optional[str] = None  # free-form notes for the run


# ============================================================================
# Helpers
# ============================================================================

def _require_endpoint():
    """Return the D2 trainer endpoint URL or raise 503 if not configured."""
    if not D2_ENDPOINT_URL:
        raise HTTPException(
            status_code=503,
            detail=(
                "D2_ENDPOINT_URL not configured. δ² trainer is not yet "
                "deployed. Set D2_ENDPOINT_URL in .env once the trainer "
                "container is running."
            ),
        )
    return D2_ENDPOINT_URL.rstrip("/")


# ============================================================================
# Status / Health
# ============================================================================

@router.get("/status")
async def status():
    """
    Return whether the δ² trainer is configured and reachable.

    Shape:
      {
        "configured": bool,                        # is D2_ENDPOINT_URL set?
        "reachable":  bool,                        # /health responds?
        "endpoint":   str | null,
        "model_loaded": bool | null,               # from /health response
        "current_optimizer": str | null,           # adam | delta2 | controller
        "training_status": str | null,             # idle | running | failed
        "bassin_size": int | null,                 # number of stored frictions
      }
    """
    out = {
        "configured": bool(D2_ENDPOINT_URL),
        "reachable": False,
        "endpoint": D2_ENDPOINT_URL or None,
        "model_loaded": None,
        "current_optimizer": None,
        "training_status": None,
        "bassin_size": None,
    }
    if not D2_ENDPOINT_URL:
        return out

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(f"{D2_ENDPOINT_URL.rstrip('/')}/health")
            if r.status_code == 200:
                out["reachable"] = True
                data = r.json()
                out["model_loaded"] = data.get("model_loaded")
                out["current_optimizer"] = data.get("current_optimizer")
                out["training_status"] = data.get("training_status")
                out["bassin_size"] = data.get("bassin_size")
    except Exception as exc:
        logger.warning(f"D2 status probe failed: {exc}")
    return out


# ============================================================================
# Inference
# ============================================================================

@router.post("/generate")
async def generate(req: GenerateRequest):
    """
    Forward a generation request to the δ² trainer.

    The trainer's /generate endpoint streams tokens via SSE. This route is
    BLOCKING (returns the full response). For streaming, use the
    avatar WebSocket which proxies through `app/avatar/llm.py`.

    Body fields are passed through as-is. The trainer chooses whether to
    invoke bassin recall based on the `enable_bassin_recall` flag and
    its own uncertainty threshold.
    """
    base = _require_endpoint()
    body = req.model_dump()
    try:
        async with httpx.AsyncClient(timeout=D2_TIMEOUT_SECONDS) as client:
            r = await client.post(f"{base}/generate", json=body)
            if r.status_code != 200:
                raise HTTPException(
                    status_code=r.status_code,
                    detail=f"Trainer /generate returned {r.status_code}: {r.text[:200]}",
                )
            return r.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Trainer unreachable: {e}")


# ============================================================================
# Training control
# ============================================================================

@router.post("/train/start")
async def train_start(req: TrainStartRequest):
    """
    Ask the trainer to launch a training run.

    The trainer kicks off `python d2/experiments/continual.py` with the
    specified args and returns a `run_id`. Poll /train/status to check
    progress.

    Returns:
        {"ok": True, "run_id": "...", "started_at": "..."}
    """
    base = _require_endpoint()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{base}/train/start", json=req.model_dump())
            if r.status_code != 200:
                raise HTTPException(
                    status_code=r.status_code,
                    detail=f"Trainer /train/start returned {r.status_code}: {r.text[:200]}",
                )
            return r.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Trainer unreachable: {e}")


@router.post("/train/stop")
async def train_stop():
    """Ask the trainer to stop the current training run."""
    base = _require_endpoint()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{base}/train/stop")
            if r.status_code != 200:
                raise HTTPException(
                    status_code=r.status_code,
                    detail=f"Trainer /train/stop returned {r.status_code}: {r.text[:200]}",
                )
            return r.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Trainer unreachable: {e}")


@router.get("/train/status")
async def train_status():
    """
    Return current training state.

    Shape (forwarded from trainer):
      {
        "status": "idle" | "running" | "completed" | "failed",
        "run_id": "...",
        "started_at": "...",
        "step": int,                # current training step
        "total_steps": int,
        "current_loss": float,
        "controller_stats": {...},  # only if optimizer=controller
        "bassin_stats": {...},
      }
    """
    base = _require_endpoint()
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(f"{base}/train/status")
            if r.status_code != 200:
                raise HTTPException(
                    status_code=r.status_code,
                    detail=f"Trainer /train/status returned {r.status_code}",
                )
            return r.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Trainer unreachable: {e}")


# ============================================================================
# Bassin inspection
# ============================================================================

@router.get("/bassin/stats")
async def bassin_stats():
    """
    Return summary statistics of the tension reservoir.

    Used by the dashboard to render bassin behavior over time.

    Shape:
      {
        "size": int,                          # number of stored frictions
        "tension_distribution": {...},        # histogram of tension scores
        "negation_type_counts": {              # how many of each Hegelian type
          "inessential_difference": int,
          "essential_difference": int,
          "opposition": int,
          "annihilation": int,
        },
        "by_layer": {layer_name: {...}, ...}, # per-layer breakdown
      }
    """
    base = _require_endpoint()
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(f"{base}/bassin/stats")
            if r.status_code != 200:
                raise HTTPException(
                    status_code=r.status_code,
                    detail=f"Trainer /bassin/stats returned {r.status_code}",
                )
            return r.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Trainer unreachable: {e}")


@router.post("/bassin/query")
async def bassin_query(body: dict):
    """
    Query the bassin for tensions semantically related to a given context.

    Used at inference time when the model is uncertain — the controller
    pulls the most relevant past frictions and uses them as contrastive
    context for re-generation.

    Body:
      {
        "context": "...",         # text to find related tensions for
        "top_k": 5,               # how many to retrieve
        "negation_types": [...],  # optional filter, e.g. ["opposition", "annihilation"]
      }
    """
    base = _require_endpoint()
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.post(f"{base}/bassin/query", json=body)
            if r.status_code != 200:
                raise HTTPException(
                    status_code=r.status_code,
                    detail=f"Trainer /bassin/query returned {r.status_code}",
                )
            return r.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Trainer unreachable: {e}")


# ============================================================================
# Comparison runs (read-only — historical results)
# ============================================================================

@router.get("/runs")
async def list_runs():
    """
    Return historical training runs (Adam vs δ² vs controller comparisons).

    Each run was launched via /train/start and produced a metrics JSON.
    The trainer keeps a list; this proxies to it.

    Used by the dashboard to render the "comparison plot" tab.
    """
    base = _require_endpoint()
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(f"{base}/runs")
            if r.status_code != 200:
                raise HTTPException(
                    status_code=r.status_code,
                    detail=f"Trainer /runs returned {r.status_code}",
                )
            return r.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Trainer unreachable: {e}")


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Return one historical run's full metrics."""
    base = _require_endpoint()
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(f"{base}/runs/{run_id}")
            if r.status_code != 200:
                raise HTTPException(
                    status_code=r.status_code,
                    detail=f"Trainer /runs/{run_id} returned {r.status_code}",
                )
            return r.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Trainer unreachable: {e}")
