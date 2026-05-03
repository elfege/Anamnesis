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


# ============================================================================
# /explain — Claude CLI–powered interpretation of any KPI / panel / result
# ============================================================================
#
# WHAT THIS DOES, FOR THE DUMMIES:
# Every help icon (?) and "Explain these results" button in the δ² dashboard
# tab POSTs here. The endpoint shells out to Claude CLI (running on the host
# via SSH from inside the container) with a prompt that includes:
#   - what the user clicked
#   - the current value of the relevant KPI(s)
#   - context (which optimizer, which benchmark, what's "good")
# Claude returns a plain-English explanation, written for someone who has
# never seen a continual-learning paper. The endpoint returns it as plain
# text or JSON.
#
# WHY CLAUDE CLI not the Anthropic API:
# CLAUDE_CLI_HOST + CLAUDE_CLI_PATH are already wired (see app/routes/chat.py).
# CLI uses your existing subscription; no API key needed in .env.

class ExplainRequest(BaseModel):
    """What the dashboard sends to /explain."""
    # What is the user looking at? Free-form, e.g. "BWT", "controller_stats",
    # "the runs table", "the bassin distribution", "the status panel".
    what: str
    # Optional: the actual current values, as JSON-serializable dict.
    # Example: {"adam": -0.106, "delta2_additive": -0.063, "gem": -0.017}
    values: dict | None = None
    # Optional: extra context the dashboard wants to pass through (e.g.
    # "method=controller", "benchmark=permuted_mnist").
    context: dict | None = None
    # Audience hint — affects how technical the answer is.
    # "dummy" → write for a 7-year-old as the user requested
    # "engineer" → ML-literate but new to this codebase
    # "researcher" → assumes familiarity with the literature
    audience: str = "dummy"
    # Length cap — keep responses snappy
    max_words: int = 200
    # If True: skip the MongoDB cache, run Claude CLI fresh, overwrite cache.
    force_regenerate: bool = False


def _explain_cache_key(req: ExplainRequest) -> str:
    """Stable SHA-256 over canonical JSON of the request inputs.

    `force_regenerate` is intentionally NOT part of the key — it's a
    control flag, not an input that should fork the cache.
    """
    import hashlib
    import json as _json
    payload = {
        "what": req.what,
        "values": req.values,
        "context": req.context,
        "audience": req.audience,
        "max_words": req.max_words,
    }
    canonical = _json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@router.post("/explain")
async def explain(req: ExplainRequest):
    """
    Generate a plain-English explanation of a δ² dashboard element.

    Caching:
        - On first call with a given (what, values, context, audience, max_words)
          tuple, runs Claude CLI (~10s) and stores the result in MongoDB
          (collection: d2_explanations).
        - Subsequent calls with the same inputs return the cached entry
          immediately (`cached: true` in response).
        - Pass `force_regenerate=true` to bypass the cache, run Claude
          fresh, and overwrite the cached entry.

    Returns: {
        "explanation": "...",
        "model": "claude-cli",
        "elapsed_ms": ...,
        "audience": "...",
        "cached": bool,
        "cache_key": "...",
        "generated_at": "ISO-8601"
    }
    """
    import json as _json
    import shlex
    import subprocess
    import time
    from datetime import datetime, timezone

    cache_key = _explain_cache_key(req)

    # ── Cache lookup (unless force_regenerate) ────────────────────
    try:
        from database import get_d2_explanations_collection
        cache_col = get_d2_explanations_collection()
    except Exception as e:
        cache_col = None
        logger.warning(f"d2 explain cache unavailable: {e}")

    if cache_col is not None and not req.force_regenerate:
        cached = await cache_col.find_one({"cache_key": cache_key})
        if cached and cached.get("explanation"):
            generated_at = cached.get("generated_at")
            return {
                "explanation": cached["explanation"],
                "model": cached.get("model", "claude-cli"),
                "elapsed_ms": 0,
                "audience": req.audience,
                "cached": True,
                "cache_key": cache_key,
                "generated_at": generated_at.isoformat() if hasattr(generated_at, "isoformat") else generated_at,
            }

    # Build the prompt. Heavily steered: short, dummy-grade, no jargon
    # without immediate translation.
    audience_directive = {
        "dummy": (
            "Write at a level a curious 12-year-old (or a busy adult who has "
            "never read an ML paper) can follow. No undefined jargon. If you "
            "must use a technical term, give a one-clause translation right "
            "after it. No equations. Use concrete analogies."
        ),
        "engineer": (
            "Write for an ML-literate engineer new to this codebase. Use "
            "standard terminology. No equations unless essential."
        ),
        "researcher": (
            "Write for a researcher familiar with continual-learning "
            "literature. Cite related work briefly."
        ),
    }.get(req.audience, "Be clear and concrete.")

    parts = [
        "You are explaining one element of a research dashboard for the "
        "δ² project — a continual-learning optimizer that retains structured "
        "negatives (gradient frictions classified into 4 Hegelian negation "
        "types) in a 'tension reservoir' (bassin) instead of discarding them.",
        "",
        f"The user is looking at: **{req.what}**.",
        "",
    ]
    if req.values:
        parts.append("Current values:")
        parts.append("```json")
        parts.append(_json.dumps(req.values, indent=2)[:1500])
        parts.append("```")
        parts.append("")
    if req.context:
        parts.append("Context:")
        parts.append("```json")
        parts.append(_json.dumps(req.context, indent=2)[:1500])
        parts.append("```")
        parts.append("")
    parts += [
        f"Audience: {audience_directive}",
        f"Length: {req.max_words} words MAX. Aim for less.",
        "",
        "Tell the user, in plain language:",
        "1. What this thing IS (one sentence).",
        "2. What the value(s) MEAN concretely (is it good? bad? compared to what?).",
        "3. What action the user could take if they cared.",
        "",
        "Skip preamble. Start with the answer.",
    ]
    prompt = "\n".join(parts)

    # Invoke claude CLI on the host (via SSH from inside the container).
    # This mirrors the pattern in routes/chat.py:_stream_claude_cli.
    cli_path = os.environ.get("CLAUDE_CLI_PATH", "claude")
    cli_host = os.environ.get("CLAUDE_CLI_HOST", "host.docker.internal")
    cli_user = os.environ.get("SSH_USER", os.environ.get("USER", "elfege"))

    # Use -p (print) flag, no --output-format so we get plain text
    # --tools "" to disable tool use (we just want a direct answer)
    cmd_remote = f'{cli_path} -p --tools ""'

    explanation_text: str | None = None
    elapsed_ms: int = 0

    try:
        # Try local execution first (when running directly on the host).
        # Otherwise fall back to SSH (when running inside the container).
        local_available = os.path.exists(cli_path) or subprocess.run(
            ["which", cli_path], capture_output=True, text=True
        ).returncode == 0

        if local_available:
            t0 = time.time()
            result = subprocess.run(
                [cli_path, "-p", "--tools", ""],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=60,
            )
            elapsed_ms = int((time.time() - t0) * 1000)
            if result.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Claude CLI failed: {result.stderr[:200]}",
                )
            explanation_text = result.stdout.strip()
        else:
            # SSH fallback (inside container)
            from routes.files import _ssh_client
            client = _ssh_client(cli_host, cli_user)
            t0 = time.time()
            stdin, stdout, stderr = client.exec_command(cmd_remote, timeout=60)
            stdin.write(prompt.encode("utf-8"))
            stdin.channel.shutdown_write()
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            elapsed_ms = int((time.time() - t0) * 1000)
            if not out.strip():
                raise HTTPException(
                    status_code=500,
                    detail=f"Claude CLI returned empty. stderr: {err[:200]}",
                )
            explanation_text = out.strip()

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Claude CLI timed out after 60s",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("explain endpoint failed")
        raise HTTPException(status_code=500, detail=f"explain failed: {e}")

    # ── Persist to MongoDB cache (best-effort) ────────────────────
    generated_at = datetime.now(timezone.utc)
    if cache_col is not None and explanation_text:
        try:
            await cache_col.update_one(
                {"cache_key": cache_key},
                {"$set": {
                    "cache_key": cache_key,
                    "what": req.what,
                    "values": req.values,
                    "context": req.context,
                    "audience": req.audience,
                    "max_words": req.max_words,
                    "explanation": explanation_text,
                    "model": "claude-cli",
                    "generated_at": generated_at,
                }},
                upsert=True,
            )
        except Exception as e:
            logger.warning(f"d2 explain cache write failed: {e}")

    return {
        "explanation": explanation_text,
        "model": "claude-cli",
        "elapsed_ms": elapsed_ms,
        "audience": req.audience,
        "cached": False,
        "cache_key": cache_key,
        "generated_at": generated_at.isoformat(),
    }
