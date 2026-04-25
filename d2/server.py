"""
server.py — FastAPI service exposing the δ² engine as HTTP endpoints.

WHAT THIS FILE IS, FOR THE DUMMIES:
====================================

The Anamnesis app (in app/) talks to the δ² engine over HTTP. This file
is what the Anamnesis app talks TO. It runs in a separate Docker
container that has GPU access. The container loads the model once at
startup and serves:

    GET  /health                     → liveness + model status
    POST /generate                   → text generation, optional bassin recall
    POST /train/start                → launch a continual-learning benchmark
    POST /train/stop                 → stop the current run
    GET  /train/status               → step / loss / controller stats
    GET  /bassin/stats               → bassin distribution
    POST /bassin/query               → semantic retrieval from bassin
    GET  /runs                       → list completed benchmark runs
    GET  /runs/{run_id}              → one run's full metrics

The Anamnesis app's `app/routes/anamnesis_d2.py` proxies each of these
1-to-1. So the user-facing API surface is the union of these endpoints
re-exposed at /api/d2/*.

WHY A SEPARATE SERVICE?
========================

The Anamnesis app has no GPU. It runs on dellserver (CPU-only) where
the orchestrator lives. The d2 model needs PyTorch + CUDA. So:

    container 1 (dellserver, CPU): Anamnesis app + MongoDB + crawler
    container 2 (server/RunPod, CUDA GPU): d2 trainer + inference

They communicate over HTTP. Same architecture as the avatar workers
and the QLoRA trainer.

This file is run by `d2/Dockerfile` which installs PyTorch and the d2
dependencies, then `uvicorn d2.server:app --host 0.0.0.0 --port 3015`.

CURRENT STATE:
===============

Most endpoints are SCAFFOLDED but return "not implemented" — they need
the actual model loaded and the training subprocess wired up. This file
is here so:

    1. The HTTP surface matches what app/routes/anamnesis_d2.py expects
    2. /health works immediately so the proxy probe passes
    3. The avatar UI sees δ² as "available" once D2_ENDPOINT_URL is set
       to this service's URL

The actual generation + training implementations are TODOs marked clearly
in each handler. They depend on `d2/inference.py` (already exists in
scaffolded form) and `d2/train.py` (already exists).
"""

import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# These imports require torch + the d2 module installed; safe inside the container.
# Outside (e.g. during static analysis on the orchestrator host), they will fail —
# that's fine, this file isn't meant to run there.
try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger("d2.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

app = FastAPI(title="Anamnesis-δ² Engine", version="0.1.0")


# ============================================================================
# Service state (loaded at startup, mutated by training)
# ============================================================================

class ServiceState:
    """Holds the shared state the endpoints read from / write to."""
    model = None                 # Will hold the Transformer once loaded
    optimizer = None             # Active optimizer (AdamW / DeltaSquared / Controller)
    current_optimizer_name = None  # "adam" | "delta2" | "controller"
    training_proc: Optional[subprocess.Popen] = None  # current training subprocess
    current_run_id: Optional[str] = None
    runs_dir: Path = Path(os.environ.get("D2_RUNS_DIR", "/workspace/runs"))

    @classmethod
    def init(cls):
        cls.runs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Service runs dir: {cls.runs_dir}")


# ============================================================================
# Schemas
# ============================================================================

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 200
    enable_bassin_recall: bool = True
    uncertainty_threshold: float = 1.5
    stream: bool = True


class TrainStartRequest(BaseModel):
    optimizer: str = "controller"  # "adam" | "delta2" | "controller"
    benchmark: str = "permuted_mnist"
    tasks: int = 5
    epochs: int = 1
    seed: int = 0
    notes: Optional[str] = None


class BassinQueryRequest(BaseModel):
    context: str
    top_k: int = 5
    negation_types: Optional[list[str]] = None  # filter by Hegelian type


# ============================================================================
# Lifecycle
# ============================================================================

@app.on_event("startup")
async def on_startup():
    ServiceState.init()
    if not HAS_TORCH:
        logger.warning("torch not available — endpoints will return placeholders")
    else:
        logger.info("torch available — model loading deferred to first /generate call")


# ============================================================================
# /health — liveness probe
# ============================================================================

@app.get("/health")
async def health():
    """
    Liveness + model status. Called by:
      - Anamnesis app's GET /api/d2/status
      - Avatar's GET /api/avatar/models (probe to mark δ² as available)
      - Worker registry health checks
    """
    bassin_size = 0
    if ServiceState.optimizer is not None and hasattr(ServiceState.optimizer, "get_bassin_stats"):
        try:
            stats = ServiceState.optimizer.get_bassin_stats()
            bassin_size = stats.get("total_params", 0)
        except Exception:
            pass

    training_status = "idle"
    if ServiceState.training_proc is not None:
        if ServiceState.training_proc.poll() is None:
            training_status = "running"
        else:
            training_status = "completed" if ServiceState.training_proc.returncode == 0 else "failed"

    return {
        "status": "ok",
        "service": "d2-engine",
        "version": "0.1.0",
        "torch_available": HAS_TORCH,
        "model_loaded": ServiceState.model is not None,
        "current_optimizer": ServiceState.current_optimizer_name,
        "training_status": training_status,
        "bassin_size": bassin_size,
        "current_run_id": ServiceState.current_run_id,
    }


# ============================================================================
# /generate — text generation with optional bassin recall
# ============================================================================

@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Stream tokens from the δ² model. SSE-formatted output.

    For now this is a SCAFFOLD: it returns a single-token "not implemented"
    SSE response. The real implementation hooks up to d2/inference.py once
    the model loading code is wired (TODO).
    """
    if not HAS_TORCH:
        raise HTTPException(
            status_code=503,
            detail="torch not available — service running in scaffold mode",
        )
    # TODO: load model on first call (lazy); call d2/inference.py:generate(req).
    # For now, return a placeholder so the proxy + UI can be tested end-to-end.
    async def _scaffold_stream():
        yield f"data: {json.dumps({'token': '[d2-engine scaffold] '})}\n\n"
        yield f"data: {json.dumps({'token': 'generation not yet wired. '})}\n\n"
        yield f"data: {json.dumps({'token': 'See d2/server.py TODO.'})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    if req.stream:
        return StreamingResponse(_scaffold_stream(), media_type="text/event-stream")

    # Non-streaming: just return the assembled string
    return {
        "text": "[d2-engine scaffold] generation not yet wired.",
        "tokens_generated": 0,
        "bassin_recall_triggered": False,
    }


# ============================================================================
# /train/* — training control
# ============================================================================

@app.post("/train/start")
async def train_start(req: TrainStartRequest):
    """
    Launch a continual-learning benchmark run as a subprocess.

    Runs `python d2/experiments/continual.py --method <X> --benchmark <Y> ...`
    in the background. Returns a run_id that can be polled via /train/status.
    """
    if ServiceState.training_proc is not None and ServiceState.training_proc.poll() is None:
        raise HTTPException(
            status_code=409,
            detail=f"Training already in progress: run_id={ServiceState.current_run_id}",
        )

    run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    output_path = ServiceState.runs_dir / f"{run_id}.json"

    cmd = [
        "python", "-m", "d2.experiments.continual",
        "--method", req.optimizer,
        "--benchmark", req.benchmark,
        "--tasks", str(req.tasks),
        "--epochs", str(req.epochs),
        "--seed", str(req.seed),
        "--output", str(output_path),
    ]

    log_path = ServiceState.runs_dir / f"{run_id}.log"
    log_file = open(log_path, "w")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=os.environ.get("D2_PROJECT_DIR", "/app"),
        )
    except FileNotFoundError as e:
        log_file.close()
        raise HTTPException(status_code=500, detail=f"Failed to launch trainer: {e}")

    ServiceState.training_proc = proc
    ServiceState.current_run_id = run_id

    # Save a small "started" marker
    started_at = time.time()
    meta_path = ServiceState.runs_dir / f"{run_id}.meta.json"
    meta = {
        "run_id": run_id,
        "started_at": started_at,
        "command": cmd,
        "request": req.model_dump(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(f"Started training run {run_id}: {' '.join(cmd)}")
    return {
        "ok": True,
        "run_id": run_id,
        "started_at": started_at,
        "log_file": str(log_path),
    }


@app.post("/train/stop")
async def train_stop():
    """Terminate the current training subprocess (SIGTERM, then SIGKILL after 5s)."""
    if ServiceState.training_proc is None or ServiceState.training_proc.poll() is not None:
        return {"ok": True, "message": "No training in progress"}

    proc = ServiceState.training_proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    run_id = ServiceState.current_run_id
    ServiceState.training_proc = None
    return {"ok": True, "stopped_run_id": run_id}


@app.get("/train/status")
async def train_status():
    """Return current training subprocess state."""
    if ServiceState.training_proc is None:
        return {"status": "idle"}

    rc = ServiceState.training_proc.poll()
    if rc is None:
        status = "running"
    else:
        status = "completed" if rc == 0 else "failed"

    out = {
        "status": status,
        "run_id": ServiceState.current_run_id,
        "return_code": rc,
    }

    # Try to read partial metrics from the output file (if the trainer is
    # writing them incrementally — current continual.py only writes at end).
    if ServiceState.current_run_id:
        out_path = ServiceState.runs_dir / f"{ServiceState.current_run_id}.json"
        if out_path.exists():
            try:
                out["metrics"] = json.loads(out_path.read_text())
            except Exception:
                pass

    return out


# ============================================================================
# /bassin/* — tension reservoir inspection
# ============================================================================

@app.get("/bassin/stats")
async def bassin_stats():
    """
    Return summary statistics of the in-memory bassin.

    SCAFFOLD: until the model is actually loaded with a δ²-family
    optimizer, returns zeros. The real implementation reads from
    `ServiceState.optimizer.get_bassin_stats()` (already implemented
    in d2/optimizer.py).
    """
    if ServiceState.optimizer is None:
        return {
            "size": 0,
            "tension_distribution": {},
            "negation_type_counts": {
                "inessential_difference": 0,
                "essential_difference": 0,
                "opposition": 0,
                "annihilation": 0,
            },
            "by_layer": {},
            "warning": "no optimizer loaded yet — start a training run first",
        }
    if hasattr(ServiceState.optimizer, "get_bassin_stats"):
        return ServiceState.optimizer.get_bassin_stats()
    return {"size": 0, "warning": "current optimizer has no bassin"}


@app.post("/bassin/query")
async def bassin_query(req: BassinQueryRequest):
    """
    Retrieve tensions semantically related to a given context.

    SCAFFOLD: returns empty list. The real implementation queries
    d2/bassin.py's BassinStore (which uses MongoDB vector search over
    context embeddings).
    """
    return {
        "context": req.context,
        "top_k": req.top_k,
        "results": [],
        "warning": "bassin query not yet wired — see d2/server.py TODO",
    }


# ============================================================================
# /runs — historical training runs
# ============================================================================

@app.get("/runs")
async def list_runs():
    """List metadata for all completed runs in the runs directory."""
    runs = []
    for meta_path in sorted(ServiceState.runs_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
            run_id = meta.get("run_id")
            metrics_path = ServiceState.runs_dir / f"{run_id}.json"
            meta["completed"] = metrics_path.exists()
            runs.append(meta)
        except Exception as e:
            logger.warning(f"Could not read {meta_path}: {e}")
    return {"runs": runs, "total": len(runs)}


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Return one run's metrics + metadata."""
    meta_path = ServiceState.runs_dir / f"{run_id}.meta.json"
    metrics_path = ServiceState.runs_dir / f"{run_id}.json"
    log_path = ServiceState.runs_dir / f"{run_id}.log"

    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    out = {"run_id": run_id, "meta": json.loads(meta_path.read_text())}
    if metrics_path.exists():
        try:
            out["metrics"] = json.loads(metrics_path.read_text())
        except Exception:
            pass
    if log_path.exists():
        # Last ~50 lines of the log for quick inspection
        with open(log_path) as f:
            lines = f.readlines()[-50:]
        out["log_tail"] = "".join(lines)
    return out


# ============================================================================
# Local entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("D2_SERVICE_PORT", "3015"))
    uvicorn.run(app, host="0.0.0.0", port=port)
