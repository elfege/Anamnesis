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

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Service start time — used by /host to report uptime.
_SERVICE_START_TS = time.time()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Sibling-import helper: makes `from inference import …` work for the d2/ tree
# (mirrors the pattern in d2/experiments/continual.py:142).
sys.path.insert(0, str(Path(__file__).parent))

# These imports require torch + the d2 module installed; safe inside the container.
# Outside (e.g. during static analysis on the orchestrator host), they will fail —
# that's fine, this file isn't meant to run there.
try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# tiktoken (GPT-2 BPE) for tokenization at /generate time. Optional: if missing,
# /generate returns a clear 503 instead of hard-crashing the service.
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

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
    checkpoints_dir: Path = Path(os.environ.get("D2_CHECKPOINTS_DIR", "/workspace/checkpoints"))

    # /generate state — lazy-initialized on first call
    inference_engine = None      # d2.inference.D2InferenceEngine, lazy-loaded
    tokenizer = None             # tiktoken GPT-2 encoder

    @classmethod
    def init(cls):
        cls.runs_dir.mkdir(parents=True, exist_ok=True)
        cls.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Service runs dir: {cls.runs_dir}")
        logger.info(f"Service checkpoints dir: {cls.checkpoints_dir}")

    @classmethod
    def find_default_checkpoint(cls) -> Optional[Path]:
        """
        Resolve which checkpoint /generate should load by default.

        Priority:
          1. D2_CHECKPOINT_PATH env var (explicit override)
          2. {checkpoints_dir}/best.pt
          3. Most recent .pt file under {checkpoints_dir}
        """
        env_path = os.environ.get("D2_CHECKPOINT_PATH")
        if env_path and Path(env_path).exists():
            return Path(env_path)

        best = cls.checkpoints_dir / "best.pt"
        if best.exists():
            return best

        candidates = sorted(
            cls.checkpoints_dir.rglob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    @classmethod
    def get_or_create_inference_engine(cls):
        """
        Lazy-init the D2InferenceEngine on first /generate call.

        Returns the engine if a checkpoint is available, raises HTTPException(503)
        with a helpful message if no model has been trained yet.
        """
        if cls.inference_engine is not None and cls.inference_engine.is_loaded():
            return cls.inference_engine

        ckpt = cls.find_default_checkpoint()
        if ckpt is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No d² checkpoint found. Train one first: "
                    "POST /train/start with a language-model dataset, "
                    f"or drop a .pt file into {cls.checkpoints_dir}."
                ),
            )

        # Lazy import — avoids paying torch/inference cost at startup
        from inference import D2InferenceEngine

        engine = D2InferenceEngine(
            checkpoint_path=str(ckpt),
            device="auto",
            entropy_threshold=4.0,
            enable_bassin_recall=True,
        )
        if not engine.load():
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load checkpoint at {ckpt}",
            )

        cls.inference_engine = engine
        logger.info(f"Inference engine ready: {ckpt}")
        return engine

    @classmethod
    def get_or_create_tokenizer(cls):
        """Lazy-init the GPT-2 BPE tokenizer (matches WikiText-103 prep script)."""
        if cls.tokenizer is not None:
            return cls.tokenizer
        if not HAS_TIKTOKEN:
            raise HTTPException(
                status_code=503,
                detail="tiktoken not installed in this container — cannot tokenize prompts.",
            )
        cls.tokenizer = tiktoken.get_encoding("gpt2")
        return cls.tokenizer


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


class TrainLMRequest(BaseModel):
    """Launch a language-model training run (d2/train.py on a tokenized .bin dataset)."""
    optimizer: str = "delta2"            # "adam" | "delta2"
    dataset: str = "wikitext"            # subdir under data_dir
    max_steps: int = 5000
    batch_size: Optional[int] = None     # default from config
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_embd: Optional[int] = None
    block_size: Optional[int] = None
    d2_eta: Optional[float] = None
    d2_additive_mode: Optional[bool] = None  # path B (true) vs path A (false). Default: true.
    d2_base_lr: Optional[float] = None       # gradient-descent step size when additive_mode=true
    experiment: Optional[str] = None     # checkpoint subdir name (auto if None)
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
# /host — host/machine introspection probe
# ============================================================================
#
# Returned by the unified Machines panel on the Anamnesis dashboard. Honest,
# best-effort: every field is optional, missing fields surface as null on
# the UI rather than as fake zeros. No external calls — only local probes.

@app.get("/host")
async def host_info():
    """
    Hostname, basic GPU info, role hint, and uptime for this δ² container.

    The Anamnesis app's /api/machines endpoint calls this to populate one
    card in the Machines tab. It deliberately does NOT include training
    metrics (those live in /train/status) or model state (/health).
    """
    info: dict = {
        "service": "d2-engine",
        "service_version": "0.1.0",
        "hostname": socket.gethostname(),
        "uptime_s": int(time.time() - _SERVICE_START_TS),
        "torch_available": HAS_TORCH,
        "ip": None,
        "gpus": [],
        "roles": ["d2-engine"],
    }

    # Best-effort: outbound IP via UDP-trick (does NOT actually send a packet)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            info["ip"] = s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        pass

    # GPU probe — nvidia-smi, fallback to nothing
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            stdout = b""
        for line in stdout.decode(errors="replace").strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                try:
                    info["gpus"].append({
                        "name": parts[0],
                        "vram_total_mib": int(float(parts[1])),
                        "vram_free_mib":  int(float(parts[2])),
                        "util_pct":       int(float(parts[3])),
                        "temp_c":         int(float(parts[4])),
                    })
                except ValueError:
                    pass
    except FileNotFoundError:
        info["gpu_probe_error"] = "nvidia-smi not in PATH"
    except Exception as e:
        info["gpu_probe_error"] = f"{type(e).__name__}: {str(e)[:120]}"

    # Active training run (if any) — surface run id + status from ServiceState
    try:
        if ServiceState.training_proc is not None:
            running = ServiceState.training_proc.poll() is None
            info["active_training"] = {
                "run_id": ServiceState.current_run_id,
                "status": "running" if running else (
                    "completed" if ServiceState.training_proc.returncode == 0 else "failed"
                ),
            }
    except Exception:
        pass

    return info


# ============================================================================
# /generate — text generation with optional bassin recall
# ============================================================================

@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Stream tokens from the δ² model.

    Lifecycle:
      1. Lazy-load tokenizer + inference engine on first request.
      2. Tokenize prompt with GPT-2 BPE (matches WikiText-103 prep).
      3. Generate up to `max_tokens` via D2InferenceEngine.generate.
      4. Decode + emit per-token SSE events when stream=True;
         otherwise return the full assembled string.

    Errors:
      503 — torch missing / no checkpoint trained / tiktoken missing
      500 — checkpoint load failed
    """
    if not HAS_TORCH:
        raise HTTPException(
            status_code=503,
            detail="torch not available — service running in scaffold mode",
        )

    engine = ServiceState.get_or_create_inference_engine()
    enc = ServiceState.get_or_create_tokenizer()

    prompt_ids_list = enc.encode_ordinary(req.prompt or "")
    if not prompt_ids_list:
        prompt_ids_list = [enc.eot_token] if hasattr(enc, "eot_token") else [50256]

    prompt_tensor = torch.tensor([prompt_ids_list], dtype=torch.long)
    prompt_len = prompt_tensor.size(1)

    # Update entropy threshold per request (allows UI to tune sensitivity)
    engine.entropy_threshold = req.uncertainty_threshold
    engine.enable_bassin_recall = req.enable_bassin_recall

    if not req.stream:
        # Synchronous path: generate everything, then return
        generated_ids, stats = engine.generate(
            prompt_tensor,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )
        new_tokens = generated_ids[0, prompt_len:].tolist()
        text = enc.decode(new_tokens)
        return {
            "text": text,
            "tokens_generated": stats.get("tokens_generated", len(new_tokens)),
            "bassin_recall_triggered": stats.get("bassin_recalls", 0) > 0,
            "entropy_mean": stats.get("entropy_mean"),
            "entropy_max": stats.get("entropy_max"),
            "pct_uncertain": stats.get("pct_uncertain"),
        }

    # Streaming path: incremental generation with per-token SSE
    async def _stream_tokens():
        # Generate one token at a time so we can stream as we go.
        # We re-implement the engine's loop to interleave async yields.
        idx = prompt_tensor.to(engine.device)
        bassin_recalls = 0
        emitted_so_far = ""

        try:
            for i in range(req.max_tokens):
                with torch.no_grad():
                    idx_cond = idx if idx.size(1) <= engine.model.config.block_size \
                        else idx[:, -engine.model.config.block_size:]
                    logits, _ = engine.model(idx_cond)
                    logits = logits[:, -1, :]

                    # Entropy / bassin-recall hook (logged only — see inference.py TODO)
                    from bassin import compute_entropy
                    entropy = compute_entropy(logits[0])
                    if entropy > engine.entropy_threshold and engine.enable_bassin_recall:
                        bassin_recalls += 1

                    logits = logits / max(req.temperature, 1e-5)
                    if req.top_k:
                        v, _ = torch.topk(logits, min(req.top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float("Inf")

                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat((idx, idx_next), dim=1)

                # Decode incremental delta. tiktoken decode operates on the full
                # sequence then we substring; per-token decode is unsafe for
                # multi-byte BPE pieces (would emit replacement chars).
                new_tokens = idx[0, prompt_len:].tolist()
                full_decoded = enc.decode(new_tokens)
                delta = full_decoded[len(emitted_so_far):]
                emitted_so_far = full_decoded

                if delta:
                    payload = {"token": delta, "entropy": float(entropy)}
                    yield f"data: {json.dumps(payload)}\n\n"
                    # Cooperative yield so the SSE framing flushes
                    await asyncio.sleep(0)

            yield f"data: {json.dumps({'done': True, 'tokens_generated': req.max_tokens, 'bassin_recalls': bassin_recalls})}\n\n"
        except Exception as e:
            logger.exception("generate() streaming failure")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(_stream_tokens(), media_type="text/event-stream")


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


@app.post("/train/lm/start")
async def train_lm_start(req: TrainLMRequest):
    """
    Launch a language-model training run (next-token cross-entropy on a
    tokenized .bin dataset, e.g. WikiText-103).

    Runs `python -m d2.train --optimizer X --dataset Y --output-dir Z ...`.
    Checkpoints land in {checkpoints_dir}/{experiment}/best.pt — which is
    what /generate auto-loads.
    """
    if ServiceState.training_proc is not None and ServiceState.training_proc.poll() is None:
        raise HTTPException(
            status_code=409,
            detail=f"Training already in progress: run_id={ServiceState.current_run_id}",
        )

    run_id = f"lm-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    experiment = req.experiment or f"{req.optimizer}_{req.dataset}_{run_id[3:]}"

    # train.py reads --data-dir; the prep script writes under /app/d2/data/{dataset}/
    data_dir = os.environ.get("D2_DATA_DIR", "/app/d2/data")

    cmd = [
        "python", "-m", "d2.train",
        "--optimizer", req.optimizer,
        "--dataset", req.dataset,
        "--max-steps", str(req.max_steps),
        "--data-dir", data_dir,
        "--output-dir", str(ServiceState.checkpoints_dir),
        "--experiment", experiment,
    ]
    if req.batch_size is not None:
        cmd += ["--batch-size", str(req.batch_size)]
    if req.n_layer is not None:
        cmd += ["--n-layer", str(req.n_layer)]
    if req.n_head is not None:
        cmd += ["--n-head", str(req.n_head)]
    if req.n_embd is not None:
        cmd += ["--n-embd", str(req.n_embd)]
    if req.block_size is not None:
        cmd += ["--block-size", str(req.block_size)]
    if req.d2_eta is not None:
        cmd += ["--d2-eta", str(req.d2_eta)]
    if req.d2_additive_mode is not None:
        cmd += ["--d2-additive-mode", "true" if req.d2_additive_mode else "false"]
    if req.d2_base_lr is not None:
        cmd += ["--d2-base-lr", str(req.d2_base_lr)]

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

    started_at = time.time()
    meta_path = ServiceState.runs_dir / f"{run_id}.meta.json"
    meta = {
        "run_id": run_id,
        "kind": "language_model",
        "experiment": experiment,
        "checkpoint_dir": str(ServiceState.checkpoints_dir / experiment),
        "started_at": started_at,
        "command": cmd,
        "request": req.model_dump(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # Invalidate cached inference engine so /generate picks up the new checkpoint
    ServiceState.inference_engine = None

    logger.info(f"Started LM training run {run_id}: {' '.join(cmd)}")
    return {
        "ok": True,
        "run_id": run_id,
        "experiment": experiment,
        "checkpoint_dir": str(ServiceState.checkpoints_dir / experiment),
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
# /checkpoints/personal — read-only scan of personal-track checkpoints
# ============================================================================
#
# This endpoint scans the host bind-mount at $D2_PERSONAL_CHECKPOINTS_DIR
# (default /workspace/checkpoints_personal) for personal_* run dirs and
# returns one summary object per run. It is read-only and never logs to
# any shared collection — these are personal-track artifacts and stay on
# the host they were trained on.
#
# Per the two-track architecture (see README_canonical_two_tracks.md):
#   - Bench checkpoints   → /workspace/checkpoints       (publishable)
#   - Personal checkpoints → /workspace/checkpoints_personal (private)
#
# The dashboard's /api/d2/personal-runs proxies this 1:1.
# ============================================================================

PERSONAL_CHECKPOINTS_DIR = Path(
    os.environ.get("D2_PERSONAL_CHECKPOINTS_DIR", "/workspace/checkpoints_personal")
)


def _scan_personal_run(run_dir: Path) -> Optional[dict]:
    """Read one personal_* checkpoint dir and return its summary, or None on error."""
    try:
        cfg_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.jsonl"
        cfg: dict = {}
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception as e:
                logger.warning(f"Bad config.json in {run_dir}: {e}")

        steps = 0
        tasks: set = set()
        final_train: Optional[float] = None
        final_val: Optional[float] = None
        if metrics_path.exists():
            last_line = None
            with open(metrics_path) as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if "task_idx" in rec:
                        tasks.add(rec["task_idx"])
                    last_line = rec
            if last_line is not None:
                steps = int(last_line.get("global_step", 0)) + 1
                tv = last_line.get("train_loss")
                vv = last_line.get("val_loss")
                final_train = float(tv) if tv is not None else None
                final_val = float(vv) if vv is not None else None

        # Completion time: prefer the lora_adapter_final dir mtime (written
        # only at successful end of run); fall back to metrics.jsonl mtime.
        adapter_dir = run_dir / "lora_adapter_final"
        if adapter_dir.exists():
            mtime = adapter_dir.stat().st_mtime
        elif metrics_path.exists():
            mtime = metrics_path.stat().st_mtime
        else:
            mtime = run_dir.stat().st_mtime

        from datetime import datetime, timezone as _tz
        completed_at = (
            datetime.fromtimestamp(mtime, tz=_tz.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

        return {
            "experiment": cfg.get("experiment") or run_dir.name,
            "base_model": cfg.get("base_model"),
            "optimizer": cfg.get("optimizer"),
            "steps": steps,
            "tasks": len(tasks),
            "final_train_loss": final_train,
            "final_val_loss": final_val,
            "completed_at": completed_at,
            "has_adapter": adapter_dir.exists(),
            "config": {
                k: cfg.get(k) for k in (
                    "lr", "d2_eta", "batch_size", "block_size",
                    "lora_r", "lora_alpha", "lora_target_modules",
                    "steps_per_task", "seed",
                ) if k in cfg
            },
        }
    except Exception as exc:
        logger.warning(f"Failed to scan {run_dir}: {exc}")
        return None


@app.get("/checkpoints/personal")
async def list_personal_checkpoints():
    """
    Scan $D2_PERSONAL_CHECKPOINTS_DIR for personal_* run subdirs and return
    a summary per run. Read-only, no MongoDB writes, no shared logging.

    Returns:
      {
        "dir": "/workspace/checkpoints_personal",
        "exists": bool,
        "runs": [ {experiment, base_model, optimizer, steps, tasks,
                   final_train_loss, final_val_loss, completed_at, ...}, ... ],
        "total": int,
      }
    """
    out = {
        "dir": str(PERSONAL_CHECKPOINTS_DIR),
        "exists": PERSONAL_CHECKPOINTS_DIR.exists(),
        "runs": [],
        "total": 0,
    }
    if not PERSONAL_CHECKPOINTS_DIR.exists():
        return out

    runs = []
    for sub in sorted(PERSONAL_CHECKPOINTS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        if not sub.name.startswith("personal_"):
            # Enforce the naming convention from README_canonical_two_tracks.md
            continue
        summary = _scan_personal_run(sub)
        if summary:
            runs.append(summary)
    # Sort newest-first by completed_at
    runs.sort(key=lambda r: r.get("completed_at") or "", reverse=True)
    out["runs"] = runs
    out["total"] = len(runs)
    return out


# ============================================================================
# Local entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("D2_SERVICE_PORT", "3015"))
    uvicorn.run(app, host="0.0.0.0", port=port)
