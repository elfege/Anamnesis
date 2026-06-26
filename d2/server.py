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
    # Start the LoRA idle auto-unload watchdog (belt-and-suspenders against
    # the browser-side beforeunload handler missing).
    try:
        asyncio.create_task(_idle_unload_loop())
        logger.info(
            f"LoRA idle auto-unload loop started (threshold={LORA_IDLE_UNLOAD_SECONDS}s)"
        )
    except Exception as exc:
        logger.warning(f"could not start LoRA idle loop: {exc}")


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
        # model_loaded is true if EITHER a full d² checkpoint OR a LoRA adapter is live.
        "model_loaded": (ServiceState.model is not None) or (_ACTIVE_ADAPTER is not None),
        "current_optimizer": ServiceState.current_optimizer_name,
        "training_status": training_status,
        "bassin_size": bassin_size,
        "current_run_id": ServiceState.current_run_id,
        "active_lora_adapter": _ACTIVE_ADAPTER,
        "loaded_lora_count": len(_LOADED_ADAPTERS),
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
        "cpu": None,
        "ram": None,
        "gpus": [],
        "roles": ["d2-engine"],
    }

    # CPU + RAM via psutil (lightweight, pure-Python with C bindings).
    # 1-second sample on a worker thread so the event loop stays responsive.
    try:
        import os as _os
        import psutil  # type: ignore

        loop = asyncio.get_event_loop()
        cpu_pct = await loop.run_in_executor(None, lambda: psutil.cpu_percent(interval=1.0))
        cores = psutil.cpu_count(logical=True) or 0
        try:
            la1, la5, _la15 = _os.getloadavg()
        except (OSError, AttributeError):
            la1, la5 = None, None
        info["cpu"] = {
            "percent": round(cpu_pct, 1),
            "cores": cores,
            "load_1": round(la1, 2) if la1 is not None else None,
            "load_5": round(la5, 2) if la5 is not None else None,
        }
        vm = psutil.virtual_memory()
        info["ram"] = {
            "used_gb":  round(vm.used  / (1024 ** 3), 2),
            "total_gb": round(vm.total / (1024 ** 3), 2),
            "percent":  round(vm.percent, 1),
        }
    except ImportError:
        info["cpu_probe_error"] = "psutil not installed"
    except Exception as e:
        info["cpu_probe_error"] = f"{type(e).__name__}: {str(e)[:120]}"

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

    # GPU probe — nvidia-smi (rich query: util, temp, power, used/free/total).
    # Patch 2026-06-08 (per sibling MSG-321): previously stderr was DEVNULL and
    # returncode was ignored, so an NVML driver/library mismatch (exit 255 with
    # empty stdout) looked indistinguishable from "no GPU". Now we capture
    # stderr and surface the actual failure as gpu_probe_error.
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,"
            "utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stderr = b""
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=2.0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            stdout = b""
        for line in stdout.decode(errors="replace").strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                try:
                    total = int(float(parts[1]))
                    used  = int(float(parts[2]))
                    free  = int(float(parts[3]))
                    info["gpus"].append({
                        "name": parts[0],
                        "vram_total_mib": total,
                        "vram_used_mib":  used,
                        "vram_free_mib":  free,
                        "vram_percent":   round(100.0 * used / total, 1) if total else None,
                        "util_pct":       int(float(parts[4])),
                        "temp_c":         int(float(parts[5])),
                        "power_w":        int(float(parts[6])) if parts[6] not in ("[Not Supported]", "N/A") else None,
                    })
                except ValueError:
                    pass
        # If nvidia-smi exited non-zero AND parsed no GPUs, the probe failed —
        # don't conflate that with "no GPU on this host." Surface the actual
        # stderr line so the dashboard shows the real reason (e.g. driver/library
        # version mismatch from an upgrade without reboot).
        if not info["gpus"] and proc.returncode not in (0, None) and "gpu_probe_error" not in info:
            stderr_lines = stderr.decode(errors="replace").strip().splitlines()
            info["gpu_probe_error"] = stderr_lines[0] if stderr_lines else f"nvidia-smi exited {proc.returncode}"
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
    global _LAST_INFERENCE_TS
    if not HAS_TORCH:
        raise HTTPException(
            status_code=503,
            detail="torch not available — service running in scaffold mode",
        )

    # ── LoRA path: if a PEFT adapter is loaded, use it ────────────
    if _ACTIVE_ADAPTER is not None and _ACTIVE_ADAPTER in _LOADED_ADAPTERS:
        return await _generate_via_lora(req)

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


# ============================================================================
# LoRA hot-loading — load/unload PEFT adapters for chat-time inference
# ============================================================================
#
# These endpoints let the dashboard's "Load for chat" buttons (Personal
# Benchmarks panel) wrap the gpt2-medium base + a trained LoRA adapter
# into a single live model that /generate uses, then unload it again
# when the user closes /chat (sendBeacon).
#
# State model:
#   _LOADED_ADAPTERS  : {adapter_id: {model, tokenizer, base_model, adapter_path, loaded_at}}
#   _ACTIVE_ADAPTER   : adapter_id of the one /generate should use
#   _LAST_INFERENCE_TS: monotonic timestamp of last /generate call (for idle auto-unload)
#
# At most one adapter is loaded at a time on tight VRAM. Loading a different
# one auto-unloads the previous one. A background task scans every 60s and
# unloads if idle > LORA_IDLE_UNLOAD_SECONDS (env-tunable, default 1800).
#
# Path-traversal defense: only paths under the bench/personal checkpoint
# roots are accepted. Base-model allowlist limits to GPT-2 family for now.
# ============================================================================

LORA_ALLOWED_BASE_MODELS = {"gpt2-medium", "gpt2-large", "gpt2"}
LORA_ALLOWED_PATH_ROOTS = (
    "/workspace/checkpoints_personal",
    "/workspace/checkpoints_bench",
)
LORA_IDLE_UNLOAD_SECONDS = int(os.environ.get("LORA_IDLE_UNLOAD_SECONDS", "1800"))

_LOADED_ADAPTERS: dict = {}
_ACTIVE_ADAPTER: Optional[str] = None
_LAST_INFERENCE_TS: float = 0.0
_LORA_LOCK = asyncio.Lock()


def _vram_used_mb() -> int:
    if not HAS_TORCH:
        return 0
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        pass
    return 0


def _validate_adapter_path(adapter_path: str) -> Path:
    p = Path(adapter_path).resolve()
    s = str(p)
    if not any(s == root or s.startswith(root + "/") for root in LORA_ALLOWED_PATH_ROOTS):
        raise HTTPException(
            status_code=400,
            detail=(
                f"adapter_path must be under one of {LORA_ALLOWED_PATH_ROOTS}; "
                f"got {s} (path-traversal defense)."
            ),
        )
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"adapter_path does not exist: {s}")
    if not (p / "adapter_config.json").exists():
        raise HTTPException(
            status_code=400,
            detail=f"missing adapter_config.json in {s} — not a PEFT adapter dir",
        )
    return p


def _do_unload_adapter(adapter_id: Optional[str]) -> dict:
    """Synchronous unload helper — frees model, drops cache. Idempotent."""
    global _ACTIVE_ADAPTER
    before = _vram_used_mb()
    target = adapter_id or _ACTIVE_ADAPTER
    if target is None or target not in _LOADED_ADAPTERS:
        return {"ok": True, "unloaded": None, "vram_freed_mb": 0, "note": "nothing loaded"}
    entry = _LOADED_ADAPTERS.pop(target, None)
    if entry is not None:
        try:
            del entry["model"]
        except Exception:
            pass
        try:
            del entry["tokenizer"]
        except Exception:
            pass
    if _ACTIVE_ADAPTER == target:
        _ACTIVE_ADAPTER = None
    if HAS_TORCH:
        try:
            import gc as _gc
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
    after = _vram_used_mb()
    freed = max(0, before - after)
    logger.info(f"LoRA unload: {target} freed ~{freed} MB (now {after} MB used)")
    return {"ok": True, "unloaded": target, "vram_freed_mb": freed}


class LoadLoraRequest(BaseModel):
    base_model: str
    adapter_path: str
    adapter_id: Optional[str] = None


class UnloadLoraRequest(BaseModel):
    adapter_id: Optional[str] = None


@app.post("/load_lora")
async def load_lora(req: LoadLoraRequest):
    """
    Load a PEFT LoRA adapter on top of a base model and register it as the
    active adapter for /generate to use.

    Idempotent on adapter_id: re-loading the same id is a no-op. Loading a
    different id auto-unloads the previous one (one adapter at a time).
    """
    global _ACTIVE_ADAPTER
    if not HAS_TORCH:
        raise HTTPException(status_code=503, detail="torch not available")
    if req.base_model not in LORA_ALLOWED_BASE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"base_model must be one of {sorted(LORA_ALLOWED_BASE_MODELS)}; got {req.base_model!r}",
        )
    adapter_path = _validate_adapter_path(req.adapter_path)
    adapter_id = req.adapter_id or adapter_path.parent.name

    async with _LORA_LOCK:
        # Idempotent fast path
        if adapter_id in _LOADED_ADAPTERS and _ACTIVE_ADAPTER == adapter_id:
            entry = _LOADED_ADAPTERS[adapter_id]
            return {
                "ok": True,
                "adapter_id": adapter_id,
                "base_model": entry["base_model"],
                "adapter_path": entry["adapter_path"],
                "vram_used_mb": _vram_used_mb(),
                "loaded_at": entry["loaded_at"],
                "note": "already loaded",
            }

        # If a different adapter is active, unload it first.
        if _ACTIVE_ADAPTER is not None and _ACTIVE_ADAPTER != adapter_id:
            _do_unload_adapter(_ACTIVE_ADAPTER)

        # Heavy import — only here, never at module-import time
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"transformers/peft not installed in this container: {exc}",
            )

        logger.info(f"LoRA load: base={req.base_model} adapter={adapter_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(req.base_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            base = AutoModelForCausalLM.from_pretrained(
                req.base_model,
                torch_dtype=dtype,
            )
            if torch.cuda.is_available():
                base = base.to("cuda")
            model = PeftModel.from_pretrained(base, str(adapter_path))
            model.eval()
        except Exception as exc:
            logger.exception("LoRA load failed")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"LoRA load failed: {exc}")

        loaded_at = time.time()
        _LOADED_ADAPTERS[adapter_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "base_model": req.base_model,
            "adapter_path": str(adapter_path),
            "loaded_at": loaded_at,
        }
        _ACTIVE_ADAPTER = adapter_id

    used = _vram_used_mb()
    logger.info(f"LoRA loaded: {adapter_id} (~{used} MB VRAM in use)")
    return {
        "ok": True,
        "adapter_id": adapter_id,
        "base_model": req.base_model,
        "adapter_path": str(adapter_path),
        "vram_used_mb": used,
        "loaded_at": loaded_at,
    }


@app.post("/unload_lora")
async def unload_lora(req: Optional[UnloadLoraRequest] = None):
    """Unload the named adapter (or the active one). Idempotent."""
    target = req.adapter_id if req is not None else None
    async with _LORA_LOCK:
        return _do_unload_adapter(target)


@app.get("/lora_status")
async def lora_status():
    """Return what adapters are loaded and which is active."""
    return {
        "loaded_adapters": [
            {
                "adapter_id": aid,
                "base_model": entry["base_model"],
                "adapter_path": entry["adapter_path"],
                "loaded_at": entry["loaded_at"],
            }
            for aid, entry in _LOADED_ADAPTERS.items()
        ],
        "active_adapter": _ACTIVE_ADAPTER,
        "vram_used_mb": _vram_used_mb(),
        "last_inference_ts": _LAST_INFERENCE_TS or None,
        "idle_unload_seconds": LORA_IDLE_UNLOAD_SECONDS,
    }


async def _idle_unload_loop():
    """Background task: every 60s, unload the active LoRA if idle too long."""
    global _ACTIVE_ADAPTER
    while True:
        try:
            await asyncio.sleep(60)
            if _ACTIVE_ADAPTER is None:
                continue
            if _LAST_INFERENCE_TS == 0:
                entry = _LOADED_ADAPTERS.get(_ACTIVE_ADAPTER)
                ref_ts = entry["loaded_at"] if entry else 0
            else:
                ref_ts = _LAST_INFERENCE_TS
            idle = time.time() - ref_ts
            if idle > LORA_IDLE_UNLOAD_SECONDS:
                logger.info(
                    f"LoRA idle auto-unload: {_ACTIVE_ADAPTER} idle for "
                    f"{int(idle)}s > {LORA_IDLE_UNLOAD_SECONDS}s"
                )
                async with _LORA_LOCK:
                    _do_unload_adapter(_ACTIVE_ADAPTER)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning(f"idle_unload_loop tick failed: {exc}")


async def _generate_via_lora(req: "GenerateRequest"):
    """Run /generate against the currently-loaded LoRA adapter.

    Uses the HF tokenizer (matches the base) and the PEFT-wrapped model.
    Streams via SSE if req.stream is True, else returns a single JSON.
    """
    global _LAST_INFERENCE_TS
    entry = _LOADED_ADAPTERS[_ACTIVE_ADAPTER]
    model = entry["model"]
    tokenizer = entry["tokenizer"]
    device = next(model.parameters()).device

    enc_ids = tokenizer.encode(req.prompt or "", return_tensors="pt").to(device)
    if enc_ids.numel() == 0:
        # Fallback to BOS/EOS token
        eos = tokenizer.eos_token_id or 50256
        enc_ids = torch.tensor([[eos]], dtype=torch.long, device=device)
    prompt_len = enc_ids.size(1)

    if not req.stream:
        with torch.no_grad():
            out = model.generate(
                enc_ids,
                max_new_tokens=req.max_tokens,
                do_sample=True,
                temperature=max(req.temperature, 1e-5),
                top_k=req.top_k or 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = out[0, prompt_len:].tolist()
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        _LAST_INFERENCE_TS = time.time()
        return {
            "text": text,
            "tokens_generated": len(new_tokens),
            "bassin_recall_triggered": False,
            "adapter_id": _ACTIVE_ADAPTER,
            "via": "lora",
        }

    async def _stream_lora():
        global _LAST_INFERENCE_TS
        idx = enc_ids
        emitted_so_far = ""
        try:
            for _ in range(req.max_tokens):
                with torch.no_grad():
                    out = model(input_ids=idx)
                    logits = out.logits[:, -1, :] / max(req.temperature, 1e-5)
                    if req.top_k:
                        v, _ix = torch.topk(logits, min(req.top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float("Inf")
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    nxt = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat([idx, nxt], dim=1)
                new_tokens = idx[0, prompt_len:].tolist()
                full = tokenizer.decode(new_tokens, skip_special_tokens=True)
                delta = full[len(emitted_so_far):]
                emitted_so_far = full
                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"
                    await asyncio.sleep(0)
                eos_id = tokenizer.eos_token_id
                if eos_id is not None and int(nxt.item()) == eos_id:
                    break
            _LAST_INFERENCE_TS = time.time()
            yield f"data: {json.dumps({'done': True, 'adapter_id': _ACTIVE_ADAPTER, 'via': 'lora'})}\n\n"
        except Exception as exc:
            logger.exception("LoRA streaming failed")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(_stream_lora(), media_type="text/event-stream")


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
