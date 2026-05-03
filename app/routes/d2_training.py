"""
d2_training.py — Training Launcher + Live Status routes for the δ² engine.

WHAT THIS DOES, FOR THE DUMMIES:
================================

The δ² engine (anamnesis-d2 container on `server`, port 3015) holds the
inference model, the bassin, and the metrics from past runs. But spinning
up a *new* training run was, until now, a curl-and-pray exercise — you had
to SSH to server, docker exec into the trainer, and fire one of the
overnight scripts by hand.

This module surfaces TWO concrete one-click runs to the dashboard:

  - "bench"     → permuted-MNIST with delta2_additive (5 tasks, ~2 min)
  - "personal"  → gpt2-medium + LoRA on the chronological Anamnesis corpus
                  (300 steps × 2 tasks, ~15 min)

Both runs are launched OUTSIDE the app container, by SSHing to the GPU host
and `docker exec -d`-ing the trainer container with `nohup bash -c "..."`.
This mirrors how the overnight scripts were already being invoked — we are
not inventing a new protocol, just exposing it behind an authenticated POST.

WHY NOT POST DIRECTLY TO THE D² ENGINE?
========================================

The d² engine has /train/start and /train/lm/start endpoints, but those run
either `experiments.continual` with hardcoded shape, or `train.py` (the old
nanoGPT pretraining loop) — neither one matches the personal-LoRA pipeline
that finetune_lora.py implements. The morning report is explicit: "DO NOT
touch the d² training code, the optimizer, the bassin, finetune_lora.py".

So the app launches the SAME script that personal_arms_only_2026_05_04.sh
launches (the canonical one), via docker exec. No new code in d2/.

ACTIVE-RUN DETECTION:
=====================

We scan log files at /workspace/checkpoints_personal/_*.log and
/workspace/checkpoints_bench/_*.log on the GPU host. The "active" run is
the one whose log file has been modified within the last 90 seconds AND
whose corresponding script process is still running (checked via pgrep
on the docker host). Reading log files is explicitly allowed by the
constraint set.

PRIVACY / SAFETY:
=================

- One GPU, one slot. /start refuses with 409 if any run is already active.
- Stop requires confirmation_token=="STOP" body field (same shape as the
  RunPod stop pattern).
- dry_run=true validates everything but never fires.
- No MongoDB writes. No external requests. SSH only to the configured
  GPU host (D2_SSH_HOST, default "server" via ~/.ssh/config).
"""

import json
import logging
import os
import re
import shlex
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("anamnesis.routes.d2_training")
router = APIRouter(prefix="/api/d2/training", tags=["d2-training"])


# ============================================================================
# Configuration
# ============================================================================

# SSH host alias for the GPU box that runs the anamnesis-d2 container.
# Defaults to "server" — assumes ~/.ssh/config in the app container resolves it.
D2_SSH_HOST = os.environ.get("D2_SSH_HOST", "server")
D2_SSH_USER = os.environ.get("D2_SSH_USER", os.environ.get("SSH_USER", "elfege"))

# The container name on the GPU host where d² training runs.
D2_DOCKER_CONTAINER = os.environ.get("D2_DOCKER_CONTAINER", "anamnesis-d2")

# Where the trainer writes log files (inside the container).
D2_LOG_DIRS = [
    "/workspace/checkpoints_personal",
    "/workspace/checkpoints_bench",
]

# How recently a log file must have been touched to count as "active".
ACTIVE_WINDOW_SECONDS = 120

# Window between heartbeats on the SSE log stream.
LOG_STREAM_TICK_SECONDS = 2.0


# ============================================================================
# Schemas
# ============================================================================

class StartRunRequest(BaseModel):
    """POST /api/d2/training/start"""
    kind: str  # "bench" | "personal"
    experiment_name: Optional[str] = None  # default auto-generated
    dry_run: bool = False  # validate command + lock state but do not fire


class StopRunRequest(BaseModel):
    """POST /api/d2/training/stop"""
    confirmation_token: str  # must equal "STOP"


# ============================================================================
# SSH helper (paramiko-backed, mirrors routes/files._ssh_client)
# ============================================================================

def _ssh_exec(command: str, timeout: int = 30) -> tuple[int, str, str]:
    """
    Run `command` on D2_SSH_HOST via paramiko.
    Returns (exit_code, stdout, stderr).
    Raises HTTPException(503) if SSH itself fails.
    """
    # We deliberately import the helper from routes.files so we share its
    # ~/.ssh/config + key-discovery logic. Keeps SSH config in one place.
    from routes.files import _ssh_client

    client = _ssh_client(D2_SSH_HOST, D2_SSH_USER)
    try:
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err
    finally:
        try:
            client.close()
        except Exception:
            pass


# ============================================================================
# Run command builders
# ============================================================================

def _build_bench_command(experiment_name: str, log_path: str) -> str:
    """
    Return the `bash -c "..."` payload that, when run inside anamnesis-d2,
    fires a permuted-MNIST bench run with delta2_additive and writes
    everything to `log_path`.
    """
    # Mirrors d2/scripts/pareto_v3_alpha_sweep_2026_05_04.sh in spirit:
    # one short bench run, delta2_additive, sane defaults.
    out_dir = f"/workspace/checkpoints_bench/{experiment_name}"
    payload = (
        f"mkdir -p {shlex.quote(out_dir)} && "
        f"cd /workspace && "
        f"exec >>{shlex.quote(log_path)} 2>&1 && "
        f"echo === bench run start $(date -Iseconds) === && "
        f"python -m d2.experiments.continual "
        f"  --method delta2_additive --benchmark permuted_mnist "
        f"  --tasks 5 --epochs 1 --d2-eta 1e-2 "
        f"  --output {shlex.quote(out_dir)}/run.json "
        f"  --device cuda --seed 0 ; "
        f"echo === bench run end $(date -Iseconds) exit=$? ==="
    )
    return payload


def _build_personal_command(experiment_name: str, log_path: str) -> str:
    """
    Return the `bash -c "..."` payload that fires the canonical personal-LoRA
    run. Mirrors d2/scripts/personal_arms_only_2026_05_04.sh exactly:
    gpt2-medium + LoRA on c_attn, 300 steps, fp16, bassin via --optimizer delta2.
    """
    out_dir = "/workspace/checkpoints_personal"
    payload = (
        f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"mkdir -p {shlex.quote(out_dir)} && "
        f"cd /workspace && "
        f"exec >>{shlex.quote(log_path)} 2>&1 && "
        f"echo === personal run start $(date -Iseconds) === && "
        f"nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader 2>/dev/null || true && "
        f"python -m d2.finetune_lora "
        f"  --base-model gpt2-medium "
        f"  --lora-target-modules c_attn "
        f"  --dtype fp16 "
        f"  --no-load-in-4bit "
        f"  --data-dir /workspace/data_personal/anamnesis_chronological "
        f"  --output-dir {shlex.quote(out_dir)} "
        f"  --steps-per-task 300 "
        f"  --block-size 256 "
        f"  --batch-size 1 "
        f"  --eval-interval 50 "
        f"  --eval-batches 4 "
        f"  --experiment {shlex.quote(experiment_name)} "
        f"  --optimizer delta2 "
        f"  --d2-eta 1e-6 ; "
        f"echo === personal run end $(date -Iseconds) exit=$? ==="
    )
    return payload


def _build_docker_exec(payload: str) -> str:
    """
    Wrap `payload` in `docker exec -d <container> bash -c '...'` so it runs
    detached from the SSH session. We use single-quotes inside docker exec
    and shlex-escape with double-quote-aware quoting.
    """
    # docker exec -d returns immediately; the process keeps running inside
    # the container under PID 1's reaper. Output already redirected via the
    # `exec >>logfile 2>&1` inside the payload.
    return (
        f"docker exec -d {shlex.quote(D2_DOCKER_CONTAINER)} "
        f"bash -c {shlex.quote(payload)}"
    )


# ============================================================================
# Active-run detection
# ============================================================================

def _scan_for_active_run() -> Optional[dict]:
    """
    Look for an in-progress training run on the GPU host.

    Strategy:
      1. List log files matching /workspace/checkpoints_*/_*.log inside the
         container, with mtime + size + most recent step line.
      2. Filter to those modified within ACTIVE_WINDOW_SECONDS.
      3. Cross-check that a python -m d2.* (finetune_lora or experiments.continual)
         is currently running inside the container — if yes, the latest
         recently-touched log belongs to that run.

    Returns dict like:
      {
        "active": True,
        "kind": "personal"|"bench",
        "log_path": "...",
        "experiment_name": "...",
        "started_at": "ISO",
        "elapsed_s": int,
        "step": int|None,
        "total_steps": int|None,
        "last_train_loss": float|None,
        "last_val_loss": float|None,
        "last_log_lines": [str, ...]  # last 20 lines
      }

    Returns {"active": False, ...} when no active run.
    """
    # Step 1 — list logs (mtime + size). Use stat in epoch seconds.
    cmd_list = (
        f"docker exec {shlex.quote(D2_DOCKER_CONTAINER)} bash -c '"
        f"for d in {' '.join(D2_LOG_DIRS)}; do "
        f"  for f in \"$d\"/_*.log; do "
        f"    [ -f \"$f\" ] && stat -c \"%Y %s %n\" \"$f\"; "
        f"  done 2>/dev/null; "
        f"done'"
    )
    try:
        rc, out, err = _ssh_exec(cmd_list, timeout=10)
    except HTTPException:
        return None
    # rc=1 is OK if output is non-empty: the bash for-loop returns the exit
    # code of its last iteration, and an empty checkpoints_bench dir makes
    # the literal glob _*.log fail [ -f ] for every iteration. We only treat
    # rc != 0 as fatal when there's also no output to parse.
    if rc != 0 and not (out or "").strip():
        logger.debug(f"_scan_for_active_run: log listing rc={rc} err={err[:120]}")
        return None

    now_epoch = time.time()
    candidates: list[dict] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # "<mtime> <size> <path>"
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            continue
        try:
            mtime = int(parts[0])
            size = int(parts[1])
        except ValueError:
            continue
        path = parts[2]
        age_s = now_epoch - mtime
        candidates.append({"path": path, "mtime": mtime, "size": size, "age_s": age_s})

    # Sort by mtime DESC, take the most recent
    candidates.sort(key=lambda c: c["mtime"], reverse=True)
    if not candidates:
        return {"active": False, "log_path": None, "candidates": []}

    most_recent = candidates[0]
    is_recent = most_recent["age_s"] <= ACTIVE_WINDOW_SECONDS

    # Step 2 — check process is still alive
    proc_running = False
    if is_recent:
        cmd_proc = (
            f"docker exec {shlex.quote(D2_DOCKER_CONTAINER)} bash -c "
            f"\"pgrep -f 'python -m d2\\.(finetune_lora|experiments\\.continual)' >/dev/null && echo RUNNING || echo IDLE\""
        )
        try:
            rc2, out2, _ = _ssh_exec(cmd_proc, timeout=10)
            proc_running = "RUNNING" in (out2 or "")
        except HTTPException:
            proc_running = False

    active = is_recent and proc_running

    # Step 3 — parse the log: kind, experiment, last step/loss, last lines
    kind = None
    experiment_name = None
    started_at_iso = None
    step = None
    last_train_loss = None
    last_val_loss = None
    last_lines: list[str] = []

    log_path = most_recent["path"]
    if "/checkpoints_personal/" in log_path:
        kind = "personal"
    elif "/checkpoints_bench/" in log_path:
        kind = "bench"

    # Tail the log
    cmd_tail = (
        f"docker exec {shlex.quote(D2_DOCKER_CONTAINER)} "
        f"bash -c {shlex.quote('tail -200 ' + shlex.quote(log_path))}"
    )
    try:
        rc3, out3, _ = _ssh_exec(cmd_tail, timeout=10)
        if rc3 == 0:
            tail_lines = out3.splitlines()
            last_lines = tail_lines[-20:]
            # Parse "  step    N | train X | val Y" — last occurrence wins
            step_re = re.compile(r"step\s+(\d+)\s*\|\s*train\s+([\d.eE+-]+)\s*\|\s*val\s+([\d.eE+-]+)")
            for ln in tail_lines:
                m = step_re.search(ln)
                if m:
                    step = int(m.group(1))
                    try:
                        last_train_loss = float(m.group(2))
                        last_val_loss = float(m.group(3))
                    except ValueError:
                        pass
            # Parse experiment name from the docker run wrapper; fall back to filename
            for ln in tail_lines:
                m = re.search(r"--experiment\s+(\S+)", ln)
                if m:
                    experiment_name = m.group(1).strip("'\"")
                    break
            if not experiment_name:
                # Filename pattern: _personal_arms_2026-05-04.log → personal_arms_2026-05-04
                experiment_name = Path(log_path).stem.lstrip("_")
            # Parse "=== ... start <ISO> ===" for started_at
            for ln in tail_lines:
                m = re.search(r"start\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\s=]*)", ln)
                if m:
                    started_at_iso = m.group(1)
                    break
    except HTTPException:
        pass

    if not started_at_iso:
        # Use mtime of the file as a crude estimate
        started_at_iso = (
            datetime.fromtimestamp(most_recent["mtime"], tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

    # Pick total_steps based on kind (matches the canonical script defaults)
    total_steps = None
    if kind == "personal":
        # 300 steps/task × 2 tasks (chronological corpus has 2 tasks)
        total_steps = 600
    elif kind == "bench":
        # 5 tasks × 1 epoch on permuted-MNIST — train loss is logged per
        # batch but there is no fixed total. Leave None; UI will show step.
        total_steps = None

    return {
        "active": active,
        "kind": kind,
        "log_path": log_path,
        "experiment_name": experiment_name,
        "started_at": started_at_iso,
        "elapsed_s": int(most_recent["age_s"]) if most_recent else None,
        "step": step,
        "total_steps": total_steps,
        "last_train_loss": last_train_loss,
        "last_val_loss": last_val_loss,
        "last_log_lines": last_lines,
        "process_running": proc_running,
        "log_age_s": int(most_recent["age_s"]) if most_recent else None,
    }


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/status")
async def training_status():
    """
    Return the latest training run's state.

    Shape: see _scan_for_active_run for the keys. When no log is found at all,
    returns {"active": False, "log_path": null, "candidates": []}.
    """
    try:
        st = _scan_for_active_run()
        if st is None:
            return {
                "active": False,
                "log_path": None,
                "error": "scan failed (SSH or docker exec)",
            }
        return st
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("training_status failed")
        raise HTTPException(status_code=500, detail=f"training_status failed: {exc}")


@router.post("/start")
async def training_start(req: StartRunRequest):
    """
    Launch a training run.

    Body: {"kind": "bench" | "personal", "experiment_name": "...", "dry_run": false}

    Returns: {
        "ok": true,
        "run_id": "...",
        "kind": "bench" | "personal",
        "experiment_name": "...",
        "log_path": "...",
        "command_remote": "<the docker exec command>",  # for transparency
        "dry_run": bool
    }

    Errors:
      400 — invalid kind
      409 — another run is already active (one GPU, no parallelism)
      503 — SSH unreachable
    """
    if req.kind not in ("bench", "personal"):
        raise HTTPException(status_code=400, detail=f"kind must be 'bench' or 'personal'")

    # Refuse to fire if a run is already active — single GPU constraint.
    st = _scan_for_active_run()
    if st and st.get("active"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"A {st.get('kind') or 'training'} run is already in progress "
                f"(log: {st.get('log_path')}). Stop it before starting a new one."
            ),
        )

    # Build the command + log path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{req.kind}_{timestamp}_{uuid.uuid4().hex[:6]}"
    experiment_name = req.experiment_name or run_id

    if req.kind == "bench":
        log_path = f"/workspace/checkpoints_bench/_{run_id}.log"
        payload = _build_bench_command(experiment_name, log_path)
    else:  # personal
        log_path = f"/workspace/checkpoints_personal/_{run_id}.log"
        payload = _build_personal_command(experiment_name, log_path)

    docker_cmd = _build_docker_exec(payload)

    if req.dry_run:
        return {
            "ok": True,
            "run_id": run_id,
            "kind": req.kind,
            "experiment_name": experiment_name,
            "log_path": log_path,
            "command_remote": docker_cmd,
            "dry_run": True,
            "note": "dry_run=true: command was validated but NOT executed.",
        }

    # Pre-create the log file inside the container so the first /status poll
    # finds it immediately (otherwise there's a race where the user clicks
    # Start, refreshes, sees no active run, and panics).
    pre_touch_cmd = (
        f"docker exec {shlex.quote(D2_DOCKER_CONTAINER)} "
        f"bash -c {shlex.quote('mkdir -p ' + shlex.quote(str(Path(log_path).parent)) + ' && touch ' + shlex.quote(log_path))}"
    )
    try:
        _ssh_exec(pre_touch_cmd, timeout=10)
    except HTTPException as exc:
        raise HTTPException(status_code=503, detail=f"Cannot reach GPU host to create log file: {exc.detail}")

    # Fire the actual training command (detached via docker exec -d).
    try:
        rc, out, err = _ssh_exec(docker_cmd, timeout=15)
    except HTTPException as exc:
        raise HTTPException(status_code=503, detail=f"Cannot reach GPU host to start run: {exc.detail}")

    if rc != 0:
        raise HTTPException(
            status_code=500,
            detail=f"docker exec failed (rc={rc}): {err[:300] or out[:300]}",
        )

    logger.info(f"d2 training started: kind={req.kind} run_id={run_id} log={log_path}")
    return {
        "ok": True,
        "run_id": run_id,
        "kind": req.kind,
        "experiment_name": experiment_name,
        "log_path": log_path,
        "command_remote": docker_cmd,
        "dry_run": False,
    }


@router.post("/stop")
async def training_stop(req: StopRunRequest):
    """
    Stop the currently active training run.

    Body: {"confirmation_token": "STOP"}

    The confirmation token must literally equal "STOP" — guards against
    accidental kill from a misfired AJAX call.

    Returns: {"ok": true, "killed": bool, "log_path": "...", "stderr": "..."}
    """
    if req.confirmation_token != "STOP":
        raise HTTPException(
            status_code=400,
            detail='confirmation_token must be exactly "STOP"',
        )

    st = _scan_for_active_run()
    if not st or not st.get("active"):
        return {
            "ok": True,
            "killed": False,
            "log_path": st.get("log_path") if st else None,
            "note": "No active training run found — nothing to kill.",
        }

    # Kill any python -m d2.finetune_lora / d2.experiments.continual processes
    # inside the container. pkill -f matches on the full command line.
    kill_cmd = (
        f"docker exec {shlex.quote(D2_DOCKER_CONTAINER)} bash -c "
        f"\"pkill -TERM -f 'python -m d2\\.(finetune_lora|experiments\\.continual)' ; "
        f"sleep 2 ; "
        f"pkill -KILL -f 'python -m d2\\.(finetune_lora|experiments\\.continual)' 2>/dev/null ; "
        f"echo done\""
    )
    try:
        rc, out, err = _ssh_exec(kill_cmd, timeout=15)
    except HTTPException as exc:
        raise HTTPException(status_code=503, detail=f"Cannot reach GPU host: {exc.detail}")

    logger.info(f"d2 training stop sent: rc={rc} log={st.get('log_path')}")
    return {
        "ok": True,
        "killed": True,
        "log_path": st.get("log_path"),
        "stderr": err[:200],
    }


@router.get("/logs/stream")
async def logs_stream():
    """
    Server-Sent Events stream of the active run's log tail.

    Each tick (every LOG_STREAM_TICK_SECONDS) we:
      - Re-scan for the active run
      - Emit a JSON payload with state + last 20 lines
      - If no active run for >5 ticks, emit one final "idle" event and close

    Usage from JS:
        const es = new EventSource('/api/d2/training/logs/stream');
        es.onmessage = (ev) => { const st = JSON.parse(ev.data); ... }
    """
    async def gen() -> AsyncIterator[str]:
        idle_ticks = 0
        max_idle_ticks = 5  # ~10s of "no run" before we close the stream

        # Emit a hello immediately so the client knows the stream is alive
        yield f"event: hello\ndata: {json.dumps({'msg': 'log stream connected'})}\n\n"

        while True:
            try:
                # Run blocking SSH in a thread so we don't stall the event loop
                st = await asyncio.to_thread(_scan_for_active_run)
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc), 'active': False})}\n\n"
                await asyncio.sleep(LOG_STREAM_TICK_SECONDS)
                continue

            if st is None:
                yield f"data: {json.dumps({'error': 'scan failed', 'active': False})}\n\n"
            else:
                yield f"data: {json.dumps(st)}\n\n"
                if not st.get("active"):
                    idle_ticks += 1
                    if idle_ticks >= max_idle_ticks:
                        # Final goodbye — frontend will reconnect on next /start
                        yield f"event: idle\ndata: {json.dumps({'msg': 'no active run; closing stream'})}\n\n"
                        return
                else:
                    idle_ticks = 0

            await asyncio.sleep(LOG_STREAM_TICK_SECONDS)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
