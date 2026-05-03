"""
training_catalog.py — Declarative registry of every training task this project
can fire from the dashboard.

WHY THIS EXISTS
================

Before this file, "what training can the project do" was implicit knowledge —
it lived in scattered shell scripts under d2/scripts/, in two hard-coded
buttons in the δ² tab (Bench / Personal), and in the per-machine trainer
sidecars surfaced in the Training tab. The user had to remember (or grep)
which script did which sweep, where it ran, how long it took, and what its
output looked like.

The Training Catalog flips that around: there is now ONE place that names,
describes, and parametrizes every canonical training task. The dashboard
renders a card per entry, lets the user click Start, and uses the existing
d2_training launcher under the hood. New tasks go here, get a card for free.

DESIGN
======

- Pure-Python data, no DB writes — the catalog itself is build-time config.
- Each entry references either:
    (a) a `command_template` string that gets fired via docker exec on the
        target host (mirrors the manual `bash /app/d2/scripts/...` pattern), OR
    (b) a `kind` string ("bench" / "personal") that delegates to the existing
        d2_training.training_start() builder. Used for the two arms that
        already have first-class command builders.
- "Last run" data comes from the same log scan that d2_training uses, so
  the card status badge is always consistent with the Live Training Status
  panel — no separate source of truth.
- Conflict + duplicate protection live HERE (the catalog is the only entry
  point users hit), not duplicated into d2_training.

"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("anamnesis.training_catalog")
router = APIRouter(prefix="/api/training", tags=["training-catalog"])


# ============================================================================
# Catalog
# ============================================================================
#
# Each entry shape:
#   id                  — short unique slug (used in URLs)
#   title               — short, dashboard-card title
#   purpose             — one-sentence "why this exists"
#   resource            — "cpu" | "local-gpu" | "runpod-gpu"
#   target_host         — "server" | "office" | "runpod" — where it would fire
#                         (drives conflict-detection scope)
#   docker_container    — container name on target_host where the script runs
#   command_template    — bash command to exec inside the container.
#                         Use {experiment} / {log_path} placeholders. The
#                         catalog pre-creates the log file so /status sees
#                         it immediately.
#   delegate_kind       — if set ("bench" or "personal"), instead of running
#                         a script, we hand off to d2_training.training_start
#                         with that kind. Used for the two arms that already
#                         have first-class command builders.
#   log_glob            — where to look for run history (mtime + path).
#   estimated_minutes   — honest wall-time on `target_host`. None if unknown.
#   estimated_cost_usd  — total USD cost; 0 for local; honest RunPod estimate
#                         (community 3090 ≈ $0.34/hr, A100 PCIe ≈ $1.89/hr).
#   produces            — output dir / artifact path the run writes to.
#   help_text           — what the help (?) icon explains. Plain prose.

TRAINING_CATALOG: list[dict] = [
    # ── Pareto sweeps (permuted-MNIST) ────────────────────────────────────
    {
        "id": "pareto_eta_meaningful",
        "title": "Pareto sweep — η meaningful range",
        "purpose": (
            "Probe whether δ²-additive's η has any measurable effect once it's "
            "in the same magnitude as the gradient term."
        ),
        "resource": "local-gpu",
        "target_host": "server",
        "docker_container": "anamnesis-d2",
        "command_template": "bash /app/d2/scripts/pareto_meaningful_eta_2026_05_04.sh",
        "log_glob": "/workspace/checkpoints_personal/_pareto_v2_*.log",
        "estimated_minutes": 15,
        "estimated_cost_usd": 0.0,
        "produces": "/workspace/checkpoints_bench/pareto_v2_2026-05-04/",
        "help_text": (
            "Sweeps δ²-additive's η across {1e-3, 3e-3, 1e-2, 3e-2, 1e-1} "
            "with 3 seeds each on permuted-MNIST. This is the range where η·tanh(B) "
            "is comparable to the gradient step — anywhere lower and δ² is "
            "effectively SGD. Adam baseline included."
        ),
    },
    {
        "id": "pareto_v3_alpha_sweep",
        "title": "Pareto v3 — α₂ low/mid range",
        "purpose": (
            "Sweep α₂ (bassin growth rate) to grow B's magnitude into the "
            "range where tanh actually bends."
        ),
        "resource": "local-gpu",
        "target_host": "server",
        "docker_container": "anamnesis-d2",
        "command_template": "bash /app/d2/scripts/pareto_v3_alpha_sweep_2026_05_04.sh",
        "log_glob": "/workspace/checkpoints_personal/_pareto_v3_*.log",
        "estimated_minutes": 12,
        "estimated_cost_usd": 0.0,
        "produces": "/workspace/checkpoints_bench/pareto_v3_2026-05-04/",
        "help_text": (
            "Sweeps α₂ ∈ {1e-2, 1e-1, 1, 10} × 3 seeds. Yesterday's η-only sweep "
            "showed bit-identical results because the bassin never grew enough "
            "for tanh to non-linearize. This sweep grows it — α₁ is held at "
            "α₂ × 0.1, η fixed at 1e-2."
        ),
    },
    {
        "id": "pareto_v4_alpha_high",
        "title": "Pareto v4 — α₂ HIGH range",
        "purpose": (
            "v3 found only tiny shifts at α₂=10. v4 pushes α₂ to {1e2, 1e3, 1e4} "
            "where tanh(B) should genuinely bend (B > 0.1)."
        ),
        "resource": "local-gpu",
        "target_host": "server",
        "docker_container": "anamnesis-d2",
        "command_template": "bash /app/d2/scripts/pareto_v4_alpha_high_2026_05_04.sh",
        "log_glob": "/workspace/checkpoints_personal/_pareto_v4_*.log",
        "estimated_minutes": 9,
        "estimated_cost_usd": 0.0,
        "produces": "/workspace/checkpoints_bench/pareto_v4_2026-05-04/",
        "help_text": (
            "3 alphas × 3 seeds at α₂ ∈ {100, 1000, 10000}. Same MLP+permuted-MNIST "
            "setup. If δ² shows no benefit here either, the additive form is "
            "negative-result confirmed for this benchmark."
        ),
    },
    # ── Personal arms (gpt2-medium + LoRA) ────────────────────────────────
    {
        "id": "personal_arms_delta2_adam",
        "title": "Personal arms — δ² then Adam (gpt2-medium LoRA)",
        "purpose": (
            "End-to-end personal-corpus run: δ² and Adam on the chronological "
            "Anamnesis episodes, gpt2-medium + LoRA on c_attn."
        ),
        "resource": "local-gpu",
        "target_host": "server",
        "docker_container": "anamnesis-d2",
        # Delegate to the existing builder so we don't duplicate the recipe.
        "delegate_kind": "personal",
        "log_glob": "/workspace/checkpoints_personal/_personal_*.log",
        "estimated_minutes": 30,
        "estimated_cost_usd": 0.0,
        "produces": "/workspace/checkpoints_personal/personal_*/",
        "help_text": (
            "Fires d2/scripts/personal_arms_only_2026_05_04.sh equivalent: "
            "300 steps × 2 tasks, batch=1 block=256, fp16. PRIVATE TRACK — "
            "outputs never published, never logged to MongoDB."
        ),
    },
    # ── Bench permuted-MNIST single shot ──────────────────────────────────
    {
        "id": "bench_permuted_mnist_smoke",
        "title": "Bench smoke — permuted-MNIST (5 tasks)",
        "purpose": (
            "Two-minute smoke test: δ²-additive on permuted-MNIST. Used to "
            "verify the d² engine is alive after restarts or upgrades."
        ),
        "resource": "local-gpu",
        "target_host": "server",
        "docker_container": "anamnesis-d2",
        "delegate_kind": "bench",
        "log_glob": "/workspace/checkpoints_bench/_*.log",
        "estimated_minutes": 2,
        "estimated_cost_usd": 0.0,
        "produces": "/workspace/checkpoints_bench/<run_id>/run.json",
        "help_text": (
            "Fires the canonical bench builder (d2.experiments.continual + "
            "delta2_additive + permuted_mnist + 5 tasks + 1 epoch). Produces "
            "one run.json with ACC/BWT/FWT. Useful as a heartbeat for the "
            "training pipeline after container restarts."
        ),
    },
    # ── Multi-seed sensitivity ────────────────────────────────────────────
    {
        "id": "multi_seed_sensitivity",
        "title": "Multi-seed sensitivity rerun (Pareto v2)",
        "purpose": (
            "Resume the Pareto v2 sweep that the d² container restart "
            "interrupted on 2026-05-04 — fills in missing η × seed cells."
        ),
        "resource": "local-gpu",
        "target_host": "server",
        "docker_container": "anamnesis-d2",
        "command_template": "bash /app/d2/scripts/pareto_v2_resume_2026_05_04.sh",
        "log_glob": "/workspace/checkpoints_personal/_pareto_v2_resume_*.log",
        "estimated_minutes": 10,
        "estimated_cost_usd": 0.0,
        "produces": "/workspace/checkpoints_bench/pareto_v2_2026-05-04/",
        "help_text": (
            "Reruns the 8 cells lost when agent #3 restarted the d² container "
            "mid-sweep: η=1e-2 seed 2, η=3e-2 × 3, η=1e-1 × 3, plus the Adam "
            "baseline. Uses identical hyperparameters to the original v2 sweep "
            "so results are mergeable."
        ),
    },
    # ── WikiText-103 LM bench (NEW — script + entry, NOT auto-fired) ─────
    {
        "id": "bench_wikitext103",
        "title": "Bench WikiText-103 — δ² vs Adam (LM)",
        "purpose": (
            "Real LM workload on WikiText-103. Tests whether δ²'s structured "
            "retention helps next-token prediction more than it helped MLP+MNIST."
        ),
        "resource": "local-gpu",
        "target_host": "server",
        "docker_container": "anamnesis-d2",
        "command_template": "bash /app/d2/scripts/bench_wikitext103_2026_05_04.sh",
        "log_glob": "/workspace/checkpoints_bench/_bench_wikitext103_*.log",
        # Honest estimate: ~2 hr/arm on the 1660 SUPER, ~25 min on 3090
        "estimated_minutes": 240,
        "estimated_cost_usd": 0.0,
        "produces": "/workspace/checkpoints_bench/wikitext103_2026-05-04/",
        "help_text": (
            "δ²-additive (η=1e-6) vs Adam (lr=3e-4) on WikiText-103, GPT-2 small "
            "(124M), 2000 steps, block 512, batch 4. PREREQ: data must be "
            "tokenized first via d2/data/prepare_wikitext.py. Script aborts "
            "if /workspace/d2/data/wikitext/train.bin is missing — by design, "
            "to prevent silently downloading 500MB on the GPU host."
        ),
    },
]


# ============================================================================
# In-memory duplicate-protection ledger
# ============================================================================
#
# Maps task_id → (last_started_epoch, last_run_id). Lives in this process's
# memory. We deliberately do NOT persist this to MongoDB: duplicates are about
# protecting against double-clicks within seconds, not about long-term audit.
# A 60-second window is enough to absorb double-clicks, accidental retries,
# and over-eager scripted callers, without preventing legitimate "I want to
# rerun the same task with a different seed" use cases.

_RECENT_STARTS: dict[str, tuple[float, str]] = {}
DUPLICATE_WINDOW_SECONDS = 60


# ============================================================================
# Schemas
# ============================================================================

class StartRequest(BaseModel):
    dry_run: bool = False
    confirmation_token: str = ""


# ============================================================================
# Helpers
# ============================================================================

def _entry_for(task_id: str) -> dict:
    for e in TRAINING_CATALOG:
        if e["id"] == task_id:
            return e
    raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")


def _ssh_exec_for_target(target_host: str, command: str, timeout: int = 30) -> tuple[int, str, str]:
    """
    Run `command` on the target host via paramiko, sharing the SSH config logic
    used by routes.files. Returns (rc, stdout, stderr).
    """
    from routes.files import _ssh_client
    user = os.environ.get("SSH_USER", "elfege")
    client = _ssh_client(target_host, user)
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


def _list_history(entry: dict) -> list[dict]:
    """
    List historical runs for `entry` by globbing log files on the target host.
    Returns newest-first list of {path, mtime_epoch, age_s, size}.

    Returns [] if the host is unreachable (best-effort — a missing history
    must NOT block the catalog from rendering).
    """
    glob_pat = entry.get("log_glob")
    if not glob_pat:
        return []
    container = entry.get("docker_container")
    target = entry.get("target_host")
    # Use docker exec when a container is named, otherwise host shell directly.
    if container:
        cmd = (
            f"docker exec {shlex.quote(container)} bash -c "
            f"{shlex.quote('for f in ' + glob_pat + '; do [ -f \"$f\" ] && stat -c \"%Y %s %n\" \"$f\"; done 2>/dev/null')}"
        )
    else:
        cmd = (
            f"bash -c {shlex.quote('for f in ' + glob_pat + '; do [ -f \"$f\" ] && stat -c \"%Y %s %n\" \"$f\"; done 2>/dev/null')}"
        )
    try:
        rc, out, err = _ssh_exec_for_target(target, cmd, timeout=8)
    except Exception as exc:
        logger.debug(f"_list_history({entry['id']}): SSH failed: {exc}")
        return []
    if rc != 0 and not (out or "").strip():
        return []
    now_epoch = time.time()
    out_list: list[dict] = []
    for ln in (out or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split(maxsplit=2)
        if len(parts) != 3:
            continue
        try:
            mtime = int(parts[0])
            size = int(parts[1])
        except ValueError:
            continue
        out_list.append({
            "path": parts[2],
            "mtime_epoch": mtime,
            "age_s": int(now_epoch - mtime),
            "size_bytes": size,
        })
    out_list.sort(key=lambda d: d["mtime_epoch"], reverse=True)
    return out_list


def _classify_status(history: list[dict], entry: dict) -> dict:
    """
    Reduce log history to a status badge.
    Returns: {"state": "never"|"last"|"running",
              "last_run_age_s": int|None,
              "run_count": int,
              "last_run_path": str|None}
    """
    if not history:
        return {"state": "never", "last_run_age_s": None, "run_count": 0, "last_run_path": None}
    most_recent = history[0]
    # We consider a run "running" if its log mtime is within 120s. We do NOT
    # cross-check pgrep here (that would 1 SSH per card and slow the page).
    # The Live Training Status panel does the authoritative pgrep check.
    state = "running" if most_recent["age_s"] <= 120 else "last"
    return {
        "state": state,
        "last_run_age_s": most_recent["age_s"],
        "run_count": len(history),
        "last_run_path": most_recent["path"],
    }


def _check_conflict(entry: dict) -> Optional[str]:
    """
    Conflict protection: refuse to start if a conflicting run is already active
    on the target host's GPU.

    Strategy:
      - We re-use d2_training._scan_for_active_run() for the d² container on
        `server`. If the entry targets that same {host, container}, an active
        run there is a conflict.
      - For non-conflicting cases (different host, or CPU-only task), allow.
        (We do not currently have CPU-only tasks but the field exists.)

    Returns the conflict reason string, or None if no conflict.
    """
    target = entry.get("target_host")
    container = entry.get("docker_container")
    resource = entry.get("resource", "local-gpu")

    # Only the d² container on `server` is currently scanned authoritatively
    if target == "server" and container == "anamnesis-d2":
        try:
            from routes.d2_training import _scan_for_active_run
            st = _scan_for_active_run()
        except Exception as exc:
            logger.debug(f"_check_conflict scan failed: {exc}")
            return None
        if st and st.get("active"):
            return (
                f"Conflict: a {st.get('kind') or 'training'} run is already "
                f"active on {target}/{container} (log: {st.get('log_path')}). "
                f"Stop it first or wait for it to finish."
            )
        return None

    # CPU-only or other-host tasks: no conflict scope yet. Allow.
    if resource == "cpu":
        return None
    # Different GPU host: allowed (different physical resource).
    return None


def _check_duplicate(task_id: str) -> Optional[str]:
    """
    Duplicate protection: refuse to start the same task_id twice within
    DUPLICATE_WINDOW_SECONDS. In-memory ledger.

    Returns the duplicate reason string, or None if no recent duplicate.
    """
    entry = _RECENT_STARTS.get(task_id)
    if not entry:
        return None
    last_ts, last_run_id = entry
    age = time.time() - last_ts
    if age < DUPLICATE_WINDOW_SECONDS:
        remaining = int(DUPLICATE_WINDOW_SECONDS - age)
        return (
            f"Duplicate: task '{task_id}' was started {int(age)}s ago "
            f"(run {last_run_id}). Wait {remaining}s, or use Re-run last "
            f"if you really want a second one."
        )
    return None


def _record_start(task_id: str, run_id: str) -> None:
    _RECENT_STARTS[task_id] = (time.time(), run_id)


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/catalog")
async def list_catalog():
    """
    Return every catalog entry with last-run info merged in. Never errors
    on a per-entry SSH failure — those entries just get state="unknown".
    """
    items: list[dict] = []
    for entry in TRAINING_CATALOG:
        try:
            history = _list_history(entry)
            badge = _classify_status(history, entry)
        except Exception as exc:
            logger.exception(f"catalog entry {entry['id']}: history failed")
            history = []
            badge = {"state": "unknown", "last_run_age_s": None, "run_count": 0, "last_run_path": None}
        item = {
            "id": entry["id"],
            "title": entry["title"],
            "purpose": entry["purpose"],
            "resource": entry["resource"],
            "target_host": entry["target_host"],
            "estimated_minutes": entry.get("estimated_minutes"),
            "estimated_cost_usd": entry.get("estimated_cost_usd", 0.0),
            "produces": entry.get("produces"),
            "help_text": entry.get("help_text", ""),
            "delegate_kind": entry.get("delegate_kind"),
            "uses_script": bool(entry.get("command_template")),
            "status": badge,
            "history_count": len(history),
            "history_latest": history[0] if history else None,
        }
        items.append(item)
    return {"items": items, "duplicate_window_seconds": DUPLICATE_WINDOW_SECONDS}


@router.get("/catalog/{task_id}")
async def get_catalog_entry(task_id: str):
    entry = _entry_for(task_id)
    history = _list_history(entry)
    return {
        "entry": entry,
        "history": history,
    }


def _fire_entry(entry: dict, dry_run: bool) -> dict:
    """
    Build + (optionally) execute the command for this catalog entry.
    Returns the dict that the start endpoint will send back.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{entry['id']}_{timestamp}_{uuid.uuid4().hex[:6]}"

    # Branch A — delegate to existing d2_training builder
    if entry.get("delegate_kind"):
        from routes.d2_training import (
            _build_bench_command, _build_personal_command, _build_docker_exec,
            _ssh_exec, D2_DOCKER_CONTAINER,
        )
        kind = entry["delegate_kind"]
        if kind == "bench":
            log_path = f"/workspace/checkpoints_bench/_{run_id}.log"
            payload = _build_bench_command(run_id, log_path)
        elif kind == "personal":
            log_path = f"/workspace/checkpoints_personal/_{run_id}.log"
            payload = _build_personal_command(run_id, log_path)
        else:
            raise HTTPException(status_code=500, detail=f"unknown delegate_kind: {kind}")
        docker_cmd = _build_docker_exec(payload)
        if dry_run:
            return {
                "ok": True, "run_id": run_id, "log_path": log_path,
                "command_remote": docker_cmd, "dry_run": True,
                "note": "dry_run=true: command was validated but NOT executed.",
            }
        # Pre-touch log + fire
        pre_touch = (
            f"docker exec {shlex.quote(D2_DOCKER_CONTAINER)} bash -c "
            f"{shlex.quote('mkdir -p ' + shlex.quote(str(Path(log_path).parent)) + ' && touch ' + shlex.quote(log_path))}"
        )
        try:
            _ssh_exec(pre_touch, timeout=10)
            rc, out, err = _ssh_exec(docker_cmd, timeout=15)
        except HTTPException as exc:
            raise HTTPException(status_code=503, detail=f"Cannot reach GPU host: {exc.detail}")
        if rc != 0:
            raise HTTPException(status_code=500, detail=f"docker exec failed: {err[:200] or out[:200]}")
        return {"ok": True, "run_id": run_id, "log_path": log_path, "command_remote": docker_cmd, "dry_run": False}

    # Branch B — fire a script template
    template = entry.get("command_template")
    if not template:
        raise HTTPException(status_code=500, detail=f"entry has neither delegate_kind nor command_template")
    container = entry["docker_container"]
    target = entry["target_host"]
    # Tag the run with experiment + log path. Most existing scripts hardcode
    # their own log path internally — we still pass an env var EXPERIMENT so
    # any future script can pick it up cheaply.
    inner = f"EXPERIMENT={shlex.quote(run_id)} {template}"
    docker_cmd = (
        f"docker exec -d {shlex.quote(container)} bash -c {shlex.quote(inner)}"
    )
    if dry_run:
        return {
            "ok": True, "run_id": run_id, "log_path": entry.get("log_glob"),
            "command_remote": docker_cmd, "dry_run": True,
            "note": "dry_run=true: command was validated but NOT executed.",
        }
    try:
        rc, out, err = _ssh_exec_for_target(target, docker_cmd, timeout=15)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Cannot reach {target}: {exc}")
    if rc != 0:
        raise HTTPException(status_code=500, detail=f"docker exec failed (rc={rc}): {err[:200] or out[:200]}")
    return {"ok": True, "run_id": run_id, "log_path": entry.get("log_glob"), "command_remote": docker_cmd, "dry_run": False}


@router.post("/catalog/{task_id}/start")
async def start_task(task_id: str, req: StartRequest):
    """
    Start a catalog task.

    Body: {"dry_run": bool, "confirmation_token": "START"}

    Conflict + duplicate protection both apply. Returns 409 with
    `reason_if_blocked` on either failure mode.
    """
    entry = _entry_for(task_id)

    # Confirmation token guards against accidental fires from misconfigured
    # callers (the dashboard's confirm() prompt makes the user type/click).
    # dry_run skips the token check — tests need to validate without a token.
    if not req.dry_run and req.confirmation_token != "START":
        raise HTTPException(status_code=400, detail="confirmation_token must equal \"START\"")

    # Duplicate first (cheap, in-memory) — even dry runs are checked, so a
    # caller probing dry_run repeatedly doesn't accidentally hide the dup
    # from a real fire that follows.
    dup_reason = _check_duplicate(task_id)
    if dup_reason:
        raise HTTPException(status_code=409, detail=dup_reason)

    # Conflict (one round-trip SSH). Skip on dry_run only when the caller is
    # explicitly testing the schema — by default we DO check, so the dry_run
    # output reflects what would actually happen.
    conflict_reason = _check_conflict(entry)
    if conflict_reason:
        raise HTTPException(status_code=409, detail=conflict_reason)

    result = _fire_entry(entry, dry_run=req.dry_run)
    if result.get("ok") and not req.dry_run:
        _record_start(task_id, result["run_id"])
        logger.info(f"catalog task {task_id} started: run_id={result['run_id']}")
    result["task_id"] = task_id
    return result


@router.post("/catalog/{task_id}/rerun")
async def rerun_task(task_id: str, req: StartRequest):
    """
    Re-run the most recent run of `task_id` with the same hyperparameters.

    Today, all catalog entries are deterministic (the script or the delegated
    builder hardcodes its hyperparams). So re-run is functionally identical to
    start — but the endpoint exists so the UI can phrase it correctly and so
    we have a hook for a future world where entries carry a `last_args` blob.
    """
    entry = _entry_for(task_id)
    history = _list_history(entry)
    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"No prior run of {task_id} found — use Start instead.",
        )
    return await start_task(task_id, req)
