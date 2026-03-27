import asyncio
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from typing import Optional

from config import LOG_FILE, TRAIN_SCRIPT, TRAIN_DIR, VENV_PYTHON, GPU_TYPE

_state: dict = {
    "proc": None,
    "pid": None,
    "started_at": None,
}

_TQDM_RE = re.compile(
    r"(\d+)%\|[█▉▊▋▌▍▎▏ ]+\|\s*(\d+)/(\d+)\s+\[([0-9:]+)<([^,]+),\s*([0-9.]+)s/it\]"
)
_HF_RE = re.compile(
    r"\{'loss':\s*'([0-9.]+)'.*?'learning_rate':\s*'([0-9.e+-]+)'.*?'mean_token_accuracy':\s*'([0-9.]+)'.*?'epoch':\s*'([0-9.]+)'\}"
)


def is_running() -> bool:
    proc = _state["proc"]
    if proc is None:
        return False
    return proc.poll() is None


def start_training(resume: Optional[str] = None) -> dict:
    if is_running():
        return {"error": "already running", "pid": _state["pid"]}

    cmd = [VENV_PYTHON, TRAIN_SCRIPT]
    if resume:
        cmd += ["--resume_from_checkpoint", resume]

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if GPU_TYPE == "rocm":
        env.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

    os.makedirs(TRAIN_DIR, exist_ok=True)
    log_fh = open(LOG_FILE, "a")

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=TRAIN_DIR,
    )

    _state["proc"] = proc
    _state["pid"] = proc.pid
    _state["started_at"] = datetime.now(timezone.utc).isoformat()
    return {"started": True, "pid": proc.pid}


def stop_training() -> dict:
    proc = _state["proc"]
    if proc is None or proc.poll() is not None:
        return {"error": "not running"}
    proc.terminate()
    return {"stopped": True, "pid": _state["pid"]}


def parse_log() -> dict:
    progress = {}
    latest_metrics = {}
    history = []

    try:
        with open(LOG_FILE, "r", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {"progress": progress, "latest_metrics": latest_metrics, "history": history}

    for line in reversed(lines):
        if not progress:
            m = _TQDM_RE.search(line)
            if m:
                pct, step, total, elapsed, remaining, sec_per_step = m.groups()
                progress = {
                    "pct": int(pct),
                    "step": int(step),
                    "total": int(total),
                    "elapsed": elapsed,
                    "eta": remaining.strip(),
                    "sec_per_step": float(sec_per_step),
                }

        if not latest_metrics:
            m = _HF_RE.search(line)
            if m:
                loss, lr, acc, epoch = m.groups()
                latest_metrics = {
                    "loss": float(loss),
                    "lr": float(lr),
                    "accuracy": float(acc),
                    "epoch": float(epoch),
                }

        if progress and latest_metrics:
            break

    for line in lines:
        m = _HF_RE.search(line)
        if m:
            loss, lr, acc, epoch = m.groups()
            history.append({
                "loss": float(loss),
                "lr": float(lr),
                "accuracy": float(acc),
                "epoch": float(epoch),
            })

    return {
        "progress": progress,
        "latest_metrics": latest_metrics,
        "history": history,
    }


def get_exit_code() -> Optional[int]:
    proc = _state["proc"]
    if proc is None:
        return None
    return proc.poll()
