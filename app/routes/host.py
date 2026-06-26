"""
host.py — local host introspection probe for the Anamnesis app container.

Symmetric to d2/server.py:/host and trainers/app/main.py:/host so the
unified Resource Status / Machines panel can render telemetry for the
machine that runs the Anamnesis app itself (typically dellserver).

Honest, best-effort: every field is optional. If something can't be probed
it stays null / empty rather than reporting a fake zero.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger("anamnesis.routes.host")

router = APIRouter(tags=["host"])

_SERVICE_START_TS = time.time()


def _outbound_ip() -> str | None:
    """UDP-trick to discover our outbound IP without sending a packet."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return None


async def _nvidia_smi_probe() -> tuple[list[dict[str, Any]], str | None]:
    """Best-effort nvidia-smi probe. Returns (gpus, error)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,"
            "utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=2.0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return [], "nvidia-smi timeout"
    except FileNotFoundError:
        return [], "nvidia-smi not in PATH"
    except Exception as e:
        return [], f"{type(e).__name__}: {str(e)[:120]}"

    gpus: list[dict[str, Any]] = []
    for line in stdout.decode(errors="replace").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            try:
                total = int(float(parts[1]))
                used = int(float(parts[2]))
                free = int(float(parts[3]))
                gpus.append({
                    "name": parts[0],
                    "vram_total_mib": total,
                    "vram_used_mib": used,
                    "vram_free_mib": free,
                    "vram_percent": round(100.0 * used / total, 1) if total else None,
                    "util_pct": int(float(parts[4])),
                    "temp_c": int(float(parts[5])),
                    "power_w": int(float(parts[6])) if parts[6] not in ("[Not Supported]", "N/A") else None,
                })
            except ValueError:
                pass
    # nvidia-smi can exit non-zero with empty stdout and the reason on stderr
    # (e.g. NVML driver/library mismatch). Don't let that read as "no GPU" —
    # surface it so the panel distinguishes "no GPU" from "probe failed".
    if not gpus and proc.returncode not in (0, None):
        lines = stderr.decode(errors="replace").strip().splitlines()
        return [], (lines[0] if lines else f"nvidia-smi exited {proc.returncode}")
    return gpus, None


@router.get("/host")
async def host_info() -> dict[str, Any]:
    """
    Hostname, CPU%, load avg, RAM usage, per-GPU stats (if any), uptime, role.

    The Anamnesis Resource Status panel calls this to populate one card per
    machine in the Machines section. Returns null fields rather than fake
    zeros when data isn't available.
    """
    info: dict[str, Any] = {
        "service": "anamnesis-app",
        "hostname": socket.gethostname(),
        "uptime_s": int(time.time() - _SERVICE_START_TS),
        "ip": _outbound_ip(),
        "cpu": None,
        "ram": None,
        "gpus": [],
        "roles": ["anamnesis-app"],
    }

    # CPU + RAM — psutil is the canonical lightweight cross-platform probe.
    try:
        import psutil  # type: ignore

        # 1-second CPU sample. Off-thread so we don't block the event loop.
        loop = asyncio.get_event_loop()
        cpu_pct = await loop.run_in_executor(None, lambda: psutil.cpu_percent(interval=1.0))
        cores = psutil.cpu_count(logical=True) or 0
        try:
            la1, la5, _la15 = os.getloadavg()
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
            "used_gb": round(vm.used / (1024 ** 3), 2),
            "total_gb": round(vm.total / (1024 ** 3), 2),
            "percent": round(vm.percent, 1),
        }
    except ImportError:
        info["cpu_probe_error"] = "psutil not installed"
    except Exception as e:
        info["cpu_probe_error"] = f"{type(e).__name__}: {str(e)[:120]}"

    # GPU — Anamnesis app box (dellserver) usually has none; probe anyway.
    gpus, gpu_err = await _nvidia_smi_probe()
    info["gpus"] = gpus
    if gpu_err and not gpus:
        info["gpu_probe_error"] = gpu_err

    return info
