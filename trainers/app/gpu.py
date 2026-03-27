import asyncio
import json


async def get_gpu_stats(gpu_type: str) -> dict:
    if gpu_type == "rocm":
        return await _rocm_stats()
    return await _cuda_stats()


async def _rocm_stats() -> dict:
    try:
        proc = await asyncio.create_subprocess_exec(
            "rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower", "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        data = json.loads(stdout.decode())
        # rocm-smi JSON structure: {"card0": {...}} or {"GPU[0]": {...}}
        card = next(iter(data.values()))
        return {
            "gpu_pct": float(card.get("GPU use (%)", card.get("GPU Use (%)", 0))),
            "vram_used_mb": _to_mb(card.get("VRAM Total Used Memory (B)", card.get("vram_used", 0))),
            "vram_total_mb": _to_mb(card.get("VRAM Total Memory (B)", card.get("vram_total", 0))),
            "temp_c": float(card.get("Temperature (Sensor edge) (C)", card.get("temp", 0))),
            "power_w": float(card.get("Average Graphics Package Power (W)", card.get("power", 0))),
        }
    except Exception:
        return {}


async def _cuda_stats() -> dict:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        parts = [p.strip() for p in stdout.decode().strip().split(",")]
        return {
            "gpu_pct": float(parts[0]),
            "vram_used_mb": float(parts[1]),
            "vram_total_mb": float(parts[2]),
            "temp_c": float(parts[3]),
            "power_w": float(parts[4]),
        }
    except Exception:
        return {}


def _to_mb(value) -> float:
    try:
        b = float(value)
        return round(b / 1024 / 1024, 1)
    except Exception:
        return 0.0
