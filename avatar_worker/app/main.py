"""Avatar GPU worker — thin FastAPI service exposing XTTS, Demucs, SadTalker.

Intentionally minimal: no persistence, no auth, accepts raw multipart uploads
and returns raw media bytes. Same image runs on CUDA/ROCm/CPU — backend
selected by the TORCH_INDEX_URL build arg and GPU_TYPE env var.
"""
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import config
import xtts
import demucs_extract
import sadtalker
import musetalk

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("worker")

app = FastAPI(title="Avatar GPU Worker", version="0.2.0")


# ─── Health / discovery ────────────────────────────────────────

def _vram_mb() -> tuple[Optional[int], Optional[int]]:
    """Return (total_mb, free_mb) or (None, None) if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        free, _ = torch.cuda.mem_get_info(0)
        free_mb = free // (1024 * 1024)
        return int(total), int(free_mb)
    except Exception:
        return None, None


_SERVICE_START_TS = __import__("time").time()


@app.get("/host")
async def host_telemetry():
    """Box-level telemetry — CPU/RAM/GPU/hostname. Mirrors the canonical
    /host shape exposed by app/routes/host.py on anamnesis-app, so the
    Anamnesis Resource Status panel's probe parser (app/routes/resources.py)
    renders this row identically to dellserver's and server's. Aligned
    2026-06-08 — prior shape was flat fields (cpu_count, ram_total_mb...)
    which the parser ignored, showing the row as ok=true but cpu/ram/gpus=null.

    Schema contract (all sub-fields optional — return null/empty rather than
    fake zeros when data isn't available):
      service, hostname, uptime_s, machine, worker_id, gpu_type,
      cpu  = {percent, cores, load_1, load_5},
      ram  = {used_gb, total_gb, percent},
      gpus = [{name, vram_total_mib, vram_used_mib, vram_free_mib, vram_percent,
               util_pct, temp_c, power_w}],
      roles = [str],
      cpu_probe_error / gpu_probe_error  (optional)
    """
    import socket
    import platform
    import time as _time

    info: dict = {
        "service": "avatar-worker",
        "hostname": socket.gethostname(),
        "uptime_s": int(_time.time() - _SERVICE_START_TS),
        "platform": platform.platform(),
        "machine": config.MACHINE_NAME,
        "worker_id": config.WORKER_ID,
        "gpu_type": config.GPU_TYPE,
        "cpu": None,
        "ram": None,
        "gpus": [],
        "roles": ["avatar-worker", "xtts", "sadtalker", "demucs", "musetalk"],
    }

    # CPU + RAM via psutil. Canonical nested-dict shape the dashboard expects.
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.5)
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
    except Exception as exc:
        info["cpu_probe_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"

    # GPU via torch (works on both CUDA and ROCm — torch.cuda is the API
    # surface AMD ROCm uses too). util_pct/temp_c/power_w aren't exposed
    # by torch; mark null. Future: shell out to rocm-smi --json for those.
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            free, _ = torch.cuda.mem_get_info(0)
            used = total - free
            info["gpus"] = [{
                "name": name,
                "vram_total_mib": int(total // (1024 * 1024)),
                "vram_used_mib": int(used // (1024 * 1024)),
                "vram_free_mib": int(free // (1024 * 1024)),
                "vram_percent": round(100.0 * used / total, 1) if total else None,
                "util_pct": None,
                "temp_c": None,
                "power_w": None,
            }]
    except Exception as exc:
        info["gpu_probe_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"

    return info


@app.get("/health")
async def health():
    """Rich health response — schema agreed with Anamnesis side (intercom MSG-119)."""
    sadtalker_ready = await sadtalker.is_ready()
    musetalk_ready = await musetalk.is_ready()
    capabilities = ["xtts", "demucs"]
    if sadtalker_ready:
        capabilities.append("sadtalker")
    if musetalk_ready:
        capabilities.append("musetalk")

    vram_total, vram_free = _vram_mb()

    return {
        "status": "ok",
        "worker_id": config.WORKER_ID,
        "machine": config.MACHINE_NAME,
        "gpu_type": config.GPU_TYPE,
        "capabilities": capabilities,
        "vram_total_mb": vram_total,
        "vram_free_mb": vram_free,
        "model_loaded": {
            "xtts": xtts.is_loaded(),
            "sadtalker": sadtalker_ready,
            "musetalk": musetalk_ready,
        },
        # Legacy fields kept for back-compat with older Anamnesis clients
        "backends": ["xtts", "demucs", "sadtalker", "musetalk"],
        "sadtalker_ready": sadtalker_ready,
        "musetalk_ready": musetalk_ready,
    }


# ─── XTTS ─────────────────────────────────────────────────────

@app.post("/xtts/synthesize")
async def xtts_synthesize(
    text: str = Form(...),
    language: str = Form("en"),
    speaker: UploadFile = File(...),
    want_mp3: bool = Form(True),
):
    """Clone the voice in `speaker` and synthesize `text`. Returns WAV or MP3."""
    tmp = Path(tempfile.mkdtemp(prefix="xtts_"))
    speaker_path = tmp / "speaker_normalized.wav"
    out_wav = tmp / "out.wav"

    suffix = Path(speaker.filename or "upload").suffix or ".bin"
    raw = tmp / f"speaker_raw{suffix}"
    with open(raw, "wb") as f:
        f.write(await speaker.read())
    _to_wav(str(raw), str(speaker_path))

    try:
        await xtts.synthesize(text, str(speaker_path), str(out_wav), language=language)
    except Exception as e:
        logger.exception("XTTS synthesis failed")
        raise HTTPException(status_code=500, detail=str(e))

    if want_mp3:
        out_mp3 = tmp / "out.mp3"
        _wav_to_mp3(str(out_wav), str(out_mp3))
        return FileResponse(str(out_mp3), media_type="audio/mpeg", filename="out.mp3")
    return FileResponse(str(out_wav), media_type="audio/wav", filename="out.wav")


@app.post("/xtts/unload")
async def xtts_unload_endpoint():
    """Explicit unload — lets Anamnesis free VRAM ahead of heavy ops."""
    unloaded = xtts.unload()
    return {"unloaded": unloaded, "now_loaded": xtts.is_loaded()}


# ─── Demucs ───────────────────────────────────────────────────

@app.post("/demucs/extract")
async def demucs_route(
    audio: UploadFile = File(...),
    trim_seconds: float = Form(15.0),
):
    tmp = Path(tempfile.mkdtemp(prefix="demucs_"))
    src = tmp / f"src{Path(audio.filename or 'upload').suffix or '.mp3'}"
    with open(src, "wb") as f:
        f.write(await audio.read())
    try:
        vocals = await demucs_extract.extract_vocals(
            str(src), out_dir=str(tmp / "out"), trim_seconds=trim_seconds
        )
    except Exception as e:
        logger.exception("Demucs failed")
        raise HTTPException(status_code=500, detail=str(e))
    return FileResponse(vocals, media_type="audio/wav", filename="vocals.wav")


# ─── SadTalker ────────────────────────────────────────────────

@app.post("/sadtalker/animate")
async def sadtalker_route(
    audio: UploadFile = File(...),
    image: UploadFile = File(...),
):
    """Produce lip-synced video from audio + reference face.

    On tight-VRAM hosts we unload XTTS first to free ~3GB before loading
    SadTalker (~4GB peak at 512px). Next XTTS call pays the reload cost.
    Governed by UNLOAD_XTTS_BEFORE_SADTALKER env var — default true on
    machines where `config.AUTO_UNLOAD_XTTS` is True.
    """
    tmp = Path(tempfile.mkdtemp(prefix="sadtalker_"))
    audio_path = tmp / f"audio{Path(audio.filename or 'audio.wav').suffix or '.wav'}"
    image_path = tmp / f"image{Path(image.filename or 'image.png').suffix or '.png'}"
    out_dir = tmp / "out"

    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    with open(image_path, "wb") as f:
        f.write(await image.read())

    if config.AUTO_UNLOAD_XTTS and xtts.is_loaded():
        unloaded = xtts.unload()
        logger.info(f"Auto-unloaded XTTS before SadTalker: {unloaded}")

    try:
        mp4 = await sadtalker.animate(str(audio_path), str(image_path), str(out_dir))
    except Exception as e:
        logger.exception("SadTalker failed")
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(mp4, media_type="video/mp4", filename="anim.mp4")


@app.post("/sadtalker/cancel")
async def sadtalker_cancel():
    """Kill any in-flight SadTalker subprocess. Idempotent."""
    killed = await sadtalker.cancel_all()
    return {"killed": killed}


# ─── MuseTalk ─────────────────────────────────────────────────

@app.post("/musetalk/animate")
async def musetalk_route(
    audio: UploadFile = File(...),
    image: UploadFile = File(...),
):
    """Produce lip-synced video from audio + reference face via MuseTalk.

    Same multipart contract as /sadtalker/animate so the orchestrator's
    MuseTalkClient can use the same call pattern. Unlike SadTalker, MuseTalk
    needs ~real-time on a 16GB ROCm card — typically <10s for a 3s clip
    after the one-time MIOpen kernel autotune is warmed."""
    tmp = Path(tempfile.mkdtemp(prefix="musetalk_"))
    audio_path = tmp / f"audio{Path(audio.filename or 'audio.wav').suffix or '.wav'}"
    image_path = tmp / f"image{Path(image.filename or 'image.png').suffix or '.png'}"
    out_dir = tmp / "out"

    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    with open(image_path, "wb") as f:
        f.write(await image.read())

    # MuseTalk's UNet (~1.5GB fp16) + VAE (~300MB) + face_alignment fits
    # comfortably in 16GB alongside XTTS — no auto-unload needed on office.
    # On smaller cards, callers should /xtts/unload before invoking.

    try:
        mp4 = await musetalk.animate(str(audio_path), str(image_path), str(out_dir))
    except Exception as e:
        logger.exception("MuseTalk failed")
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(mp4, media_type="video/mp4", filename="anim.mp4")


@app.post("/musetalk/cancel")
async def musetalk_cancel():
    """Kill any in-flight MuseTalk subprocess. Idempotent."""
    killed = await musetalk.cancel_all()
    return {"killed": killed}


# ─── helpers ──────────────────────────────────────────────────

def _to_wav(src: str, dst: str, sample_rate: int = 24000):
    """Normalize arbitrary audio → mono 24kHz WAV."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", str(sample_rate),
         "-acodec", "pcm_s16le", dst],
        check=True, capture_output=True,
    )


def _wav_to_mp3(src_wav: str, dst_mp3: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_wav, "-codec:a", "libmp3lame", "-q:a", "2", dst_mp3],
        check=True, capture_output=True,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
