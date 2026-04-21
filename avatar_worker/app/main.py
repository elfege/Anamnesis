"""Avatar GPU worker — thin FastAPI service exposing XTTS, Demucs, SadTalker.

Intentionally minimal: no persistence, no auth, accepts raw multipart uploads
and returns raw media bytes. Same image runs on CUDA/ROCm/CPU — backend
selected by the TORCH_INDEX_URL build arg and GPU_TYPE env var.
"""
import logging
import os
import shutil
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

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("worker")

app = FastAPI(title="Avatar GPU Worker", version="0.1.0")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpu_type": config.GPU_TYPE,
        "machine": config.MACHINE_NAME,
        "backends": ["xtts", "demucs", "sadtalker"],
        "sadtalker_ready": await sadtalker.is_ready(),
    }


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

    # Write speaker reference (may be WAV or anything ffmpeg can read → normalize)
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


@app.post("/demucs/extract")
async def demucs_route(
    audio: UploadFile = File(...),
    trim_seconds: float = Form(15.0),
):
    """Extract vocals from a song; returns trimmed vocals WAV."""
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


@app.post("/sadtalker/animate")
async def sadtalker_route(
    audio: UploadFile = File(...),
    image: UploadFile = File(...),
):
    """Produce lip-synced video from audio + reference face."""
    tmp = Path(tempfile.mkdtemp(prefix="sadtalker_"))
    audio_path = tmp / f"audio{Path(audio.filename or 'audio.wav').suffix or '.wav'}"
    image_path = tmp / f"image{Path(image.filename or 'image.png').suffix or '.png'}"
    out_dir = tmp / "out"

    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    with open(image_path, "wb") as f:
        f.write(await image.read())

    try:
        mp4 = await sadtalker.animate(str(audio_path), str(image_path), str(out_dir))
    except Exception as e:
        logger.exception("SadTalker failed")
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(mp4, media_type="video/mp4", filename="anim.mp4")


# ── helpers ─────────────────────────────────────────────────

def _to_wav(src: str, dst: str, sample_rate: int = 24000):
    """Normalize arbitrary audio → mono 24kHz WAV."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", src,
            "-ac", "1",
            "-ar", str(sample_rate),
            "-acodec", "pcm_s16le",
            dst,
        ],
        check=True,
        capture_output=True,
    )


def _wav_to_mp3(src_wav: str, dst_mp3: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_wav, "-codec:a", "libmp3lame", "-q:a", "2", dst_mp3],
        check=True,
        capture_output=True,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
