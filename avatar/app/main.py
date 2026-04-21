"""
Avatar service — FastAPI app with WebSocket + voice management.

Endpoints:
  GET  /                           → Chat UI
  GET  /health                     → Service health + backend info
  GET  /media/{path}               → Serve generated audio/video files
  WS   /ws                         → Chat WebSocket (streams tokens + audio + video)
  POST /api/chat                   → REST fallback

  GET  /api/voices                 → List available voices (edge presets + cloned)
  POST /api/voices/upload          → Upload a sample (file|song|record). Multipart.
  POST /api/voices/{slug}/preview  → Generate a short preview of the voice
  DELETE /api/voices/{slug}        → Remove a cloned voice
"""
import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
import tts
import voices
from pipeline import AvatarPipeline

# ─── Logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("avatar")

# ─── App ─────────────────────────────────────────────────────────
app = FastAPI(title=f"Avatar — {config.PERSONA_NAME}", version="0.2.0")

STATIC_DIR = Path(__file__).parent / "static"
MEDIA_DIR = Path("/tmp/avatar_media")
MEDIA_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_pipeline: Optional[AvatarPipeline] = None


def _get_pipeline() -> AvatarPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AvatarPipeline()
    return _pipeline


# ─── Core routes ─────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/health")
async def health():
    pipeline = _get_pipeline()
    return {
        "status": "ok",
        "persona": config.PERSONA_NAME,
        "machine": config.MACHINE_NAME,
        "gpu_type": config.GPU_TYPE,
        "backends": pipeline.backends,
        "default_voice_id": config.DEFAULT_VOICE_ID,
    }


@app.post("/api/chat")
async def chat_rest(body: dict):
    message = body.get("message", "").strip()
    voice_id = body.get("voice_id")
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    pipeline = _get_pipeline()
    result = await pipeline.process(message, voice_id=voice_id)

    response = {"text": result.text, "timings": result.timings}
    if result.error:
        response["error"] = result.error
    if result.audio_path:
        audio_name = f"audio_{id(result)}.mp3"
        _serve_file(result.audio_path, audio_name)
        response["audio_url"] = f"/media/{audio_name}"
    if result.video_path:
        video_name = f"video_{id(result)}.mp4"
        _serve_file(result.video_path, video_name)
        response["video_url"] = f"/media/{video_name}"
    return response


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    """
    Client → {"message": "hello", "voice_id": "edge:en-US-AvaNeural"}
    Server → streaming {type: token|audio|video|done|error, ...}
    """
    await ws.accept()
    pipeline = _get_pipeline()
    logger.info("WebSocket connected")

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            message = data.get("message", "").strip()
            voice_id = data.get("voice_id")
            if not message:
                await ws.send_json({"type": "error", "message": "Empty message"})
                continue

            async def on_token(token: str):
                await ws.send_json({"type": "token", "data": token})

            async def on_audio(path: str):
                audio_name = f"audio_{hash(path) & 0xFFFFFFFF:08x}.mp3"
                _serve_file(path, audio_name)
                await ws.send_json({"type": "audio", "url": f"/media/{audio_name}"})

            async def on_video(path: str):
                video_name = f"video_{hash(path) & 0xFFFFFFFF:08x}.mp4"
                _serve_file(path, video_name)
                await ws.send_json({"type": "video", "url": f"/media/{video_name}"})

            result = await pipeline.process(
                message,
                voice_id=voice_id,
                on_token=on_token,
                on_audio=on_audio,
                on_video=on_video,
            )

            done_msg = {"type": "done", "timings": result.timings}
            if result.error:
                done_msg["error"] = result.error
            await ws.send_json(done_msg)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ─── Voice management API ───────────────────────────────────────

@app.get("/api/voices")
async def list_voices():
    reg = voices.get_registry()
    return {
        **reg.list_all(),
        "default_voice_id": config.DEFAULT_VOICE_ID,
    }


@app.post("/api/voices/upload")
async def upload_voice(
    file: UploadFile = File(...),
    name: str = Form(...),
    kind: str = Form("file"),           # "file" | "song" | "record"
    language: str = Form("en"),
    notes: Optional[str] = Form(None),
):
    """
    Upload an audio sample and register it as a cloned voice.

    kind=file    → treat as ready-to-use speech sample
    kind=song    → run Demucs to isolate vocals, then register vocals.wav
    kind=record  → browser-recorded WebM/Opus; converted to WAV
    """
    if kind not in ("file", "song", "record"):
        raise HTTPException(status_code=400, detail=f"Invalid kind: {kind}")

    reg = voices.get_registry()

    # Save upload to a temp file (preserving extension so ffmpeg/demucs can sniff it)
    suffix = Path(file.filename or "upload").suffix.lower() or ".bin"
    tmpdir = Path(tempfile.mkdtemp(prefix="voice_upload_"))
    upload_path = tmpdir / f"upload{suffix}"
    with open(upload_path, "wb") as f:
        f.write(await file.read())
    logger.info(f"Voice upload: name={name}, kind={kind}, bytes={upload_path.stat().st_size}")

    try:
        # Decide what WAV to register
        if kind == "song":
            from audio.demucs_extract import extract_vocals
            # Demucs wants a file it can decode — mp3/wav/flac all work
            wav_to_register = await extract_vocals(str(upload_path), out_dir=str(tmpdir / "demucs"))
        elif kind == "record":
            # Browser typically sends webm/opus; convert to 24kHz mono WAV for XTTS
            wav_to_register = str(tmpdir / "recorded.wav")
            await asyncio.to_thread(_convert_to_wav, str(upload_path), wav_to_register)
        else:  # file
            # Accept WAV/MP3/FLAC; normalize to a clean mono 24kHz WAV for XTTS
            wav_to_register = str(tmpdir / "normalized.wav")
            await asyncio.to_thread(_convert_to_wav, str(upload_path), wav_to_register)

        voice = reg.add(
            name=name,
            source_wav_path=wav_to_register,
            source=kind,
            language=language,
            original_filename=file.filename,
            notes=notes,
        )
        return {"ok": True, "voice": voice.to_public()}

    except Exception as e:
        logger.exception("Voice upload failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/voices/{slug}")
async def delete_voice(slug: str):
    reg = voices.get_registry()
    ok = reg.delete(slug)
    if not ok:
        raise HTTPException(status_code=404, detail="Voice not found")
    return {"ok": True}


@app.post("/api/voices/{slug}/preview")
async def preview_voice(slug: str, body: dict):
    text = (body or {}).get("text") or "Hello, I'm here. This is a short preview of my voice."
    reg = voices.get_registry()
    voice = reg.get_cloned(slug)
    if voice is None:
        raise HTTPException(status_code=404, detail="Voice not found")

    out_dir = Path(tempfile.mkdtemp(prefix="voice_preview_"))
    out_path = out_dir / "preview.mp3"
    voice_spec = reg.resolve(voice.id)

    try:
        await tts.synthesize_with_voice(voice_spec, text, str(out_path))
    except Exception as e:
        logger.exception("Preview failed")
        raise HTTPException(status_code=500, detail=str(e))

    # Serve via /media/
    preview_name = f"preview_{slug}_{id(out_path):x}.mp3"
    _serve_file(str(out_path), preview_name)
    return {"audio_url": f"/media/{preview_name}"}


@app.post("/api/preview-edge")
async def preview_edge_voice(body: dict):
    """Preview an edge-tts preset (no slug — the voice is encoded in voice_id)."""
    text = body.get("text") or "Hello, this is a quick preview."
    voice_id = body.get("voice_id")
    if not voice_id or not voice_id.startswith("edge:"):
        raise HTTPException(status_code=400, detail="voice_id must be an 'edge:*' id")

    reg = voices.get_registry()
    spec = reg.resolve(voice_id)

    out_dir = Path(tempfile.mkdtemp(prefix="voice_preview_"))
    out_path = out_dir / "preview.mp3"
    try:
        await tts.synthesize_with_voice(spec, text, str(out_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    preview_name = f"preview_edge_{hash(voice_id) & 0xFFFFFFFF:08x}.mp3"
    _serve_file(str(out_path), preview_name)
    return {"audio_url": f"/media/{preview_name}"}


# ─── Media serving ───────────────────────────────────────────────

@app.get("/media/{filename}")
async def serve_media(filename: str):
    path = MEDIA_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    media_type = "audio/mpeg" if filename.endswith(".mp3") else "video/mp4"
    return FileResponse(str(path), media_type=media_type)


def _serve_file(src: str, dest_name: str):
    dest = MEDIA_DIR / dest_name
    if dest.exists():
        dest.unlink()
    os.symlink(src, dest)


# ─── Helpers ─────────────────────────────────────────────────────

def _convert_to_wav(src: str, dst: str, sample_rate: int = 24000):
    """Normalize arbitrary audio input to mono 24kHz WAV via ffmpeg."""
    import subprocess
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", src,
            "-ac", "1",                         # mono
            "-ar", str(sample_rate),            # resample
            "-acodec", "pcm_s16le",
            dst,
        ],
        check=True,
        capture_output=True,
    )


# ─── Entrypoint ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
