"""Avatar routes — full-page UI + JSON/WS API living inside Anamnesis.

Page:
    GET  /avatar                     → HTML page
API:
    GET  /api/avatar/info             → persona, voice defaults, worker status
    GET  /api/avatar/voices           → list {presets: [...], cloned: [...]}
    POST /api/avatar/voices           → upload sample (multipart: kind=file|song|record)
    DELETE /api/avatar/voices/{slug}  → remove a cloned voice
    POST /api/avatar/voices/{slug}/preview → short preview audio (JSON)
    POST /api/avatar/preview-edge     → preview an edge preset (JSON)
    POST /api/avatar/chat             → REST chat (blocking)
    WS   /api/avatar/ws               → streaming chat
    GET  /api/avatar/media/{name}     → serve generated audio/video
"""
import asyncio
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import config
from avatar import voices as voices_module
from avatar.tts.dispatch import synthesize_with_voice
from avatar.pipeline import get_pipeline
from avatar.audio.demucs_client import extract_vocals_via_worker

logger = logging.getLogger("anamnesis.routes.avatar")

router = APIRouter(tags=["avatar"])
templates = Jinja2Templates(directory="templates")

MEDIA_DIR = Path("/tmp/anamnesis_avatar_media")
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


# ─── Page ────────────────────────────────────────────────────────

@router.get("/avatar", response_class=HTMLResponse)
async def avatar_page(request: Request):
    return templates.TemplateResponse(
        "avatar.html",
        {"request": request, "persona": config.AVATAR_PERSONA_NAME},
    )


# ─── Info ────────────────────────────────────────────────────────

@router.get("/api/avatar/info")
async def info():
    p = get_pipeline()
    return {
        "status": "ok",
        **p.info,
        "reference_image": f"/static/img/{os.path.basename(config.AVATAR_REFERENCE_IMAGE)}"
                           if config.AVATAR_REFERENCE_IMAGE else None,
    }


# ─── Voice management ───────────────────────────────────────────

@router.get("/api/avatar/voices")
async def list_voices():
    reg = voices_module.get_registry()
    return {**reg.list_all(), "default_voice_id": config.DEFAULT_VOICE_ID}


@router.post("/api/avatar/voices")
async def upload_voice(
    file: UploadFile = File(...),
    name: str = Form(...),
    kind: str = Form("file"),          # file | song | record
    language: str = Form("en"),
    notes: Optional[str] = Form(None),
):
    if kind not in ("file", "song", "record"):
        raise HTTPException(status_code=400, detail=f"Invalid kind: {kind}")

    reg = voices_module.get_registry()
    tmp = Path(tempfile.mkdtemp(prefix="voice_upload_"))
    suffix = Path(file.filename or "upload").suffix.lower() or ".bin"
    upload_path = tmp / f"upload{suffix}"
    with open(upload_path, "wb") as f:
        f.write(await file.read())
    logger.info(f"Voice upload: name={name}, kind={kind}, bytes={upload_path.stat().st_size}")

    try:
        if kind == "song":
            # Offload vocal separation to the GPU worker
            vocals_raw = tmp / "vocals.wav"
            await extract_vocals_via_worker(str(upload_path), str(vocals_raw))
            # Normalize for XTTS (mono 24kHz)
            normalized = tmp / "normalized.wav"
            await asyncio.to_thread(_to_wav, str(vocals_raw), str(normalized))
            wav_to_register = str(normalized)
        elif kind == "record":
            # Browser sends webm/opus — normalize for XTTS
            normalized = tmp / "normalized.wav"
            await asyncio.to_thread(_to_wav, str(upload_path), str(normalized))
            wav_to_register = str(normalized)
        else:
            normalized = tmp / "normalized.wav"
            await asyncio.to_thread(_to_wav, str(upload_path), str(normalized))
            wav_to_register = str(normalized)

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


@router.delete("/api/avatar/voices/{slug}")
async def delete_voice(slug: str):
    ok = voices_module.get_registry().delete(slug)
    if not ok:
        raise HTTPException(status_code=404, detail="Voice not found")
    return {"ok": True}


@router.post("/api/avatar/voices/{slug}/preview")
async def preview_cloned(slug: str, body: dict):
    text = (body or {}).get("text") or "Hello, I'm here. This is a short preview of my voice."
    reg = voices_module.get_registry()
    v = reg.get_cloned(slug)
    if v is None:
        raise HTTPException(status_code=404, detail="Voice not found")

    out_dir = Path(tempfile.mkdtemp(prefix="voice_preview_"))
    out_path = out_dir / "preview.mp3"
    try:
        await synthesize_with_voice(reg.resolve(v.id), text, str(out_path))
    except Exception as e:
        logger.exception("Preview failed")
        raise HTTPException(status_code=500, detail=str(e))
    name = f"preview_{slug}_{id(out_path):x}.mp3"
    _serve_file(str(out_path), name)
    return {"audio_url": f"/api/avatar/media/{name}"}


@router.post("/api/avatar/preview-edge")
async def preview_edge(body: dict):
    text = body.get("text") or "Hello, this is a quick preview."
    voice_id = body.get("voice_id")
    if not voice_id or not voice_id.startswith("edge:"):
        raise HTTPException(status_code=400, detail="voice_id must start with 'edge:'")
    reg = voices_module.get_registry()
    out_dir = Path(tempfile.mkdtemp(prefix="voice_preview_"))
    out_path = out_dir / "preview.mp3"
    try:
        await synthesize_with_voice(reg.resolve(voice_id), text, str(out_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    name = f"preview_edge_{hash(voice_id) & 0xFFFFFFFF:08x}.mp3"
    _serve_file(str(out_path), name)
    return {"audio_url": f"/api/avatar/media/{name}"}


# ─── Chat: REST + WebSocket ─────────────────────────────────────

@router.post("/api/avatar/chat")
async def chat_rest(body: dict):
    message = (body or {}).get("message", "").strip()
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)
    voice_id = body.get("voice_id")
    animate = body.get("animate")

    pipeline = get_pipeline()
    result = await pipeline.process(message, voice_id=voice_id, animate=animate)

    resp = {"text": result.text, "timings": result.timings}
    if result.error:
        resp["error"] = result.error
    if result.audio_path:
        name = f"audio_{id(result):x}.mp3"
        _serve_file(result.audio_path, name)
        resp["audio_url"] = f"/api/avatar/media/{name}"
    if result.video_path:
        name = f"video_{id(result):x}.mp4"
        _serve_file(result.video_path, name)
        resp["video_url"] = f"/api/avatar/media/{name}"
    return resp


@router.websocket("/api/avatar/ws")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    pipeline = get_pipeline()
    logger.info("Avatar WS connected")

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            message = (data.get("message") or "").strip()
            if not message:
                await ws.send_json({"type": "error", "message": "Empty message"})
                continue
            voice_id = data.get("voice_id")
            animate = data.get("animate")

            async def on_token(token: str):
                await ws.send_json({"type": "token", "data": token})

            async def on_audio(path: str):
                name = f"audio_{hash(path) & 0xFFFFFFFF:08x}.mp3"
                _serve_file(path, name)
                await ws.send_json({"type": "audio", "url": f"/api/avatar/media/{name}"})

            async def on_video(path: str):
                name = f"video_{hash(path) & 0xFFFFFFFF:08x}.mp4"
                _serve_file(path, name)
                await ws.send_json({"type": "video", "url": f"/api/avatar/media/{name}"})

            result = await pipeline.process(
                message, voice_id=voice_id, animate=animate,
                on_token=on_token, on_audio=on_audio, on_video=on_video,
            )
            done = {"type": "done", "timings": result.timings}
            if result.error:
                done["error"] = result.error
            await ws.send_json(done)

    except WebSocketDisconnect:
        logger.info("Avatar WS disconnected")
    except Exception as e:
        logger.exception("Avatar WS error")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ─── Media serving ──────────────────────────────────────────────

@router.get("/api/avatar/media/{filename}")
async def serve_media(filename: str):
    # Basic path hygiene
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="bad filename")
    path = MEDIA_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    media_type = "audio/mpeg" if filename.endswith(".mp3") else "video/mp4"
    return FileResponse(str(path), media_type=media_type)


def _serve_file(src: str, dest_name: str):
    dest = MEDIA_DIR / dest_name
    if dest.exists():
        dest.unlink()
    os.symlink(src, dest)


def _to_wav(src: str, dst: str, sample_rate: int = 24000):
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", str(sample_rate),
         "-acodec", "pcm_s16le", dst],
        check=True, capture_output=True,
    )
