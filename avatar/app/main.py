"""
Avatar service — FastAPI app with WebSocket for real-time chat + animated persona.

Endpoints:
  GET  /                  → Chat UI
  GET  /health            → Service health + backend info
  GET  /media/{path}      → Serve generated audio/video files
  WS   /ws                → WebSocket: send message, receive streaming text + media URLs
  POST /api/chat          → REST fallback: send message, get response + media
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from pipeline import AvatarPipeline

# ─── Logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("avatar")

# ─── App ─────────────────────────────────────────────────────────
app = FastAPI(title=f"Avatar — {config.PERSONA_NAME}", version="0.1.0")

# Serve static files (reference image, generated media)
STATIC_DIR = Path(__file__).parent / "static"
MEDIA_DIR = Path("/tmp/avatar_media")
MEDIA_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── Pipeline (lazy init) ───────────────────────────────────────
_pipeline: AvatarPipeline | None = None


def _get_pipeline() -> AvatarPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AvatarPipeline()
    return _pipeline


# ─── Routes ──────────────────────────────────────────────────────

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
    }


@app.post("/api/chat")
async def chat_rest(body: dict):
    """REST endpoint — for clients that don't support WebSocket."""
    message = body.get("message", "").strip()
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    pipeline = _get_pipeline()
    result = await pipeline.process(message)

    response = {
        "text": result.text,
        "timings": result.timings,
    }
    if result.error:
        response["error"] = result.error
    if result.audio_path:
        # Copy to serveable location
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
    WebSocket protocol:
      Client sends: {"message": "hello"}
      Server sends (streaming):
        {"type": "token", "data": "Hello"}       — LLM tokens
        {"type": "audio", "url": "/media/..."}    — TTS audio ready
        {"type": "video", "url": "/media/..."}    — Animated video ready
        {"type": "done", "timings": {...}}         — Pipeline complete
        {"type": "error", "message": "..."}        — Error
    """
    await ws.accept()
    pipeline = _get_pipeline()
    logger.info("WebSocket connected")

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            message = data.get("message", "").strip()
            if not message:
                await ws.send_json({"type": "error", "message": "Empty message"})
                continue

            # Stream tokens back in real-time
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


# ─── Media serving ───────────────────────────────────────────────

@app.get("/media/{filename}")
async def serve_media(filename: str):
    path = MEDIA_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    media_type = "audio/mpeg" if filename.endswith(".mp3") else "video/mp4"
    return FileResponse(str(path), media_type=media_type)


def _serve_file(src: str, dest_name: str):
    """Symlink a generated file into the media dir for serving."""
    dest = MEDIA_DIR / dest_name
    if dest.exists():
        dest.unlink()
    os.symlink(src, dest)


# ─── Entrypoint ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
