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

import httpx

import config
from avatar import voices as voices_module
from avatar.tts.dispatch import synthesize_with_voice
from avatar.pipeline import get_pipeline
from avatar.audio.demucs_client import extract_vocals_via_worker
from avatar.workers import probe_all_workers
from avatar.llm import ANAMNESIS_GPT_ENDPOINTS
from database import (
    list_chat_sessions,
    get_chat_session,
    delete_chat_session,
    rename_chat_session,
)

logger = logging.getLogger("anamnesis.routes.avatar")

router = APIRouter(tags=["avatar"])
templates = Jinja2Templates(directory="templates")

MEDIA_DIR = Path("/tmp/anamnesis_avatar_media")
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


# ─── Page ────────────────────────────────────────────────────────

@router.get("/avatar", response_class=HTMLResponse)
async def avatar_page(request: Request):
    # Cache-bust the static portrait image — browsers aggressively cache static
    # PNG urls and a "hard refresh" doesn't always bust them. Stamp mtime as a
    # query string so the URL changes whenever the file does.
    display_path = "/app/static/img/belle_display.png"
    try:
        display_v = str(int(os.path.getmtime(display_path)))
    except OSError:
        display_v = "0"
    return templates.TemplateResponse(
        "avatar.html",
        {
            "request": request,
            "persona": config.AVATAR_PERSONA_NAME,
            "display_image_version": display_v,
        },
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
    session_id = body.get("session_id")
    backend = body.get("backend") or "ollama"
    model = body.get("model")
    # Per-service worker selection (Phase 2). Accept both new fields plus the
    # legacy `preferred_worker` as a fallback so old clients keep working.
    preferred_tts_worker = body.get("preferred_tts_worker")
    preferred_animation_worker = body.get("preferred_animation_worker")
    preferred_worker = body.get("preferred_worker")
    animation_engine = body.get("animation_engine") or "sadtalker"
    no_fallback = bool(body.get("no_fallback"))

    pipeline = get_pipeline()
    result = await pipeline.process(
        message, voice_id=voice_id, animate=animate, session_id=session_id,
        backend=backend, model=model,
        preferred_tts_worker=preferred_tts_worker,
        preferred_animation_worker=preferred_animation_worker,
        preferred_worker=preferred_worker,
        animation_engine=animation_engine,
        no_fallback=no_fallback,
    )

    resp = {"text": result.text, "timings": result.timings, "session_id": result.session_id}
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

    # Current in-flight pipeline task — user can cancel via {type: "stop"}
    current_task: Optional[asyncio.Task] = None

    async def run_pipeline(data: dict):
        """Run a single pipeline cycle and stream results over the WS."""
        message = (data.get("message") or "").strip()
        if not message:
            await ws.send_json({"type": "error", "message": "Empty message"})
            return
        voice_id = data.get("voice_id")
        animate = data.get("animate")
        session_id = data.get("session_id")
        backend = data.get("backend") or "ollama"
        model = data.get("model")
        # Per-service worker selection (Phase 2). Both new fields + legacy
        # fallback. Frontend sends preferred_worker: null when on the new
        # split UI; the per-service fields carry the actual choice.
        preferred_tts_worker = data.get("preferred_tts_worker")
        preferred_animation_worker = data.get("preferred_animation_worker")
        preferred_worker = data.get("preferred_worker")
        animation_engine = data.get("animation_engine") or "sadtalker"
        no_fallback = bool(data.get("no_fallback"))
        # Advanced overrides — None means "use server default".
        system_prompt_override = data.get("system_prompt_override") or None
        sampling = data.get("sampling") or {}

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

        try:
            result = await pipeline.process(
                message, voice_id=voice_id, animate=animate, session_id=session_id,
                backend=backend, model=model,
                preferred_tts_worker=preferred_tts_worker,
                preferred_animation_worker=preferred_animation_worker,
                preferred_worker=preferred_worker,
                animation_engine=animation_engine,
                no_fallback=no_fallback,
                system_prompt_override=system_prompt_override,
                sampling=sampling,
                on_token=on_token, on_audio=on_audio, on_video=on_video,
            )
            done = {"type": "done", "timings": result.timings, "session_id": result.session_id}
            if result.error:
                done["error"] = result.error
            await ws.send_json(done)
        except asyncio.CancelledError:
            # User pressed stop — notify client and swallow the cancellation
            try:
                await ws.send_json({"type": "stopped"})
            except Exception:
                pass
            raise

    # Track which session this connection has a render in flight for. We
    # only fire the worker-side cancel when the user EXPLICITLY hits Stop,
    # not on page reload / new message preemption. Reason: cancel_all() is
    # global — it kills every SadTalker subprocess on every worker, not just
    # one tied to this WS. The previous over-eager calls were killing every
    # render mid-stream (operator confirmed via debug logs 2026-06-11 02:45 EDT).
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)

            # Control message: explicit Stop button. This is the ONE place
            # we broadcast cancel to every engine — user wants every GPU job
            # dead now, on every worker, across every engine they might have
            # been routed to.
            if data.get("type") == "stop":
                if current_task and not current_task.done():
                    current_task.cancel()
                # Animation engines: SadTalker is persistent on pipeline;
                # MuseTalk is constructed per-request, but its cancel_all
                # only walks config.AVATAR_WORKER_ENDPOINTS so a fresh
                # instance is fine.
                async def _broadcast_cancel():
                    from avatar.animation.musetalk_client import MuseTalkClient
                    try:
                        await pipeline._sadtalker.cancel_all()
                    except Exception:
                        pass
                    try:
                        await MuseTalkClient().cancel_all()
                    except Exception:
                        pass
                asyncio.create_task(_broadcast_cancel())
                continue

            # New message arrived — cancel the previous local task so we don't
            # double-stream tokens. Do NOT touch the remote SadTalker subprocess:
            # if the user is still happy to receive the older video, let it
            # finish; if not, they'll hit Stop or close the tab and the next
            # message-driven render will queue. Killing it here was breaking
            # consecutive sends.
            if current_task and not current_task.done():
                current_task.cancel()
                try:
                    await current_task
                except (asyncio.CancelledError, Exception):
                    pass

            current_task = asyncio.create_task(run_pipeline(data))

    except WebSocketDisconnect:
        logger.info("Avatar WS disconnected")
        if current_task and not current_task.done():
            current_task.cancel()
        # Intentionally NOT calling _sadtalker.cancel_all() here. A common
        # cause of disconnect is "user navigated away and came back" (the
        # avatar tab gets a fresh WS each time). Killing the render on
        # disconnect meant every reload wiped the in-flight job.
    except Exception as e:
        logger.exception("Avatar WS error")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ─── Model + hardware probes (Phase 7.2) ─────────────────────────

@router.get("/api/avatar/models")
async def list_backends_and_models():
    """Return available LLM backends and their models.

    Shape:
      {
        "default_backend": "ollama",
        "default_model": "...",
        "backends": {
          "ollama":        {"available": bool, "models": [...], "endpoint": "..."},
          "claude":        {"available": bool, "models": ["claude-..."]},
          "anamnesis_gpt": {"available": bool, "endpoints": [...]},
        }
      }
    """
    out = {
        "default_backend": "ollama",
        "default_model": config.OLLAMA_DEFAULT_MODEL,
        "backends": {},
    }

    # Ollama — probe first reachable endpoint for its tag list
    ollama_info = {"available": False, "models": [], "endpoint": None}
    try:
        for url, label, _ in config.OLLAMA_ENDPOINTS:
            try:
                async with httpx.AsyncClient(timeout=4.0) as client:
                    r = await client.get(f"{url}/api/tags")
                    if r.status_code == 200:
                        ollama_info["available"] = True
                        ollama_info["endpoint"] = label
                        ollama_info["models"] = [m["name"] for m in r.json().get("models", [])]
                        break
            except Exception:
                continue
    except Exception as e:
        ollama_info["error"] = str(e)
    out["backends"]["ollama"] = ollama_info

    # Claude API — available iff key present
    out["backends"]["claude"] = {
        "available": bool(config.ANTHROPIC_API_KEY),
        "models": [config.CLAUDE_MODEL] if config.ANTHROPIC_API_KEY else [],
    }

    # AnamnesisGPT — available iff an endpoint responds
    ana_info = {"available": False, "endpoints": [], "models": []}
    for url in ANAMNESIS_GPT_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{url}/health")
                if r.status_code == 200:
                    ana_info["available"] = True
                    ana_info["endpoints"].append(url)
                    ana_info["models"] = ["anamnesis-gpt"]
        except Exception:
            continue
    out["backends"]["anamnesis_gpt"] = ana_info

    # δ² — available iff D2_ENDPOINT_URL is set AND the trainer responds
    d2_endpoint = os.environ.get("D2_ENDPOINT_URL", "").rstrip("/")
    d2_info = {"available": False, "endpoint": d2_endpoint or None, "models": []}
    if d2_endpoint:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{d2_endpoint}/health")
                if r.status_code == 200:
                    d2_info["available"] = True
                    d2_info["models"] = ["d2"]
        except Exception:
            pass
    out["backends"]["d2"] = d2_info

    # Together.ai — available iff key set; probe /v1/models for live list
    together_info = {"available": False, "models": [], "endpoint": config.TOGETHER_BASE_URL}
    if config.TOGETHER_API_KEY:
        together_info["available"] = True
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                r = await client.get(
                    f"{config.TOGETHER_BASE_URL}/models",
                    headers={"Authorization": f"Bearer {config.TOGETHER_API_KEY}"},
                )
                if r.status_code == 200:
                    # Filter to chat-capable models; prefer instruct variants
                    raw = r.json() if isinstance(r.json(), list) else r.json().get("data", [])
                    together_info["models"] = sorted(
                        m["id"] for m in raw if isinstance(m, dict) and "id" in m
                    )[:80]    # cap at 80 — UI dropdown sanity
        except Exception as e:
            together_info["error"] = str(e)[:120]
    out["backends"]["together"] = together_info

    # RunPod — available iff a pod endpoint is registered (active pod)
    runpod_endpoint = config.RUNPOD_ENDPOINT_URL.rstrip("/") if config.RUNPOD_ENDPOINT_URL else ""
    runpod_info = {
        "available": False,
        "endpoint": runpod_endpoint or None,
        "models": [config.RUNPOD_DEFAULT_MODEL] if runpod_endpoint else [],
    }
    if runpod_endpoint:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{runpod_endpoint}/models")
                if r.status_code == 200:
                    runpod_info["available"] = True
                    raw = r.json() if isinstance(r.json(), list) else r.json().get("data", [])
                    runpod_info["models"] = sorted(
                        m["id"] for m in raw if isinstance(m, dict) and "id" in m
                    ) or runpod_info["models"]
        except Exception as e:
            runpod_info["error"] = str(e)[:120]
    out["backends"]["runpod"] = runpod_info

    return out


@router.get("/api/avatar/workers")
async def list_workers():
    """Probe all configured GPU workers and return reachability + capabilities.

    Shape:
      {
        "workers": [
          {"url": "...", "label": "...", "reachable": bool,
           "worker_id": "...", "gpu_type": "cuda|rocm|cpu",
           "capabilities": ["xtts", "sadtalker", "demucs"],
           "error": null | str}
        ]
      }
    """
    workers = await probe_all_workers()
    return {"workers": workers}


# ─── Sessions (reuses chat_sessions collection with backend="avatar") ─

@router.get("/api/avatar/sessions")
async def avatar_list_sessions(limit: int = 50):
    sessions = await list_chat_sessions(limit=limit, backend="avatar")
    return {"sessions": sessions}


@router.get("/api/avatar/sessions/{session_id}")
async def avatar_get_session(session_id: str):
    doc = await get_chat_session(session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")
    # Guard: only return avatar sessions via this endpoint
    if doc.get("backend") != "avatar":
        raise HTTPException(status_code=404, detail="Not an avatar session")
    return doc


@router.delete("/api/avatar/sessions/{session_id}")
async def avatar_delete_session(session_id: str):
    doc = await get_chat_session(session_id)
    if not doc or doc.get("backend") != "avatar":
        raise HTTPException(status_code=404, detail="Session not found")
    await delete_chat_session(session_id)
    return {"ok": True}


@router.patch("/api/avatar/sessions/{session_id}/title")
async def avatar_rename_session(session_id: str, body: dict):
    title = (body or {}).get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title required")
    doc = await get_chat_session(session_id)
    if not doc or doc.get("backend") != "avatar":
        raise HTTPException(status_code=404, detail="Session not found")
    await rename_chat_session(session_id, title)
    return {"ok": True}


# ─── RunPod provisioning trigger ────────────────────────────────
#
# UI button calls this to spin up a RunPod avatar worker with a chosen GPU
# tier. anamnesis-app runs in a container; deploy_runpod.sh lives on the host
# (dellserver). Bridge via SSH using the mounted ssh keys + SSH_HOST_DELLSERVER
# env (which is `host.docker.internal`).
#
# This is a long-running operation (1-5 min to provision a pod). Returns
# synchronously with a long timeout. Failure modes surfaced verbatim from
# the script's stderr.
@router.post("/api/avatar/runpod/provision")
async def runpod_provision(body: dict):
    """Provision (or restart) a RunPod avatar pod with a chosen GPU tier.

    Body: {"gpu_tier": "rtx3090|rtx4090|a100|h100", "profile": "avatar"}

    Triggers `RUNPOD_PROFILE=<profile> ./deploy_runpod.sh start --gpu <tier>` on
    the dellserver host. The script writes the new pod URL back to .env
    (AVATAR_WORKER_URL_5) and to the worker_registry collection so anamnesis-app
    picks it up on next worker probe.

    NOTE — the deploy script depends on the avatar worker image being
    published to ghcr.io as `anamnesis-avatar-worker:cuda-runpod`. If the
    image isn't published, RunPod's pull will fail and this endpoint returns
    the script's stderr verbatim so the operator sees the gap.
    """
    gpu_tier = (body or {}).get("gpu_tier", "rtx4090").strip().lower()
    profile = (body or {}).get("profile", "avatar").strip().lower()
    if gpu_tier not in ("rtx3090", "rtx4090", "a100", "h100"):
        raise HTTPException(status_code=400, detail=f"unsupported gpu_tier: {gpu_tier!r}")
    if profile not in ("avatar", "trainer", "d2"):
        raise HTTPException(status_code=400, detail=f"unsupported profile: {profile!r}")

    # Honest bridge: return the exact command for the operator to run on the
    # host. Container-to-host SSH would work but needs ssh-client installed +
    # bind-mounted key file with strict perms (0600) — both fixable but require
    # an image rebuild we're not doing this session. Trigger-file watcher (the
    # NVR-restart pattern) is the cleaner future direction; until then, manual
    # paste keeps the operator in control of when money starts being spent.
    cmd = (
        f"cd ~/0_GENESIS_PROJECT/0_ANAMNESIS && "
        f"RUNPOD_PROFILE={profile} ./deploy_runpod.sh start --gpu {gpu_tier}"
    )
    stop_cmd = (
        f"cd ~/0_GENESIS_PROJECT/0_ANAMNESIS && "
        f"RUNPOD_PROFILE={profile} ./deploy_runpod.sh stop"
    )
    return {
        "ok": True,
        "manual_required": True,
        "gpu_tier": gpu_tier,
        "profile": profile,
        "command": cmd,
        "stop_command": stop_cmd,
        "instructions": (
            "Copy the command and run it in a dellserver terminal. "
            "The script provisions the pod, writes the URL to .env (AVATAR_WORKER_URL_5), "
            "and registers it with worker_registry. anamnesis-app picks up the new URL "
            "on its next worker probe (every 30s). To stop billing, run the stop_command."
        ),
        "notes": [
            "Pod takes 30s-3min to provision.",
            "Image dependency: deploy_runpod.sh tries to pull ghcr.io/elfege/anamnesis-avatar-worker:cuda-runpod. "
            "If that's not published yet, RunPod will fail at the pull step — publish or switch to a build-on-pod flow first.",
            "Cost: pay-per-second once running. RTX 4090 ~$0.40/hr, A100 ~$1.50/hr, H100 ~$2.50/hr.",
        ],
    }


# ─── RunPod pod registry (MongoDB-backed) ───────────────────────
# Why this lives outside .env: a RunPod pod_id changes on every create.
# Storing it in AWS Secrets / .env forced a 5-step manual chain on every
# pod cycle (AWS edit → pull_env.sh → docker restart → UI refresh). The
# registry stores pod_id+port; URL is derived. UI/CLI mutate via these
# endpoints, config.AVATAR_WORKER_ENDPOINTS picks up changes on next access.

@router.get("/api/avatar/runpod/pods")
async def runpod_pods_list():
    from avatar import runpod_pods
    return {"pods": runpod_pods.list_pods_sync()}


@router.post("/api/avatar/runpod/pods")
async def runpod_pods_add(body: dict):
    """Register a RunPod pod. Body: {pod_id, port, label, gpu_type?}.

    URL is derived server-side: https://<pod_id>-<port>.proxy.runpod.net.
    Upsert by pod_id so re-posting the same id refreshes the label/gpu_type.
    """
    from avatar import runpod_pods
    body = body or {}
    try:
        pod = await runpod_pods.add_pod(
            pod_id=body.get("pod_id", "").strip(),
            port=body.get("port", 3013),
            label=body.get("label", "").strip(),
            gpu_type=body.get("gpu_type"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "pod": pod}


@router.delete("/api/avatar/runpod/pods/{pod_id}")
async def runpod_pods_delete(pod_id: str):
    from avatar import runpod_pods
    removed = await runpod_pods.delete_pod(pod_id)
    if not removed:
        raise HTTPException(status_code=404, detail="pod not found")
    return {"ok": True, "deleted": pod_id}


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


# ─── Debug terminal — tail the in-memory ring buffer ─────────────

@router.get("/api/avatar/debug-logs")
async def debug_logs(since_seq: int = -1, limit: int = 500):
    """Frontend polls this every ~1s with the seq of the last line it saw.

    Returns the new entries plus the current high-water mark. since_seq=-1 means
    "first call — give me a tailful so the panel boots populated."
    """
    import debug_logs as _dl
    return _dl.fetch(since_seq=since_seq, limit=limit)


# ─── Emergency stop — the panic button ───────────────────────────
# Intentionally a plain HTTP POST (NOT WebSocket). The whole point of this
# endpoint is to be callable when the WS pipeline is itself the problem.
# Implementation deliberately lives in avatar/emergency.py — see header
# comment there before refactoring this.

@router.post("/api/avatar/emergency-stop")
async def emergency_stop():
    from avatar import emergency
    return await emergency.panic()


# ─── Advanced settings (sampling + persona override) — DB-backed ─

@router.get("/api/avatar/advanced-settings")
async def avatar_advanced_get():
    from avatar import advanced_settings
    return await advanced_settings.get_settings()


@router.put("/api/avatar/advanced-settings")
async def avatar_advanced_put(payload: dict):
    from avatar import advanced_settings
    return await advanced_settings.update_settings(payload)
