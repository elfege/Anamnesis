# Avatar refactor plan — integrate into Anamnesis with swappable GPU worker

**Date:** 2026-04-21
**Goal:** kill the avatar-app monolith. Avatar becomes a full-page section of the Anamnesis UI, with GPU work delegated to a swappable REST worker (same pattern as trainers).

---

## Problems with current state

| Problem | Impact |
|---|---|
| Avatar runs as a monolith on office | When office sleeps/reboots, Belle goes dark. Anamnesis UI doesn't. |
| SadTalker checkpoints in `/tmp/SadTalker/` | `tmpfiles.d` cleans `/tmp` after 10 days; on some systems at reboot. Fragile. |
| Separate UI app on port 3012 | Inconsistent with "one UI, many backends" Anamnesis pattern. User explicitly asked: no separate UI. |
| No swappable GPU backend | Can't fail over from office (AMD) to server (NVIDIA) without editing code. Inconsistent with trainer env-driven backend selection. |

---

## Target architecture

```
┌───────────────── dellserver (always-on) ─────────────────┐
│                                                           │
│  anamnesis-app (3010)                                     │
│    ├─ existing routes: /dashboard, /trainers, /chat, …    │
│    ├─ NEW routes: /avatar (full page, own template)       │
│    ├─ NEW routes: /api/avatar/*                           │
│    ├─ avatar/voices.py     — registry (WAV samples on vol)│
│    ├─ avatar/pipeline.py   — orchestrator                 │
│    ├─ avatar/tts/edge.py   — runs in-process (no GPU)     │
│    └─ clients that call GPU worker over HTTP              │
│                                                           │
└─────────────────────────┬─────────────────────────────────┘
                          │ HTTP
         ┌────────────────┴────────────────┐
         │                                 │
┌────────▼─────────────┐         ┌─────────▼──────────────┐
│ office (RX 6800)     │         │ server (NVIDIA)        │
│ avatar-worker-office │         │ avatar-worker-server   │
│  :3013               │         │  :3013                 │
│  POST /xtts/...      │         │  POST /xtts/...        │
│  POST /demucs/...    │         │  POST /demucs/...      │
│  POST /sadtalker/... │         │  POST /sadtalker/...   │
└──────────────────────┘         └────────────────────────┘
```

- GPU worker: **same image, different docker-compose per machine** (CUDA vs ROCm — trainers pattern).
- Anamnesis env: `AVATAR_WORKER_URL_1`, `AVATAR_WORKER_URL_2`, … (mirrors existing `OLLAMA_URL_N` fallback chain).
- Worker runs on whichever GPU is available; dellserver doesn't care which.

---

## Directory layout

```
0_ANAMNESIS/
├── app/                             # existing Anamnesis app
│   ├── routes/
│   │   ├── avatar.py                # NEW — page + API routes
│   │   └── …
│   ├── avatar/                      # NEW — avatar modules
│   │   ├── voices.py                # (moved from avatar/app/)
│   │   ├── pipeline.py              # orchestrator
│   │   ├── tts/
│   │   │   ├── base.py
│   │   │   ├── edge.py              # in-process
│   │   │   └── xtts_client.py       # NEW — HTTP → worker
│   │   └── animation/
│   │       └── sadtalker_client.py  # NEW — HTTP → worker
│   └── templates/
│       ├── dashboard.html           # + "Avatar" tab link
│       └── avatar.html              # NEW — full-page, Home btn → /dashboard
│
├── avatar_worker/                   # NEW — GPU worker
│   ├── Dockerfile                   # shared image
│   ├── docker-compose.office.yml    # ROCm
│   ├── docker-compose.server.yml    # CUDA
│   ├── docker-compose.cpu.yml       # fallback (dellserver, slow)
│   ├── requirements.txt
│   └── app/
│       ├── main.py                  # FastAPI
│       ├── xtts.py
│       ├── sadtalker.py
│       └── demucs_extract.py
│
├── avatar/                          # TO BE DELETED after migration
└── docker-compose.yml               # existing master (anamnesis-app, -mongo)
```

---

## Worker REST API (same on every machine)

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/health` | — | `{status, gpu_type, machine, backends: [xtts, demucs, sadtalker]}` |
| POST | `/xtts/synthesize` | multipart: `text`, `language`, `speaker.wav` | `audio/wav` bytes |
| POST | `/demucs/extract` | multipart: `audio.*` (mp3/wav) | `audio/wav` (vocals, trimmed) |
| POST | `/sadtalker/animate` | multipart: `audio.wav`, `image.png` | `video/mp4` bytes |

All three endpoints accept raw uploads and return raw media — no base64 bloat.

---

## Anamnesis-side config (additions to `app/config.py`)

```python
# Avatar GPU worker endpoints — tries in order, uses first reachable.
# Mirrors OLLAMA_URL_N fallback pattern.
def _build_avatar_worker_endpoints():
    endpoints = []
    for i in range(1, 10):
        url = os.environ.get(f"AVATAR_WORKER_URL_{i}")
        if not url: break
        label = os.environ.get(f"AVATAR_WORKER_LABEL_{i}", f"worker-{i}")
        endpoints.append((url, label))
    return endpoints

AVATAR_WORKER_ENDPOINTS = _build_avatar_worker_endpoints()
AVATAR_ENABLED = bool(AVATAR_WORKER_ENDPOINTS) or os.environ.get("AVATAR_ALLOW_EDGE_ONLY", "true") == "true"
```

If no worker is configured, edge-tts (in-process) still works — UI hides the "cloned voices" and "animation" options.

---

## Storage

| Thing | Location | Notes |
|-------|----------|-------|
| Cloned voice WAVs | dellserver volume `anamnesis_voices:/app/voices` | lives with the orchestrator; shipped to worker per request |
| SadTalker checkpoints | **`~/models/SadTalker/` on office** (persistent) | moved out of `/tmp/` |
| XTTS model | HF cache volume on each worker machine | ~1.8GB, one-time download |
| Demucs model | torch hub cache on each worker machine | ~2.5GB, one-time download |
| Generated audio/video | dellserver `/tmp/anamnesis_media/` | short-lived, served via `/media/` |

---

## Execution order

1. **Write plan** ← you are here.
2. Build `avatar_worker/` (Dockerfile, main.py, xtts/sadtalker/demucs modules, compose per machine).
3. Move SadTalker on office: `/tmp/SadTalker/` → `~/models/SadTalker/`.
4. Deploy worker on office (`docker-compose.office.yml`, ROCm). Smoke-test all 3 endpoints.
5. Move voice/pipeline code into `app/avatar/`, rewrite `tts/xtts_client.py` + `animation/sadtalker_client.py` to call the worker.
6. Add `app/routes/avatar.py` with `/avatar` page + `/api/avatar/*` routes to the existing FastAPI app.
7. Build `app/templates/avatar.html` — full-page avatar UI (chat + voices), Home button → `/dashboard`.
8. Add "Avatar" tab link to `dashboard.html`.
9. Rebuild `anamnesis-app`. Smoke-test from the Anamnesis UI.
10. **Retirement**:
    - Stop + remove `avatar-office` container.
    - Delete `avatar/` top-level dir.
    - Port registry: remove 3012 (office), add 3013 avatar-worker (office + server).
    - Intercom MSG-113 → RESOLVED (3012 retired; 3013 reachable only on LAN, no proxy routing needed).

---

## Non-goals (explicitly)

- **Not building a proxy config** — avatar lives inside Anamnesis, reached via same hostname/port as the rest of it.
- **Not pre-installing two GPU workers** — only office first. Server is an optional fallback you can bring up later.
- **Not touching the d² work** — completely separate task tree.
- **Not fixing the SadTalker numpy<2 bug separately** — it's already pinned in the new Dockerfile; worker build will pick it up for free.

---

## Open questions — ANSWERED 2026-04-21

- [x] Single GPU worker container with XTTS+Demucs+SadTalker? → **yes, one container, one image.**
- [x] Avatar UI is its own template (full page)? → **yes, Home button goes back to /dashboard.**
- [x] Drop `avatar/` top-level directory? → **yes, after migration.**
