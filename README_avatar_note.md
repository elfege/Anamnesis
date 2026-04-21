# Avatar — architecture cheat-sheet

Quick reference for how Belle works after the 2026-04-21 refactor. For the full design, see `docs/plans/avatar_refactor_plan.md`.

---

## One-line summary

**Avatar UI lives inside Anamnesis** (dellserver:3010/avatar). GPU work (voice cloning, animation) is delegated to a **thin REST worker** (`avatar_worker/`) running on any GPU machine — same machine-swap pattern as `trainers/`.

---

## Where things live

```
┌──────────── dellserver (always-on) ─────────────┐
│ anamnesis-app — port 3010                       │
│   /avatar                → full-page UI          │
│   /api/avatar/{info,voices,chat,ws,media}        │
│   app/avatar/            → orchestrator          │
│   app/routes/avatar.py   → router                │
│   app/templates/avatar.html → page               │
│   Volume anamnesis_voices → /app/voices          │
│     (cloned voice WAV + metadata JSON sidecars)  │
│   Bind ./samples → /app/samples (rw)             │
│     (source material, visible in VS Code)        │
└─────────────────────┬───────────────────────────┘
                      │ HTTP (AVATAR_WORKER_URL_N chain)
     ┌────────────────┴─────────────────┐
     │                                  │
┌────▼──────────── office ──────┐  ┌────▼──── server ─────┐
│ avatar-worker-office (:3013)  │  │ avatar-worker-server │
│ AMD RX 6800 / ROCm            │  │ NVIDIA 1660 / CUDA   │
│ SadTalker: ~/models/SadTalker │  │ (lighter — fast TTS) │
│ XTTS  + Demucs + SadTalker    │  │ XTTS  + Demucs       │
└───────────────────────────────┘  └──────────────────────┘
```

### Key paths (repo root = `~/0_GENESIS_PROJECT/0_ANAMNESIS/`)

| Path | Purpose |
|------|---------|
| `app/avatar/` | Orchestrator: voices registry, pipeline, HTTP clients (xtts/sadtalker/demucs) |
| `app/routes/avatar.py` | FastAPI routes (page + JSON/WS API) |
| `app/templates/avatar.html` | Full-page UI (Home button → `/dashboard`) |
| `app/static/img/belle_reference.png` | Avatar face image served to the UI |
| `samples/personas/Belle/*.png` | Source Belle images (gitignored, VS Code-visible) |
| `samples/audio/parolesonore/*.mp3` | Source vocals (gitignored) |
| `avatar_worker/` | GPU worker subproject (runs on office/server) |
| `avatar_worker/docker-compose.office.yml` | ROCm |
| `avatar_worker/docker-compose.server.yml` | CUDA |
| `docs/plans/avatar_refactor_plan.md` | Full plan + rationale |

---

## Voice system

- **Edge TTS presets** — ~13 preset voices (English/French). Run in-process on dellserver. Fast (~300ms–1s). No GPU needed.
- **Cloned voices (XTTS v2)** — built from a speaker reference (6–15s clean audio). Reference WAV kept in the `anamnesis_voices` volume. Every synthesis request ships the reference + text to the GPU worker, which returns MP3.
- Voice-ID format: `edge:en-US-AvaNeural` | `cloned:<slug>`.
- Worker fallback chain: env `AVATAR_WORKER_URL_1`, `AVATAR_WORKER_URL_2`, … (same idea as `OLLAMA_URL_N`). First reachable wins, next request tries from the top again.

### How Belle (ParoleSonore) was made

1. User ran Demucs on `je_suis_pas_accro_FULL.mp3` → `vocals.mp3` (a cappella). Files at `samples/audio/parolesonore/`.
2. Uploaded `vocals.mp3` via `POST /api/avatar/voices` with `kind=file, language=fr` → registered as `cloned:belle-parolesonore`.
3. Every chat request with `voice_id=cloned:belle-parolesonore` ships this reference to the worker for XTTS synthesis.

### Performance (as of 2026-04-21)

| Voice | Backend | Where | Latency |
|-------|---------|-------|---------|
| `edge:en-US-AvaNeural` | in-process | dellserver | ~300 ms |
| `edge:fr-FR-DeniseNeural` | in-process | dellserver | ~1 s |
| `cloned:belle-parolesonore` | XTTS via worker | office (ROCm) | 10–70 s |
| `cloned:belle-parolesonore` | XTTS via worker | server (CUDA) | ~5 s expected |

Note: ROCm XTTS is slow because MIOpen kernels aren't as optimized as cuDNN for the transformer attention patterns XTTS uses. Switching to the CUDA worker on `server` gives the big speedup.

---

## Animation

- SadTalker (audio-driven, single-stage, 256/512px output).
- Runs on office worker (ROCm). Checkpoints at `~/models/SadTalker/` on office — moved there from `/tmp/` during the refactor.
- Triggered by `animate=true` on a chat request. Pipeline: edge/XTTS → WAV → SadTalker `animate` endpoint → MP4 → UI video element.
- First run is slow (cold model load + ffmpeg).

---

## Scripts (menu-driven)

All three scripts at the repo root support `--test` (dry-run), `--action=...` (skip menu, for cron/alias), and both local and remote targets.

| Script | Menu |
|--------|------|
| `./start.sh` | local / everything / restart one / status / stop all |
| `./deploy.sh` | rebuild local / everything / single / remotes only / prune |
| `./stop.sh` | local / remote / all |

Aliases (`.bash_aliases`): `avatar`, `codeavatar`, `startavatar`, `stopavatar`, `logavatar`, `deployavatar` now all point to `avatar_worker/` on office.

---

## Known issues / gotchas

- **ROCm XTTS is 10× slower than CUDA** — hence moving the worker to NVIDIA for voice cloning.
- **llama3.2's French is mediocre.** Persona prompt was tightened (2026-04-21) to force the model to match the user's language and not hallucinate "Beauty and the Beast" tropes from the name "Belle". Better quality via Claude API backend (already wired, just switch `LLM_BACKEND` env).
- **Cold-path XTTS takes ~30 s** on first call (model download ~1.8 GB + load). Subsequent calls reuse the cache (Docker volume `avatar_worker_hf_cache`).
- **SadTalker requires `numpy<2`** (uses removed `np.VisibleDeprecationWarning`). Pinned in `avatar_worker/Dockerfile`.
- **Coqui TTS requires `transformers<5` + `coqui-tts[codec]`** for torch 2.8+. Pinned in Dockerfile.
- **GTX 1660 SUPER is 6 GB.** Fine for XTTS alone; SadTalker may not fit alongside — if you ever run animation on `server`, do it with the model unloaded from the TTS side.

---

## Docker env (anamnesis-app) — what drives the avatar

```
AVATAR_WORKER_URL_1=http://192.168.10.15:3013       # CUDA (primary)
AVATAR_WORKER_LABEL_1=server · 1660 SUPER CUDA
AVATAR_WORKER_URL_2=http://192.168.10.110:3013      # ROCm (fallback + SadTalker)
AVATAR_WORKER_LABEL_2=office · RX 6800 ROCm

AVATAR_EDGE_VOICE=en-US-AvaNeural
DEFAULT_VOICE_ID=edge:en-US-AvaNeural
AVATAR_PERSONA_NAME=Belle
AVATAR_ANIMATE_DEFAULT=auto        # true iff a worker is reachable AND reference image exists
VOICES_DIR=/app/voices
```

---

## Quick links

- Chat with Belle: http://192.168.10.20:3010/avatar
- Dashboard:       http://192.168.10.20:3010/dashboard
- Worker healths:
  - `curl http://192.168.10.15:3013/health` (server, CUDA)
  - `curl http://192.168.10.110:3013/health` (office, ROCm)

---

*Last updated: 2026-04-21 after moving from monolith to orchestrator/worker split, and before the CUDA move.*
