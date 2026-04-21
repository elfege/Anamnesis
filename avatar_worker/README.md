# avatar_worker — GPU service for avatar pipeline

Thin FastAPI service that exposes three GPU-bound endpoints used by the
Anamnesis avatar page:

| Endpoint | Purpose |
|----------|---------|
| `POST /xtts/synthesize` | Clone voice (speaker_wav) + text → audio |
| `POST /demucs/extract`  | Song → isolated vocals (for cloning) |
| `POST /sadtalker/animate` | Audio + reference image → lip-synced MP4 |
| `GET /health`           | Status + GPU type + SadTalker readiness |

Follows the `trainers/` pattern: **one image, per-machine compose file**.

## Deploy

### Office (AMD RX 6800, ROCm)
Requires SadTalker at `~/models/SadTalker/` (checkpoints included).
```bash
docker compose -f docker-compose.office.yml up -d --build
```

### Server (NVIDIA, CUDA)
```bash
docker compose -f docker-compose.server.yml up -d --build
```

### CPU fallback
```bash
docker compose -f docker-compose.cpu.yml up -d --build
```

## Configuration (env)

| Variable | Default | Notes |
|----------|---------|-------|
| `WORKER_PORT` | `3013` | |
| `GPU_TYPE` | `cpu` | Set by compose file: `cuda`, `rocm`, `cpu`. |
| `MACHINE_NAME` | `unknown` | Used in `/health`. |
| `SADTALKER_DIR` | `/opt/SadTalker` | Bind-mount target. |
| `SADTALKER_HOST_DIR` | `/home/elfege/models/SadTalker` | Host path (compose-level). |
| `XTTS_MODEL_NAME` | `tts_models/multilingual/multi-dataset/xtts_v2` | |
| `DEMUCS_MODEL` | `htdemucs` | |

## Smoke test

```bash
# Health
curl http://localhost:3013/health

# XTTS — needs any speech WAV as speaker reference
curl -F text='Hello, testing.' -F speaker=@sample.wav -F language=en \
  http://localhost:3013/xtts/synthesize -o out.mp3

# Demucs
curl -F audio=@song.mp3 http://localhost:3013/demucs/extract -o vocals.wav

# SadTalker
curl -F audio=@speech.wav -F image=@face.png \
  http://localhost:3013/sadtalker/animate -o anim.mp4
```
