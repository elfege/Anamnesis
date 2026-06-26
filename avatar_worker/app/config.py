"""Worker config — everything env-driven."""
import os

# Hardware identity
GPU_TYPE = os.environ.get("GPU_TYPE", "cpu")              # cuda | rocm | cpu
MACHINE_NAME = os.environ.get("MACHINE_NAME", "unknown")

# Unique ID for this worker instance (Anamnesis-side probe uses this as stable key).
# Defaults to "<machine>-<gpu_type>" but can be overridden for multiple workers
# on the same machine (e.g. "server-cuda-1660", "server-cuda-3090").
WORKER_ID = os.environ.get("WORKER_ID", f"{MACHINE_NAME}-{GPU_TYPE}")

# Server
HOST = os.environ.get("WORKER_HOST", "0.0.0.0")
PORT = int(os.environ.get("WORKER_PORT", "3013"))

# ── XTTS ──────────────────────────────────────────────────────
XTTS_MODEL_NAME = os.environ.get(
    "XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2"
)

# ── Demucs ────────────────────────────────────────────────────
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")

# ── SadTalker ─────────────────────────────────────────────────
SADTALKER_DIR = os.environ.get("SADTALKER_DIR", "/opt/SadTalker")
SADTALKER_CHECKPOINT_DIR = os.environ.get(
    "SADTALKER_CHECKPOINT_DIR", "/opt/SadTalker/checkpoints"
)
# Render resolution. 512 = best quality (~5GB VRAM peak). 256 = lower quality
# but fits in 6GB cards alongside the SadTalker weights themselves. Default
# conservative 256 — machines with headroom set SADTALKER_SIZE=512.
SADTALKER_SIZE = int(os.environ.get("SADTALKER_SIZE", "256"))
# GFPGAN face enhancer adds ~1GB VRAM; skip on tight hosts.
# Set to "" to disable, or "gfpgan" / "RestoreFormer" to enable.
SADTALKER_ENHANCER = os.environ.get("SADTALKER_ENHANCER", "")

# On tight-VRAM hosts (e.g. GTX 1660 SUPER 6GB), unload XTTS before running
# SadTalker so both fit sequentially. Cost: ~5s reload on next XTTS call.
AUTO_UNLOAD_XTTS = os.environ.get("AUTO_UNLOAD_XTTS", "true").lower() in ("true", "1", "yes")

# ── MuseTalk ──────────────────────────────────────────────────
# Tencent's diffusion-based real-time lip-sync model (~real-time on 16GB ROCm
# vs SadTalker's 30-120s). Repo lives at /opt/MuseTalk (bind-mounted) and the
# pinned deps (diffusers 0.30.2, accelerate 0.28.0, decord, face_alignment)
# live in a PYTHONUSERBASE overlay so they don't clobber XTTS's transformers 5.x.
MUSETALK_DIR = os.environ.get("MUSETALK_DIR", "/opt/MuseTalk")
MUSETALK_VERSION = os.environ.get("MUSETALK_VERSION", "v15")  # "v15" or "v1"
MUSETALK_USE_FLOAT16 = os.environ.get("MUSETALK_USE_FLOAT16", "true").lower() in ("true", "1", "yes")
# PYTHONUSERBASE path passed to subprocess so it picks up the overlay. Compose
# already exports this env var globally; setting here as fallback for dev runs.
MUSETALK_PYENV = os.environ.get("PYTHONUSERBASE", "/opt/MuseTalk_pyenv")
