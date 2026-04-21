"""Worker config — everything env-driven."""
import os

# Hardware identity
GPU_TYPE = os.environ.get("GPU_TYPE", "cpu")              # cuda | rocm | cpu
MACHINE_NAME = os.environ.get("MACHINE_NAME", "unknown")

# Server
HOST = os.environ.get("WORKER_HOST", "0.0.0.0")
PORT = int(os.environ.get("WORKER_PORT", "3013"))

# XTTS
XTTS_MODEL_NAME = os.environ.get(
    "XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2"
)

# Demucs
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")

# SadTalker
SADTALKER_DIR = os.environ.get("SADTALKER_DIR", "/opt/SadTalker")
SADTALKER_CHECKPOINT_DIR = os.environ.get(
    "SADTALKER_CHECKPOINT_DIR", "/opt/SadTalker/checkpoints"
)
