"""
Avatar service configuration — all settings driven by environment variables.
Same pattern as trainers/app/config.py: Docker compose sets GPU_TYPE, backends, etc.
"""
import os

# ─── Hardware ────────────────────────────────────────────────────
GPU_TYPE = os.environ.get("GPU_TYPE", "cuda")          # cuda | rocm | cpu
MACHINE_NAME = os.environ.get("MACHINE_NAME", "unknown")

# ─── LLM Backend ────────────────────────────────────────────────
# ollama | claude | trainer | d2
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")

# Ollama settings
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# Claude API settings
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# Trainer inference endpoint
TRAINER_URL = os.environ.get("TRAINER_URL", "http://host.docker.internal:3011")

# d² model (future)
D2_MODEL_PATH = os.environ.get("D2_MODEL_PATH", "")
D2_CHECKPOINT = os.environ.get("D2_CHECKPOINT", "")

# ─── TTS Backend ─────────────────────────────────────────────────
# edge | xtts | fish
TTS_BACKEND = os.environ.get("TTS_BACKEND", "edge")
TTS_VOICE = os.environ.get("TTS_VOICE", "en-US-AnaNeural")  # Edge TTS voice
TTS_RATE = os.environ.get("TTS_RATE", "+0%")                 # Speed adjustment

# XTTS / Fish Speech (future)
TTS_MODEL_URL = os.environ.get("TTS_MODEL_URL", "")
TTS_SPEAKER_WAV = os.environ.get("TTS_SPEAKER_WAV", "")      # Voice cloning reference

# ─── Animation Backend ──────────────────────────────────────────
# liveportrait | none
ANIM_BACKEND = os.environ.get("ANIM_BACKEND", "liveportrait")
ANIM_REFERENCE_IMAGE = os.environ.get("ANIM_REFERENCE_IMAGE", "/app/static/reference.png")

# LivePortrait settings
LIVEPORTRAIT_CHECKPOINT_DIR = os.environ.get(
    "LIVEPORTRAIT_CHECKPOINT_DIR", "/models/liveportrait"
)

# ─── Server ──────────────────────────────────────────────────────
HOST = os.environ.get("AVATAR_HOST", "0.0.0.0")
PORT = int(os.environ.get("AVATAR_PORT", "3012"))

# ─── Persona ─────────────────────────────────────────────────────
PERSONA_NAME = os.environ.get("PERSONA_NAME", "Aria")
PERSONA_SYSTEM_PROMPT = os.environ.get("PERSONA_SYSTEM_PROMPT", (
    "You are Aria, a thoughtful and curious conversational partner. "
    "You speak naturally, with warmth and intelligence. "
    "Keep responses concise — 1-3 sentences — since they will be spoken aloud."
))
