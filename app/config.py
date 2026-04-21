import os

# MongoDB
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb://localhost:5438"
)
MONGO_DB = os.environ.get("MONGO_DB", "anamnesis")

# Embedding model
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "1024"))

# CPU throttle: percentage of logical cores available to embedding workers (1-100)
EMBEDDING_CPU_PCT = max(1, min(100, int(os.environ.get("EMBEDDING_CPU_PCT", "50"))))

# Collection and index names
COLLECTION_NAME = "episodes"
VECTOR_INDEX_NAME = "episode_vector_index"

# Search defaults
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Ollama fallback chain: tries endpoints in order, uses first reachable one.
# Configure via env vars: OLLAMA_URL_1, OLLAMA_URL_2, OLLAMA_URL_3
# Each OLLAMA_LABEL_N / OLLAMA_GPU_N describes the endpoint.
# Falls back to host.docker.internal if no env vars are set.
def _build_ollama_endpoints():
    endpoints = []
    for i in range(1, 10):
        url = os.environ.get(f"OLLAMA_URL_{i}")
        if not url:
            break
        label = os.environ.get(f"OLLAMA_LABEL_{i}", f"endpoint-{i}")
        has_gpu = os.environ.get(f"OLLAMA_GPU_{i}", "false").lower() in ("true", "1", "yes")
        endpoints.append((url, label, has_gpu))
    if not endpoints:
        # Default: local host only
        endpoints.append(("http://host.docker.internal:11434", "local", False))
    return endpoints

OLLAMA_ENDPOINTS = _build_ollama_endpoints()
# Legacy single-URL override (if set, skips fallback chain entirely)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "")
OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_DEFAULT_MODEL", "llama3.2")

# Anthropic / Claude API (optional)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# Claude CLI (subscription backend, $0 per message)
# Path to claude binary on the host — accessed via SSH from inside the container
CLAUDE_CLI_HOST = os.environ.get("CLAUDE_CLI_HOST", "localhost")
CLAUDE_CLI_PATH = os.environ.get("CLAUDE_CLI_PATH", "claude")

# ─── Avatar ─────────────────────────────────────────────────────
# Avatar GPU worker endpoints — tries in order, uses first reachable.
# Mirrors OLLAMA_URL_N fallback pattern.
def _build_avatar_worker_endpoints():
    endpoints = []
    for i in range(1, 10):
        url = os.environ.get(f"AVATAR_WORKER_URL_{i}")
        if not url:
            break
        label = os.environ.get(f"AVATAR_WORKER_LABEL_{i}", f"avatar-worker-{i}")
        endpoints.append((url, label))
    return endpoints

AVATAR_WORKER_ENDPOINTS = _build_avatar_worker_endpoints()

# Voice storage (cloned WAV samples + metadata sidecar JSON)
VOICES_DIR = os.environ.get("VOICES_DIR", "/app/voices")

# Default voice ID (used when no voice is specified per-request)
AVATAR_EDGE_VOICE = os.environ.get("AVATAR_EDGE_VOICE", "en-US-AvaNeural")
AVATAR_EDGE_RATE = os.environ.get("AVATAR_EDGE_RATE", "+0%")
DEFAULT_VOICE_ID = os.environ.get("DEFAULT_VOICE_ID", f"edge:{AVATAR_EDGE_VOICE}")

# Reference image for lip-sync animation (mounted into container)
AVATAR_REFERENCE_IMAGE = os.environ.get("AVATAR_REFERENCE_IMAGE", "/app/static/img/belle_reference.png")

# Animation on by default if a worker is configured and a reference image exists
_auto_animate = bool(AVATAR_WORKER_ENDPOINTS) and os.path.exists(AVATAR_REFERENCE_IMAGE)
AVATAR_ANIMATE_DEFAULT = os.environ.get("AVATAR_ANIMATE_DEFAULT", "auto").lower()
if AVATAR_ANIMATE_DEFAULT == "auto":
    AVATAR_ANIMATE_DEFAULT = _auto_animate
else:
    AVATAR_ANIMATE_DEFAULT = AVATAR_ANIMATE_DEFAULT in ("true", "1", "yes")

# Persona (overridable via env)
AVATAR_PERSONA_NAME = os.environ.get("AVATAR_PERSONA_NAME", "Belle")
AVATAR_PERSONA_SYSTEM_PROMPT = os.environ.get(
    "AVATAR_PERSONA_SYSTEM_PROMPT",
    "You are Belle, a thoughtful and curious conversational partner. "
    "You speak naturally, with warmth and intelligence. "
    "Keep responses concise — 1-3 sentences — since they will be spoken aloud.",
)
