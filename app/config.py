import os

# MongoDB
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb://localhost:5438"
)
MONGO_DB = os.environ.get("MONGO_DB", "anamnesis")

# Embedding model
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "384"))

# Collection and index names
COLLECTION_NAME = "episodes"
VECTOR_INDEX_NAME = "episode_vector_index"

# Search defaults
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Ollama (runs on host, accessed from container via host.docker.internal)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_DEFAULT_MODEL", "llama3.2")

# Anthropic / Claude API (optional)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
