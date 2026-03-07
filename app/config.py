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
