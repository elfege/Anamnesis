"""
Model registry for Anamnesis.

Stores model metadata (free/paid, capabilities, context size) in MongoDB.
Pre-seeded with known models on startup. Extendable via API.
"""

import logging
from typing import Optional

from database import get_settings_collection

logger = logging.getLogger("anamnesis.models_registry")

# Pre-seeded model definitions
KNOWN_MODELS = [
    # Ollama models (all free, local)
    {
        "model_id": "llama3.2:latest",
        "provider": "ollama",
        "display_name": "Llama 3.2 (3B)",
        "free": True,
        "parameters": "3B",
        "context_window": 128000,
        "notes": "Fast, lightweight, good for summarization",
    },
    {
        "model_id": "llama3.1:8b",
        "provider": "ollama",
        "display_name": "Llama 3.1 (8B)",
        "free": True,
        "parameters": "8B",
        "context_window": 128000,
        "notes": "Better quality than 3B, still runs on CPU",
    },
    {
        "model_id": "qwen2.5:7b",
        "provider": "ollama",
        "display_name": "Qwen 2.5 (7B)",
        "free": True,
        "parameters": "7B",
        "context_window": 32768,
        "notes": "Strong multilingual, good at structured output",
    },
    {
        "model_id": "qwen2.5:14b",
        "provider": "ollama",
        "display_name": "Qwen 2.5 (14B)",
        "free": True,
        "parameters": "14B",
        "context_window": 32768,
        "notes": "High quality, needs ~10GB RAM",
    },
    {
        "model_id": "mistral:7b",
        "provider": "ollama",
        "display_name": "Mistral (7B)",
        "free": True,
        "parameters": "7B",
        "context_window": 32768,
        "notes": "Fast, efficient, good reasoning",
    },
    {
        "model_id": "gemma2:9b",
        "provider": "ollama",
        "display_name": "Gemma 2 (9B)",
        "free": True,
        "parameters": "9B",
        "context_window": 8192,
        "notes": "Google's open model, solid quality",
    },
    {
        "model_id": "phi3:medium",
        "provider": "ollama",
        "display_name": "Phi-3 Medium (14B)",
        "free": True,
        "parameters": "14B",
        "context_window": 128000,
        "notes": "Microsoft, excellent for its size, large context",
    },
    {
        "model_id": "deepseek-r1:8b",
        "provider": "ollama",
        "display_name": "DeepSeek R1 (8B)",
        "free": True,
        "parameters": "8B",
        "context_window": 64000,
        "notes": "Strong reasoning and code understanding",
    },
    # Claude API models (paid)
    {
        "model_id": "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "display_name": "Claude Haiku 4.5",
        "free": False,
        "parameters": "N/A",
        "context_window": 200000,
        "cost_per_1k_input": 0.001,
        "cost_per_1k_output": 0.005,
        "notes": "Cheapest Claude, ~$0.002/summary",
    },
    {
        "model_id": "claude-sonnet-4-6",
        "provider": "anthropic",
        "display_name": "Claude Sonnet 4.6",
        "free": False,
        "parameters": "N/A",
        "context_window": 200000,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "notes": "Higher quality, ~$0.01/summary",
    },
]


async def seed_models_registry():
    """Seed known models into MongoDB if not already present."""
    coll = get_settings_collection()

    for model in KNOWN_MODELS:
        await coll.update_one(
            {"_id": f"model:{model['model_id']}"},
            {"$setOnInsert": model},
            upsert=True,
        )
    logger.info(f"Models registry seeded: {len(KNOWN_MODELS)} models")


async def get_all_models() -> list[dict]:
    """Return all registered models."""
    coll = get_settings_collection()
    models = []
    async for doc in coll.find({"model_id": {"$exists": True}}):
        doc.pop("_id", None)
        models.append(doc)
    return models


async def get_model_info(model_id: str) -> Optional[dict]:
    """Get info for a specific model."""
    coll = get_settings_collection()
    doc = await coll.find_one({"_id": f"model:{model_id}"})
    if doc:
        doc.pop("_id", None)
        return doc
    return None


async def is_free_model(model_id: str) -> bool:
    """Check if a model is free (local/Ollama) or paid (API)."""
    info = await get_model_info(model_id)
    if info:
        return info.get("free", False)
    # Unknown models: assume Ollama models are free, others are not
    return "claude" not in model_id.lower() and "anthropic" not in model_id.lower()


async def is_free_backend(backend: str, model: str = "") -> bool:
    """Check if the current backend+model combination is free."""
    if backend == "claude":
        return False
    if backend == "ollama":
        return True
    # Check model registry as fallback
    return await is_free_model(model)


# ─── Embedding Models ────────────────────────────────────────────

EMBEDDING_MODELS = [
    {
        "model_id": "all-MiniLM-L6-v2",
        "display_name": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "notes": "Fast, lightweight. Original default.",
    },
    {
        "model_id": "all-MiniLM-L12-v2",
        "display_name": "all-MiniLM-L12-v2",
        "dimensions": 384,
        "notes": "Slightly better than L6, same dims, minimal overhead.",
    },
    {
        "model_id": "all-mpnet-base-v2",
        "display_name": "all-mpnet-base-v2",
        "dimensions": 768,
        "notes": "Higher quality, 2× dims, moderate CPU cost.",
    },
    {
        "model_id": "BAAI/bge-large-en-v1.5",
        "display_name": "bge-large-en-v1.5",
        "dimensions": 1024,
        "notes": "Best quality. 1024 dims. Heavier CPU. GPU recommended.",
    },
]


def get_embedding_model_info(model_id: str) -> Optional[dict]:
    """Return embedding model metadata by model_id."""
    for m in EMBEDDING_MODELS:
        if m["model_id"] == model_id:
            return m
    return None
