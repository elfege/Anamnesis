"""Thin LLM wrapper for avatar — streams from first reachable Ollama endpoint.

Uses the same endpoint discovery as routes/chat.py. Kept simple: no tool use,
just persona-driven chat.
"""
import json
import logging
from typing import AsyncIterator

import httpx

from config import OLLAMA_ENDPOINTS, OLLAMA_URL, OLLAMA_DEFAULT_MODEL

logger = logging.getLogger("anamnesis.avatar.llm")


async def _find_ollama_endpoint() -> tuple[str, str, bool] | None:
    if OLLAMA_URL:
        return (OLLAMA_URL, "custom", False)
    for url, label, has_gpu in OLLAMA_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/api/tags")
                if resp.status_code == 200:
                    return (url, label, has_gpu)
        except Exception:
            continue
    return None


async def stream_reply(
    system: str,
    user_message: str,
    model: str = None,
) -> AsyncIterator[tuple[str, str]]:
    """Yield (kind, text) pairs. kind ∈ {token, endpoint, error}."""
    model = model or OLLAMA_DEFAULT_MODEL
    endpoint = await _find_ollama_endpoint()
    if not endpoint:
        yield ("error", "No Ollama endpoint reachable")
        return

    base, label, has_gpu = endpoint
    yield ("endpoint", label)

    payload = {
        "model": model,
        "stream": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", f"{base}/api/chat", json=payload) as r:
                if r.status_code != 200:
                    text = await r.aread()
                    yield ("error", f"Ollama {r.status_code}: {text.decode()[:200]}")
                    return
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    msg = data.get("message", {}) or {}
                    content = msg.get("content", "")
                    if content:
                        yield ("token", content)
                    if data.get("done"):
                        return
    except Exception as exc:
        yield ("error", str(exc))


async def unload_model(model: str = None):
    """Ask Ollama to drop the model from memory (frees VRAM)."""
    model = model or OLLAMA_DEFAULT_MODEL
    endpoint = await _find_ollama_endpoint()
    if not endpoint:
        return
    base = endpoint[0]
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(f"{base}/api/generate", json={"model": model, "keep_alive": 0})
    except Exception as e:
        logger.warning(f"Ollama unload failed: {e}")
