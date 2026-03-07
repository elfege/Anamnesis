import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

import httpx
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import (
    OLLAMA_URL,
    OLLAMA_DEFAULT_MODEL,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
)
from database import get_episodes_collection, vector_search
from embedding import get_embedding

logger = logging.getLogger("anamnesis.routes.chat")

router = APIRouter(tags=["chat"])

# ─── In-memory session history ────────────────────────────────────
# {session_id: [{"role": "user"|"assistant", "content": "..."}]}
_sessions: dict[str, list[dict]] = defaultdict(list)
MAX_HISTORY = 20  # messages to keep per session


# ─── Pydantic models ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    backend: str = "ollama"          # "ollama" | "claude"
    model: Optional[str] = None      # overrides OLLAMA_DEFAULT_MODEL
    session_id: Optional[str] = None
    top_k: int = 3                   # memories to retrieve


# ─── Prompt construction ──────────────────────────────────────────

SYSTEM_BASE = (
    "You are Anamnesis, an AI assistant with episodic memory. "
    "You have access to memories distilled from past sessions. "
    "Use relevant memories as context without being slavish about them. "
    "Be direct, thoughtful, and honest."
)


def _memory_block(episodes: list[dict]) -> str:
    if not episodes:
        return ""
    lines = ["\n\n--- Relevant memories from past sessions ---"]
    for ep in episodes:
        score = ep.get("similarity_score", 0)
        lines.append(f"[{score:.0%} match] {ep.get('summary', '')}")
    lines.append("--- End of memories ---")
    return "\n".join(lines)


# ─── Episode storage ──────────────────────────────────────────────

async def _store_chat_episode(
    session_id: str, user_msg: str, assistant_msg: str, model_label: str
):
    """Embed and persist the exchange as an episode."""
    try:
        collection = get_episodes_collection()
        summary = (
            f"Chat [{model_label}] — "
            f"User: {user_msg[:120]}{'...' if len(user_msg) > 120 else ''}"
        )
        raw_exchange = f"User: {user_msg}\n\nAssistant: {assistant_msg}"
        embedding = get_embedding(summary)
        episode_id = f"chat-{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc)
        await collection.insert_one({
            "episode_id": episode_id,
            "instance": "anamnesis-chat",
            "project": "chat",
            "summary": summary,
            "raw_exchange": raw_exchange,
            "tags": ["chat", model_label.split(":")[0], session_id[:8]],
            "embedding": embedding,
            "timestamp": now,
            "retrieval_count": 0,
            "last_retrieved": None,
        })
        logger.info(f"Stored chat episode: {episode_id}")
    except Exception as exc:
        logger.warning(f"Could not store chat episode: {exc}")


# ─── Ollama streaming generator ───────────────────────────────────

async def _stream_ollama(
    system: str,
    messages: list[dict],
    model: str,
    session_id: str,
    user_msg: str,
) -> AsyncIterator[str]:
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": model,
        "stream": True,
        "messages": [{"role": "system", "content": system}] + messages,
    }

    full_response: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield f"data: {json.dumps({'error': f'Ollama {resp.status_code}: {body.decode()}'})}\n\n"
                    return

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            full_response.append(token)
                            yield f"data: {json.dumps({'token': token})}\n\n"
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    except httpx.ConnectError:
        yield f"data: {json.dumps({'error': 'Cannot connect to Ollama. Is it running? (http://localhost:11434)'})}\n\n"
        return
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        return

    # Finalise session and store episode
    reply = "".join(full_response)
    _sessions[session_id].append({"role": "assistant", "content": reply})
    _trim_history(session_id)
    await _store_chat_episode(session_id, user_msg, reply, f"ollama:{model}")
    yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"


# ─── Claude streaming generator ───────────────────────────────────

async def _stream_claude(
    system: str,
    messages: list[dict],
    session_id: str,
    user_msg: str,
) -> AsyncIterator[str]:
    if not ANTHROPIC_API_KEY:
        yield f"data: {json.dumps({'error': 'ANTHROPIC_API_KEY not set in .env'})}\n\n"
        return

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 4096,
        "system": system,
        "messages": messages,
        "stream": True,
    }

    full_response: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield f"data: {json.dumps({'error': f'Claude API {resp.status_code}: {body.decode()}'})}\n\n"
                    return

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        event = json.loads(raw)
                        if event.get("type") == "content_block_delta":
                            token = event.get("delta", {}).get("text", "")
                            if token:
                                full_response.append(token)
                                yield f"data: {json.dumps({'token': token})}\n\n"
                    except json.JSONDecodeError:
                        continue

    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        return

    reply = "".join(full_response)
    _sessions[session_id].append({"role": "assistant", "content": reply})
    _trim_history(session_id)
    await _store_chat_episode(session_id, user_msg, reply, f"claude:{CLAUDE_MODEL}")
    yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"


def _trim_history(session_id: str):
    if len(_sessions[session_id]) > MAX_HISTORY:
        _sessions[session_id] = _sessions[session_id][-MAX_HISTORY:]


# ─── Endpoints ────────────────────────────────────────────────────

@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream a chat response (SSE). Retrieves relevant memories first."""

    session_id = req.session_id or uuid.uuid4().hex

    # Retrieve relevant memories
    try:
        query_vector = get_embedding(req.message)
        memories = await vector_search(query_vector=query_vector, top_k=req.top_k)
    except Exception:
        memories = []

    system = SYSTEM_BASE + _memory_block(memories)

    # Append user message and snapshot history for this turn
    _sessions[session_id].append({"role": "user", "content": req.message})
    messages = list(_sessions[session_id])

    if req.backend == "claude":
        gen = _stream_claude(system, messages, session_id, req.message)
    else:
        model = req.model or OLLAMA_DEFAULT_MODEL
        gen = _stream_ollama(system, messages, model, session_id, req.message)

    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/chat/models")
async def list_ollama_models():
    """List locally available Ollama models."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                return {"models": models, "default": OLLAMA_DEFAULT_MODEL}
            return {"models": [], "default": OLLAMA_DEFAULT_MODEL, "error": f"Ollama {resp.status_code}"}
    except httpx.ConnectError:
        return {"models": [], "default": OLLAMA_DEFAULT_MODEL, "error": "Cannot connect to Ollama"}
    except Exception as exc:
        return {"models": [], "default": OLLAMA_DEFAULT_MODEL, "error": str(exc)}


@router.get("/api/chat/balance")
async def get_anthropic_balance():
    """
    Attempt to fetch Anthropic credit balance via their admin API.
    Returns whatever the API provides; falls back to a console link if unavailable.
    """
    if not ANTHROPIC_API_KEY:
        return {"available": False, "reason": "ANTHROPIC_API_KEY not configured"}

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        for path in ["/v1/billing/credits", "/v1/usage"]:
            try:
                resp = await client.get(
                    f"https://api.anthropic.com{path}", headers=headers
                )
                if resp.status_code == 200:
                    return {"available": True, "data": resp.json(), "source": path}
            except Exception:
                continue

    return {
        "available": False,
        "reason": "Balance API not accessible with this key — check console.anthropic.com",
        "console_url": "https://console.anthropic.com/settings/billing",
    }


@router.delete("/api/chat/session/{session_id}")
async def clear_session(session_id: str):
    """Clear the conversation history for a session."""
    _sessions.pop(session_id, None)
    return {"cleared": True, "session_id": session_id}
