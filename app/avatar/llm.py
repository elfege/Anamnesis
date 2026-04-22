"""LLM wrapper for avatar — streams from one of three backends.

Backends:
  - "ollama"         → local Ollama endpoint (default)
  - "claude"         → Anthropic Claude API (requires ANTHROPIC_API_KEY)
  - "anamnesis_gpt"  → the fine-tuned / δ² model via trainer inference endpoint

All backends yield (kind, text) tuples, where kind ∈ {"token", "endpoint", "error"}.
Backend selection happens per-request — no global state change.
"""
import json
import logging
import os
import re
from typing import AsyncIterator, Optional

import httpx

from config import (
    OLLAMA_ENDPOINTS,
    OLLAMA_URL,
    OLLAMA_DEFAULT_MODEL,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
)

logger = logging.getLogger("anamnesis.avatar.llm")

# AnamnesisGPT endpoints (reused from routes/anamnesis_gpt.py config)
_NANOGPT_URLS_RAW = os.environ.get("NANOGPT_URLS", os.environ.get("NANOGPT_URL", ""))
ANAMNESIS_GPT_ENDPOINTS = [u.strip().rstrip("/") for u in _NANOGPT_URLS_RAW.split(",") if u.strip()]


# ─── Ollama ─────────────────────────────────────────────────────

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


# ─── Language detection (applied to all backends) ───────────────

_FRENCH_HINTS = re.compile(
    r"\b(bonjour|salut|merci|comment|tu\s+vas|je\s+suis|qu['’]est|s['’]il|ça|c['’]est|t['’]aime|pourquoi|"
    r"français|vois|parle|aime|fais|veux)\b|\b(ais|ait|ez|ont|ent)\b\s|é|è|ê|à|ç|û|ù|ô",
    re.IGNORECASE,
)
_ENGLISH_HINTS = re.compile(
    r"\b(hello|hi|hey|what|why|how|where|when|who|the|you|your|are|is|what's|don't|i'm|i\s+am|please|could|would|should)\b",
    re.IGNORECASE,
)

def _detect_language(text: str) -> str:
    t = text.strip()
    if not t:
        return ""
    fr = len(_FRENCH_HINTS.findall(t))
    en = len(_ENGLISH_HINTS.findall(t))
    if fr > en:
        return "fr"
    if en > fr:
        return "en"
    return ""


def _lang_instruction(lang: str) -> str:
    return {
        "en": "\n\nReply in English. Do not use French.",
        "fr": "\n\nRéponds en français. Don't use English.",
    }.get(lang, "")


# ─── Backend: Ollama ────────────────────────────────────────────

async def _stream_ollama(
    system: str,
    user_message: str,
    model: Optional[str],
    previous_messages: Optional[list[dict]],
) -> AsyncIterator[tuple[str, str]]:
    model = model or OLLAMA_DEFAULT_MODEL
    endpoint = await _find_ollama_endpoint()
    if not endpoint:
        yield ("error", "No Ollama endpoint reachable")
        return
    base, label, _ = endpoint
    yield ("endpoint", f"ollama:{label}")

    messages = [{"role": "system", "content": system + _lang_instruction(_detect_language(user_message))}]
    if previous_messages:
        messages.extend(previous_messages)
    messages.append({"role": "user", "content": user_message})

    payload = {"model": model, "stream": True, "messages": messages}

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
                    content = (data.get("message") or {}).get("content", "")
                    if content:
                        yield ("token", content)
                    if data.get("done"):
                        return
    except Exception as exc:
        yield ("error", str(exc))


# ─── Backend: Claude API ────────────────────────────────────────

async def _stream_claude(
    system: str,
    user_message: str,
    model: Optional[str],
    previous_messages: Optional[list[dict]],
) -> AsyncIterator[tuple[str, str]]:
    if not ANTHROPIC_API_KEY:
        yield ("error", "ANTHROPIC_API_KEY not set on the server")
        return
    model = model or CLAUDE_MODEL
    yield ("endpoint", f"claude:{model}")

    # Claude wants system as a separate field, messages = [user/assistant turns only]
    messages = list(previous_messages or [])
    messages.append({"role": "user", "content": user_message})

    system_with_lang = system + _lang_instruction(_detect_language(user_message))

    payload = {
        "model": model,
        "max_tokens": 1024,
        "system": system_with_lang,
        "messages": messages,
        "stream": True,
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", "https://api.anthropic.com/v1/messages",
                headers=headers, json=payload,
            ) as r:
                if r.status_code != 200:
                    body = await r.aread()
                    yield ("error", f"Claude {r.status_code}: {body.decode()[:200]}")
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        return
                    try:
                        ev = json.loads(raw)
                    except Exception:
                        continue
                    if ev.get("type") == "content_block_delta":
                        delta = ev.get("delta", {})
                        if delta.get("type") == "text_delta":
                            token = delta.get("text", "")
                            if token:
                                yield ("token", token)
    except Exception as exc:
        yield ("error", str(exc))


# ─── Backend: AnamnesisGPT (δ² or QLoRA adapter) ────────────────

async def _stream_anamnesis_gpt(
    system: str,
    user_message: str,
    model: Optional[str],
    previous_messages: Optional[list[dict]],
) -> AsyncIterator[tuple[str, str]]:
    if not ANAMNESIS_GPT_ENDPOINTS:
        yield ("error", "NANOGPT_URLS not configured — AnamnesisGPT unavailable")
        return

    # Build a flat prompt from system + prior turns + current user message.
    # The trainer's /generate endpoint accepts a single prompt string.
    parts = [f"[SYSTEM] {system + _lang_instruction(_detect_language(user_message))}"]
    for m in (previous_messages or []):
        role = m.get("role", "")
        content = m.get("content", "")
        parts.append(f"[{role.upper()}] {content}")
    parts.append(f"[USER] {user_message}")
    parts.append("[ASSISTANT]")
    prompt = "\n\n".join(parts)

    body = {"prompt": prompt, "max_tokens": 512, "temperature": 0.8, "top_k": 200, "stream": True}

    for url in ANAMNESIS_GPT_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", f"{url}/generate", json=body) as r:
                    if r.status_code != 200:
                        text = await r.aread()
                        logger.warning(f"AnamnesisGPT {url} returned {r.status_code}: {text[:200]}")
                        continue
                    yield ("endpoint", f"anamnesis_gpt:{url}")
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw:
                            continue
                        try:
                            ev = json.loads(raw)
                        except Exception:
                            continue
                        if ev.get("done"):
                            return
                        token = ev.get("token", "")
                        if token:
                            yield ("token", token)
                    return  # success on this endpoint
        except Exception as exc:
            logger.warning(f"AnamnesisGPT endpoint {url} failed: {exc}")
            continue

    yield ("error", f"No AnamnesisGPT endpoint reachable. Tried: {', '.join(ANAMNESIS_GPT_ENDPOINTS)}")


# ─── Public dispatcher ──────────────────────────────────────────

async def stream_reply(
    system: str,
    user_message: str,
    model: Optional[str] = None,
    previous_messages: Optional[list[dict]] = None,
    backend: str = "ollama",
) -> AsyncIterator[tuple[str, str]]:
    """Dispatch to the requested backend. Yields (kind, text) tuples."""
    if backend == "ollama":
        async for evt in _stream_ollama(system, user_message, model, previous_messages):
            yield evt
    elif backend == "claude":
        async for evt in _stream_claude(system, user_message, model, previous_messages):
            yield evt
    elif backend in ("anamnesis_gpt", "anamnesis-gpt", "d2"):
        async for evt in _stream_anamnesis_gpt(system, user_message, model, previous_messages):
            yield evt
    else:
        yield ("error", f"Unknown backend: {backend!r}")


# ─── Utility: unload Ollama model (frees VRAM between turns) ────

async def unload_model(model: Optional[str] = None):
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
