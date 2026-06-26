"""LLM wrapper for avatar — streams from one of three backends.

Backends:
  - "ollama"         → local Ollama endpoint (default)
  - "claude"         → Anthropic Claude API (requires ANTHROPIC_API_KEY)
  - "anamnesis_gpt"  → the fine-tuned / δ² model via trainer inference endpoint

All backends yield (kind, text) tuples, where kind ∈ {"token", "endpoint", "error"}.
Backend selection happens per-request — no global state change.
"""
import asyncio
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

# δ² engine endpoint (dedicated trainer container running d2/inference.py).
# Different model + optimizer than the AnamnesisGPT QLoRA fine-tune, so it
# gets its own URL. May be unset if the δ² trainer isn't deployed yet.
D2_ENDPOINT = os.environ.get("D2_ENDPOINT_URL", "").rstrip("/")


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
    sampling: Optional[dict] = None,
) -> AsyncIterator[tuple[str, str]]:
    # Read-timeout policy:
    #   - 12s when the model is already loaded (ollama emits headers + tokens
    #     within ~2s once warm). Catches the zombie-runner wedge fast.
    #   - 90s on cold load (model not in /api/ps). Small models like llama3.2
    #     load in 15-30s on RX 6800, but qwen2.5:14b can take 60s+ to swap
    #     into VRAM from disk. 90s covers both without false-positiving a
    #     legitimate cold load as a wedge.
    WARM_READ_TIMEOUT_S = 12.0
    COLD_READ_TIMEOUT_S = 90.0

    model = model or OLLAMA_DEFAULT_MODEL

    if OLLAMA_URL:
        candidates = [(OLLAMA_URL, "custom")]
    else:
        candidates = [(u, l) for (u, l, _gpu) in OLLAMA_ENDPOINTS]
    if not candidates:
        yield ("error", "No Ollama endpoints configured")
        return

    messages = [{"role": "system", "content": system + _lang_instruction(_detect_language(user_message))}]
    if previous_messages:
        messages.extend(previous_messages)
    messages.append({"role": "user", "content": user_message})
    payload = {"model": model, "stream": True, "messages": messages}
    # Layer-2 overrides (advanced settings): only keys with non-None values
    # are forwarded. Ollama applies its defaults to anything we omit.
    if sampling:
        opts = {k: v for k, v in sampling.items()
                if v is not None and k in ("temperature", "top_p", "top_k", "repeat_penalty")}
        if opts:
            payload["options"] = opts
            logger.info(f"Ollama sampling overrides: {opts}")

    last_err: Optional[str] = None
    for base, label in candidates:
        # Probe: is ollama up AND is our model already loaded?
        # /api/tags = reachability. /api/ps = currently-resident models.
        # The ps probe lets us distinguish a slow cold-load (legit, give it time)
        # from a wedged runner (zombie subprocess, no time will help).
        model_warm = False
        try:
            async with httpx.AsyncClient(timeout=4.0) as c:
                resp = await c.get(f"{base}/api/tags")
                if resp.status_code != 200:
                    last_err = f"{label}: /api/tags {resp.status_code}"
                    logger.warning(f"Ollama probe {label} failed: {last_err}")
                    continue
                try:
                    ps_resp = await c.get(f"{base}/api/ps")
                    if ps_resp.status_code == 200:
                        loaded = [m.get("name", "") for m in ps_resp.json().get("models", [])]
                        # ollama suffixes ":latest" sometimes; match by prefix.
                        model_warm = any(n.split(":")[0] == model.split(":")[0] for n in loaded)
                except Exception:
                    pass  # /api/ps optional — fall back to cold-timeout
        except Exception as exc:
            last_err = f"{label}: probe failed ({exc})"
            logger.warning(f"Ollama probe {label} failed: {exc}")
            continue

        read_timeout = WARM_READ_TIMEOUT_S if model_warm else COLD_READ_TIMEOUT_S
        if not model_warm:
            logger.info(f"Ollama {label}: model {model!r} not in /api/ps — using cold-load timeout {COLD_READ_TIMEOUT_S}s")
        chat_timeout = httpx.Timeout(connect=5.0, read=read_timeout, write=10.0, pool=5.0)
        got_first = False
        try:
            async with httpx.AsyncClient(timeout=chat_timeout) as client:
                async with client.stream("POST", f"{base}/api/chat", json=payload) as r:
                    if r.status_code != 200:
                        body = await r.aread()
                        last_err = f"{label}: chat HTTP {r.status_code} {body.decode(errors='replace')[:200]}"
                        logger.warning(f"Ollama {label} chat returned {r.status_code}")
                        continue
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except Exception:
                            continue
                        content = (data.get("message") or {}).get("content", "")
                        if content:
                            if not got_first:
                                yield ("endpoint", f"ollama:{label}")
                                got_first = True
                            yield ("token", content)
                        if data.get("done"):
                            return
                    if got_first:
                        return  # full stream completed cleanly
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
            last_err = f"{label}: chat timed out ({type(exc).__name__})"
            logger.warning(f"Ollama {label} chat timeout — falling through to next endpoint")
            # Fire-and-forget self-heal. Doesn't block this request; next request
            # finds the endpoint restarted (cooldown-gated to avoid thrashing).
            try:
                from avatar import recovery
                asyncio.create_task(recovery.try_heal(base, reason=f"chat {type(exc).__name__}"))
            except Exception as heal_exc:
                logger.warning(f"recovery scheduling failed: {heal_exc}")
            continue
        except Exception as exc:
            last_err = f"{label}: chat failed ({exc})"
            logger.warning(last_err)
            continue

    yield ("error", f"All Ollama endpoints failed. Last: {last_err or 'unknown'}")


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
    # Use 'User:'/'Assistant:' (familiar from training data) instead of bracketed
    # [USER]/[ASSISTANT] which models echo literally as if they were tokens.
    parts = [f"System: {system + _lang_instruction(_detect_language(user_message))}"]
    for m in (previous_messages or []):
        role = (m.get("role", "") or "").capitalize() or "User"
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
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
                    # The 1.5B demo has no reliable EOS, and the trainer's
                    # /generate accepts no `stop` param (confirmed w/ the δ²
                    # chat, intercom MSG-319). Left unbounded it role-plays
                    # BOTH sides of the dialogue — emitting literal "User:" /
                    # "Assistant:" turn labels — until max_tokens guillotines
                    # it mid-word. Fix client-side: cut the stream at the first
                    # turn label. Buffer a short tail so a label split across
                    # token boundaries ("Assis" + "tant:") is still caught.
                    stops = ("User:", "Assistant:", "System:")
                    holdback = max(len(s) for s in stops) - 1
                    buf = ""

                    def _earliest_stop(text: str) -> int:
                        idx = -1
                        for s in stops:
                            i = text.find(s)
                            if i != -1 and (idx == -1 or i < idx):
                                idx = i
                        return idx

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
                            break
                        token = ev.get("token", "")
                        if not token:
                            continue
                        buf += token
                        cut = _earliest_stop(buf)
                        if cut != -1:
                            if cut > 0:
                                yield ("token", buf[:cut])
                            return  # drop the hallucinated continuation
                        if len(buf) > holdback:
                            yield ("token", buf[:-holdback])
                            buf = buf[-holdback:]
                    if buf:
                        yield ("token", buf)  # flush tail (no stop label seen)
                    return  # success on this endpoint
        except Exception as exc:
            logger.warning(f"AnamnesisGPT endpoint {url} failed: {exc}")
            continue

    yield ("error", f"No AnamnesisGPT endpoint reachable. Tried: {', '.join(ANAMNESIS_GPT_ENDPOINTS)}")


# ─── Backend: δ² engine (dedicated trainer) ─────────────────────

async def _stream_d2(
    system: str,
    user_message: str,
    model: Optional[str],
    previous_messages: Optional[list[dict]],
) -> AsyncIterator[tuple[str, str]]:
    """
    Stream from the δ² trainer container.

    Distinct from _stream_anamnesis_gpt:
      - Talks to D2_ENDPOINT_URL (one trainer, not a failover chain)
      - The trainer's /generate accepts an `enable_bassin_recall` flag
        which is true by default — this is what makes δ² different from
        ordinary text generation: when the model is uncertain, it pulls
        from the bassin and re-generates with contrastive context.

    Falls back gracefully (yields "error") if D2_ENDPOINT_URL isn't set.
    """
    if not D2_ENDPOINT:
        yield ("error", "D2_ENDPOINT_URL not configured — δ² trainer not deployed")
        return

    # Use 'User:'/'Assistant:' format — base models echo literal [USER]/[ASSISTANT]
    # tokens as if they were actual response content (real bug observed in chat).
    parts = [f"System: {system + _lang_instruction(_detect_language(user_message))}"]
    for m in (previous_messages or []):
        role = (m.get("role", "") or "").capitalize() or "User"
        parts.append(f"{role}: {m.get('content', '')}")
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    prompt = "\n\n".join(parts)

    body = {
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.8,
        "top_k": 200,
        "enable_bassin_recall": True,
        "uncertainty_threshold": 1.5,
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{D2_ENDPOINT}/generate", json=body) as r:
                if r.status_code != 200:
                    text = await r.aread()
                    yield ("error", f"d2 {r.status_code}: {text.decode()[:200]}")
                    return
                yield ("endpoint", f"d2:{D2_ENDPOINT}")
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
    except Exception as exc:
        yield ("error", f"d2 endpoint error: {exc}")


# ─── OpenAI-compatible streaming (Together.ai, RunPod, generic vLLM) ──

async def _stream_openai_compat(
    system: str,
    user_message: str,
    model: Optional[str],
    previous_messages: Optional[list[dict]],
    base_url: str,
    api_key: str,
    backend_label: str,
) -> AsyncIterator[tuple[str, str]]:
    """Stream from any OpenAI-compatible /v1/chat/completions endpoint."""
    if not base_url:
        yield ("error", f"{backend_label} backend not configured (no base_url)")
        return
    msgs = [{"role": "system", "content": system}]
    if previous_messages:
        msgs.extend(previous_messages)
    msgs.append({"role": "user", "content": user_message})
    payload = {
        "model": model or "",
        "messages": msgs,
        "stream": True,
        "max_tokens": 256,
        "temperature": 0.85,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{base_url.rstrip('/')}/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode(errors="replace")[:200]
                    yield ("error", f"{backend_label} HTTP {r.status_code}: {body}")
                    return
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[len("data: "):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0]["delta"].get("content")
                        if delta:
                            yield ("token", delta)
                    except Exception:
                        continue
    except Exception as exc:
        yield ("error", f"{backend_label} endpoint error: {exc}")


# ─── Public dispatcher ──────────────────────────────────────────

async def stream_reply(
    system: str,
    user_message: str,
    model: Optional[str] = None,
    previous_messages: Optional[list[dict]] = None,
    backend: str = "ollama",
    sampling: Optional[dict] = None,
) -> AsyncIterator[tuple[str, str]]:
    """Dispatch to the requested backend. Yields (kind, text) tuples.

    sampling: optional dict of ollama-style overrides — keys among
    {temperature, top_p, top_k, repeat_penalty}. Only non-None values
    are forwarded. Currently honored by the ollama backend only; other
    backends ignore (they have their own knobs and we haven't wired them
    through yet)."""
    if backend == "ollama":
        async for evt in _stream_ollama(system, user_message, model, previous_messages, sampling=sampling):
            yield evt
    elif backend == "claude":
        async for evt in _stream_claude(system, user_message, model, previous_messages):
            yield evt
    elif backend in ("anamnesis_gpt", "anamnesis-gpt"):
        async for evt in _stream_anamnesis_gpt(system, user_message, model, previous_messages):
            yield evt
    elif backend == "d2":
        async for evt in _stream_d2(system, user_message, model, previous_messages):
            yield evt
    elif backend == "together":
        async for evt in _stream_openai_compat(
            system, user_message, model, previous_messages,
            base_url=config.TOGETHER_BASE_URL,
            api_key=config.TOGETHER_API_KEY,
            backend_label="together",
        ):
            yield evt
    elif backend == "runpod":
        async for evt in _stream_openai_compat(
            system, user_message, model, previous_messages,
            base_url=config.RUNPOD_ENDPOINT_URL,
            api_key=config.RUNPOD_API_KEY,
            backend_label="runpod",
        ):
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
