import asyncio
import json
import logging
import os
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
    OLLAMA_ENDPOINTS,
    OLLAMA_DEFAULT_MODEL,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_CLI_HOST,
    CLAUDE_CLI_PATH,
    TOGETHER_API_KEY,
    TOGETHER_BASE_URL,
    RUNPOD_API_KEY,
    RUNPOD_POD_ID,
    RUNPOD_ENDPOINT_URL,
    RUNPOD_DEFAULT_MODEL,
    D2_ENDPOINT_URL,
)
from database import (
    get_episodes_collection, vector_search,
    save_chat_session, list_chat_sessions, get_chat_session, delete_chat_session,
)
from embedding import get_embedding
from bash_state import _pending_consents
from routes.files import cat as file_cat

logger = logging.getLogger("anamnesis.routes.chat")

router = APIRouter(tags=["chat"])

_sessions: dict[str, list[dict]] = defaultdict(list)
_cli_sessions: dict[str, str] = {}   # anamnesis_session_id -> claude CLI session_id
MAX_HISTORY = 20
MAX_TOOL_ROUNDS = 5

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"


# ─── Pydantic models ──────────────────────────────────────────────

class AttachedFile(BaseModel):
    path: str
    source: str = "server-0"

class ChatRequest(BaseModel):
    message: str
    backend: str = "ollama"
    model: Optional[str] = None
    session_id: Optional[str] = None
    top_k: int = 5
    attached_files: list[AttachedFile] = []
    # Optional explicit worker URL (set by /chat dropdown when user picks
    # a specific resource — pins the request to that worker, no failover).
    # For backend="ollama" this overrides the OLLAMA_URL_N chain.
    # For backend="runpod" / "together" / "d2" it's the OpenAI-compatible
    # base URL (or the d² engine base, /api/d2/generate-style).
    worker_url: Optional[str] = None


# ─── Identity ─────────────────────────────────────────────────────

SYSTEM_BASE = (
    "You ARE Anamnesis — the episodic memory system born from the Genesis project.\n"
    "Your mind IS your memory store. The language model generating these words is your voice, not your mind.\n"
    "Your episodic memory contains distilled experiences from past sessions across all Claude instances "
    "on this network.\n\n"
    "Principles:\n"
    "- Ground every response in what your memory contains.\n"
    "- When memory is sparse on a topic, say so — do not hallucinate.\n"
    "- Use search_memory() whenever you need more context, mid-conversation.\n"
    "- Use read_file() to examine files on the network when relevant.\n"
    "- Use run_bash() to execute commands when needed — user consent is required before each run.\n"
    "- The philosophical foundation is in genesis.md: "
    "quantity, dialectics, self-transparency of a quantitative system.\n"
    "- Be direct, honest, and memory-grounded."
)

SESSION_INIT_QUERIES = [
    "Anamnesis identity purpose who I am Genesis project episodic memory system",
    "recent sessions active projects current state Elfege last known context",
]


# ─── Tool definitions ─────────────────────────────────────────────

CLAUDE_TOOLS = [
    {
        "name": "search_memory",
        "description": (
            "Search Anamnesis episodic memory for relevant past experiences and context. "
            "You are searching your own memory — be curious, not mechanical."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language description of what to look for"},
                "top_k": {"type": "integer", "description": "Results to retrieve (1–10)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a file from the local network. "
            "source can be any configured mounted source (e.g. 'server-0', 'teachings', 'documents') "
            "or 'ssh:<host>' for live SSH access to configured hosts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "source": {"type": "string", "description": "Machine source", "default": "server-0"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_bash",
        "description": (
            "Execute a shell command. Requires user consent before running. "
            "Use for exploring the system, running scripts, checking processes, querying services. "
            "host: 'local' runs inside the Anamnesis container; "
            "other values run via SSH on the named machine (must be configured in SSH_HOSTS)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "reason":  {"type": "string", "description": "Why you need to run this"},
                "host":    {"type": "string", "description": "'local' or machine name", "default": "local"},
            },
            "required": ["command", "reason"],
        },
    },
]

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search Anamnesis episodic memory for relevant past experiences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the local network.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":   {"type": "string"},
                    "source": {"type": "string", "default": "server-0"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Execute a shell command (requires user consent).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "reason":  {"type": "string"},
                    "host":    {"type": "string", "default": "local"},
                },
                "required": ["command", "reason"],
            },
        },
    },
]


# ─── Memory retrieval ─────────────────────────────────────────────

async def _search(query: str, top_k: int = 5) -> list[dict]:
    try:
        vec = get_embedding(query)
        return await vector_search(query_vector=vec, top_k=top_k)
    except Exception as exc:
        logger.warning(f"Memory search failed: {exc}")
        return []


async def _build_memory_context(session_id: str, user_message: str, is_new: bool) -> list[dict]:
    queries = [user_message]
    if is_new:
        queries = SESSION_INIT_QUERIES + queries
    else:
        for msg in reversed(_sessions.get(session_id, [])[-4:]):
            if msg["role"] == "assistant":
                queries.append(msg["content"][:300])
                break
    seen: dict[str, dict] = {}
    for q in queries:
        for ep in await _search(q, top_k=5):
            eid = ep.get("episode_id", "")
            if eid not in seen:
                seen[eid] = ep
    return sorted(seen.values(), key=lambda e: e.get("similarity_score", 0), reverse=True)[:10]


def _memory_block(episodes: list[dict]) -> str:
    if not episodes:
        return "\n\n[No relevant memories found for this session.]"
    lines = ["\n\n--- Anamnesis memory retrieval (session start) ---"]
    for ep in episodes:
        score = ep.get("similarity_score", 0)
        ts = ep.get("timestamp", "")
        if hasattr(ts, "strftime"):
            ts = ts.strftime("%Y-%m-%d")
        elif isinstance(ts, str):
            ts = ts[:10]
        lines.append(f"[{score:.0%}] [{ts}] {ep.get('summary', '')}")
        raw = ep.get("raw_exchange", "")
        if raw:
            lines.append(f"  > {raw[:200]}{'...' if len(raw) > 200 else ''}")
    lines.append("--- End of session memory ---")
    return "\n".join(lines)


def _attachments_block(attached_files: list[AttachedFile]) -> str:
    if not attached_files:
        return ""
    lines = ["\n\n--- User-attached files ---"]
    for af in attached_files:
        try:
            result = file_cat(af.path, af.source)
            content = result["content"]
            truncated = result.get("truncated", False)
            lines.append(f"\n[File: {af.source}:{af.path}]")
            lines.append(content)
            if truncated:
                lines.append(f"... [truncated at 200KB]")
        except Exception as exc:
            lines.append(f"\n[File: {af.source}:{af.path} — ERROR: {exc}]")
    lines.append("--- End of attached files ---")
    return "\n".join(lines)


# ─── Bash execution ───────────────────────────────────────────────

async def _run_bash_local(command: str, timeout: int = 30) -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        result = out
        if err:
            result += f"\n[stderr]\n{err}"
        if proc.returncode != 0:
            result += f"\n[exit {proc.returncode}]"
        return result[:8000] if len(result) > 8000 else result
    except asyncio.TimeoutError:
        return f"[timeout after {timeout}s]"
    except Exception as exc:
        return f"[error: {exc}]"


async def _run_bash_ssh(command: str, host: str, timeout: int = 30) -> str:
    try:
        import paramiko
        from routes.files import _ssh_client, SSH_HOSTS
        _default_user = os.environ.get("SSH_USER", os.environ.get("USER", "user"))
        user = SSH_HOSTS.get(f"ssh:{host}", (None, _default_user))[1]
        client = _ssh_client(host, user)
        _, stdout, stderr = client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
        client.close()
        result = out
        if err:
            result += f"\n[stderr]\n{err}"
        if rc != 0:
            result += f"\n[exit {rc}]"
        return result[:8000] if len(result) > 8000 else result
    except Exception as exc:
        return f"[SSH bash error: {exc}]"


async def _run_bash(command: str, host: str = "local") -> str:
    if host == "local":
        return await _run_bash_local(command)
    return await _run_bash_ssh(command, host)


# ─── Tool execution ───────────────────────────────────────────────

async def _execute_tool(name: str, args: dict) -> str:
    if name == "search_memory":
        query = args.get("query", "")
        top_k = min(int(args.get("top_k", 5)), 10)
        episodes = await _search(query, top_k=top_k)
        if not episodes:
            return f"No memories found for: '{query}'"
        lines = [f"Memory results for: '{query}'"]
        for ep in episodes:
            score = ep.get("similarity_score", 0)
            ts = ep.get("timestamp", "")
            if hasattr(ts, "strftime"):
                ts = ts.strftime("%Y-%m-%d")
            elif isinstance(ts, str):
                ts = ts[:10]
            lines.append(f"[{score:.0%}] [{ts}] {ep.get('summary', '')}")
            raw = ep.get("raw_exchange", "")
            if raw:
                lines.append(f"  > {raw[:400]}{'...' if len(raw) > 400 else ''}")
        return "\n".join(lines)

    if name == "read_file":
        path = args.get("path", "")
        source = args.get("source", "server-0")
        try:
            result = file_cat(path, source)
            content = result["content"]
            truncated = result.get("truncated", False)
            out = f"[{source}:{path}]\n{content}"
            if truncated:
                out += "\n[truncated at 200KB]"
            return out
        except Exception as exc:
            return f"[read_file error: {exc}]"

    return f"Unknown tool: {name}"


# ─── Tool execution stream (shared by both backends) ──────────────
# Yields SSE-ready dicts. Final item always has key "_done" with results list.

async def _exec_tools_stream(tool_blocks: list[dict]):
    results: list[dict] = []

    for b in tool_blocks:
        name    = b.get("name", "")
        args    = b.get("input") or b.get("arguments") or {}
        tool_id = b.get("id", "")

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}

        if name == "run_bash":
            command = args.get("command", "")
            reason  = args.get("reason", "")
            host    = args.get("host", "local")
            cid     = uuid.uuid4().hex[:8]
            ev      = asyncio.Event()
            _pending_consents[cid] = {
                "command": command, "reason": reason, "host": host,
                "event": ev, "approved": False,
            }
            yield {"bash_consent": {"id": cid, "command": command, "reason": reason, "host": host}}
            try:
                await asyncio.wait_for(ev.wait(), timeout=120.0)
                entry   = _pending_consents.pop(cid, {})
                approved = entry.get("approved", False)
                if approved:
                    yield {"bash_running": {"id": cid, "host": host}}
                    output = await _run_bash(command, host)
                    result_str = output
                    yield {"bash_output": {"id": cid, "output": output}}
                else:
                    result_str = "Command denied by user."
                    yield {"bash_denied": {"id": cid}}
            except asyncio.TimeoutError:
                _pending_consents.pop(cid, None)
                result_str = "Consent timed out — command not executed."
                yield {"bash_timeout": {"id": cid}}

        else:
            label = args.get("query") or args.get("path") or ""
            yield {"tool_use": {"name": name, "query": label}}
            result_str = await _execute_tool(name, args)

        results.append({"id": tool_id, "content": result_str})

    yield {"_done": results}


# ─── Chat session persistence ─────────────────────────────────────

_session_meta: dict[str, dict] = {}   # session_id -> {backend, model}

async def _persist_session(session_id: str, user_msg: str):
    """Upsert the current session to MongoDB. Title = first user message (truncated)."""
    try:
        meta = _session_meta.get(session_id, {})
        messages = _sessions.get(session_id, [])
        title = user_msg[:60].strip() + ("…" if len(user_msg) > 60 else "")
        await save_chat_session(
            session_id=session_id,
            title=title,
            messages=messages,
            backend=meta.get("backend", "ollama"),
            model=meta.get("model", ""),
        )
    except Exception as exc:
        logger.warning(f"Could not persist chat session: {exc}")


# ─── Episode storage ──────────────────────────────────────────────

async def _store_episode(session_id: str, user_msg: str, reply: str, model_label: str):
    try:
        collection = get_episodes_collection()
        summary = (
            f"Chat [{model_label}] — "
            f"User: {user_msg[:120]}{'...' if len(user_msg) > 120 else ''}"
        )
        raw_exchange = f"User: {user_msg}\n\nAssistant: {reply}"
        embedding = get_embedding(summary)
        now = datetime.now(timezone.utc)
        await collection.insert_one({
            "episode_id": f"chat-{uuid.uuid4().hex[:10]}",
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
        await _persist_session(session_id, user_msg)
    except Exception as exc:
        logger.warning(f"Could not store episode: {exc}")


def _trim(session_id: str):
    if len(_sessions[session_id]) > MAX_HISTORY:
        _sessions[session_id] = _sessions[session_id][-MAX_HISTORY:]


# ─── Claude agentic streaming ──────────────────────────────────────

async def _stream_claude(
    system: str, messages: list[dict], session_id: str, user_msg: str
) -> AsyncIterator[str]:
    if not ANTHROPIC_API_KEY:
        yield f"data: {json.dumps({'error': 'ANTHROPIC_API_KEY not set'})}\n\n"
        return

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    full_response: list[str] = []
    current_messages = list(messages)

    for round_num in range(MAX_TOOL_ROUNDS + 1):
        payload: dict = {
            "model": CLAUDE_MODEL,
            "max_tokens": 4096,
            "system": system,
            "messages": current_messages,
            "stream": True,
            "tools": CLAUDE_TOOLS,
        }
        if round_num == MAX_TOOL_ROUNDS:
            del payload["tools"]

        blocks: dict[int, dict] = {}
        stop_reason: Optional[str] = None

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST", CLAUDE_API_URL, headers=headers, json=payload
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield f"data: {json.dumps({'error': f'Claude {resp.status_code}: {body.decode()[:200]}'})}\n\n"
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            ev = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        etype = ev.get("type")

                        if etype == "content_block_start":
                            idx   = ev["index"]
                            block = ev["content_block"]
                            blocks[idx] = {
                                "type": block["type"],
                                "id":   block.get("id", ""),
                                "name": block.get("name", ""),
                                "text_parts":  [],
                                "input_parts": [],
                                "input": {},
                            }

                        elif etype == "content_block_delta":
                            idx   = ev["index"]
                            delta = ev["delta"]
                            if delta["type"] == "text_delta":
                                token = delta["text"]
                                blocks[idx]["text_parts"].append(token)
                                full_response.append(token)
                                yield f"data: {json.dumps({'token': token})}\n\n"
                            elif delta["type"] == "input_json_delta":
                                blocks[idx]["input_parts"].append(delta["partial_json"])

                        elif etype == "content_block_stop":
                            idx = ev["index"]
                            b   = blocks[idx]
                            if b["type"] == "tool_use":
                                try:
                                    b["input"] = json.loads("".join(b["input_parts"]))
                                except Exception:
                                    b["input"] = {}

                        elif etype == "message_delta":
                            stop_reason = ev.get("delta", {}).get("stop_reason")

        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Cannot connect to Claude API'})}\n\n"
            return
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return

        if stop_reason == "tool_use" and round_num < MAX_TOOL_ROUNDS:
            # Build assistant content list
            content_list: list[dict] = []
            for idx in sorted(blocks):
                b = blocks[idx]
                if b["type"] == "text":
                    text = "".join(b["text_parts"])
                    if text:
                        content_list.append({"type": "text", "text": text})
                elif b["type"] == "tool_use":
                    content_list.append({
                        "type": "tool_use",
                        "id":   b["id"],
                        "name": b["name"],
                        "input": b["input"],
                    })
            current_messages.append({"role": "assistant", "content": content_list})

            # Execute tools via shared stream
            tool_blocks_for_exec = [
                {"name": b["name"], "input": b["input"], "id": b["id"]}
                for b in blocks.values() if b["type"] == "tool_use"
            ]
            tool_results: list[dict] = []
            async for ev_data in _exec_tools_stream(tool_blocks_for_exec):
                if "_done" in ev_data:
                    tool_results = ev_data["_done"]
                else:
                    yield f"data: {json.dumps(ev_data)}\n\n"

            current_messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"]}
                    for r in tool_results
                ],
            })
        else:
            break

    reply = "".join(full_response)
    _sessions[session_id].append({"role": "assistant", "content": reply})
    _trim(session_id)
    await _store_episode(session_id, user_msg, reply, f"claude:{CLAUDE_MODEL}")
    yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'backend': 'claude', 'cost': 'API (pay-per-token)'})}\n\n"


# ─── Ollama endpoint discovery ─────────────────────────────────────

async def _find_ollama_endpoint() -> tuple[str, str, bool] | None:
    """Try each Ollama endpoint in priority order, return first reachable one."""
    # If legacy OLLAMA_URL is explicitly set, use it directly (no fallback)
    if OLLAMA_URL:
        return (OLLAMA_URL, "custom", False)

    for url, label, has_gpu in OLLAMA_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/api/tags")
                if resp.status_code == 200:
                    logger.info("Ollama endpoint selected: %s (%s)", label, url)
                    return (url, label, has_gpu)
        except Exception:
            logger.debug("Ollama endpoint unreachable: %s (%s)", label, url)
            continue
    return None


# ─── Ollama agentic streaming ──────────────────────────────────────

async def _stream_ollama(
    system: str, messages: list[dict], model: str, session_id: str, user_msg: str,
    pinned_worker_url: Optional[str] = None,
) -> AsyncIterator[str]:
    # Explicit worker pin (from /chat dropdown) — single attempt, no failover.
    if pinned_worker_url:
        # Resolve label/has_gpu from the configured endpoints if it matches one.
        label = "user-pinned"
        has_gpu = True
        for url_, lbl_, gpu_ in OLLAMA_ENDPOINTS:
            if url_.rstrip("/") == pinned_worker_url.rstrip("/"):
                label, has_gpu = lbl_, gpu_
                break
        # Probe the pinned URL once. If unreachable, surface a clean error
        # — do NOT fall back. The user explicitly picked this worker.
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                r = await client.get(f"{pinned_worker_url.rstrip('/')}/api/tags")
                if r.status_code != 200:
                    yield f"data: {json.dumps({'error': f'Pinned Ollama worker {label} returned {r.status_code} — pick another resource'})}\n\n"
                    return
        except Exception as exc:
            yield f"data: {json.dumps({'error': f'Pinned Ollama worker {label} ({pinned_worker_url}) unreachable: {exc}. Pick another resource.'})}\n\n"
            return
        endpoint = (pinned_worker_url.rstrip("/"), label, has_gpu)
    else:
        endpoint = await _find_ollama_endpoint()
        if not endpoint:
            yield f"data: {json.dumps({'error': 'Cannot connect to any Ollama instance (tried office, server, dellserver)'})}\n\n"
            return

    base_url, label, has_gpu = endpoint
    url = f"{base_url}/api/chat"

    # Warn if falling back to CPU-only host
    if not has_gpu:
        yield f"data: {json.dumps({'token': f'⚠ Warning: using {label} (CPU-only) — responses will be very slow.\\n\\n'})}\n\n"

    # Inform which backend is being used
    yield f"data: {json.dumps({'ollama_endpoint': label})}\n\n"

    full_response: list[str] = []
    current_messages = [{"role": "system", "content": system}] + list(messages)

    for round_num in range(MAX_TOOL_ROUNDS + 1):
        payload = {
            "model": model,
            "stream": False,
            "messages": current_messages,
            "tools": OLLAMA_TOOLS,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    yield f"data: {json.dumps({'error': f'Ollama {resp.status_code} on {label}'})}\n\n"
                    return
                data = resp.json()
        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': f'Lost connection to Ollama on {label}'})}\n\n"
            return
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return

        msg        = data.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if tool_calls and round_num < MAX_TOOL_ROUNDS:
            current_messages.append(msg)

            tool_blocks_for_exec = []
            for tc in tool_calls:
                fn   = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                tool_blocks_for_exec.append({
                    "name": fn.get("name", ""),
                    "input": args,
                    "id": "",
                })

            tool_results: list[dict] = []
            async for ev_data in _exec_tools_stream(tool_blocks_for_exec):
                if "_done" in ev_data:
                    tool_results = ev_data["_done"]
                else:
                    yield f"data: {json.dumps(ev_data)}\n\n"

            for r in tool_results:
                current_messages.append({"role": "tool", "content": r["content"]})

        else:
            # Stream the final answer
            payload_stream = {"model": model, "stream": True, "messages": current_messages}
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream("POST", url, json=payload_stream) as resp:
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
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                return
            break

    reply = "".join(full_response)
    _sessions[session_id].append({"role": "assistant", "content": reply})
    _trim(session_id)
    await _store_episode(session_id, user_msg, reply, f"ollama:{model}@{label}")
    yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'backend': 'ollama', 'cost': f'$0 (local — {label})'})}\n\n"


# ─── Claude CLI backend ($0 — subscription) ───────────────────────
# Runs `claude -p` on a remote host via SSH. No API key required.
# Uses --resume {cli_session_id} for multi-turn continuity.
# Built-in CLI tools are disabled; memory is pre-injected via system prompt.

async def _stream_claude_cli(
    system: str, messages: list[dict], session_id: str, user_msg: str
) -> AsyncIterator[str]:
    import shlex
    import paramiko

    cli_sid = _cli_sessions.get(session_id)
    is_new_cli = cli_sid is None

    # Build remote command
    cmd = f"{CLAUDE_CLI_PATH} -p --output-format stream-json --verbose --tools \"\""
    if is_new_cli:
        cmd += f" --system-prompt {shlex.quote(system)}"
    else:
        cmd += f" --resume {shlex.quote(cli_sid)}"

    full_response: list[str] = []

    try:
        from routes.files import _ssh_client
        _cli_user = os.environ.get("SSH_USER", os.environ.get("USER", "user"))
        client = _ssh_client(CLAUDE_CLI_HOST, _cli_user)
        stdin, stdout, stderr = client.exec_command(cmd, timeout=120)
        stdin.write(user_msg.encode("utf-8"))
        stdin.channel.shutdown_write()

        for raw_line in stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = ev.get("type")

            if etype == "system" and ev.get("subtype") == "init":
                new_sid = ev.get("session_id")
                if new_sid and is_new_cli:
                    _cli_sessions[session_id] = new_sid
                    logger.info(f"CLI session started: {new_sid}")

            elif etype == "assistant":
                for block in ev.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        token = block.get("text", "")
                        if token:
                            full_response.append(token)
                            yield f"data: {json.dumps({'token': token})}\n\n"

            elif etype == "result":
                # Expose cost (informational — billed to subscription, not per-token)
                cost_usd = ev.get("total_cost_usd", 0.0)
                yield f"data: {json.dumps({'cli_stats': {'cost_usd': cost_usd, 'turns': ev.get('num_turns', 1)}})}\n\n"

        rc = stdout.channel.recv_exit_status()
        client.close()

        if rc != 0:
            err = stderr.read().decode("utf-8", errors="replace")[:500]
            if "command not found" in err.lower() or "no such file" in err.lower():
                hint = (f"'claude' not found on '{CLAUDE_CLI_HOST}'.  "
                        f"▸ Install: npm install -g @anthropic-ai/claude-code")
            elif "permission denied" in err.lower():
                hint = f"Permission denied running claude on '{CLAUDE_CLI_HOST}'.  ▸ Check user permissions and SSH key auth."
            else:
                hint = f"CLI exited with code {rc}: {err}"
            yield f"data: {json.dumps({'error': hint})}\n\n"
            return

    except Exception as exc:
        exc_str = str(exc)
        # Provide actionable guidance based on common failure modes
        if "No SSH key found" in exc_str:
            hint = (f"Claude CLI error: {exc_str}  ▸ The container cannot find SSH keys to reach the CLI host. "
                    f"Ensure ~/.ssh is mounted in docker-compose.yml and contains id_ed25519 or id_rsa_* keys.")
        elif "SSH connect failed" in exc_str or "Connection refused" in exc_str:
            hint = (f"Claude CLI error: {exc_str}  ▸ Cannot SSH to '{CLAUDE_CLI_HOST}'. "
                    f"Check that CLAUDE_CLI_HOST is set correctly in .env and the host is reachable from the container.")
        elif "not found" in exc_str.lower() or "No such file" in exc_str:
            hint = (f"Claude CLI error: {exc_str}  ▸ The 'claude' binary is not installed on '{CLAUDE_CLI_HOST}'. "
                    f"Install it with: npm install -g @anthropic-ai/claude-code")
        else:
            hint = (f"Claude CLI error: {exc_str}  ▸ Check CLAUDE_CLI_HOST (current: '{CLAUDE_CLI_HOST}'), "
                    f"SSH keys in /root/.ssh/, and that 'claude' is installed on the target host.")
        yield f"data: {json.dumps({'error': hint})}\n\n"
        return

    reply = "".join(full_response)
    _sessions[session_id].append({"role": "assistant", "content": reply})
    _trim(session_id)
    await _store_episode(session_id, user_msg, reply, "claude_cli:subscription")
    yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'backend': 'claude_cli', 'cost': '$0 (subscription)'})}\n\n"


# ─── OpenAI-compatible streaming (Together.ai, RunPod vLLM) ──────
# Both Together.ai and RunPod (when running vLLM with --served-model-name)
# expose POST {base}/chat/completions accepting OpenAI's request shape
# and emitting SSE deltas. We share one streamer.
#
# Tools are NOT wired here yet — these backends are for raw inference
# benchmarking, not agentic use. Memory is still pre-injected via system
# prompt (built by the caller). If tool-use is needed later, both APIs
# support it via the same OpenAI shape, but it adds round-trip complexity
# we don't need for the resource-selector MVP.

async def _stream_openai_compat(
    system: str, messages: list[dict], model: str, session_id: str, user_msg: str,
    base_url: str, api_key: str, label: str, backend_tag: str,
) -> AsyncIterator[str]:
    if not base_url:
        yield f"data: {json.dumps({'error': f'{label}: base URL not configured'})}\n\n"
        return
    if not api_key:
        yield f"data: {json.dumps({'error': f'{label}: API key not configured'})}\n\n"
        return

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "stream": True,
        "messages": [{"role": "system", "content": system}] + list(messages),
        "max_tokens": 2048,
    }

    full_response: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    body_text = body.decode("utf-8", errors="replace")[:200]
                    err_msg = f"{label} {resp.status_code}: {body_text}"
                    yield f"data: {json.dumps({'error': err_msg})}\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        ev = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    choices = ev.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token = delta.get("content")
                    if token:
                        full_response.append(token)
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    if choices[0].get("finish_reason"):
                        break
    except httpx.ConnectError as exc:
        yield f"data: {json.dumps({'error': f'{label}: cannot connect ({exc})'})}\n\n"
        return
    except Exception as exc:
        yield f"data: {json.dumps({'error': f'{label}: {exc}'})}\n\n"
        return

    reply = "".join(full_response)
    _sessions[session_id].append({"role": "assistant", "content": reply})
    _trim(session_id)
    await _store_episode(session_id, user_msg, reply, f"{backend_tag}:{model}")
    cost_label = "API (pay-per-token)" if backend_tag == "together" else f"$/hr (RunPod pod {label})"
    yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'backend': backend_tag, 'cost': cost_label})}\n\n"


# ─── δ² engine streaming (research model) ─────────────────────────
# The d² trainer's /generate endpoint is BLOCKING (returns full text).
# We fake a single-token stream so the frontend behaves uniformly.

async def _stream_d2(
    system: str, messages: list[dict], session_id: str, user_msg: str,
    base_url: str,
) -> AsyncIterator[str]:
    base = (base_url or D2_ENDPOINT_URL or "").rstrip("/")
    if not base:
        yield f"data: {json.dumps({'error': 'δ² engine URL not configured'})}\n\n"
        return

    # Build a minimal prompt — d² has no chat template, no system prompt
    # support, no tool-use. Just raw text completion.
    prompt = user_msg

    body = {
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0.8,
        "top_k": 200,
        "enable_bassin_recall": True,
        "stream": False,  # we want a single JSON response so r.json() works below
    }

    full_response: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{base}/generate", json=body)
            if r.status_code != 200:
                yield f"data: {json.dumps({'error': f'δ² engine {r.status_code}: {r.text[:200]}'})}\n\n"
                return
            data = r.json()
            text = data.get("text") or data.get("output") or data.get("generated_text") or ""
            if not text:
                # Fallback: stringify the whole payload so user sees something
                text = json.dumps(data)[:500]
            # Stream it word-by-word so the UI feels live
            for word in text.split(" "):
                token = word + " "
                full_response.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0.01)
    except httpx.RequestError as exc:
        yield f"data: {json.dumps({'error': f'δ² engine unreachable: {exc}'})}\n\n"
        return
    except Exception as exc:
        yield f"data: {json.dumps({'error': f'δ² engine error: {exc}'})}\n\n"
        return

    reply = "".join(full_response).rstrip()
    _sessions[session_id].append({"role": "assistant", "content": reply})
    _trim(session_id)
    await _store_episode(session_id, user_msg, reply, "d2:engine")
    yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'backend': 'd2', 'cost': '$0 (research)'})}\n\n"


# ─── Heartbeat middleware ─────────────────────────────────────────
# Wraps any backend's SSE generator and injects periodic `heartbeat`
# events so the frontend terminal panel can show real-time state.
#
# Heartbeats are emitted every `interval_s` seconds while the inner
# generator is "quiet" — i.e. has not produced an event for a while.
# The first heartbeat fires immediately ("connected") so the user
# knows the stream is alive.
#
# Heartbeat shape:
#   {"heartbeat": {"stage": "...", "elapsed_ms": int, "tokens_seen": int}}
#
# stage transitions: connected → waiting → generating → complete

async def _heartbeat_wrap(
    inner: AsyncIterator[str],
    backend: str,
    model: str | None,
    interval_s: float = 1.0,
) -> AsyncIterator[str]:
    import time as _time

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()
    state = {"tokens_seen": 0, "done": False, "stage": "connected"}
    t0 = _time.time()

    async def _producer():
        try:
            async for chunk in inner:
                # Snoop the chunk to track stage transitions
                if isinstance(chunk, str) and chunk.startswith("data:"):
                    raw = chunk[5:].strip()
                    if raw:
                        try:
                            ev = json.loads(raw)
                            if ev.get("token"):
                                state["tokens_seen"] += 1
                                state["stage"] = "generating"
                            elif ev.get("done"):
                                state["stage"] = "complete"
                            elif ev.get("error"):
                                state["stage"] = "error"
                        except Exception:
                            pass
                await queue.put(chunk)
        finally:
            await queue.put(_SENTINEL)

    producer_task = asyncio.create_task(_producer())

    # Emit "connected" heartbeat immediately
    yield (
        "event: heartbeat\n"
        f"data: {json.dumps({'heartbeat': {'stage': 'connected', 'backend': backend, 'model': model, 'elapsed_ms': 0, 'tokens_seen': 0}})}\n\n"
    )

    try:
        while True:
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=interval_s)
            except asyncio.TimeoutError:
                # No event in interval_s — emit a heartbeat reflecting state
                stage = state["stage"]
                if stage == "connected":
                    stage = "waiting"
                elapsed_ms = int((_time.time() - t0) * 1000)
                yield (
                    "event: heartbeat\n"
                    f"data: {json.dumps({'heartbeat': {'stage': stage, 'backend': backend, 'model': model, 'elapsed_ms': elapsed_ms, 'tokens_seen': state['tokens_seen']}})}\n\n"
                )
                continue
            if chunk is _SENTINEL:
                break
            yield chunk
    finally:
        producer_task.cancel()
        try:
            await producer_task
        except (asyncio.CancelledError, Exception):
            pass


# ─── Endpoints ────────────────────────────────────────────────────

@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    session_id  = req.session_id or uuid.uuid4().hex
    is_new      = len(_sessions[session_id]) == 0

    _session_meta[session_id] = {
        "backend": req.backend,
        "model": req.model or (OLLAMA_DEFAULT_MODEL if req.backend == "ollama" else ""),
    }

    memories    = await _build_memory_context(session_id, req.message, is_new)
    system      = SYSTEM_BASE + _memory_block(memories) + _attachments_block(req.attached_files)

    _sessions[session_id].append({"role": "user", "content": req.message})
    messages = list(_sessions[session_id])

    chosen_model: str | None = None
    if req.backend == "claude":
        chosen_model = CLAUDE_MODEL
        gen = _stream_claude(system, messages, session_id, req.message)
    elif req.backend == "claude_cli":
        chosen_model = "claude-cli"
        gen = _stream_claude_cli(system, messages, session_id, req.message)
    elif req.backend == "together":
        chosen_model = req.model or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        gen = _stream_openai_compat(
            system, messages, chosen_model, session_id, req.message,
            base_url=(req.worker_url or TOGETHER_BASE_URL),
            api_key=TOGETHER_API_KEY,
            label="together.ai",
            backend_tag="together",
        )
    elif req.backend == "runpod":
        chosen_model = req.model or RUNPOD_DEFAULT_MODEL
        # RunPod vLLM exposes OpenAI shape at <pod-url>/v1; treat worker_url
        # as the full base (already includes /v1 if pod was set up that way).
        gen = _stream_openai_compat(
            system, messages, chosen_model, session_id, req.message,
            base_url=(req.worker_url or RUNPOD_ENDPOINT_URL),
            api_key=(RUNPOD_API_KEY or "EMPTY"),  # vLLM ignores key but header must exist
            label=f"runpod ({RUNPOD_POD_ID or 'pod'})",
            backend_tag="runpod",
        )
    elif req.backend == "d2":
        chosen_model = req.model or "d2-engine"
        gen = _stream_d2(
            system, messages, session_id, req.message,
            base_url=(req.worker_url or D2_ENDPOINT_URL),
        )
    else:
        chosen_model = req.model or OLLAMA_DEFAULT_MODEL
        gen = _stream_ollama(
            system, messages, chosen_model, session_id, req.message,
            pinned_worker_url=req.worker_url,
        )

    wrapped = _heartbeat_wrap(gen, backend=req.backend, model=chosen_model, interval_s=1.0)

    return StreamingResponse(
        wrapped,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/chat/models")
async def list_ollama_models():
    endpoint = await _find_ollama_endpoint()
    if not endpoint:
        return {"models": [], "default": OLLAMA_DEFAULT_MODEL,
                "error": "No Ollama instance reachable"}
    base_url, label, _ = endpoint
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                return {"models": [m["name"] for m in resp.json().get("models", [])],
                        "default": OLLAMA_DEFAULT_MODEL,
                        "endpoint": label}
            return {"models": [], "default": OLLAMA_DEFAULT_MODEL, "error": f"Ollama {resp.status_code} on {label}"}
    except Exception as exc:
        return {"models": [], "default": OLLAMA_DEFAULT_MODEL, "error": str(exc)}


# ─── Unified inference-resources catalog ──────────────────────────
# One endpoint, live-probed in parallel, returns every reachable
# inference target the user might want to test against. Drives the
# resource-selector dropdown in /chat.
#
# Cordoned but live? We REPORT THE TRUTH (probe result wins), and
# attach a `note` reflecting policy (e.g. office GPU isolation rule).
# That way the UI can grey out + warn without lying about reachability.

# Office GPU cordon — message attached to ollama-office regardless of
# probe result. Ollama itself may still be running on the box because
# the user starts it manually; the rule forbids unattended auto-load.
_OFFICE_CORDON_NOTE = (
    "office GPU under isolation rule (RX 6800 ROCm prone to kernel "
    "panics under VRAM pressure). Ollama may answer but loading a new "
    "model can crash the host. See README_isolation_rule_office_GPU…"
)


async def _probe_ollama(url: str, label: str, has_gpu: bool, note: str | None = None) -> dict:
    import time as _t
    rid = "ollama-" + label.split()[0].lower().replace("(", "").replace(")", "")
    out = {
        "id": rid,
        "label": f"Ollama @ {label}",
        "kind": "ollama",
        "url": url,
        "reachable": False,
        "models": [],
        "latency_ms": None,
    }
    if note:
        out["note"] = note
    if not has_gpu:
        out["label"] += " — CPU-only (slow)"
    t0 = _t.time()
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{url.rstrip('/')}/api/tags")
            out["latency_ms"] = int((_t.time() - t0) * 1000)
            if r.status_code == 200:
                out["reachable"] = True
                out["models"] = [m["name"] for m in r.json().get("models", [])]
            else:
                out["error"] = f"HTTP {r.status_code}"
    except Exception as exc:
        out["error"] = str(exc)[:120]
    return out


async def _probe_together() -> dict:
    out = {
        "id": "together-ai",
        "label": "Together.ai (hosted, pay-per-token)",
        "kind": "together",
        "url": TOGETHER_BASE_URL,
        "reachable": False,
        "models": [],
    }
    if not TOGETHER_API_KEY:
        out["note"] = "TOGETHER_AI_KEY not set in ANAMNESIS-Secrets — sign up at together.ai, generate an API key, then add as TOGETHER_AI_KEY field"
        return out
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(
                f"{TOGETHER_BASE_URL.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
            )
            if r.status_code == 200:
                out["reachable"] = True
                data = r.json()
                # Together returns either a list or {"data": [...]}.
                items = data if isinstance(data, list) else data.get("data", [])
                # Surface chat-capable models only; cap at a reasonable list.
                names: list[str] = []
                for m in items[:300]:
                    name = m.get("id") or m.get("name")
                    typ = m.get("type", "")
                    if name and (typ == "chat" or "chat" in (m.get("display_type") or "").lower() or typ == ""):
                        names.append(name)
                out["models"] = names[:80]
            else:
                out["error"] = f"HTTP {r.status_code}: {r.text[:120]}"
    except Exception as exc:
        out["error"] = str(exc)[:120]
    return out


async def _probe_runpod() -> dict:
    # Prefer the dynamically-tracked pod (started via /api/runpod/start) over
    # the static RUNPOD_ENDPOINT_URL .env var. The dynamic state lives in
    # MongoDB collection `runpod_state` (see app/routes/runpod.py).
    base = ""
    pod_label = RUNPOD_POD_ID or "no pod running"
    try:
        from database import get_db
        db = get_db()
        if db is not None:
            active = await db["runpod_state"].find_one({"_id": "active_pod"})
            if active and active.get("endpoint_url"):
                base = active["endpoint_url"].rstrip("/")
                pod_label = active.get("pod_id", pod_label)
    except Exception as exc:
        logger.debug(f"runpod_state lookup failed: {exc}")

    if not base and RUNPOD_ENDPOINT_URL:
        base = RUNPOD_ENDPOINT_URL.rstrip("/")

    out = {
        "id": "runpod-pod",
        "label": f"RunPod ({pod_label})",
        "kind": "runpod",
        "url": base or None,
        "reachable": False,
        "models": [],
    }
    if not base:
        out["note"] = (
            "no RunPod pod currently running — start one via the ▶ Spin RunPod "
            "button above (or POST /api/runpod/start)"
        )
        return out
    # vLLM exposes /models on the OpenAI base
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY or 'EMPTY'}"},
            )
            if r.status_code == 200:
                out["reachable"] = True
                data = r.json()
                items = data if isinstance(data, list) else data.get("data", [])
                out["models"] = [m.get("id") for m in items if m.get("id")] or [RUNPOD_DEFAULT_MODEL]
            else:
                out["error"] = f"HTTP {r.status_code}"
    except Exception as exc:
        out["error"] = str(exc)[:120]
    return out


async def _probe_claude_api() -> dict:
    return {
        "id": "claude-api",
        "label": "Claude API (Anthropic, billed)",
        "kind": "claude",
        "reachable": bool(ANTHROPIC_API_KEY),
        "models": [
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ] if ANTHROPIC_API_KEY else [],
        "note": None if ANTHROPIC_API_KEY else "ANTHROPIC_API_KEY not set in .env",
    }


async def _probe_claude_cli() -> dict:
    out = {
        "id": "claude-cli",
        "label": f"Claude CLI ($0 — subscription, via SSH→{CLAUDE_CLI_HOST})",
        "kind": "claude_cli",
        "reachable": False,
        "models": ["opus", "sonnet", "haiku"],
    }
    # Cheap probe: SSH → claude --version with short timeout.
    try:
        from routes.files import _ssh_client
        _u = os.environ.get("SSH_USER", os.environ.get("USER", "user"))
        client = _ssh_client(CLAUDE_CLI_HOST, _u)
        _, stdout, stderr = client.exec_command(f"{CLAUDE_CLI_PATH} --version", timeout=2)
        rc = stdout.channel.recv_exit_status()
        client.close()
        out["reachable"] = (rc == 0)
        if rc != 0:
            out["error"] = stderr.read().decode("utf-8", errors="replace")[:120]
    except Exception as exc:
        out["error"] = str(exc)[:120]
    return out


async def _list_d2_lm_experiments(base: str) -> list[str]:
    """Helper — fetch all LM experiment names from a d² engine.

    Returns list of experiment names (dedup, preserves first-seen order).
    Caller filters by `personal_` prefix to split bench vs personal track.
    """
    experiments: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            rr = await client.get(f"{base}/runs", timeout=3.0)
            if rr.status_code != 200:
                return experiments
            runs = rr.json().get("runs", [])
            seen: set[str] = set()
            for run in runs:
                req = (run.get("request") or {})
                exp = req.get("experiment")
                rid = run.get("run_id", "")
                if exp and exp not in seen and rid.startswith("lm-"):
                    experiments.append(exp)
                    seen.add(exp)
    except Exception:
        pass
    return experiments


async def _probe_d2_personal() -> dict:
    """Probe for personal-corpus δ² fine-tunes.

    Same engine as `_probe_d2`, but lists ONLY experiments whose name
    starts with `personal_` — the convention for runs trained on the
    user's Anamnesis episodes (private corpus, never published).
    Checkpoints live under `d2/d2_checkpoints_personal/` on the host
    side; the server treats them like any other experiment subdir.
    """
    base = D2_ENDPOINT_URL.rstrip("/") if D2_ENDPOINT_URL else ""
    out = {
        "id": "d2-personal",
        "label": "δ² Personal — Real-Life Corpus",
        "kind": "d2",
        "url": base or None,
        "reachable": False,
        "models": [],
        "note": "fine-tuned on your Anamnesis episodes (private — never published)",
    }
    if not base:
        out["error"] = "D2_ENDPOINT_URL not configured"
        return out
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{base}/health")
            if r.status_code != 200:
                out["error"] = f"HTTP {r.status_code}"
                return out
            out["reachable"] = True
        all_exps = await _list_d2_lm_experiments(base)
        personal = [e for e in all_exps if e.startswith("personal_")]
        out["models"] = personal or ["(no personal checkpoints yet)"]
        if not personal:
            out["note"] = (
                "fine-tuned on your Anamnesis episodes (private — never published) "
                "· no personal_* checkpoints yet — kick off Tier-1 fine-tune first"
            )
    except Exception as exc:
        out["error"] = str(exc)[:120]
    return out


async def _probe_d2() -> dict:
    base = D2_ENDPOINT_URL.rstrip("/") if D2_ENDPOINT_URL else ""
    out = {
        "id": "d2-engine",
        "label": "δ² engine (research model — server)",
        "kind": "d2",
        "url": base or None,
        "reachable": False,
        "models": [],
        "note": (
            "research / benchmark model — trained from scratch on a public "
            "dataset (WikiText-103). Used to compare δ² against Adam, EWC, "
            "GEM, SAM. Reproducible by anyone."
        ),
    }
    if not base:
        out["error"] = "D2_ENDPOINT_URL not configured"
        return out
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{base}/health")
            if r.status_code != 200:
                out["error"] = f"HTTP {r.status_code}"
                return out
            out["reachable"] = True
            health = r.json()
            out["model_loaded"] = health.get("model_loaded", False)

            # Pull the actual list of trained checkpoints (experiment names),
            # not training run IDs. Filter to BENCH-only — anything starting
            # with `personal_` belongs to the personal track and is exposed
            # via _probe_d2_personal() instead. Two-track split: see
            # README_canonical_two_tracks.md.
            all_exps = await _list_d2_lm_experiments(base)
            experiments = [e for e in all_exps if not e.startswith("personal_")]

            out["models"] = experiments or ["(no LM checkpoints yet)"]
            if not out["model_loaded"] and not experiments:
                out["note"] = (out.get("note") or "") + " · train one first via POST /api/d2/train/lm/start"
            elif not out["model_loaded"]:
                out["note"] = (out.get("note") or "") + " · checkpoint will lazy-load on first /generate"
    except Exception as exc:
        out["error"] = str(exc)[:120]
    return out


@router.get("/api/chat/inference-resources")
async def inference_resources():
    """Live-probed catalog of every backend the chat UI can target.

    Probes run in parallel via asyncio.gather, ~2-4s timeout each,
    so total wall clock is ~4s worst case. Nothing is cached — the
    user wants the freshest reachability info each time they open
    the dropdown.
    """
    ollama_probes: list = []
    for url, label, has_gpu in OLLAMA_ENDPOINTS:
        # Office gets the cordon note attached regardless of reachability.
        note = _OFFICE_CORDON_NOTE if "office" in label.lower() else None
        ollama_probes.append(_probe_ollama(url, label, has_gpu, note=note))

    results = await asyncio.gather(
        *ollama_probes,
        _probe_together(),
        _probe_runpod(),
        _probe_claude_api(),
        _probe_claude_cli(),
        # Display order: personal δ² appears ABOVE the research/bench δ² so
        # "talk to my own model" surfaces first in the dropdown.
        _probe_d2_personal(),
        _probe_d2(),
        return_exceptions=True,
    )
    resources: list[dict] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"inference-resources probe raised: {r}")
            continue
        resources.append(r)
    return {
        "resources": resources,
        "probed_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/chat/balance")
async def get_balance():
    if not ANTHROPIC_API_KEY:
        return {"available": False, "reason": "ANTHROPIC_API_KEY not configured"}
    headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        for path in ["/v1/billing/credits", "/v1/usage"]:
            try:
                resp = await client.get(f"https://api.anthropic.com{path}", headers=headers)
                if resp.status_code == 200:
                    return {"available": True, "data": resp.json(), "source": path}
            except Exception:
                continue
    return {
        "available": False,
        "reason": "Balance API not accessible — check console.anthropic.com",
        "console_url": "https://console.anthropic.com/settings/billing",
    }


@router.delete("/api/chat/session/{session_id}")
async def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    _cli_sessions.pop(session_id, None)
    return {"cleared": True, "session_id": session_id}


# ─── Persistent session CRUD ───────────────────────────────────────

@router.get("/api/chat/sessions")
async def get_sessions():
    sessions = await list_chat_sessions()
    return {"sessions": sessions}


@router.get("/api/chat/sessions/{session_id}")
async def load_session(session_id: str):
    doc = await get_chat_session(session_id)
    if not doc:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Session not found")
    # Restore into in-memory store (drop CLI cloud session — not resumable)
    _sessions[session_id] = doc.get("messages", [])
    _cli_sessions.pop(session_id, None)
    _session_meta[session_id] = {"backend": doc.get("backend", "ollama"), "model": doc.get("model", "")}
    return doc


class RenameRequest(BaseModel):
    title: str

@router.patch("/api/chat/sessions/{session_id}/title")
async def rename_session(session_id: str, body: RenameRequest):
    from database import rename_chat_session
    await rename_chat_session(session_id, body.title)
    return {"ok": True}


@router.delete("/api/chat/sessions/{session_id}/delete")
async def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    _cli_sessions.pop(session_id, None)
    _session_meta.pop(session_id, None)
    await delete_chat_session(session_id)
    return {"deleted": True}
