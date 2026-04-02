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
    OLLAMA_DEFAULT_MODEL,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_CLI_HOST,
    CLAUDE_CLI_PATH,
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


# ─── Ollama agentic streaming ──────────────────────────────────────

async def _stream_ollama(
    system: str, messages: list[dict], model: str, session_id: str, user_msg: str
) -> AsyncIterator[str]:
    url = f"{OLLAMA_URL}/api/chat"
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
                    yield f"data: {json.dumps({'error': f'Ollama {resp.status_code}'})}\n\n"
                    return
                data = resp.json()
        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Cannot connect to Ollama'})}\n\n"
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
    await _store_episode(session_id, user_msg, reply, f"ollama:{model}")
    yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'backend': 'ollama', 'cost': '$0 (local)'})}\n\n"


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
            yield f"data: {json.dumps({'error': f'CLI exit {rc}: {err}'})}\n\n"
            return

    except Exception as exc:
        yield f"data: {json.dumps({'error': f'Claude CLI error: {exc}'})}\n\n"
        return

    reply = "".join(full_response)
    _sessions[session_id].append({"role": "assistant", "content": reply})
    _trim(session_id)
    await _store_episode(session_id, user_msg, reply, "claude_cli:subscription")
    yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'backend': 'claude_cli', 'cost': '$0 (subscription)'})}\n\n"


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

    if req.backend == "claude":
        gen = _stream_claude(system, messages, session_id, req.message)
    elif req.backend == "claude_cli":
        gen = _stream_claude_cli(system, messages, session_id, req.message)
    else:
        model = req.model or OLLAMA_DEFAULT_MODEL
        gen   = _stream_ollama(system, messages, model, session_id, req.message)

    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/chat/models")
async def list_ollama_models():
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            if resp.status_code == 200:
                return {"models": [m["name"] for m in resp.json().get("models", [])],
                        "default": OLLAMA_DEFAULT_MODEL}
            return {"models": [], "default": OLLAMA_DEFAULT_MODEL, "error": f"Ollama {resp.status_code}"}
    except httpx.ConnectError:
        return {"models": [], "default": OLLAMA_DEFAULT_MODEL, "error": "Cannot connect to Ollama"}
    except Exception as exc:
        return {"models": [], "default": OLLAMA_DEFAULT_MODEL, "error": str(exc)}


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
