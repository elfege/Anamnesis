"""
JSONL Conversation Log Ingester for Anamnesis.

Reads Claude Code JSONL conversation logs from project directories,
filters for high-value exchanges (error→fix cycles, architectural decisions,
user corrections, discoveries), uses Claude API to summarize them into
episodes, and ingests into MongoDB with deduplication.

Source directories:
  - /sources/*/. claude/projects/*/  (mounted read-only in container)
  - Local machine dirs staged via sync_sources.sh

Exchange types that qualify as "high-value":
  - Error → fix cycles (tool errors followed by successful resolution)
  - User corrections ("no", "don't", "instead", "not that")
  - Architectural decisions (file creation, config changes, design discussions)
  - Discoveries (new findings about the codebase or system)
"""

import asyncio
import hashlib
import json
import logging
import multiprocessing
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from config import OLLAMA_URL, OLLAMA_ENDPOINTS, OLLAMA_DEFAULT_MODEL, ANTHROPIC_API_KEY, CLAUDE_MODEL
from database import get_episodes_collection, get_settings_collection
from embedding import get_embedding

logger = logging.getLogger("anamnesis.jsonl_ingester")

# ─── Configuration ──────────────────────────────────────────────

# Runtime config — loaded exclusively from DB. Empty until load_jsonl_source_roots().
JSONL_SOURCE_ROOTS: dict[str, str] = {}

# Min exchange length (chars) to consider for summarization
MIN_EXCHANGE_LENGTH = 200

# Max raw exchange length to send to summarization
MAX_EXCHANGE_FOR_SUMMARY = 12000

# Defaults (overridable via settings API)
_TOTAL_CORES = multiprocessing.cpu_count()
DEFAULT_SETTINGS = {
    "summarization_backend": "ollama",        # "ollama" | "claude"
    "ollama_model": OLLAMA_DEFAULT_MODEL,     # which Ollama model
    "max_exchanges_per_run": 0,               # 0 = unlimited
    "cpu_core_pct": 70,                       # percentage of logical cores
}

# Thread pool for CPU-bound work (embedding), sized to core limit
_embedding_pool: ThreadPoolExecutor | None = None


def _get_core_limit(pct: int) -> int:
    """Calculate actual core count from percentage."""
    return max(1, int(_TOTAL_CORES * pct / 100))


def _worker_init():
    """Called in each ThreadPoolExecutor worker — pin torch to 1 thread per worker."""
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _ensure_embedding_pool(pct: int) -> ThreadPoolExecutor:
    """Create or resize the embedding thread pool.

    Each worker is initialized with torch.set_num_threads(1) so that
    N workers at pct% of cores means exactly N CPU threads total — not N×(all cores).
    """
    global _embedding_pool
    cores = _get_core_limit(pct)
    if _embedding_pool is None or _embedding_pool._max_workers != cores:
        if _embedding_pool is not None:
            _embedding_pool.shutdown(wait=False)
        _embedding_pool = ThreadPoolExecutor(max_workers=cores, initializer=_worker_init)
        logger.info(f"Embedding thread pool: {cores} workers ({pct}% of {_TOTAL_CORES} cores), 1 torch thread each")

    return _embedding_pool


# ─── Persistent Settings ──────────────────────────────────────

async def get_jsonl_settings() -> dict:
    """Load JSONL ingester settings from MongoDB, with defaults."""
    try:
        coll = get_settings_collection()
        doc = await coll.find_one({"_id": "jsonl_ingester"})
        if doc:
            settings = {**DEFAULT_SETTINGS}
            for k in DEFAULT_SETTINGS:
                if k in doc:
                    settings[k] = doc[k]
            return settings
    except Exception as e:
        logger.warning(f"Could not load settings: {e}")
    return {**DEFAULT_SETTINGS}


async def update_jsonl_settings(updates: dict) -> dict:
    """Update JSONL ingester settings in MongoDB. Returns merged settings."""
    valid_keys = set(DEFAULT_SETTINGS.keys())
    filtered = {k: v for k, v in updates.items() if k in valid_keys}

    if not filtered:
        return await get_jsonl_settings()

    coll = get_settings_collection()
    await coll.update_one(
        {"_id": "jsonl_ingester"},
        {"$set": filtered},
        upsert=True,
    )
    return await get_jsonl_settings()


async def load_jsonl_source_roots():
    """Load JSONL source roots from MongoDB. Populates module-level JSONL_SOURCE_ROOTS.
    If no config exists yet, creates an empty entry — configure via dashboard UI."""
    global JSONL_SOURCE_ROOTS
    coll = get_settings_collection()
    doc = await coll.find_one({"_id": "jsonl_source_roots"})
    if doc and doc.get("roots"):
        JSONL_SOURCE_ROOTS = doc["roots"]
        logger.info(f"JSONL source roots loaded: {list(JSONL_SOURCE_ROOTS.keys())}")
    else:
        # Seed empty — user must configure via dashboard
        await coll.update_one(
            {"_id": "jsonl_source_roots"},
            {"$setOnInsert": {"roots": {}}},
            upsert=True,
        )
        JSONL_SOURCE_ROOTS = {}
        logger.warning("No JSONL source roots configured — add them via the dashboard")


async def save_jsonl_source_roots(roots: dict):
    """Persist JSONL source roots to MongoDB."""
    global JSONL_SOURCE_ROOTS
    coll = get_settings_collection()
    await coll.update_one(
        {"_id": "jsonl_source_roots"},
        {"$set": {"roots": roots}},
        upsert=True,
    )
    JSONL_SOURCE_ROOTS = roots


async def get_jsonl_source_roots_config() -> dict:
    """Return current JSONL source roots for the API."""
    return {
        "roots": JSONL_SOURCE_ROOTS,
    }


# Summarization prompt
SUMMARIZE_PROMPT = """You are an episode extractor for an AI memory system called Anamnesis.

Given a conversation exchange between a user and Claude, extract the key lesson, decision, fix, or insight as a concise episode summary.

Rules:
- Focus on WHAT was learned, decided, or fixed — not the mechanics of how Claude searched files
- Include the project context and technical specifics (file names, error messages, config values)
- Keep it under 300 words
- If the exchange is routine (just reading files, running commands with no interesting outcome), respond with exactly: SKIP
- If it contains a user correction or preference, frame it as a behavioral lesson
- If it contains an error→fix cycle, describe the root cause and solution
- If it contains an architectural decision, capture the rationale

Output format:
SUMMARY: <your summary>
TAGS: <comma-separated tags>

Or if not worth storing:
SKIP"""

# Patterns that indicate high-value exchanges
_CORRECTION_PATTERNS = re.compile(
    r"\b(no[,.]?\s+(?:don'?t|not|instead|rather|actually|wrong|stop|quit))"
    r"|(?:don'?t\s+\w+)"
    r"|(?:instead\s+(?:of|do|use))"
    r"|(?:that'?s\s+(?:not|wrong|incorrect))"
    r"|(?:fix\s+(?:this|that|the|it))"
    r"|(?:the\s+(?:problem|issue|bug)\s+(?:is|was))",
    re.IGNORECASE,
)

_ERROR_PATTERNS = re.compile(
    r"\b(error|exception|traceback|failed|failure|crash|broken|bug|issue"
    r"|cannot|can't|couldn't|unable|refused|denied|timeout|404|500|502"
    r"|ENOENT|EACCES|EPERM|segfault|OOM|killed)\b",
    re.IGNORECASE,
)

_DECISION_PATTERNS = re.compile(
    r"\b(decided|decision|approach|architecture|design|strategy|chose|picked"
    r"|trade-?off|alternative|option|migrate|refactor|restructure|overhaul"
    r"|implement|proposal|plan)\b",
    re.IGNORECASE,
)

# ─── State ──────────────────────────────────────────────────────

_ingester_state = {
    "running": False,
    "stop_requested": False,
    "last_run": None,
    "last_run_duration_seconds": 0,
    "episodes_ingested_last_run": 0,
    "total_episodes_ingested": 0,
    "files_processed": 0,
    "exchanges_evaluated": 0,
    "exchanges_skipped": 0,
    "errors": [],
    "last_interrupted": None,
}

_STATE_DOC_ID = "jsonl_ingester_state"


async def _load_persisted_state():
    """Load ingester state from MongoDB on startup. Detect interrupted runs."""
    try:
        coll = get_settings_collection()
        doc = await coll.find_one({"_id": _STATE_DOC_ID})

        if doc:
            was_running = doc.get("running", False)

            # Restore cumulative stats
            _ingester_state["total_episodes_ingested"] = doc.get("total_episodes_ingested", 0)
            _ingester_state["last_run"] = doc.get("last_run")
            _ingester_state["last_run_duration_seconds"] = doc.get("last_run_duration_seconds", 0)
            _ingester_state["episodes_ingested_last_run"] = doc.get("episodes_ingested_last_run", 0)
            _ingester_state["files_processed"] = doc.get("files_processed", 0)
            _ingester_state["exchanges_evaluated"] = doc.get("exchanges_evaluated", 0)
            _ingester_state["exchanges_skipped"] = doc.get("exchanges_skipped", 0)

            if was_running:
                interrupted_at = doc.get("last_run") or "unknown"
                logger.warning(
                    f"Previous JSONL ingestion run was interrupted (started {interrupted_at}). "
                    "Marking as not running. No data corruption — dedup covers partial runs."
                )
                _ingester_state["last_interrupted"] = interrupted_at
                # Clear the stale running flag in DB
                await coll.update_one(
                    {"_id": _STATE_DOC_ID},
                    {"$set": {"running": False, "last_interrupted": interrupted_at}},
                )

        # Seed total_episodes_ingested from actual DB count if persisted value is 0
        # (handles first startup after persistence was added, or counter drift)
        if _ingester_state["total_episodes_ingested"] == 0:
            try:
                episodes_coll = get_episodes_collection()
                actual_count = await episodes_coll.count_documents(
                    {"episode_id": {"$regex": "^jsonl_"}}
                )
                if actual_count > 0:
                    _ingester_state["total_episodes_ingested"] = actual_count
                    logger.info(
                        f"Seeded total_episodes_ingested from DB: {actual_count} JSONL episodes"
                    )
                    await _persist_state()
            except Exception as e:
                logger.warning(f"Could not seed episode count: {e}")

    except Exception as e:
        logger.warning(f"Could not load persisted ingester state: {e}")


async def _persist_state():
    """Save current ingester state to MongoDB."""
    try:
        coll = get_settings_collection()
        await coll.update_one(
            {"_id": _STATE_DOC_ID},
            {"$set": {
                "running": _ingester_state["running"],
                "last_run": _ingester_state["last_run"],
                "last_run_duration_seconds": _ingester_state["last_run_duration_seconds"],
                "episodes_ingested_last_run": _ingester_state["episodes_ingested_last_run"],
                "total_episodes_ingested": _ingester_state["total_episodes_ingested"],
                "files_processed": _ingester_state["files_processed"],
                "exchanges_evaluated": _ingester_state["exchanges_evaluated"],
                "exchanges_skipped": _ingester_state["exchanges_skipped"],
                "last_interrupted": _ingester_state.get("last_interrupted"),
            }},
            upsert=True,
        )
    except Exception as e:
        logger.warning(f"Could not persist ingester state: {e}")


async def _reconcile_orphans():
    """Find episodes missing from crawl_state and add their dedup entries.

    This handles the case where insert_one succeeded but _mark_as_ingested
    didn't (process killed between the two operations).
    """
    try:
        episodes_coll = get_episodes_collection()
        crawl_state = await _get_crawl_state_collection()

        # Find JSONL-sourced episodes
        cursor = episodes_coll.find(
            {"episode_id": {"$regex": "^jsonl_"}},
            {"episode_id": 1, "raw_exchange": 1},
        )

        reconciled = 0
        async for doc in cursor:
            episode_id = doc["episode_id"]
            raw = doc.get("raw_exchange", "")
            if not raw:
                continue

            content_hash = _content_hash(raw)
            existing = await crawl_state.find_one({"content_hash": content_hash})
            if not existing:
                await crawl_state.update_one(
                    {"content_hash": content_hash},
                    {"$set": {
                        "content_hash": content_hash,
                        "episode_id": episode_id,
                        "source_name": "jsonl_ingester",
                        "ingested_at": datetime.now(timezone.utc),
                        "reconciled": True,
                    }},
                    upsert=True,
                )
                reconciled += 1

        if reconciled:
            logger.info(f"Reconciled {reconciled} orphaned JSONL episodes in crawl_state")

    except Exception as e:
        logger.warning(f"Orphan reconciliation failed: {e}")


async def initialize_ingester():
    """Called during app startup. Loads persisted state and reconciles orphans."""
    await _load_persisted_state()
    await _reconcile_orphans()


def get_ingester_status() -> dict:
    return {**_ingester_state}


def stop_jsonl_ingestion() -> dict:
    """Request the current ingestion run to stop after the current exchange."""
    if not _ingester_state["running"]:
        return {"status": "not_running"}
    _ingester_state["stop_requested"] = True
    return {"status": "stop_requested"}


# ─── JSONL Parsing ──────────────────────────────────────────────

def _parse_jsonl_file(filepath: str) -> list[dict]:
    """Parse a JSONL file into a list of message records.

    Filters for 'user' and 'assistant' type messages only.
    Returns records sorted by timestamp.
    """
    messages = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = record.get("type")
                if msg_type not in ("user", "assistant"):
                    continue

                messages.append(record)
    except Exception as e:
        logger.error(f"Failed to read JSONL file {filepath}: {e}")
        return []

    return messages


def _extract_text_from_message(record: dict) -> str:
    """Extract readable text from a user or assistant message record."""
    message = record.get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    # Content is a list of content blocks
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type", "")

        if block_type == "text":
            text = block.get("text", "")
            # Skip system-reminder tags
            if "<system-reminder>" in text:
                text = re.sub(
                    r"<system-reminder>.*?</system-reminder>",
                    "",
                    text,
                    flags=re.DOTALL,
                ).strip()
            # Skip ide_selection tags
            if "<ide_selection>" in text:
                text = re.sub(
                    r"<ide_selection>.*?</ide_selection>",
                    "",
                    text,
                    flags=re.DOTALL,
                ).strip()
            if text:
                parts.append(text)

        elif block_type == "tool_result":
            tool_content = block.get("content", "")
            is_error = block.get("is_error", False)
            if is_error and tool_content:
                parts.append(f"[TOOL ERROR]: {tool_content[:500]}")
            elif tool_content and len(tool_content) < 500:
                parts.append(f"[tool output]: {tool_content}")

        elif block_type == "tool_use":
            tool_name = block.get("name", "unknown")
            tool_input = block.get("input", {})
            # Only include tool uses that are interesting (not routine reads)
            if tool_name in ("Edit", "Write", "NotebookEdit"):
                parts.append(f"[{tool_name}]: {json.dumps(tool_input)[:500]}")
            elif tool_name == "Bash":
                cmd = tool_input.get("command", "")
                parts.append(f"[Bash]: {cmd[:300]}")

        elif block_type == "thinking":
            # Include thinking for context but truncated
            thinking = block.get("thinking", "")
            if thinking and len(thinking) > 100:
                parts.append(f"[thinking]: {thinking[:500]}")

    return "\n".join(parts)


def _group_into_exchanges(messages: list[dict]) -> list[dict]:
    """Group sequential user→assistant message pairs into exchanges.

    An exchange = one user message + the following assistant response(s).
    Tool results that follow are included with the assistant turn.
    """
    exchanges = []
    current_exchange = None

    for record in messages:
        msg_type = record.get("type")
        role = record.get("message", {}).get("role", "")

        if msg_type == "user" and role == "user":
            # Check if this is a tool_result (continuation of assistant turn)
            content = record.get("message", {}).get("content", "")
            is_tool_result = False
            if isinstance(content, list):
                is_tool_result = any(
                    b.get("type") == "tool_result" for b in content if isinstance(b, dict)
                )

            if is_tool_result and current_exchange:
                # Tool result — append to current exchange
                text = _extract_text_from_message(record)
                if text:
                    current_exchange["parts"].append(f"[tool result]: {text[:500]}")
            else:
                # New user message — start new exchange
                if current_exchange and current_exchange["parts"]:
                    exchanges.append(current_exchange)

                text = _extract_text_from_message(record)
                current_exchange = {
                    "user_text": text,
                    "parts": [f"USER: {text}"] if text else [],
                    "timestamp": record.get("timestamp", ""),
                    "session_id": record.get("sessionId", ""),
                    "project_dir": record.get("cwd", ""),
                    "git_branch": record.get("gitBranch", ""),
                }

        elif msg_type == "assistant" and current_exchange is not None:
            text = _extract_text_from_message(record)
            if text:
                current_exchange["parts"].append(f"ASSISTANT: {text}")

    # Don't forget the last exchange
    if current_exchange and current_exchange["parts"]:
        exchanges.append(current_exchange)

    return exchanges


def _score_exchange(exchange: dict) -> float:
    """Score an exchange for value. Higher = more worth ingesting.

    Returns 0.0 for routine exchanges, >1.0 for high-value ones.
    """
    full_text = "\n".join(exchange["parts"])

    if len(full_text) < MIN_EXCHANGE_LENGTH:
        return 0.0

    score = 0.0

    # User corrections are very valuable
    if _CORRECTION_PATTERNS.search(exchange.get("user_text", "")):
        score += 3.0

    # Error patterns in the exchange
    error_matches = len(_ERROR_PATTERNS.findall(full_text))
    if error_matches >= 2:
        score += 2.0
    elif error_matches >= 1:
        score += 1.0

    # Decision/architecture patterns
    decision_matches = len(_DECISION_PATTERNS.findall(full_text))
    if decision_matches >= 2:
        score += 2.0
    elif decision_matches >= 1:
        score += 0.5

    # Tool errors followed by fixes
    if "[TOOL ERROR]" in full_text:
        score += 1.5

    # Length bonus (longer exchanges tend to have more substance)
    if len(full_text) > 2000:
        score += 0.5
    if len(full_text) > 5000:
        score += 0.5

    # Multi-turn exchanges (more back-and-forth = more interesting)
    turns = full_text.count("USER:") + full_text.count("ASSISTANT:")
    if turns >= 4:
        score += 1.0

    return score


def _content_hash(text: str) -> str:
    """SHA-256 hash for deduplication."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _extract_project_name(project_dir: str) -> str:
    """Extract project name from cwd path like /home/user/0_NVR → 0_NVR."""
    if not project_dir:
        return "unknown"
    # Get the last meaningful path component (skip generic path parts)
    parts = project_dir.rstrip("/").split("/")
    _skip = {"home", "root", "sources", ""}
    for part in reversed(parts):
        if part and part.lower() not in _skip and not part.startswith("."):
            return part
    return "unknown"


def _extract_instance_from_path(source_root: str) -> str:
    """Map source root path to instance name.

    Looks for 'server-N' patterns in the path, falling back to 'unknown'.
    """
    import re
    match = re.search(r"(server-\d+)", source_root)
    if match:
        return match.group(1)
    return "unknown"


# ─── Summarization Backends ───────────────────────────────────

def _parse_summary_response(reply: str) -> Optional[dict]:
    """Parse SUMMARY:/TAGS: format from any backend response."""
    if not reply or reply.strip().startswith("SKIP"):
        return None

    summary_match = re.search(r"SUMMARY:\s*(.+?)(?=\nTAGS:|$)", reply, re.DOTALL)
    tags_match = re.search(r"TAGS:\s*(.+)", reply)

    if not summary_match:
        return {"summary": reply[:1000], "tags": ["jsonl", "auto-extracted"]}

    summary = summary_match.group(1).strip()
    tags = []
    if tags_match:
        tags = [t.strip() for t in tags_match.group(1).split(",") if t.strip()]

    return {"summary": summary, "tags": tags}


async def _summarize_with_ollama(exchange_text: str, model: str) -> Optional[dict]:
    """Use local Ollama to summarize. Free, no API costs."""
    if len(exchange_text) > MAX_EXCHANGE_FOR_SUMMARY:
        exchange_text = exchange_text[:MAX_EXCHANGE_FOR_SUMMARY] + "\n\n[... truncated ...]"

    # Try Ollama endpoints in fallback order
    ollama_urls = [OLLAMA_URL] if OLLAMA_URL else [ep[0] for ep in OLLAMA_ENDPOINTS]
    last_err = None
    for ollama_url in ollama_urls:
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "options": {"num_thread": _get_core_limit(
                            (await get_jsonl_settings()).get("cpu_core_pct", 70)
                        )},
                        "messages": [
                            {
                                "role": "user",
                                "content": f"{SUMMARIZE_PROMPT}\n\n---\n\nEXCHANGE:\n{exchange_text}",
                            }
                        ],
                    },
                )
                response.raise_for_status()

            reply = response.json().get("message", {}).get("content", "").strip()
            return _parse_summary_response(reply)
        except httpx.ConnectError:
            last_err = f"Cannot connect to Ollama at {ollama_url}"
            logger.debug(last_err)
            continue
        except Exception as e:
            last_err = str(e)
            logger.debug(f"Ollama summarization failed on {ollama_url}: {e}")
            continue

    logger.error(f"All Ollama endpoints failed: {last_err}")
    return None


async def _summarize_with_claude(exchange_text: str) -> Optional[dict]:
    """Use Claude API for summarization. Billed per token."""
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set — cannot use Claude backend")
        return None

    if len(exchange_text) > MAX_EXCHANGE_FOR_SUMMARY:
        exchange_text = exchange_text[:MAX_EXCHANGE_FOR_SUMMARY] + "\n\n[... truncated ...]"

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": f"{SUMMARIZE_PROMPT}\n\n---\n\nEXCHANGE:\n{exchange_text}",
            }
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        reply = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                reply += block.get("text", "")

        return _parse_summary_response(reply.strip())

    except Exception as e:
        logger.error(f"Claude API summarization failed: {e}")
        return None


async def _summarize_exchange(exchange_text: str, settings: dict) -> Optional[dict]:
    """Route to the configured summarization backend."""
    backend = settings.get("summarization_backend", "ollama")
    if backend == "claude":
        return await _summarize_with_claude(exchange_text)
    else:
        model = settings.get("ollama_model", OLLAMA_DEFAULT_MODEL)
        return await _summarize_with_ollama(exchange_text, model)


# ─── Ingestion ──────────────────────────────────────────────────

async def _get_crawl_state_collection():
    """Get the crawl_state collection for dedup tracking."""
    from database import _db
    return _db["crawl_state"]


async def _is_already_ingested(content_hash: str) -> bool:
    """Check if content has already been ingested."""
    crawl_state = await _get_crawl_state_collection()
    existing = await crawl_state.find_one({"content_hash": content_hash})
    return existing is not None


async def _mark_as_ingested(content_hash: str, episode_id: str):
    """Record content hash as ingested."""
    crawl_state = await _get_crawl_state_collection()
    await crawl_state.update_one(
        {"content_hash": content_hash},
        {
            "$set": {
                "content_hash": content_hash,
                "episode_id": episode_id,
                "source_name": "jsonl_ingester",
                "ingested_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )


async def _ingest_exchange(
    exchange: dict,
    machine: str,
    project: str,
    summary_data: dict,
) -> bool:
    """Ingest a summarized exchange as an episode. Returns True if ingested."""
    raw_text = "\n".join(exchange["parts"])
    content_hash = _content_hash(raw_text)

    if await _is_already_ingested(content_hash):
        return False

    # Build episode ID
    ts = exchange.get("timestamp", "")[:10].replace("-", "")
    session_short = exchange.get("session_id", "")[:8]
    hash_short = content_hash[:8]
    episode_id = f"jsonl_{machine}_{project}_{ts}_{session_short}_{hash_short}"

    collection = get_episodes_collection()

    # Check if episode_id already exists
    existing = await collection.find_one({"episode_id": episode_id})
    if existing:
        await _mark_as_ingested(content_hash, episode_id)
        return False

    # Generate embedding in thread to avoid blocking the event loop
    summary = summary_data["summary"]
    try:
        embedding_vector = await asyncio.to_thread(get_embedding, summary)
    except Exception as e:
        logger.error(f"Embedding failed for {episode_id}: {e}")
        return False

    tags = summary_data.get("tags", []) + [
        "jsonl",
        "auto-extracted",
        f"machine:{machine}",
        f"project:{project}",
    ]

    now = datetime.now(timezone.utc)
    document = {
        "episode_id": episode_id,
        "instance": machine,
        "project": project,
        "summary": summary,
        "raw_exchange": raw_text[:50000],  # Cap raw storage at 50KB
        "tags": tags,
        "embedding": embedding_vector,
        "timestamp": now,
        "retrieval_count": 0,
        "last_retrieved": None,
    }

    await collection.insert_one(document)
    await _mark_as_ingested(content_hash, episode_id)

    logger.info(f"Ingested JSONL episode: {episode_id}")
    return True


# ─── Main Ingestion Flow ───────────────────────────────────────

async def _process_jsonl_file(filepath: str, machine: str, settings: dict) -> dict:
    """Process a single JSONL file. Returns stats dict."""
    stats = {"exchanges": 0, "ingested": 0, "skipped": 0, "errors": 0}

    messages = _parse_jsonl_file(filepath)
    if not messages:
        return stats

    exchanges = _group_into_exchanges(messages)

    for exchange in exchanges:
        if _ingester_state["stop_requested"]:
            break

        stats["exchanges"] += 1

        # Yield event loop so API stays responsive during long ingestion runs
        await asyncio.sleep(0)

        # Score the exchange
        score = _score_exchange(exchange)
        if score < 1.0:
            stats["skipped"] += 1
            continue

        # Check dedup before summarization call
        raw_text = "\n".join(exchange["parts"])
        content_hash = _content_hash(raw_text)
        if await _is_already_ingested(content_hash):
            stats["skipped"] += 1
            continue

        # Extract project name from the exchange metadata
        project = _extract_project_name(exchange.get("project_dir", ""))

        # Summarize using configured backend
        exchange_text = "\n".join(exchange["parts"])
        summary_data = await _summarize_exchange(exchange_text, settings)

        if summary_data is None:
            if score >= 3.0:
                user_text = exchange.get("user_text", "")[:500]
                summary_data = {
                    "summary": f"[Auto-extracted, no summarization] {user_text}",
                    "tags": ["unsummarized"],
                }
            else:
                stats["skipped"] += 1
                continue

        try:
            if await _ingest_exchange(exchange, machine, project, summary_data):
                stats["ingested"] += 1
            else:
                stats["skipped"] += 1
        except Exception as e:
            logger.error(f"Failed to ingest exchange from {filepath}: {e}")
            stats["errors"] += 1

    return stats


async def run_jsonl_ingestion(
    source_roots: Optional[dict] = None,
    max_exchanges: Optional[int] = None,
) -> dict:
    """Run a full JSONL ingestion cycle across all source roots.

    Settings (model, CPU cap, exchange limit) are loaded from MongoDB at start.

    Args:
        source_roots: Override default source roots (for testing)
        max_exchanges: Override max_exchanges_per_run from settings

    Returns:
        Summary stats dict
    """
    if _ingester_state["running"]:
        return {"status": "already_running"}

    _ingester_state["running"] = True
    _ingester_state["stop_requested"] = False
    _ingester_state["last_interrupted"] = None
    start_time = datetime.now(timezone.utc)
    _ingester_state["last_run"] = start_time.isoformat()
    await _persist_state()

    # Load settings from DB
    settings = await get_jsonl_settings()
    roots = source_roots or JSONL_SOURCE_ROOTS
    limit = max_exchanges or settings.get("max_exchanges_per_run", 0)
    # 0 = unlimited
    if limit <= 0:
        limit = 999999

    # Prepare thread pool with configured CPU limit
    _ensure_embedding_pool(settings.get("cpu_core_pct", 70))

    backend_label = settings.get("summarization_backend", "ollama")
    if backend_label == "ollama":
        backend_label += f":{settings.get('ollama_model', OLLAMA_DEFAULT_MODEL)}"
    logger.info(f"JSONL ingestion starting — backend: {backend_label}, limit: {limit}")

    total_stats = {
        "files_processed": 0,
        "exchanges_evaluated": 0,
        "episodes_ingested": 0,
        "exchanges_skipped": 0,
        "errors": [],
    }
    total_ingested = 0

    try:
        for machine, root_path in roots.items():
            if not os.path.isdir(root_path):
                logger.debug(f"JSONL source not found (skipping): {root_path}")
                continue

            for project_dir in sorted(Path(root_path).iterdir()):
                if not project_dir.is_dir():
                    continue

                jsonl_files = sorted(project_dir.glob("*.jsonl"))
                if not jsonl_files:
                    continue

                for jsonl_file in jsonl_files:
                    if total_ingested >= limit:
                        logger.info(f"Reached ingestion limit ({limit}), stopping")
                        break

                    if _ingester_state["stop_requested"]:
                        logger.info("Stop requested — halting ingestion after current file")
                        break

                    logger.info(f"Processing: {jsonl_file.name} ({machine})")
                    try:
                        stats = await _process_jsonl_file(
                            str(jsonl_file), machine, settings
                        )
                        total_stats["files_processed"] += 1
                        total_stats["exchanges_evaluated"] += stats["exchanges"]
                        total_stats["episodes_ingested"] += stats["ingested"]
                        total_stats["exchanges_skipped"] += stats["skipped"]
                        total_ingested += stats["ingested"]

                        if stats["errors"]:
                            total_stats["errors"].append(
                                f"{jsonl_file.name}: {stats['errors']} errors"
                            )
                    except Exception as e:
                        error_msg = f"{jsonl_file.name}: {e}"
                        logger.error(f"JSONL processing error: {error_msg}")
                        total_stats["errors"].append(error_msg)

                    # Update state live so polling can see progress
                    _ingester_state["files_processed"] = total_stats["files_processed"]
                    _ingester_state["exchanges_evaluated"] = total_stats["exchanges_evaluated"]
                    _ingester_state["episodes_ingested_last_run"] = total_stats["episodes_ingested"]
                    _ingester_state["exchanges_skipped"] = total_stats["exchanges_skipped"]
                    _ingester_state["errors"] = total_stats["errors"]
                    await _persist_state()

                if total_ingested >= limit:
                    break
            if total_ingested >= limit:
                break

    finally:
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        _ingester_state["running"] = False
        _ingester_state["last_run"] = start_time.isoformat()
        _ingester_state["last_run_duration_seconds"] = round(duration, 2)
        _ingester_state["episodes_ingested_last_run"] = total_stats["episodes_ingested"]
        _ingester_state["total_episodes_ingested"] += total_stats["episodes_ingested"]
        _ingester_state["files_processed"] = total_stats["files_processed"]
        _ingester_state["exchanges_evaluated"] = total_stats["exchanges_evaluated"]
        _ingester_state["exchanges_skipped"] = total_stats["exchanges_skipped"]
        _ingester_state["errors"] = total_stats["errors"]
        await _persist_state()

    logger.info(
        f"JSONL ingestion complete: {total_stats['episodes_ingested']} episodes from "
        f"{total_stats['files_processed']} files, "
        f"{total_stats['exchanges_evaluated']} exchanges evaluated, "
        f"{total_stats['exchanges_skipped']} skipped "
        f"({duration:.1f}s, {len(total_stats['errors'])} errors)"
    )

    total_stats["duration_seconds"] = round(duration, 2)
    total_stats["status"] = "completed"
    return total_stats
