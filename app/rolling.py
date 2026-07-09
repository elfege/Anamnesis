"""Rolling per-chat episode + Anamnesis-enforced bloat ceiling.

Per intercom V2 MSG-526 (2026-07-09, operator directive relayed by
dellserver-smart-home): every chat/agent session maintains ONE rolling
Anamnesis episode that captures the entire chat's memory across compactions.
Anamnesis owns the max-tokens ceiling + auto-compaction so writing agents
don't have to manage it.

Contract (my proposal in MSG-527, awaiting canonical PM.5):
- ID: ep_rolling_<machine>_<handle>_<session_short>
  session_short = first 8 chars of Claude Code session UUID (matches
  app/jsonl_ingester.py:773 convention).
- Ceiling: ROLLING_TOKEN_CEILING (32K tokens ~= 128K chars).
- Trigger point: 100K chars (safety margin below ceiling).
- Compaction: INLINE rewrite (not split-to-continuation) — preserves the
  one-episode-per-chat invariant. LLM summarizes oldest ROLLING_COMPACT_TAIL_PCT
  of content into a "prior arc" block, keeps the recent tail verbatim.
- superseded_by / consolidated_from stay reserved for CROSS-episode merges
  from consolidation.py / consolidation_r2.py — NOT used for rolling compaction.

Token estimation: char/4 approximation. Cheap, portable, no new deps. Ok
because the ceiling is a soft limit — 10-20% variance vs true tiktoken counts
is acceptable at 32K.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from database import get_episodes_collection
from embedding import get_embedding

logger = logging.getLogger("anamnesis.rolling")

# ─── Tunables ───────────────────────────────────────────────────
# Char-based approximation of token counts. Rule-of-thumb: 1 token ≈ 4 chars
# for English text. For the 32K-token ceiling, that's 128K chars.
ROLLING_TOKEN_CEILING = 32_000
ROLLING_CHAR_CEILING = 128_000
# Trigger compaction below the ceiling so we don't overshoot on the write
# that gets us there.
ROLLING_COMPACT_TRIGGER_CHARS = 100_000
# Percentage of the summary (measured from the head — oldest content) that
# gets sent through the LLM for compaction into a "prior arc" block.
ROLLING_COMPACT_HEAD_PCT = 0.60
# Target length of the compacted "prior arc" block as a fraction of the
# original head-slice length. 0.10 = 10% (10:1 compression).
ROLLING_COMPACT_TARGET_RATIO = 0.10

# Standardized markers for the compaction boundary. The compactor puts the
# summarized head above the marker, the verbatim tail below.
ROLLING_COMPACT_MARKER = "\n\n=== end of compacted prior arc — recent verbatim below ===\n\n"


def estimate_tokens(text: str) -> int:
    """Char/4 approximation. Fast, portable, no external deps."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def derive_rolling_id(machine: str, handle: str, session_id: str) -> str:
    """ID contract from MSG-527 spec.

    session_short = first 8 chars of the session_id (typically a UUID).
    Falls back gracefully on short/empty inputs.
    """
    m = (machine or "unknown").strip().lower()
    h = (handle or "unknown").strip()
    s = (session_id or "").strip()
    session_short = s[:8] if len(s) >= 8 else (s or "nosession")
    return f"ep_rolling_{m}_{h}_{session_short}"


async def _llm_compact_head(head_text: str, target_chars: int) -> Optional[str]:
    """Ask Ollama to compress head_text down to ~target_chars.

    Reuses the existing Ollama fallback chain via httpx directly rather than
    importing jsonl_ingester._summarize_with_ollama (that helper has a fixed
    exchange-summary schema; we want prose here).
    """
    import httpx
    from config import OLLAMA_DEFAULT_MODEL, OLLAMA_ENDPOINTS

    prompt = (
        f"Compact the following conversation excerpt into a dense prose 'prior arc' "
        f"summary of about {target_chars} characters. Preserve concrete details "
        f"(names, file paths, decisions, errors, version numbers) verbatim. "
        f"Use neutral third-person voice. No bullet points. No preamble.\n\n"
        f"EXCERPT:\n{head_text}"
    )
    body = {
        "model": OLLAMA_DEFAULT_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }
    for url, _ in OLLAMA_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=180.0) as c:
                r = await c.post(f"{url}/api/chat", json=body)
                r.raise_for_status()
            content = r.json().get("message", {}).get("content", "").strip()
            if content:
                return content
        except Exception as exc:
            logger.debug(f"Rolling compaction failed on {url}: {exc}")
            continue
    return None


async def _compact_rolling(episode_id: str) -> Optional[dict]:
    """Compact one over-ceiling rolling episode. Called async (fire-and-forget)
    by upsert_rolling_episode. Idempotent — if the episode is under the
    trigger by the time we run, no-op.
    """
    coll = get_episodes_collection()
    doc = await coll.find_one({"episode_id": episode_id})
    if not doc:
        logger.warning(f"Rolling compaction target vanished: {episode_id}")
        return None

    summary = doc.get("summary", "")
    if len(summary) < ROLLING_COMPACT_TRIGGER_CHARS:
        logger.debug(f"Rolling {episode_id} under trigger by compact time — skip")
        return None

    # Determine split point. If the marker is already present, we compact
    # only the pre-marker section (leaving prior compacted arcs stacked).
    if ROLLING_COMPACT_MARKER in summary:
        # Advance the head window to include everything up to the LAST marker
        # so we keep the recent-verbatim tail intact.
        _, tail = summary.rsplit(ROLLING_COMPACT_MARKER, 1)
        head_full = summary[: len(summary) - len(tail) - len(ROLLING_COMPACT_MARKER)]
    else:
        head_full = summary
        tail = ""

    # Only compact ROLLING_COMPACT_HEAD_PCT of the head; keep the newer part
    # (the transition zone between head and tail) verbatim in the tail.
    split = int(len(head_full) * ROLLING_COMPACT_HEAD_PCT)
    head_slice = head_full[:split]
    transition = head_full[split:]
    target_chars = max(1000, int(len(head_slice) * ROLLING_COMPACT_TARGET_RATIO))

    logger.info(f"Rolling compaction: {episode_id} head_slice={len(head_slice)} → ~{target_chars}")
    compacted = await _llm_compact_head(head_slice, target_chars)
    if not compacted:
        logger.error(f"Rolling compaction LLM failed for {episode_id}")
        return None

    new_summary = compacted + ROLLING_COMPACT_MARKER + transition + tail
    new_embedding = await asyncio.to_thread(get_embedding, new_summary)
    await coll.update_one(
        {"episode_id": episode_id},
        {"$set": {
            "summary": new_summary,
            "embedding": new_embedding,
            "last_compacted_at": datetime.now(timezone.utc),
            "_rolling_compact_ratio": round(len(compacted) / max(1, len(head_slice)), 3),
        }},
    )
    logger.info(
        f"Rolling compaction done: {episode_id} "
        f"{len(head_slice)} → {len(compacted)} "
        f"(new total {len(new_summary)} chars)"
    )
    return {
        "episode_id": episode_id,
        "compacted_chars": len(compacted),
        "head_slice_chars": len(head_slice),
        "new_total_chars": len(new_summary),
    }


async def upsert_rolling_episode(
    handle: str,
    session_id: str,
    delta: str,
    machine: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """Create-or-append the rolling episode for (machine, handle, session_id).

    Returns dict:
      {
        "episode_id": str,
        "created": bool,
        "total_chars": int,
        "estimated_tokens": int,
        "compaction_triggered": bool,   # True → auto-compaction fired async
      }
    """
    if not handle or not delta:
        raise ValueError("handle and delta are required")

    # Machine name resolution: prefer the caller-supplied value, then the
    # HOST_MACHINE env var (docker-compose.yml sets this to "dellserver" for
    # anamnesis-app), then socket.gethostname() as last resort — which
    # inside the container is the docker id, not useful for humans, so
    # callers really should pass machine= explicitly.
    if not machine:
        import os, socket
        machine = os.environ.get("HOST_MACHINE") or socket.gethostname()

    episode_id = derive_rolling_id(machine, handle, session_id)
    coll = get_episodes_collection()
    now = datetime.now(timezone.utc)

    existing = await coll.find_one({"episode_id": episode_id})
    if existing:
        # Append delta to summary. Preserve the marker structure — new content
        # ALWAYS goes at the end (verbatim tail).
        existing_summary = existing.get("summary", "")
        new_summary = existing_summary.rstrip() + "\n\n" + delta.strip()
        created = False
    else:
        new_summary = delta.strip()
        created = True

    # Re-embed on every append (small cost; keeps search current with the
    # rolling head).
    new_embedding = await asyncio.to_thread(get_embedding, new_summary)

    base_tags = list(set((tags or []) + ["rolling", f"machine:{machine}", f"handle:{handle}"]))

    if created:
        doc = {
            "episode_id": episode_id,
            "instance": handle,
            "project": "rolling",
            "summary": new_summary,
            "raw_exchange": None,
            "tags": base_tags,
            "embedding": new_embedding,
            "timestamp": now,
            "retrieval_count": 0,
            "last_retrieved": None,
            "rolling": True,
            "rolling_machine": machine,
            "rolling_handle": handle,
            "rolling_session_id": session_id,
            "rolling_created_at": now,
            "rolling_updated_at": now,
        }
        await coll.insert_one(doc)
    else:
        # Merge tags (keep any manually-added ones from prior calls).
        merged_tags = list(set(existing.get("tags", []) + base_tags))
        await coll.update_one(
            {"episode_id": episode_id},
            {"$set": {
                "summary": new_summary,
                "embedding": new_embedding,
                "tags": merged_tags,
                "rolling_updated_at": now,
            }},
        )

    # Trigger compaction if the summary crossed the trigger threshold. Fire
    # and forget — the writer doesn't wait for the LLM.
    compaction_triggered = False
    if len(new_summary) >= ROLLING_COMPACT_TRIGGER_CHARS:
        asyncio.create_task(_compact_rolling(episode_id))
        compaction_triggered = True

    return {
        "episode_id": episode_id,
        "created": created,
        "total_chars": len(new_summary),
        "estimated_tokens": estimate_tokens(new_summary),
        "compaction_triggered": compaction_triggered,
    }


async def list_rolling_episodes(limit: int = 50) -> list[dict]:
    """List all rolling episodes, newest updated first. Handy for the dashboard."""
    coll = get_episodes_collection()
    cursor = coll.find(
        {"rolling": True},
        {"embedding": 0, "raw_exchange": 0},
    ).sort("rolling_updated_at", -1).limit(limit)
    out = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["total_chars"] = len(doc.get("summary", ""))
        doc["estimated_tokens"] = estimate_tokens(doc.get("summary", ""))
        # Trim summary for list display — full text via GET one
        s = doc.get("summary", "")
        doc["summary_preview"] = (s[:400] + "…") if len(s) > 400 else s
        doc.pop("summary", None)
        out.append(doc)
    return out
