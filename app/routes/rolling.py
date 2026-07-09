"""Routes for the rolling per-chat episode + Anamnesis-enforced ceiling.

Per MSG-526 + MSG-527 spec (2026-07-09). PM.5 canonical pending.

  POST /api/episodes/rolling/upsert  {handle, session_id, delta, machine?, tags?}
  GET  /api/episodes/rolling         — list rolling episodes (metadata + preview)
  GET  /api/episodes/rolling/{episode_id} — full one
"""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse

from database import get_episodes_collection
from rolling import (
    ROLLING_COMPACT_TRIGGER_CHARS,
    ROLLING_TOKEN_CEILING,
    list_rolling_episodes,
    upsert_rolling_episode,
)

router = APIRouter(prefix="/api/episodes/rolling", tags=["rolling"])


@router.post("/upsert")
async def rolling_upsert(body: dict, response: Response):
    """Create-or-append the rolling episode for (machine, handle, session_id).

    Body:
      {
        "handle":      str  (required) — agent identity, e.g. "dellserver-anamnesis:d2"
        "session_id":  str  (required) — Claude Code session UUID (first 8 chars used)
        "delta":       str  (required) — new turn content to append
        "machine":     str  (optional) — defaults to the anamnesis-app hostname
        "tags":        list (optional) — extra tags to union onto the episode
      }

    Response headers:
      X-Anamnesis-Compacted-Pending: true  — set when the write crossed the
      compaction trigger; the async LLM compaction has been scheduled and
      will inline-rewrite the summary. Subsequent GETs may show the compacted
      form. Writers don't need to wait.

    Response body:
      {
        "episode_id":            str,
        "created":               bool,     # true iff this was a fresh episode
        "total_chars":           int,      # new summary length in chars
        "estimated_tokens":      int,      # char/4 approximation
        "compaction_triggered":  bool,     # same as the header
        "ceiling_chars":         int,      # for the writer to know the budget
        "ceiling_tokens":        int,
      }
    """
    handle = (body or {}).get("handle", "").strip()
    session_id = (body or {}).get("session_id", "").strip()
    delta = (body or {}).get("delta", "")
    machine = (body or {}).get("machine")
    tags = (body or {}).get("tags") or None

    if not handle:
        raise HTTPException(status_code=400, detail="handle is required")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not delta or not isinstance(delta, str):
        raise HTTPException(status_code=400, detail="delta must be a non-empty string")

    result = await upsert_rolling_episode(
        handle=handle,
        session_id=session_id,
        delta=delta,
        machine=machine,
        tags=tags,
    )
    if result["compaction_triggered"]:
        response.headers["X-Anamnesis-Compacted-Pending"] = "true"
    return {
        **result,
        "ceiling_chars": ROLLING_COMPACT_TRIGGER_CHARS,
        "ceiling_tokens": ROLLING_TOKEN_CEILING,
    }


@router.get("/all")
async def rolling_list(limit: int = 50):
    """List rolling episodes newest-updated first. Metadata + preview only.

    Under /all (not just the prefix) to avoid collision with the /{episode_id}
    single-episode route below — FastAPI's routing is greedy on path params.
    """
    return {
        "ceiling_chars": ROLLING_COMPACT_TRIGGER_CHARS,
        "ceiling_tokens": ROLLING_TOKEN_CEILING,
        "episodes": await list_rolling_episodes(limit=limit),
    }


@router.get("/one/{episode_id:path}")
async def rolling_get(episode_id: str):
    """Full rolling episode by id."""
    coll = get_episodes_collection()
    doc = await coll.find_one({"episode_id": episode_id, "rolling": True}, {"embedding": 0})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Rolling episode '{episode_id}' not found")
    doc["_id"] = str(doc["_id"])
    return doc
