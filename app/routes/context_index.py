"""
Context Index — living keyword/reference index for Claude instances.

Stores a structured index of projects, infrastructure, personal context,
and technical preferences. All Claude instances read this at session start
and update it during wrap-up when new context emerges.

Stored as a single mutable document in the settings collection.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from database import get_settings_collection

logger = logging.getLogger("anamnesis.context_index")
router = APIRouter(tags=["context-index"])

DOC_ID = "context_index"


# ─── Models ──────────────────────────────────────────────────────

class IndexEntry(BaseModel):
    keyword: str
    description: str
    category: str = Field(
        ...,
        description="One of: projects, infrastructure, personal, technical",
    )
    machine: Optional[str] = None
    path: Optional[str] = None
    extra: Optional[dict] = None


class ContextIndexResponse(BaseModel):
    entries: list[IndexEntry] = []
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None


class ContextIndexUpdate(BaseModel):
    entries: list[IndexEntry] = Field(
        ..., description="Entries to add or update (matched by keyword+category)",
    )
    instance_id: str = Field(
        ..., description="Instance ID of the Claude instance making the update",
    )


class ContextIndexDelete(BaseModel):
    keywords: list[str] = Field(
        ..., description="Keywords to remove (all categories)",
    )


# ─── Endpoints ───────────────────────────────────────────────────

@router.get("/api/context-index", response_model=ContextIndexResponse)
async def get_context_index():
    """Return the full context keyword index."""
    col = get_settings_collection()
    doc = await col.find_one({"_id": DOC_ID})
    if not doc:
        return ContextIndexResponse()
    return ContextIndexResponse(
        entries=doc.get("entries", []),
        updated_at=doc.get("updated_at", ""),
        updated_by=doc.get("updated_by", ""),
    )


@router.put("/api/context-index", response_model=ContextIndexResponse)
async def update_context_index(req: ContextIndexUpdate):
    """Merge entries into the context index (upsert by keyword+category)."""
    col = get_settings_collection()
    doc = await col.find_one({"_id": DOC_ID})
    existing = doc.get("entries", []) if doc else []

    # Build lookup for existing entries
    lookup = {}
    for e in existing:
        key = (e["keyword"], e["category"])
        lookup[key] = e

    # Merge new entries
    for entry in req.entries:
        key = (entry.keyword, entry.category)
        lookup[key] = entry.model_dump()

    merged = list(lookup.values())
    now = datetime.now(timezone.utc).isoformat()

    await col.update_one(
        {"_id": DOC_ID},
        {"$set": {
            "entries": merged,
            "updated_at": now,
            "updated_by": req.instance_id,
        }},
        upsert=True,
    )

    logger.info(
        f"Context index updated by {req.instance_id}: "
        f"{len(req.entries)} entries merged, {len(merged)} total"
    )
    return ContextIndexResponse(
        entries=merged, updated_at=now, updated_by=req.instance_id,
    )


@router.delete("/api/context-index")
async def delete_context_entries(req: ContextIndexDelete):
    """Remove entries by keyword (all categories)."""
    col = get_settings_collection()
    doc = await col.find_one({"_id": DOC_ID})
    if not doc:
        return {"deleted": 0}

    existing = doc.get("entries", [])
    keywords_set = set(req.keywords)
    filtered = [e for e in existing if e["keyword"] not in keywords_set]
    removed = len(existing) - len(filtered)

    await col.update_one(
        {"_id": DOC_ID},
        {"$set": {"entries": filtered}},
    )

    return {"deleted": removed, "remaining": len(filtered)}
