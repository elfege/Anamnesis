"""Avatar Advanced settings — sampling params + persona override, persisted in MongoDB.

Replaces the localStorage approach (rejected by operator 2026-06-11 ~03:50 EDT
on canonical "data-driven apps use the database" grounds). Single document in
the existing `settings` collection, keyed by `_id = "avatar_advanced"`.

Schema (all keys nullable — null means "use server / ollama default"):
  {
    "_id": "avatar_advanced",
    "temperature": float | null,
    "top_p": float | null,
    "top_k": int | null,
    "repeat_penalty": float | null,
    "system_prompt_override": str | null,
    "updated_at": ISO datetime
  }
"""
from typing import Optional
import datetime

from database import get_settings_collection

DOC_ID = "avatar_advanced"

# Field set + permitted types. Anything outside this is silently dropped.
_FIELDS = {
    "temperature":            (int, float, type(None)),
    "top_p":                  (int, float, type(None)),
    "top_k":                  (int, type(None)),
    "repeat_penalty":         (int, float, type(None)),
    "system_prompt_override": (str, type(None)),
}


def _strip_doc(doc: Optional[dict]) -> dict:
    """Return a clean dict with all known fields (None for missing) and no _id."""
    out = {k: None for k in _FIELDS}
    if doc:
        for k in _FIELDS:
            if k in doc:
                out[k] = doc[k]
    return out


async def get_settings() -> dict:
    coll = get_settings_collection()
    doc = await coll.find_one({"_id": DOC_ID})
    return _strip_doc(doc)


async def update_settings(values: dict) -> dict:
    """Validate + upsert. Empty-string strings normalise to None so that
    'clear the field' from the frontend = 'use default'."""
    clean: dict = {}
    for k, allowed_types in _FIELDS.items():
        if k not in values:
            continue
        v = values[k]
        # Normalize: "" → None, top_k → int
        if isinstance(v, str) and v.strip() == "":
            v = None
        if k == "top_k" and isinstance(v, float):
            v = int(v)
        if not isinstance(v, allowed_types):
            continue  # silently drop bad type
        clean[k] = v
    clean["updated_at"] = datetime.datetime.utcnow().isoformat()

    coll = get_settings_collection()
    await coll.update_one(
        {"_id": DOC_ID},
        {"$set": clean},
        upsert=True,
    )
    return await get_settings()
