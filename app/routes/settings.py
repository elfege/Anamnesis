"""
settings.py — Centralized settings API.

Endpoints (all under /api/settings):

    GET    /api/settings/categories         schema (no values)
    GET    /api/settings                    all resolved values (secrets redacted)
    GET    /api/settings/{category}         one category
    PUT    /api/settings/{category}/{key}   upsert, body {"value": ...}
    DELETE /api/settings/{category}/{key}   revert (delete Mongo doc)
    POST   /api/settings/restart            apply pending restart-required changes
    GET    /api/settings/restart/stream     SSE — restart progress

Storage: Mongo `settings` collection. Documents:
    {
      "_id": "<category>.<key>",
      "category": str, "key": str, "value": Any,
      "source": "ui" | "env-bootstrap" | "default",
      "secret": bool, "restart_required": bool,
      "updated_at": datetime, "updated_by": str,
    }

Secrets: redacted in responses unless ?reveal=true AND request comes from
loopback (127.0.0.1 / ::1). Best-effort only.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Body
from fastapi.responses import StreamingResponse

from database import get_settings_collection, get_episodes_collection
from settings_schema import SETTINGS_SCHEMA, get_key_def
from config_resolver import resolve, ENV_ALIASES

logger = logging.getLogger("anamnesis.routes.settings")

router = APIRouter(prefix="/api/settings", tags=["settings"])

REDACTED = "<redacted>"


def _is_localhost(request: Request) -> bool:
    """Best-effort loopback check (the container will see proxies as 127.0.0.1)."""
    if request.client is None:
        return False
    return request.client.host in ("127.0.0.1", "::1", "localhost")


def _redact(value: Any, secret: bool, reveal: bool) -> Any:
    if secret and not reveal:
        if value in (None, ""):
            return ""
        return REDACTED
    return value


async def _resolve_one(category: str, key: str, *, reveal: bool) -> dict:
    kd = get_key_def(category, key)
    if kd is None:
        raise HTTPException(404, f"Unknown setting {category}.{key}")
    value = resolve(category, key)
    return {
        "category": category,
        "key": key,
        "value": _redact(value, kd.get("secret", False), reveal),
        "secret": bool(kd.get("secret", False)),
        "restart_required": bool(kd.get("restart_required", False)),
        "type": kd.get("type", "string"),
        "default": _redact(kd.get("default"), kd.get("secret", False), reveal),
        "options": kd.get("options"),
        "description": kd.get("description", ""),
    }


# ─── GET /categories ─────────────────────────────────────────────

@router.get("/categories")
async def categories():
    """Return the schema (no values), for UI rendering."""
    out: dict[str, dict] = {}
    for cat_name, cat in SETTINGS_SCHEMA.items():
        out[cat_name] = {
            "description": cat.get("description", ""),
            "keys": [
                {
                    "key": kd["key"],
                    "type": kd.get("type", "string"),
                    "secret": bool(kd.get("secret", False)),
                    "restart_required": bool(kd.get("restart_required", False)),
                    "options": kd.get("options"),
                    "description": kd.get("description", ""),
                }
                for kd in cat.get("keys", [])
            ],
        }
    return {"categories": out}


# ─── GET / (all) ─────────────────────────────────────────────────

@router.get("")
@router.get("/")
async def get_all(request: Request, reveal: bool = False):
    """Return all resolved values. Secrets redacted unless reveal=true + loopback."""
    can_reveal = reveal and _is_localhost(request)
    out: dict[str, list[dict]] = {}
    for cat_name in SETTINGS_SCHEMA:
        rows = []
        for kd in SETTINGS_SCHEMA[cat_name]["keys"]:
            rows.append(await _resolve_one(cat_name, kd["key"], reveal=can_reveal))
        out[cat_name] = rows
    # Also report which keys have a Mongo override (so UI can show "modified" badges)
    overrides = await _list_overrides()
    return {"categories": out, "overrides": overrides, "revealed": can_reveal}


async def _list_overrides() -> list[str]:
    col = get_settings_collection()
    cursor = col.find({}, {"_id": 1})
    return [doc["_id"] async for doc in cursor]


# ─── GET /{category} ─────────────────────────────────────────────

@router.get("/{category}")
async def get_category(category: str, request: Request, reveal: bool = False):
    if category not in SETTINGS_SCHEMA:
        raise HTTPException(404, f"Unknown category: {category}")
    can_reveal = reveal and _is_localhost(request)
    rows = []
    for kd in SETTINGS_SCHEMA[category]["keys"]:
        rows.append(await _resolve_one(category, kd["key"], reveal=can_reveal))
    return {"category": category, "keys": rows}


# ─── PUT /{category}/{key} ───────────────────────────────────────

@router.put("/{category}/{key}")
async def set_value(category: str, key: str, body: dict = Body(...)):
    kd = get_key_def(category, key)
    if kd is None:
        raise HTTPException(404, f"Unknown setting {category}.{key}")

    if "value" not in body:
        raise HTTPException(400, "Body must include 'value'.")
    value = body["value"]

    # Type coercion (best effort) — clients may send strings for numeric fields
    t = kd.get("type", "string")
    try:
        if isinstance(value, str):
            if t == "int":
                value = int(value)
            elif t == "float":
                value = float(value)
            elif t == "bool":
                value = value.strip().lower() in ("1", "true", "yes", "on")
    except (ValueError, TypeError) as e:
        raise HTTPException(400, f"Invalid value for type {t}: {e}")

    if t == "select":
        opts = kd.get("options") or []
        if value not in opts:
            raise HTTPException(400, f"Value must be one of {opts}.")

    col = get_settings_collection()
    now = datetime.now(timezone.utc)
    await col.update_one(
        {"_id": f"{category}.{key}"},
        {"$set": {
            "category": category,
            "key": key,
            "value": value,
            "source": "ui",
            "secret": bool(kd.get("secret", False)),
            "restart_required": bool(kd.get("restart_required", False)),
            "updated_at": now,
            "updated_by": "ui-user",
        }},
        upsert=True,
    )

    return await _resolve_one(category, key, reveal=True)


# ─── DELETE /{category}/{key} ────────────────────────────────────

@router.delete("/{category}/{key}")
async def revert_value(category: str, key: str, request: Request):
    kd = get_key_def(category, key)
    if kd is None:
        raise HTTPException(404, f"Unknown setting {category}.{key}")

    col = get_settings_collection()
    result = await col.delete_one({"_id": f"{category}.{key}"})
    return {
        "deleted": result.deleted_count > 0,
        "resolved": await _resolve_one(category, key, reveal=_is_localhost(request)),
    }


# ─── POST /restart ───────────────────────────────────────────────

# Track last restart trigger so the SSE stream can detect when the app
# comes back up.
_RESTART_STATE = {
    "triggered_at": 0.0,
    "in_progress": False,
}


@router.post("/restart")
async def restart_apply(dry_run: bool = False):
    """Apply pending restart-required changes by triggering a container restart.

    Reads from existing app/routes/restart.py mechanism:
      - writes "reboot" to /dev/shm/anamnesis-restart/trigger
      - host-side watcher (anamnesis-restart-watcher.service) runs ./start.sh

    With dry_run=true, just reports what would happen and skips the trigger.
    """
    from pathlib import Path

    # Are there any pending UI-edited keys whose flag is restart_required?
    col = get_settings_collection()
    cursor = col.find({"source": "ui", "restart_required": True})
    pending = []
    async for doc in cursor:
        pending.append({"_id": doc["_id"], "key": doc.get("key"), "category": doc.get("category")})

    if dry_run:
        return {
            "restarting": False,
            "dry_run": True,
            "pending_restart_required": pending,
            "estimated_downtime_s": 10,
        }

    trigger = Path("/dev/shm/anamnesis-restart/trigger")
    if not trigger.parent.exists():
        raise HTTPException(503, (
            f"Trigger directory {trigger.parent} does not exist. "
            "Host-side watcher missing or bind mount not configured."
        ))
    try:
        trigger.write_text("reboot")
    except PermissionError as e:
        raise HTTPException(503, f"Cannot write trigger: {e}")

    _RESTART_STATE["triggered_at"] = time.time()
    _RESTART_STATE["in_progress"] = True
    logger.warning(f"Settings restart triggered. Pending keys: {len(pending)}")

    return {
        "restarting": True,
        "estimated_downtime_s": 10,
        "pending_restart_required": pending,
    }


@router.get("/restart/stream")
async def restart_stream():
    """SSE stream — yields events while a restart is in progress."""

    async def gen():
        start = time.time()
        yield "event: status\ndata: " + '{"phase": "starting"}' + "\n\n"
        # The container itself will go down, so this stream rarely completes
        # cleanly — clients should reconnect to /health afterwards.
        for i in range(30):
            await asyncio.sleep(1)
            yield ("event: tick\ndata: "
                   + '{"phase": "waiting", "elapsed_s": ' + str(int(time.time() - start)) + '}'
                   + "\n\n")
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ─── Per-source last-ingest helper (used by Sources card) ────────

@router.get("/sources/last-ingest")
async def sources_last_ingest():
    """For each SOURCE_N slot, return the timestamp of the most-recent
    ingested episode whose source_path begins with /sources/<SOURCE_N_NAME>/."""
    col = get_episodes_collection()
    out: list[dict] = []
    for i in range(6):
        name = resolve("sources", f"SOURCE_{i}_NAME") or f"source-{i}"
        prefix = f"/sources/{name}/"
        # The episodes schema currently lacks a `source_path` field; source
        # attribution lives in `instance` (set by the crawler from mount
        # name) or appears in `raw_exchange`. Try both.
        doc = await col.find_one(
            {"$or": [
                {"source_path": {"$regex": f"^{prefix}"}},
                {"instance": name},
                {"instance": f"office-{name}"},
                {"raw_exchange": {"$regex": prefix}},
            ]},
            sort=[("timestamp", -1)],
            projection={"timestamp": 1, "source_path": 1, "episode_id": 1, "instance": 1},
        )
        out.append({
            "slot": i,
            "name": name,
            "mount_prefix": prefix,
            "last_ingested_at": doc["timestamp"].isoformat() if doc and doc.get("timestamp") else None,
            "last_episode_id": doc.get("episode_id") if doc else None,
            "matched_via": ("source_path" if doc and doc.get("source_path")
                            else ("instance" if doc and doc.get("instance") else None)),
        })
    return {"sources": out}
