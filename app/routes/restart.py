"""
restart.py — Container-side restart trigger + persisted config.

WHAT THIS DOES, FOR THE DUMMIES:
=================================

The Anamnesis app sometimes needs a full restart — for things that can't
be hot-reloaded:
   - new env vars added
   - new secrets that weren't loaded at boot
   - δ² base-model swap (model loaded at container startup)
   - authorized_machine_id change

But the container can't restart its own stack (it would kill itself
mid-restart). And we don't want to give the container access to the
Docker socket (security risk: container that can talk to docker can
escape).

THE SOLUTION (NVR pattern, well-tested):

   container writes "reboot" to /dev/shm/anamnesis-restart/trigger
              ↓
   host-side watcher (anamnesis-restart-watcher.sh) sees the write
              ↓
   watcher runs ./start.sh on the host
              ↓
   docker compose restart picks up the new config

The trigger file lives on /dev/shm (tmpfs, RAM-backed) and is
bind-mounted into the container. That's the entire trust surface
between container and host. No SSH, no sudo, no Docker socket.


CONFIG PERSISTENCE:

For changes that need to survive the restart, the app stores them in
MongoDB (collection: `anamnesis_config`) BEFORE writing the trigger.
After restart, start.sh reads from that collection and applies the
config to the new container's environment.

USAGE FROM THE UI:

    1. User changes a setting in the UI
    2. Frontend calls POST /api/anamnesis/config-and-restart with the
       new config dict
    3. This route persists to MongoDB
    4. This route writes "reboot" to the trigger
    5. Watcher restarts the container ~5 seconds later
    6. New container starts up with the new config


REQUIREMENTS:

The /dev/shm/anamnesis-restart bind mount must be added to docker-compose.yml:

    services:
      anamnesis-app:
        volumes:
          - /dev/shm/anamnesis-restart:/dev/shm/anamnesis-restart

And the host-side watcher must be installed and running:

    sudo cp deployment/anamnesis-restart-watcher.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now anamnesis-restart-watcher
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database import get_db

logger = logging.getLogger("anamnesis.routes.restart")

router = APIRouter(prefix="/api/anamnesis", tags=["anamnesis-config"])


# ============================================================================
# Constants
# ============================================================================

# Trigger file path (must match the watcher's TRIGGER_FILE)
TRIGGER_FILE = Path("/dev/shm/anamnesis-restart/trigger")

# Reboot magic string the watcher recognizes
REBOOT_MAGIC = "reboot"


# ============================================================================
# Pydantic schemas
# ============================================================================

class ConfigPayload(BaseModel):
    """A bag of config keys the UI wants to persist + apply on restart."""
    config: dict
    restart: bool = True  # if False, just persist without triggering restart


class RestartResponse(BaseModel):
    triggered: bool
    persisted_config: dict | None = None
    message: str


# ============================================================================
# MongoDB collection
# ============================================================================

def _config_collection():
    """The anamnesis_config collection — one doc per config key (upserted)."""
    db = get_db()
    if db is None:
        raise RuntimeError("MongoDB not connected")
    return db["anamnesis_config"]


# ============================================================================
# Routes
# ============================================================================

@router.post("/config-and-restart", response_model=RestartResponse)
async def config_and_restart(payload: ConfigPayload):
    """
    Persist a config dict to MongoDB and (optionally) trigger a host restart.

    The config is upserted into anamnesis_config. start.sh reads from this
    collection at boot and exports each key as an env var to the new
    container.

    If restart=true (default), writes "reboot" to the trigger file. The
    host-side watcher picks this up within ~2-5 seconds and runs ./start.sh.

    Returns immediately — the actual restart happens out-of-band. The
    caller will get a connection drop ~5-10 seconds later when the
    container goes down for the restart.
    """
    col = _config_collection()
    now = datetime.now(timezone.utc)

    # ── Persist each config key as its own document ──────────────────────
    # Why one doc per key? So we can update them independently and so a
    # malformed payload only invalidates one key, not the whole config.
    persisted = {}
    for key, value in payload.config.items():
        await col.update_one(
            {"_id": key},
            {"$set": {"value": value, "updated_at": now}},
            upsert=True,
        )
        persisted[key] = value
    logger.info(f"Persisted config keys: {list(persisted.keys())}")

    # ── If not restarting, we're done ────────────────────────────────────
    if not payload.restart:
        return RestartResponse(
            triggered=False,
            persisted_config=persisted,
            message="Config persisted. No restart triggered.",
        )

    # ── Write the reboot trigger ─────────────────────────────────────────
    if not TRIGGER_FILE.parent.exists():
        # tmpfs not mounted — watcher isn't installed or compose isn't updated
        raise HTTPException(
            status_code=503,
            detail=(
                f"Trigger directory {TRIGGER_FILE.parent} does not exist. "
                "The host-side watcher may not be installed, or the "
                "/dev/shm/anamnesis-restart bind mount is missing in "
                "docker-compose.yml. See app/routes/restart.py for setup."
            ),
        )

    try:
        TRIGGER_FILE.write_text(REBOOT_MAGIC)
    except PermissionError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Cannot write to {TRIGGER_FILE}. The container may not have "
                "write access to the bind-mounted tmpfs. Check that "
                "/dev/shm/anamnesis-restart on the host has chmod 777."
            ),
        )

    logger.warning("RESTART TRIGGERED — container will go down within ~5 seconds")
    return RestartResponse(
        triggered=True,
        persisted_config=persisted,
        message="Restart triggered. Container will restart within ~5 seconds.",
    )


@router.get("/config")
async def get_config():
    """Return the current persisted config from MongoDB."""
    col = _config_collection()
    cursor = col.find({})
    out = {}
    async for doc in cursor:
        out[doc["_id"]] = {
            "value": doc.get("value"),
            "updated_at": doc.get("updated_at"),
        }
    return {"config": out}


@router.delete("/config/{key}")
async def delete_config_key(key: str):
    """Remove one persisted config key."""
    col = _config_collection()
    result = await col.delete_one({"_id": key})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Config key not found: {key}")
    return {"ok": True, "key": key}


@router.get("/restart/status")
async def restart_status():
    """
    Report whether the restart watcher appears to be reachable.

    Cheap health check: tries to read the trigger file. If it exists and
    is readable, the bind mount is in place and the watcher is presumably
    listening. We can't directly query the watcher process from inside
    the container (no shared PID namespace, no Docker socket).
    """
    if not TRIGGER_FILE.exists():
        return {
            "available": False,
            "reason": (
                f"Trigger file {TRIGGER_FILE} does not exist. "
                "Bind mount missing or watcher hasn't initialized."
            ),
        }
    try:
        content = TRIGGER_FILE.read_text()
    except Exception as e:
        return {"available": False, "reason": f"Cannot read trigger: {e}"}
    return {
        "available": True,
        "trigger_file": str(TRIGGER_FILE),
        "current_content": content[:200],  # truncate for safety
    }
