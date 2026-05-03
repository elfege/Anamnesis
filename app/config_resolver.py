"""
config_resolver.py — Three-layer settings resolution.

Priority (highest wins):

    1. Mongo `settings` collection  (UI-edited, source: "ui")
    2. Process environment           (.env / docker-compose)
    3. Schema default                (app/settings_schema.py)

Modules that need a value the user might want to UI-edit should call
`resolve_setting(category, key)` instead of reading os.environ directly.

Usage:

    from config_resolver import resolve_setting
    backend = resolve_setting("ingestion", "INGEST_BACKEND")  # -> "ollama"

Notes:
- This module does NOT cache. Reads are O(1) from env, O(1-Mongo-roundtrip)
  from DB, so callers may want to cache themselves for hot paths. The
  settings UI invalidates by writing to Mongo.
- For Mongo lookups we use a SYNCHRONOUS pymongo handle to avoid forcing
  callers into async. The Motor handle from database.py is async; we
  obtain a delegated sync collection via `.delegate` (same pattern used
  by ensure_vector_index()).
- If MongoDB is not yet connected (early startup), Mongo lookup is skipped
  and we fall straight through to env → default.
"""

import logging
import os
from typing import Any, Optional

from settings_schema import SETTINGS_SCHEMA, get_key_def

logger = logging.getLogger("anamnesis.config_resolver")


def _mongo_settings_sync():
    """Return a sync pymongo collection handle, or None if not connected."""
    try:
        from database import get_settings_collection
        col = get_settings_collection()
        # `col` is a Motor AsyncIOMotorCollection. .delegate gives the
        # underlying pymongo.Collection (sync).
        return col.delegate
    except Exception:
        return None


def _coerce(value: Any, key_def: dict) -> Any:
    """Coerce an env-string into the declared type."""
    if value is None:
        return None
    t = key_def.get("type", "string")
    if not isinstance(value, str):
        return value
    try:
        if t == "int":
            return int(value)
        if t == "float":
            return float(value)
        if t == "bool":
            return value.strip().lower() in ("1", "true", "yes", "on")
    except (ValueError, TypeError):
        return value
    return value


def resolve_setting(
    category: str,
    key: str,
    *,
    env_alias: Optional[str] = None,
) -> Any:
    """Resolve one setting value via the three-layer priority chain.

    Args:
        category: declared category in SETTINGS_SCHEMA
        key: declared key within that category
        env_alias: if the env var name differs from `key`, pass it here.
                   (e.g. schema key 'EMBED_MODEL' but env var 'EMBEDDING_MODEL')
    """
    key_def = get_key_def(category, key)
    if key_def is None:
        # Unknown key — treat as plain env lookup with no default.
        return os.environ.get(env_alias or key)

    # 1. Mongo override
    sync_col = _mongo_settings_sync()
    if sync_col is not None:
        try:
            doc = sync_col.find_one({"_id": f"{category}.{key}"})
            if doc and "value" in doc:
                return doc["value"]
        except Exception as e:
            logger.debug(f"Mongo settings lookup failed for {category}.{key}: {e}")

    # 2. Environment
    env_var = env_alias or key
    raw = os.environ.get(env_var)
    if raw is not None and raw != "":
        return _coerce(raw, key_def)

    # 3. Default from schema
    return key_def.get("default")


def resolve_category(category: str) -> dict[str, Any]:
    """Resolve every key in a category into a {key: value} dict."""
    cat = SETTINGS_SCHEMA.get(category)
    if not cat:
        return {}
    return {kd["key"]: resolve_setting(category, kd["key"]) for kd in cat["keys"]}


# Map of (category, key) → env-var alias when the env name differs from
# the schema key. Keep this small — it's the bridge from old env names
# to the canonical schema.
ENV_ALIASES: dict[tuple[str, str], str] = {
    ("ingestion", "EMBED_MODEL"): "EMBEDDING_MODEL",
}


def resolve(category: str, key: str) -> Any:
    """Convenience wrapper that automatically applies ENV_ALIASES."""
    alias = ENV_ALIASES.get((category, key))
    return resolve_setting(category, key, env_alias=alias)


async def seed_from_env() -> int:
    """Bootstrap Mongo `settings` from env on first startup.

    For every key in the schema, if Mongo doesn't already have a doc AND
    the env variable is set to a non-empty value, insert a doc with
    source='env-bootstrap'. Returns count of seeded keys.

    Idempotent: existing Mongo docs are never overwritten here. To replace
    them, call PUT /api/settings/{category}/{key}.
    """
    from datetime import datetime, timezone
    from database import get_settings_collection

    col = get_settings_collection()
    seeded = 0
    now = datetime.now(timezone.utc)

    for cat_name, cat in SETTINGS_SCHEMA.items():
        for kd in cat["keys"]:
            key = kd["key"]
            doc_id = f"{cat_name}.{key}"

            existing = await col.find_one({"_id": doc_id})
            if existing is not None:
                continue

            env_alias = ENV_ALIASES.get((cat_name, key), key)
            raw = os.environ.get(env_alias)
            if raw is None or raw == "":
                continue

            value = _coerce(raw, kd)
            await col.insert_one({
                "_id": doc_id,
                "category": cat_name,
                "key": key,
                "value": value,
                "source": "env-bootstrap",
                "secret": bool(kd.get("secret", False)),
                "restart_required": bool(kd.get("restart_required", False)),
                "updated_at": now,
                "updated_by": "bootstrap",
            })
            seeded += 1

    if seeded:
        logger.info(f"Seeded {seeded} settings from environment.")
    return seeded
