"""
Unified scheduler for Anamnesis background tasks (crawler + JSONL ingester).

Settings are persisted in MongoDB. Free backends auto-schedule; paid backends
require manual trigger only.

Schedule presets (in seconds):
  0       = disabled (manual only)
  1800    = every 30 minutes
  3600    = every hour
  7200    = every 2 hours
  21600   = every 6 hours
  86400   = nightly (runs once per day)
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from database import get_settings_collection

logger = logging.getLogger("anamnesis.scheduler")

# Schedule presets — label -> seconds
SCHEDULE_PRESETS = {
    "disabled": 0,
    "every_30m": 1800,
    "every_1h": 3600,
    "every_2h": 7200,
    "every_6h": 21600,
    "nightly": 86400,
}

DEFAULT_CRAWLER_SCHEDULE = "every_30m"     # crawler is lightweight, 30m is fine
DEFAULT_JSONL_SCHEDULE = "nightly"         # JSONL default: nightly at 5 AM when using free backend

# Background task handles
_crawler_schedule_task: Optional[asyncio.Task] = None
_jsonl_schedule_task: Optional[asyncio.Task] = None


async def get_schedule_settings() -> dict:
    """Load schedule settings from MongoDB."""
    defaults = {
        "crawler_schedule": DEFAULT_CRAWLER_SCHEDULE,
        "jsonl_schedule": DEFAULT_JSONL_SCHEDULE,
    }
    try:
        coll = get_settings_collection()
        doc = await coll.find_one({"_id": "schedules"})
        if doc:
            for k in defaults:
                if k in doc:
                    defaults[k] = doc[k]
    except Exception as e:
        logger.warning(f"Could not load schedule settings: {e}")
    return defaults


async def update_schedule_settings(updates: dict) -> dict:
    """Update schedule settings. Returns merged settings."""
    valid_keys = {"crawler_schedule", "jsonl_schedule"}
    filtered = {k: v for k, v in updates.items() if k in valid_keys}

    # Validate preset names
    for k, v in filtered.items():
        if v not in SCHEDULE_PRESETS:
            raise ValueError(f"Invalid schedule preset: {v}")

    if filtered:
        coll = get_settings_collection()
        await coll.update_one(
            {"_id": "schedules"},
            {"$set": filtered},
            upsert=True,
        )

    return await get_schedule_settings()


def _seconds_until_hour(target_hour: int) -> int:
    """Calculate seconds until the next occurrence of target_hour in the configured timezone."""
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(os.environ.get("SCHEDULER_TIMEZONE", "America/New_York"))
    now = datetime.now(tz=tz)
    target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return int((target - now).total_seconds())


async def _scheduled_loop(name: str, run_func, settings_key: str):
    """Generic scheduled loop. Loads interval from settings each cycle."""
    while True:
        try:
            settings = await get_schedule_settings()
            preset = settings.get(settings_key, "disabled")
            interval = SCHEDULE_PRESETS.get(preset, 0)

            if interval <= 0:
                # Disabled — check again in 60s in case settings change
                await asyncio.sleep(60)
                continue

            # For nightly preset, sleep until 5 AM instead of flat 24h
            if preset == "nightly":
                wait = _seconds_until_hour(5)
                logger.info(f"Scheduler [{name}]: nightly mode, next run in {wait // 3600}h {(wait % 3600) // 60}m")
                await asyncio.sleep(wait)
            else:
                logger.info(f"Scheduler [{name}]: waiting {interval}s ({preset})")
                await asyncio.sleep(interval)

            # Re-check settings in case they changed during sleep
            settings = await get_schedule_settings()
            preset = settings.get(settings_key, "disabled")
            if SCHEDULE_PRESETS.get(preset, 0) <= 0:
                continue

            logger.info(f"Scheduler [{name}]: starting scheduled run")
            await run_func()

        except asyncio.CancelledError:
            logger.info(f"Scheduler [{name}]: cancelled")
            break
        except Exception as e:
            logger.error(f"Scheduler [{name}]: error: {e}")
            await asyncio.sleep(30)  # back off on error


def start_jsonl_scheduler(run_func):
    """Start the JSONL ingestion scheduler background task."""
    global _jsonl_schedule_task
    if _jsonl_schedule_task and not _jsonl_schedule_task.done():
        _jsonl_schedule_task.cancel()
    _jsonl_schedule_task = asyncio.create_task(
        _scheduled_loop("jsonl", run_func, "jsonl_schedule")
    )
    logger.info("JSONL scheduler started")


def stop_jsonl_scheduler():
    """Stop the JSONL scheduler."""
    global _jsonl_schedule_task
    if _jsonl_schedule_task and not _jsonl_schedule_task.done():
        _jsonl_schedule_task.cancel()
    _jsonl_schedule_task = None
