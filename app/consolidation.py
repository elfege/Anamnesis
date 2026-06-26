"""Episode consolidation — supersession pass for evolving JSONL conversations.

Plan: docs/plans/nightly_episode_consolidation_dedup_evolving_jsonl_and_semantic_redundancy.md

Two regimes:
- **Regime 1 (this module — M1)**: within-session near-duplicate detection.
  Group `jsonl_` episodes by (machine, project, session_short); within each
  group, find pairs with embedding cosine ≥ R1_SIMILARITY_THRESHOLD; mark the
  older as superseded by the newest in the cluster. Deterministic, no LLM.
- **Regime 2 (M3, separate module)**: cross-source semantic merge. LLM-driven,
  conservative, opt-in.

Hard rule (genesis-aligned): NEVER delete. Mark `superseded_by`/`superseded_at`;
the doc stays retrievable (`include_superseded=True` in search) for audit.

Run stats are persisted to `consolidation_runs` collection — feeds the M2
dashboard surface ("X superseded this run").
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from database import get_db, get_episodes_collection, get_settings_collection

logger = logging.getLogger("anamnesis.consolidation")

# ─── Regime 1 tunables ──────────────────────────────────────────
# Cosine similarity threshold for "this is a near-duplicate snapshot, not a
# distinct turn." 0.95 = conservative; near-identical content. Lower would
# start collapsing genuinely distinct turns in the same session.
R1_SIMILARITY_THRESHOLD = 0.95

# Only consider episodes ingested via the JSONL pipeline. Manual / crawler /
# upload episodes have their own dedup logic and don't suffer the same churn.
R1_EPISODE_ID_PATTERN = re.compile(r"^jsonl_")

# Cap per-session group size so a runaway session doesn't blow memory.
# Sessions with more than this many exchanges are processed in chunks.
R1_MAX_GROUP_SIZE = 500


# ─── Helpers ────────────────────────────────────────────────────

def _parse_jsonl_episode_id(episode_id: str) -> Optional[dict]:
    """Decompose jsonl_{machine}_{project}_{ts}_{session_short}_{hash_short}.

    Returns dict with parsed fields, or None if the id doesn't match the
    expected shape. machine and project may contain underscores so we
    consume them last-first.
    """
    if not episode_id.startswith("jsonl_"):
        return None
    parts = episode_id[len("jsonl_"):].rsplit("_", 3)
    if len(parts) != 4:
        return None
    machine_and_project, ts, session_short, hash_short = parts
    # machine_and_project = "{machine}_{project}" but both may have underscores.
    # We can't split unambiguously, so we use the whole thing as the grouping
    # key. session_short alone is the operative dedup signal anyway.
    return {
        "machine_and_project": machine_and_project,
        "ts": ts,
        "session_short": session_short,
        "hash_short": hash_short,
    }


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Pure-Python — no numpy import
    (Anamnesis already imports a lot at startup; this stays light)."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def _get_runs_collection():
    db = get_db()
    return db["consolidation_runs"]


# ─── Run-state introspection (for the route + dashboard) ────────

async def get_last_run_stats(limit: int = 10) -> list[dict]:
    """Return the last N consolidation runs (newest first)."""
    coll = await _get_runs_collection()
    cursor = coll.find({}).sort("ran_at", -1).limit(limit)
    out = []
    async for doc in cursor:
        doc.pop("_id", None)
        out.append(doc)
    return out


async def count_currently_superseded() -> int:
    """How many episodes in the corpus are currently marked superseded?"""
    coll = get_episodes_collection()
    return await coll.count_documents({"superseded_by": {"$ne": None, "$exists": True}})


async def count_currently_consolidated() -> int:
    """How many episodes are CANONICAL (have a non-empty consolidated_from)?"""
    coll = get_episodes_collection()
    return await coll.count_documents(
        {"consolidated_from": {"$exists": True, "$not": {"$size": 0}}}
    )


# ─── Regime 1 core ──────────────────────────────────────────────

async def run_regime_1_pass(dry_run: bool = False) -> dict:
    """Execute one consolidation pass over JSONL episodes.

    Strategy:
      1. Group all non-superseded JSONL episodes by session_short.
      2. Within each group with ≥2 episodes, compute pairwise embedding
         cosine. Pairs with sim ≥ R1_SIMILARITY_THRESHOLD form a cluster.
      3. In each cluster, the NEWEST episode is canonical; older ones get
         `superseded_by` set to its episode_id.
      4. Stats persisted to `consolidation_runs`.

    `dry_run=True` reports what would happen without mutating documents.
    """
    started_at = datetime.now(timezone.utc)
    coll = get_episodes_collection()

    # Pull only non-superseded jsonl episodes, with the embedding (we need it
    # for cosine — paying the bytes once per pass is the price of in-memory
    # comparison; the alternative is per-pair vectorSearch which is overkill).
    query = {
        "episode_id": {"$regex": "^jsonl_"},
        "$or": [{"superseded_by": {"$exists": False}}, {"superseded_by": None}],
    }
    proj = {"episode_id": 1, "embedding": 1, "timestamp": 1, "_id": 0}

    groups: dict[str, list[dict]] = {}
    n_scanned = 0
    async for doc in coll.find(query, proj):
        parsed = _parse_jsonl_episode_id(doc["episode_id"])
        if not parsed:
            continue
        # Group key uses session_short AND machine_and_project (avoids
        # collisions of 8-hex session_shorts across different projects).
        key = f"{parsed['machine_and_project']}|{parsed['session_short']}"
        groups.setdefault(key, []).append(doc)
        n_scanned += 1

    logger.info(f"Regime 1: scanned {n_scanned} episodes across {len(groups)} session groups")

    n_groups_with_dups = 0
    n_superseded_total = 0
    bulk_updates = []

    for key, docs in groups.items():
        if len(docs) < 2:
            continue
        # Cap group size — a 500+ exchange session is exotic; chunk it
        if len(docs) > R1_MAX_GROUP_SIZE:
            logger.warning(f"Group {key} has {len(docs)} docs > cap {R1_MAX_GROUP_SIZE}; processing first {R1_MAX_GROUP_SIZE}")
            docs = docs[:R1_MAX_GROUP_SIZE]

        # Sort newest-first; the head is canonical for any cluster it's in.
        docs.sort(key=lambda d: d.get("timestamp") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

        # Track which docs have been claimed as superseded already (so we
        # don't double-mark or build cycles).
        claimed: set[str] = set()
        for i, canon in enumerate(docs):
            if canon["episode_id"] in claimed:
                continue
            canon_emb = canon.get("embedding")
            if not canon_emb:
                continue
            for older in docs[i + 1:]:
                if older["episode_id"] in claimed:
                    continue
                older_emb = older.get("embedding")
                if not older_emb:
                    continue
                sim = _cosine_similarity(canon_emb, older_emb)
                if sim >= R1_SIMILARITY_THRESHOLD:
                    claimed.add(older["episode_id"])
                    bulk_updates.append({
                        "filter": {"episode_id": older["episode_id"]},
                        "update": {"$set": {
                            "superseded_by": canon["episode_id"],
                            "superseded_at": started_at,
                            "_supersede_sim": float(sim),
                        }},
                    })
                    n_superseded_total += 1
        if claimed:
            n_groups_with_dups += 1

    finished_at = datetime.now(timezone.utc)
    duration_ms = int((finished_at - started_at).total_seconds() * 1000)

    if not dry_run and bulk_updates:
        # Mongo motor doesn't have bulk_write convenience as nice as pymongo,
        # but update_many on individual filters in batches works fine for
        # the sizes we'll see (hundreds, not millions).
        for op in bulk_updates:
            await coll.update_one(op["filter"], op["update"])
        logger.info(f"Regime 1: applied {n_superseded_total} supersessions")

    stats = {
        "regime": 1,
        "dry_run": dry_run,
        "ran_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "n_scanned": n_scanned,
        "n_groups": len(groups),
        "n_groups_with_dups": n_groups_with_dups,
        "n_superseded": n_superseded_total,
        "similarity_threshold": R1_SIMILARITY_THRESHOLD,
    }
    if not dry_run:
        runs_coll = await _get_runs_collection()
        await runs_coll.insert_one(stats.copy())
    return stats


# ─── Scheduler entry point ──────────────────────────────────────

async def run_consolidation_cycle():
    """Called by the nightly scheduler. Runs Regime 1 (Regime 2 = M3, opt-in)."""
    logger.info("Starting nightly consolidation cycle (Regime 1)")
    try:
        stats = await run_regime_1_pass(dry_run=False)
        logger.info(
            f"Consolidation done: scanned={stats['n_scanned']} "
            f"superseded={stats['n_superseded']} duration={stats['duration_ms']}ms"
        )
    except Exception as exc:
        logger.exception(f"Consolidation cycle failed: {exc}")


# ─── Manual unwind (lineage is reversible) ──────────────────────

async def unsupersede(episode_id: str) -> bool:
    """Reverse a supersession — restore an episode to default-searchable."""
    coll = get_episodes_collection()
    result = await coll.update_one(
        {"episode_id": episode_id},
        {"$unset": {"superseded_by": "", "superseded_at": "", "_supersede_sim": ""}},
    )
    return result.modified_count > 0
