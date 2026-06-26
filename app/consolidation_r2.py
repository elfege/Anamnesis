"""Regime 2 — cross-source semantic merge (LLM-driven, opt-in, reversible).

Plan: docs/plans/nightly_episode_consolidation_dedup_evolving_jsonl_and_semantic_redundancy.md §2

Operates ACROSS sources (different machines / projects) — distinct from
Regime 1 which only collapses near-dup snapshots within a single session.

Strategy:
  1. Sample N candidate episodes (default: a project filter to keep cost bounded).
  2. For each candidate, $vectorSearch its K nearest neighbors with cosine
     ≥ R2_SIMILARITY_THRESHOLD AND from a DIFFERENT (machine, project) than
     the seed. Build an adjacency graph.
  3. Find connected components → clusters.
  4. For each cluster of size 2..MAX_CLUSTER_SIZE:
       - LLM-synthesize a single canonical summary
       - Insert a NEW canonical episode with consolidated_from=[ids]
       - Mark cluster members superseded_by=<new canonical id>
  5. All operations reversible via /api/consolidation/unsupersede/{id}
     and by deleting the canonical episode if needed.

OPT-IN ONLY — not wired to any scheduler. Manual trigger via
POST /api/consolidation/run_regime_2.

Defaults are conservative:
  - cosine threshold 0.92 (lower than R1's 0.95 because cross-source has more
    lexical variation but same semantic content)
  - cluster size cap 5 to keep LLM input bounded
  - sample size cap 500 per run to keep cost predictable
  - LLM backend defaults to Ollama (free); operator can override to Claude
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_ENDPOINTS,
)
from database import get_db, get_episodes_collection, VECTOR_INDEX_NAME
from embedding import get_embedding

logger = logging.getLogger("anamnesis.consolidation_r2")

R2_SIMILARITY_THRESHOLD = 0.92
R2_K_NEIGHBORS = 10
R2_MAX_CLUSTER_SIZE = 5
R2_DEFAULT_SAMPLE_SIZE = 500

_MERGE_PROMPT = """You are consolidating near-duplicate episodic memory entries
that describe the same lesson, decision, or pattern but were captured in
different sessions. Synthesize a single canonical summary that:

- Captures the union of useful information from all entries.
- Preserves any concrete file paths, command names, error messages, version
  numbers, or other specific details verbatim across entries.
- Notes when sources disagree (don't silently pick a side).
- Uses a neutral third-person voice.
- Is 2-6 sentences long.

Return STRICT JSON: {"summary": "...", "tags": ["tag1", "tag2", ...]}

Tags should be the UNION of all source episodes' meaningful tags (drop
generic ones like 'jsonl', 'auto-extracted'). Add 'consolidated' to the
tag list.
"""


async def _llm_merge_ollama(entries: list[dict]) -> Optional[dict]:
    """Synthesize a canonical episode via Ollama. Returns {summary, tags}."""
    payload = "\n\n---\n\n".join(
        f"ENTRY {i+1} (id={e['episode_id']}, instance={e['instance']}, project={e['project']}):\n{e['summary']}"
        for i, e in enumerate(entries)
    )
    body = {
        "model": OLLAMA_DEFAULT_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": _MERGE_PROMPT},
            {"role": "user", "content": payload},
        ],
    }
    for url, _ in OLLAMA_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=180.0) as c:
                r = await c.post(f"{url}/api/chat", json=body)
                r.raise_for_status()
            reply = r.json().get("message", {}).get("content", "").strip()
            # Tolerant JSON parse — model may wrap in markdown
            for marker in ("```json", "```"):
                if marker in reply:
                    reply = reply.split(marker)[1].split("```")[0] if "```" in reply else reply
            parsed = json.loads(reply)
            if "summary" in parsed:
                return parsed
        except Exception as exc:
            logger.debug(f"Ollama merge failed on {url}: {exc}")
            continue
    return None


async def _llm_merge_claude(entries: list[dict]) -> Optional[dict]:
    """Synthesize via Claude API. Costs tokens."""
    if not ANTHROPIC_API_KEY:
        return None
    payload = "\n\n---\n\n".join(
        f"ENTRY {i+1} (id={e['episode_id']}, instance={e['instance']}, project={e['project']}):\n{e['summary']}"
        for i, e in enumerate(entries)
    )
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "system": _MERGE_PROMPT,
        "messages": [{"role": "user", "content": payload}],
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
            r.raise_for_status()
        reply = ""
        for block in r.json().get("content", []):
            if block.get("type") == "text":
                reply += block.get("text", "")
        for marker in ("```json", "```"):
            if marker in reply:
                reply = reply.split(marker)[1].split("```")[0] if "```" in reply else reply
        return json.loads(reply.strip())
    except Exception as exc:
        logger.warning(f"Claude merge failed: {exc}")
        return None


async def _llm_merge(entries: list[dict], backend: str) -> Optional[dict]:
    if backend == "claude":
        return await _llm_merge_claude(entries)
    return await _llm_merge_ollama(entries)


async def _find_cross_source_neighbors(
    episode: dict,
    threshold: float,
    k: int,
) -> list[dict]:
    """For one episode, find its K nearest neighbors that are from a different
    (machine, project) and have cosine ≥ threshold. Returns dicts with
    episode_id + instance + project + summary + similarity_score."""
    coll = get_episodes_collection()
    embedding = episode.get("embedding")
    if not embedding:
        return []
    own_key = (episode.get("instance"), episode.get("project"))

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": k * 10,
                "limit": k * 3,  # oversample so cross-source filter has room
            }
        },
        {"$addFields": {"sim": {"$meta": "vectorSearchScore"}}},
        {"$match": {
            "sim": {"$gte": threshold},
            "episode_id": {"$ne": episode["episode_id"]},
            "$or": [
                {"superseded_by": {"$exists": False}},
                {"superseded_by": None},
            ],
        }},
        {"$project": {
            "_id": 0, "episode_id": 1, "instance": 1, "project": 1,
            "summary": 1, "tags": 1, "sim": 1,
        }},
    ]

    out = []
    async for doc in coll.aggregate(pipeline):
        if (doc.get("instance"), doc.get("project")) != own_key:
            out.append(doc)
    return out[:k]


def _find_connected_components(adj: dict[str, set[str]]) -> list[set[str]]:
    """Union-find over the adjacency dict → list of clusters."""
    visited: set[str] = set()
    clusters = []
    for node in adj:
        if node in visited:
            continue
        # BFS
        cluster: set[str] = set()
        queue = [node]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            cluster.add(cur)
            for nbr in adj.get(cur, ()):
                if nbr not in visited:
                    queue.append(nbr)
        if len(cluster) >= 2:
            clusters.append(cluster)
    return clusters


async def run_regime_2_pass(
    sample_size: int = R2_DEFAULT_SAMPLE_SIZE,
    similarity_threshold: float = R2_SIMILARITY_THRESHOLD,
    backend: str = "ollama",
    project_filter: Optional[str] = None,
    dry_run: bool = True,  # default DRY because LLM calls + insertions are expensive
) -> dict:
    """Run one Regime 2 pass.

    Steps:
      1. Sample non-superseded episodes (optionally filtered to a project)
      2. For each, find cross-source neighbors via $vectorSearch ≥ threshold
      3. Build graph + extract connected components (cluster size 2..R2_MAX_CLUSTER_SIZE)
      4. For each cluster, LLM-synthesize one canonical, mark members superseded
      5. Insert canonical episodes with consolidated_from=[ids]

    Returns stats dict; persists to consolidation_runs collection.
    """
    started_at = datetime.now(timezone.utc)
    coll = get_episodes_collection()

    sample_query: dict = {
        "$or": [{"superseded_by": {"$exists": False}}, {"superseded_by": None}],
    }
    if project_filter:
        sample_query["project"] = project_filter

    # Pull sample with embedding (needed for vectorSearch seeding)
    cursor = coll.find(sample_query, {
        "episode_id": 1, "instance": 1, "project": 1,
        "summary": 1, "embedding": 1, "tags": 1, "_id": 0,
    }).limit(sample_size)
    seeds = [doc async for doc in cursor]

    logger.info(f"Regime 2: seeded with {len(seeds)} episodes (filter={project_filter})")

    # Build cross-source adjacency
    adj: dict[str, set[str]] = {}
    seed_lookup = {s["episode_id"]: s for s in seeds}
    for seed in seeds:
        neighbors = await _find_cross_source_neighbors(seed, similarity_threshold, R2_K_NEIGHBORS)
        adj.setdefault(seed["episode_id"], set())
        for nbr in neighbors:
            adj[seed["episode_id"]].add(nbr["episode_id"])
            # Symmetric — also remember nbr's payload for later merge prompt
            adj.setdefault(nbr["episode_id"], set()).add(seed["episode_id"])
            if nbr["episode_id"] not in seed_lookup:
                seed_lookup[nbr["episode_id"]] = nbr

    clusters = _find_connected_components(adj)
    clusters = [c for c in clusters if 2 <= len(c) <= R2_MAX_CLUSTER_SIZE]
    logger.info(f"Regime 2: {len(clusters)} clusters in 2..{R2_MAX_CLUSTER_SIZE} range")

    n_merged = 0
    canonical_ids_created = []

    for cluster in clusters:
        entries = [seed_lookup[eid] for eid in cluster if eid in seed_lookup]
        if len(entries) < 2:
            continue
        if dry_run:
            n_merged += 1
            continue

        # LLM merge
        merged = await _llm_merge(entries, backend)
        if not merged or "summary" not in merged:
            logger.warning(f"Merge failed for cluster {cluster}; skipping")
            continue

        new_summary = merged["summary"]
        new_tags = list({*(merged.get("tags") or []), "consolidated"})
        new_episode_id = f"consolidated_r2_{started_at.strftime('%Y%m%d%H%M%S')}_{n_merged:04d}"
        try:
            new_embedding = await asyncio.to_thread(get_embedding, new_summary)
        except Exception as exc:
            logger.error(f"Embedding canonical failed: {exc}")
            continue

        canonical_doc = {
            "episode_id": new_episode_id,
            "instance": "consolidation",
            "project": "consolidated",
            "summary": new_summary,
            "raw_exchange": None,
            "tags": new_tags,
            "embedding": new_embedding,
            "timestamp": started_at,
            "retrieval_count": 0,
            "last_retrieved": None,
            "consolidated_from": list(cluster),
            "consolidated_at": started_at,
            "consolidation_regime": 2,
            "consolidation_backend": backend,
        }
        await coll.insert_one(canonical_doc)
        canonical_ids_created.append(new_episode_id)

        # Mark cluster members superseded by the new canonical
        for member_id in cluster:
            await coll.update_one(
                {"episode_id": member_id},
                {"$set": {
                    "superseded_by": new_episode_id,
                    "superseded_at": started_at,
                    "_supersede_regime": 2,
                }},
            )

        n_merged += 1

    finished_at = datetime.now(timezone.utc)
    stats = {
        "regime": 2,
        "dry_run": dry_run,
        "ran_at": started_at,
        "finished_at": finished_at,
        "duration_ms": int((finished_at - started_at).total_seconds() * 1000),
        "n_seeds": len(seeds),
        "n_clusters_in_range": len(clusters),
        "n_merged": n_merged,
        "similarity_threshold": similarity_threshold,
        "backend": backend,
        "project_filter": project_filter,
        "canonical_ids_created": canonical_ids_created,
    }
    if not dry_run:
        runs_coll = get_db()["consolidation_runs"]
        await runs_coll.insert_one(stats.copy())
    logger.info(
        f"Regime 2 done: seeds={len(seeds)} clusters={len(clusters)} "
        f"merged={n_merged} duration={stats['duration_ms']}ms dry={dry_run}"
    )
    return stats
