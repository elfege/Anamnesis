"""
d2_bassin.py — POST /api/d2/bassin/ingest endpoint.

WHAT THIS IS, FOR THE DUMMIES:
==============================

Every time the user interacts with `0_JOB_APPLICATIONS_2026` (rewrite,
critique, save draft, restore from history, memorize feedback), the
caller fires a POST here. We:

  1. Validate the schema (Pydantic).
  2. Write the FULL payload (raw text, all fields) into the Mongo
     collection `d2_bassin_ingest_log` — this is the AUTHORITATIVE
     durable record. Never lost, replayable.
  3. Embed the relevant text with the existing 1024-dim sentence-
     transformers embedder.
  4. Best-effort POST the vector + minimal metadata to the d² engine's
     `/bassin/insert` endpoint (cache for fast training-loop access).
     If the engine endpoint doesn't exist yet OR is unreachable, the
     ingest still SUCCEEDS — the durable record is what matters.

Per MSG-248: "The local DB stays the authoritative durable record;
bassin-feed is best-effort training input."

Author choices made on 2026-05-15 by Elfege via AskUserQuestion:
  - Payload: HYBRID (full text in Mongo, vector cached in bassin)
  - Endpoint: PROXY on anamnesis-app (this file)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from database import get_bassin_ingest_collection
from embedding import get_embedding

logger = logging.getLogger("anamnesis.routes.d2_bassin")

router = APIRouter(prefix="/api/d2/bassin", tags=["d2-bassin"])

D2_ENDPOINT_URL = os.environ.get("D2_ENDPOINT_URL", "").rstrip("/")


# ─── Schema ────────────────────────────────────────────────────────────

class BassinIngestRequest(BaseModel):
    """Per MSG-248 stub schema, extended for MSG-253 chain awareness."""
    kind: Literal["rewrite", "feedback", "critique", "manual", "restore-negative"]
    payload: dict
    source: str = Field(..., description="Calling app, e.g. '0_JOB_APPLICATIONS_2026'")
    app_id: Optional[str] = Field(None, description="Caller's identifier (request id, draft id…)")
    ts: Optional[str] = Field(None, description="ISO8601 UTC; server clock used if omitted")
    # ── MSG-253: chain-aware ingestion ─────────────────────────────────
    # Lets callers surface the dialectical chain (critique→rewrite→critique-of-rewrite→…)
    # explicitly instead of leaving it implicit in app_id+ts. Optional; if omitted,
    # the row is treated as chain-root. parent_ingest_id is the Mongo _id (string)
    # of the immediate predecessor in the chain.
    parent_ingest_id: Optional[str] = Field(
        None,
        description="Mongo _id of the immediate predecessor (e.g. the critique a rewrite is replying to).",
    )
    chain_layer: Optional[int] = Field(
        None, ge=0, le=64,
        description="0 = root materials; 1 = first critique; 2 = rewrite-as-reply; etc. Caller-assigned.",
    )
    relation_to_parent: Optional[Literal[
        "critiques", "rewrites", "negates", "sublates", "restores", "refuses", "amplifies"
    ]] = Field(
        None,
        description="Hegelian-flavored typing of how this row relates to its parent. Useful for chain-aware training later.",
    )


# ─── Per-kind text extractor ───────────────────────────────────────────
# Returns the canonical "text to embed" for each interaction type.
# If a kind doesn't lend itself to a single text blob (multi-field), we
# concatenate with separators — the embedder handles long inputs by truncating.

def _embed_text_for(kind: str, payload: dict) -> str:
    """Pick the meaningful text from each interaction kind."""
    if kind == "rewrite":
        # Three-tuple: before + user_feedback → after. Embed all three so
        # the bassin learns the *correction motion*, not just endpoints.
        return "\n\n".join([
            f"BEFORE: {payload.get('before_text', '')}",
            f"FEEDBACK: {payload.get('user_feedback', '')}",
            f"AFTER: {payload.get('after_text', '')}",
        ]).strip()
    if kind == "feedback":
        return "\n".join([
            f"SECTION: {payload.get('section', '')}",
            f"CONTEXT: {payload.get('company', '')} / {payload.get('role', '')}",
            f"FEEDBACK: {payload.get('user_feedback', '')}",
        ]).strip()
    if kind == "critique":
        return "\n\n".join([
            f"MATERIALS: {(payload.get('materials', '') or '')[:4000]}",
            f"JD: {(payload.get('jd', '') or '')[:2000]}",
            f"CRITIQUE: {payload.get('critique_output', '')}",
        ]).strip()
    if kind == "manual":
        return f"{payload.get('section', '')}\n\n{payload.get('text', '')}".strip()
    if kind == "restore-negative":
        # The text the user REJECTED — negative sample.
        return f"REJECTED ({payload.get('section', '')}): {payload.get('text', '')}".strip()
    return str(payload)


# ─── Best-effort engine cache push ────────────────────────────────────

async def _push_to_engine(vec: list[float], req: BassinIngestRequest) -> Optional[str]:
    """POST {vec, kind, source, ts} to engine /bassin/insert.

    Returns None on success, or a short error string. NEVER raises — the
    bassin-cache push is best-effort. The Mongo write is the truth-source.
    """
    if not D2_ENDPOINT_URL:
        return "no D2_ENDPOINT_URL"
    body = {
        "vec": vec,
        "kind": req.kind,
        "source": req.source,
        "app_id": req.app_id,
        "ts": req.ts,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(f"{D2_ENDPOINT_URL}/bassin/insert", json=body)
            if r.status_code == 200:
                return None
            if r.status_code == 404:
                # Engine doesn't have /bassin/insert yet — expected during
                # rollout. Caller doesn't need to know; the Mongo record
                # remains the durable truth.
                return "engine /bassin/insert not implemented (404) — durable record preserved"
            return f"engine returned {r.status_code}: {r.text[:120]}"
    except httpx.RequestError as e:
        return f"engine unreachable: {str(e)[:120]}"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:120]}"


# ─── Endpoint ──────────────────────────────────────────────────────────

@router.post("/ingest")
async def ingest(req: BassinIngestRequest) -> dict[str, Any]:
    """
    Accept an interaction signal from a caller (e.g. 0_JOB_APPLICATIONS_2026)
    and write it to the durable bassin-ingest log + best-effort bassin cache.

    Returns:
        {
          "ok": True,
          "ingest_id": "<mongo objectid>",
          "embedded": True|False,
          "engine_cache": null | "<error msg if engine push failed>",
        }

    Raises 422 if validation fails (Pydantic). Otherwise 200 — even if the
    engine cache push fails, the durable Mongo record is still saved.
    """
    # 1. Build the embed text + compute vector (synchronous CPU; ~50-200ms)
    text = _embed_text_for(req.kind, req.payload)
    embedded_vec: Optional[list[float]] = None
    try:
        if text:
            embedded_vec = get_embedding(text)
    except Exception as e:
        logger.warning(f"bassin/ingest embedding failed (continuing): {e}")

    # 2. Resolve timestamps
    received_at = datetime.now(timezone.utc)
    ts_dt: Optional[datetime] = None
    if req.ts:
        try:
            ts_dt = datetime.fromisoformat(req.ts.replace("Z", "+00:00"))
        except Exception:
            ts_dt = received_at
    else:
        ts_dt = received_at

    # 3. Authoritative durable write — Mongo
    coll = get_bassin_ingest_collection()
    # Resolve parent_ingest_id (validate it points at an existing doc; if it
    # doesn't, store the supplied string anyway with a warning — the chain
    # may be ahead of our knowledge during async fan-in).
    parent_id_str: Optional[str] = req.parent_ingest_id
    if parent_id_str:
        try:
            from bson import ObjectId
            parent = await coll.find_one({"_id": ObjectId(parent_id_str)}, projection={"_id": 1})
            if not parent:
                logger.info(f"bassin ingest: parent_ingest_id={parent_id_str} not found yet (storing anyway)")
        except Exception as e:
            logger.warning(f"bassin ingest: invalid parent_ingest_id={parent_id_str} ({e})")
    doc = {
        "kind": req.kind,
        "source": req.source,
        "app_id": req.app_id,
        "payload": req.payload,
        "embedding": embedded_vec,
        "embed_text": text or None,
        "ts": ts_dt,
        "received_at": received_at,
        # MSG-253 chain fields (None when caller doesn't set them)
        "parent_ingest_id": parent_id_str,
        "chain_layer": req.chain_layer,
        "relation_to_parent": req.relation_to_parent,
    }
    result = await coll.insert_one(doc)
    ingest_id = str(result.inserted_id)
    logger.info(
        f"bassin ingest: kind={req.kind} source={req.source} "
        f"app_id={req.app_id} embedded={embedded_vec is not None} _id={ingest_id}"
    )

    # 4. Best-effort engine cache push (don't fail the ingest if this errors)
    engine_err = None
    if embedded_vec is not None:
        engine_err = await _push_to_engine(embedded_vec, req)
        if engine_err:
            logger.info(f"bassin engine-cache push: {engine_err}")

    return {
        "ok": True,
        "ingest_id": ingest_id,
        "embedded": embedded_vec is not None,
        "engine_cache": engine_err,  # None on success; string explanation otherwise
    }


@router.get("/ingest/chain/{leaf_id}")
async def ingest_chain(leaf_id: str, max_depth: int = 64) -> dict[str, Any]:
    """Walk parent_ingest_id chain from leaf back to root (per MSG-253).

    Returns the chain ordered root→leaf with each step's kind, layer,
    relation_to_parent, and a short payload preview. Useful for callers
    that want to confirm their DAG is being preserved AND for future
    chain-aware training routines that need to reconstruct the full
    dialectical sequence from any endpoint.

    `max_depth` cap protects against accidental cycles (shouldn't exist
    since parent_ingest_id is a Mongo ObjectId of a prior doc, but be safe).
    """
    from bson import ObjectId
    coll = get_bassin_ingest_collection()
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    cur_id: Optional[str] = leaf_id
    while cur_id and len(chain) < max_depth:
        if cur_id in seen:
            break  # cycle defense
        seen.add(cur_id)
        try:
            doc = await coll.find_one({"_id": ObjectId(cur_id)})
        except Exception:
            break
        if not doc:
            break
        chain.append({
            "ingest_id": str(doc["_id"]),
            "kind": doc.get("kind"),
            "chain_layer": doc.get("chain_layer"),
            "relation_to_parent": doc.get("relation_to_parent"),
            "ts": doc.get("ts").isoformat() if doc.get("ts") else None,
            "source": doc.get("source"),
            "app_id": doc.get("app_id"),
            "embed_text_preview": (doc.get("embed_text") or "")[:200],
        })
        cur_id = doc.get("parent_ingest_id")
    chain.reverse()  # root → leaf
    return {"chain": chain, "depth": len(chain), "truncated": len(chain) == max_depth}


@router.get("/ingest/stats")
async def ingest_stats() -> dict[str, Any]:
    """Quick read: total count + per-kind breakdown + most recent ts.

    Useful for office-jobs to confirm its writes are landing without
    having to give it Mongo credentials.
    """
    coll = get_bassin_ingest_collection()
    total = await coll.count_documents({})
    pipeline = [{"$group": {"_id": "$kind", "n": {"$sum": 1}}}]
    by_kind: dict[str, int] = {}
    async for r in coll.aggregate(pipeline):
        by_kind[r["_id"] or "<unknown>"] = r["n"]
    most_recent = await coll.find_one(
        {}, projection={"received_at": 1, "kind": 1, "source": 1},
        sort=[("received_at", -1)],
    )
    return {
        "total": total,
        "by_kind": by_kind,
        "most_recent": (
            {
                "received_at": most_recent["received_at"].isoformat(),
                "kind": most_recent.get("kind"),
                "source": most_recent.get("source"),
            } if most_recent else None
        ),
    }
