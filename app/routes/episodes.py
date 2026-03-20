import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from models import EpisodeCreate, EpisodeOut, EpisodeSearchRequest, EpisodeSearchResult
from embedding import get_embedding, get_active_model_info, get_embedding_pool
from bson import ObjectId

from database import (
    get_episodes_collection,
    vector_search,
    increment_retrieval_count,
    save_reembed_checkpoint,
    load_reembed_checkpoint,
    clear_reembed_checkpoint,
)

# ─── Reembed state ───────────────────────────────────────────────

_reembed_state = {
    "running": False,
    "paused": False,          # paused by user or shutdown — checkpoint saved
    "pause_requested": False, # signal to the running loop
    "total": 0,
    "done": 0,
    "errors": 0,
    "started_at": None,
    "current_model": None,
    "stale": False,
    "checkpoint_id": None,    # str(_id) of last checkpointed episode
}

logger = logging.getLogger("anamnesis.routes.episodes")

router = APIRouter(prefix="/api/episodes", tags=["episodes"])


# ─── Ingest ──────────────────────────────────────────────────────

@router.post("", response_model=EpisodeOut, status_code=201)
async def create_episode(episode: EpisodeCreate):
    """Ingest a new episode: embed the summary, store in MongoDB."""

    collection = get_episodes_collection()

    # Check for duplicate episode_id
    existing = await collection.find_one({"episode_id": episode.episode_id})
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Episode '{episode.episode_id}' already exists."
        )

    # Generate embedding from the summary text
    logger.info(f"Embedding episode: {episode.episode_id}")
    embedding_vector = get_embedding(episode.summary)

    # Build the document
    now = datetime.now(timezone.utc)
    document = {
        "episode_id": episode.episode_id,
        "instance": episode.instance,
        "project": episode.project,
        "summary": episode.summary,
        "raw_exchange": episode.raw_exchange,
        "tags": episode.tags,
        "embedding": embedding_vector,
        "timestamp": now,
        "retrieval_count": 0,
        "last_retrieved": None,
    }

    result = await collection.insert_one(document)
    logger.info(f"Stored episode: {episode.episode_id} (_id: {result.inserted_id})")

    # Return without embedding (it's large and not useful in response)
    return EpisodeOut(
        episode_id=episode.episode_id,
        instance=episode.instance,
        project=episode.project,
        summary=episode.summary,
        raw_exchange=episode.raw_exchange,
        tags=episode.tags,
        timestamp=now,
        retrieval_count=0,
        last_retrieved=None,
    )


# ─── Vector Search ───────────────────────────────────────────────

@router.post("/search", response_model=list[EpisodeSearchResult])
async def search_episodes(request: EpisodeSearchRequest):
    """Search episodes by vector similarity."""

    logger.info(
        f"Searching: '{request.query_text[:80]}...' "
        f"(top_k={request.top_k}, project={request.project_filter})"
    )

    # Embed the query text
    query_vector = get_embedding(request.query_text)

    # Execute vector search
    results = await vector_search(
        query_vector=query_vector,
        top_k=request.top_k,
        project_filter=request.project_filter,
        instance_filter=request.instance_filter,
        tag_filter=request.tag_filter,
    )

    # Track retrieval counts
    retrieved_episode_ids = [r["episode_id"] for r in results]
    await increment_retrieval_count(retrieved_episode_ids)

    # Build response (exclude embedding from output)
    search_results = []
    for doc in results:
        search_results.append(EpisodeSearchResult(
            episode_id=doc["episode_id"],
            instance=doc["instance"],
            project=doc["project"],
            summary=doc["summary"],
            raw_exchange=doc.get("raw_exchange"),
            tags=doc.get("tags", []),
            timestamp=doc["timestamp"],
            retrieval_count=doc.get("retrieval_count", 0),
            last_retrieved=doc.get("last_retrieved"),
            similarity_score=doc.get("similarity_score", 0.0),
        ))

    logger.info(f"Search returned {len(search_results)} results")
    return search_results


# ─── List / Browse ───────────────────────────────────────────────

@router.get("", response_model=list[EpisodeOut])
async def list_episodes(
    project: Optional[str] = Query(None, description="Filter by project"),
    instance: Optional[str] = Query(None, description="Filter by instance"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Pagination limit"),
):
    """List episodes with optional filters and pagination."""

    collection = get_episodes_collection()

    query_filter = {}
    if project:
        query_filter["project"] = project
    if instance:
        query_filter["instance"] = instance
    if tag:
        query_filter["tags"] = tag

    cursor = collection.find(
        query_filter,
        {"embedding": 0},                                         # exclude embedding
    ).sort("timestamp", -1).skip(skip).limit(limit)

    episodes = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        episodes.append(EpisodeOut(
            episode_id=doc["episode_id"],
            instance=doc["instance"],
            project=doc["project"],
            summary=doc["summary"],
            raw_exchange=doc.get("raw_exchange"),
            tags=doc.get("tags", []),
            timestamp=doc["timestamp"],
            retrieval_count=doc.get("retrieval_count", 0),
            last_retrieved=doc.get("last_retrieved"),
        ))

    return episodes


# ─── Get Single ──────────────────────────────────────────────────

@router.get("/{episode_id}", response_model=EpisodeOut)
async def get_episode(episode_id: str):
    """Get a single episode by its episode_id."""

    collection = get_episodes_collection()
    doc = await collection.find_one(
        {"episode_id": episode_id},
        {"embedding": 0},
    )

    if not doc:
        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found.")

    return EpisodeOut(
        episode_id=doc["episode_id"],
        instance=doc["instance"],
        project=doc["project"],
        summary=doc["summary"],
        raw_exchange=doc.get("raw_exchange"),
        tags=doc.get("tags", []),
        timestamp=doc["timestamp"],
        retrieval_count=doc.get("retrieval_count", 0),
        last_retrieved=doc.get("last_retrieved"),
    )


# ─── Reembed Migration ───────────────────────────────────────────

_CHECKPOINT_INTERVAL = 25  # save checkpoint every N episodes


async def _run_reembed(resume_from_id: str | None = None):
    """Re-embed all episodes using the current model, with pause/resume support."""
    from embedding import get_active_model_info
    collection = get_episodes_collection()
    active = get_active_model_info()
    model_id = active.get("model_id")

    _reembed_state["running"] = True
    _reembed_state["paused"] = False
    _reembed_state["pause_requested"] = False
    _reembed_state["errors"] = 0
    _reembed_state["started_at"] = datetime.now(timezone.utc).isoformat()
    _reembed_state["current_model"] = model_id
    _reembed_state["stale"] = True

    # Build query — resume from checkpoint if provided
    query = {}
    if resume_from_id:
        query["_id"] = {"$gt": ObjectId(resume_from_id)}
        logger.info(f"Reembed resuming after _id={resume_from_id}")
    else:
        _reembed_state["done"] = 0

    total = await collection.count_documents({})
    _reembed_state["total"] = total
    remaining = await collection.count_documents(query)
    logger.info(f"Reembed started: {remaining} remaining / {total} total")

    last_id: str | None = resume_from_id
    cursor = collection.find(query, {"episode_id": 1, "summary": 1}).sort("_id", 1)

    async for doc in cursor:
        # Check pause signal
        if _reembed_state["pause_requested"]:
            # Save checkpoint at done-1 to be safe (re-process last one on resume)
            safe_id = last_id  # last *confirmed* id before this one
            await save_reembed_checkpoint(model_id, _reembed_state["done"], total, safe_id or str(doc["_id"]))
            _reembed_state["running"] = False
            _reembed_state["paused"] = True
            _reembed_state["pause_requested"] = False
            _reembed_state["checkpoint_id"] = safe_id
            logger.info(f"Reembed paused at {_reembed_state['done']}/{total}, checkpoint saved")
            return

        try:
            loop = asyncio.get_event_loop()
            new_vec = await loop.run_in_executor(get_embedding_pool(), get_embedding, doc["summary"])
            await collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"embedding": new_vec}},
            )
            last_id = str(doc["_id"])
            _reembed_state["done"] += 1
            _reembed_state["checkpoint_id"] = last_id

            if _reembed_state["done"] % _CHECKPOINT_INTERVAL == 0:
                await save_reembed_checkpoint(model_id, _reembed_state["done"], total, last_id)
                logger.info(f"Reembed progress: {_reembed_state['done']}/{total} (checkpoint saved)")

        except Exception as e:
            _reembed_state["errors"] += 1
            logger.error(f"Reembed failed for {doc.get('episode_id')}: {e}")

    # Completed
    _reembed_state["running"] = False
    _reembed_state["paused"] = False
    _reembed_state["stale"] = False
    await clear_reembed_checkpoint()
    logger.info(f"Reembed complete: {_reembed_state['done']} updated, {_reembed_state['errors']} errors")


async def reembed_auto_resume():
    """Called at startup — resumes from checkpoint if one exists for the active model."""
    from embedding import get_active_model_info
    checkpoint = await load_reembed_checkpoint()
    if not checkpoint:
        return
    active_model = get_active_model_info().get("model_id")
    if checkpoint.get("model_id") != active_model:
        logger.info(f"Reembed checkpoint found but model changed ({checkpoint.get('model_id')} → {active_model}), ignoring.")
        await clear_reembed_checkpoint()
        return

    resume_id = checkpoint.get("last_id")
    _reembed_state["done"] = checkpoint.get("done", 0)
    _reembed_state["total"] = checkpoint.get("total", 0)
    _reembed_state["stale"] = True
    logger.info(f"Reembed checkpoint found ({_reembed_state['done']}/{_reembed_state['total']}), auto-resuming...")
    asyncio.create_task(_run_reembed(resume_from_id=resume_id))


@router.post("/reembed")
async def reembed_episodes():
    """Start re-embedding all episodes with the current model."""
    if _reembed_state["running"]:
        return {"status": "already_running", **_reembed_state}
    _reembed_state["done"] = 0
    asyncio.create_task(_run_reembed())
    return {"status": "started"}


@router.post("/reembed/pause")
async def pause_reembed():
    """Signal the running reembed to pause and save a checkpoint."""
    if not _reembed_state["running"]:
        return {"status": "not_running"}
    _reembed_state["pause_requested"] = True
    return {"status": "pause_requested"}


@router.post("/reembed/resume")
async def resume_reembed():
    """Resume reembed from the last checkpoint."""
    if _reembed_state["running"]:
        return {"status": "already_running"}
    checkpoint = await load_reembed_checkpoint()
    if not checkpoint:
        return {"status": "no_checkpoint"}
    resume_id = checkpoint.get("last_id")
    _reembed_state["done"] = checkpoint.get("done", 0)
    _reembed_state["total"] = checkpoint.get("total", 0)
    asyncio.create_task(_run_reembed(resume_from_id=resume_id))
    return {"status": "resumed", "from": _reembed_state["done"]}


@router.get("/reembed/status")
async def reembed_status():
    """Return current reembed progress."""
    checkpoint = await load_reembed_checkpoint()
    return {
        **_reembed_state,
        "model": get_active_model_info(),
        "checkpoint": checkpoint or None,
    }


# ─── Delete ──────────────────────────────────────────────────────

@router.delete("/{episode_id}")
async def delete_episode(episode_id: str):
    """Delete an episode by its episode_id."""

    collection = get_episodes_collection()
    result = await collection.delete_one({"episode_id": episode_id})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found.")

    logger.info(f"Deleted episode: {episode_id}")
    return {"status": "deleted", "episode_id": episode_id}
