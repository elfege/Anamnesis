import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from models import EpisodeCreate, EpisodeOut, EpisodeSearchRequest, EpisodeSearchResult
from embedding import get_embedding
from database import (
    get_episodes_collection,
    vector_search,
    increment_retrieval_count,
)

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
