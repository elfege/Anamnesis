import logging
from datetime import datetime, timezone
from typing import Optional

import motor.motor_asyncio
import pymongo
from pymongo.operations import SearchIndexModel

from config import (
    MONGO_URI,
    MONGO_DB,
    COLLECTION_NAME,
    VECTOR_INDEX_NAME,
    EMBEDDING_DIMENSIONS,
)

logger = logging.getLogger("anamnesis.database")

# Global references — set during lifespan startup
_client: motor.motor_asyncio.AsyncIOMotorClient | None = None
_db = None
_episodes_collection = None


# ─── Connection ──────────────────────────────────────────────────

async def connect_to_mongo() -> motor.motor_asyncio.AsyncIOMotorClient:
    """Connect to MongoDB. Called once during FastAPI lifespan startup."""
    global _client, _db, _episodes_collection

    logger.info(f"Connecting to MongoDB: {MONGO_URI.split('@')[-1]}")  # log host only, not creds
    _client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)

    # Verify connection
    server_info = await _client.server_info()
    logger.info(f"MongoDB connected — version {server_info.get('version', 'unknown')}")

    _db = _client[MONGO_DB]
    _episodes_collection = _db[COLLECTION_NAME]

    # Create standard indexes for filtering
    await _episodes_collection.create_index("episode_id", unique=True)
    await _episodes_collection.create_index("project")
    await _episodes_collection.create_index("instance")
    await _episodes_collection.create_index("tags")
    await _episodes_collection.create_index("timestamp")

    return _client


def close_mongo():
    """Close the MongoDB connection."""
    if _client:
        _client.close()
        logger.info("MongoDB connection closed.")


def get_episodes_collection():
    """Return the episodes collection reference."""
    if _episodes_collection is None:
        raise RuntimeError("MongoDB not connected. Call connect_to_mongo() first.")
    return _episodes_collection


# ─── Vector Index ────────────────────────────────────────────────

async def ensure_vector_index():
    """Create the vector search index if it does not exist.

    Uses pymongo's synchronous SearchIndexModel because motor does not
    yet support create_search_index. We get a sync pymongo collection
    reference from the motor client's delegate.
    """
    sync_collection = _episodes_collection.delegate                # pymongo.Collection

    # Check if index already exists
    existing_indexes = list(sync_collection.list_search_indexes())
    for idx in existing_indexes:
        if idx.get("name") == VECTOR_INDEX_NAME:
            logger.info(f"Vector index '{VECTOR_INDEX_NAME}' already exists.")
            return

    logger.info(f"Creating vector search index '{VECTOR_INDEX_NAME}'...")

    search_index_model = SearchIndexModel(
        name=VECTOR_INDEX_NAME,
        type="vectorSearch",
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": EMBEDDING_DIMENSIONS,
                    "similarity": "cosine",
                },
                {                                                  # filterable fields
                    "type": "filter",
                    "path": "project",
                },
                {
                    "type": "filter",
                    "path": "instance",
                },
                {
                    "type": "filter",
                    "path": "tags",
                },
            ]
        },
    )

    sync_collection.create_search_index(search_index_model)
    logger.info(f"Vector search index '{VECTOR_INDEX_NAME}' created.")


# ─── Vector Search ───────────────────────────────────────────────

async def vector_search(
    query_vector: list[float],
    top_k: int = 5,
    project_filter: Optional[str] = None,
    instance_filter: Optional[str] = None,
    tag_filter: Optional[list[str]] = None,
) -> list[dict]:
    """Execute a $vectorSearch aggregation and return top-K episodes."""

    # Build pre-filter for $vectorSearch                           # only add clauses that are set
    pre_filter = {}
    filter_conditions = []
    if project_filter:
        filter_conditions.append({"project": {"$eq": project_filter}})
    if instance_filter:
        filter_conditions.append({"instance": {"$eq": instance_filter}})
    if tag_filter:
        filter_conditions.append({"tags": {"$in": tag_filter}})
    if filter_conditions:
        pre_filter = {"$and": filter_conditions} if len(filter_conditions) > 1 else filter_conditions[0]

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": top_k * 10,                      # oversample for better recall
                "limit": top_k,
            }
        },
        {
            "$addFields": {
                "similarity_score": {"$meta": "vectorSearchScore"}
            }
        },
    ]

    # Add filter to $vectorSearch stage if present
    if pre_filter:
        pipeline[0]["$vectorSearch"]["filter"] = pre_filter

    collection = get_episodes_collection()
    results = []
    async for doc in collection.aggregate(pipeline):
        doc["_id"] = str(doc["_id"])                               # ObjectId -> str
        results.append(doc)

    return results


# ─── Retrieval Tracking ─────────────────────────────────────────

async def increment_retrieval_count(episode_ids: list[str]):
    """Bump retrieval_count and set last_retrieved for retrieved episodes."""
    if not episode_ids:
        return

    collection = get_episodes_collection()
    now = datetime.now(timezone.utc)

    await collection.update_many(
        {"episode_id": {"$in": episode_ids}},
        {
            "$inc": {"retrieval_count": 1},
            "$set": {"last_retrieved": now},
        },
    )


# ─── Dashboard Stats ────────────────────────────────────────────

async def get_dashboard_stats() -> dict:
    """Aggregate KPIs for the dashboard."""
    collection = get_episodes_collection()

    total_episodes = await collection.count_documents({})

    # Episodes by project
    episodes_by_project_cursor = collection.aggregate([
        {"$group": {"_id": "$project", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ])
    episodes_by_project = {}
    async for doc in episodes_by_project_cursor:
        episodes_by_project[doc["_id"]] = doc["count"]

    # Episodes by instance
    episodes_by_instance_cursor = collection.aggregate([
        {"$group": {"_id": "$instance", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ])
    episodes_by_instance = {}
    async for doc in episodes_by_instance_cursor:
        episodes_by_instance[doc["_id"]] = doc["count"]

    # Total retrievals
    total_retrievals_cursor = collection.aggregate([
        {"$group": {"_id": None, "total": {"$sum": "$retrieval_count"}}},
    ])
    total_retrievals = 0
    async for doc in total_retrievals_cursor:
        total_retrievals = doc["total"]

    # Most retrieved episodes (top 10)
    most_retrieved_cursor = collection.find(
        {"retrieval_count": {"$gt": 0}},
        {"embedding": 0},                                         # exclude embedding from response
    ).sort("retrieval_count", -1).limit(10)
    most_retrieved_episodes = []
    async for doc in most_retrieved_cursor:
        doc["_id"] = str(doc["_id"])
        most_retrieved_episodes.append(doc)

    # Recent episodes (last 10)
    recent_cursor = collection.find(
        {},
        {"embedding": 0},
    ).sort("timestamp", -1).limit(10)
    recent_episodes = []
    async for doc in recent_cursor:
        doc["_id"] = str(doc["_id"])
        recent_episodes.append(doc)

    return {
        "total_episodes": total_episodes,
        "episodes_by_project": episodes_by_project,
        "episodes_by_instance": episodes_by_instance,
        "most_retrieved_episodes": most_retrieved_episodes,
        "recent_episodes": recent_episodes,
        "total_retrievals": total_retrievals,
    }
