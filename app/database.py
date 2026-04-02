import asyncio
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


def get_settings_collection():
    """Return the settings collection for persistent app configuration."""
    if _db is None:
        raise RuntimeError("MongoDB not connected. Call connect_to_mongo() first.")
    return _db["settings"]


# ─── Chat session persistence ────────────────────────────────────

def get_chat_sessions_collection():
    if _db is None:
        raise RuntimeError("MongoDB not connected.")
    return _db["chat_sessions"]


async def save_chat_session(session_id: str, title: str, messages: list, backend: str, model: str):
    col = get_chat_sessions_collection()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    await col.update_one(
        {"session_id": session_id},
        {"$set": {
            "title": title,
            "messages": messages,
            "backend": backend,
            "model": model,
            "updated_at": now,
            "message_count": len(messages),
        }, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )


async def list_chat_sessions(limit: int = 50) -> list:
    col = get_chat_sessions_collection()
    cursor = col.find({}, {"messages": 0}).sort("updated_at", -1).limit(limit)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results


async def get_chat_session(session_id: str) -> dict | None:
    col = get_chat_sessions_collection()
    doc = await col.find_one({"session_id": session_id})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


async def delete_chat_session(session_id: str):
    col = get_chat_sessions_collection()
    await col.delete_one({"session_id": session_id})


async def rename_chat_session(session_id: str, new_title: str):
    col = get_chat_sessions_collection()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    # Append old title to name_history before overwriting
    await col.update_one(
        {"session_id": session_id},
        {
            "$set": {"title": new_title, "updated_at": now},
            "$push": {"name_history": {"title": new_title, "at": now}},
        },
    )


# ─── Reembed checkpoint persistence ─────────────────────────────

async def save_reembed_checkpoint(model_id: str, done: int, total: int, last_id: str):
    """Upsert reembed checkpoint. last_id is the str(_id) of last processed episode."""
    settings = get_settings_collection()
    from datetime import datetime, timezone
    await settings.update_one(
        {"_id": "reembed_checkpoint"},
        {"$set": {
            "model_id": model_id,
            "done": done,
            "total": total,
            "last_id": last_id,
            "saved_at": datetime.now(timezone.utc),
        }},
        upsert=True,
    )


async def load_reembed_checkpoint() -> dict:
    settings = get_settings_collection()
    doc = await settings.find_one({"_id": "reembed_checkpoint"})
    if doc:
        doc.pop("_id", None)
    return doc or {}


async def clear_reembed_checkpoint():
    settings = get_settings_collection()
    await settings.delete_one({"_id": "reembed_checkpoint"})


# ─── Embedding config persistence ───────────────────────────────

async def save_embedding_config(model_id: str, cpu_pct: int = None, cpu_cores: list = None):
    """Persist active embedding model (and optionally CPU settings) to MongoDB."""
    settings = get_settings_collection()
    update = {"model_id": model_id}
    if cpu_pct is not None:
        update["cpu_pct"] = cpu_pct
    if cpu_cores is not None:
        update["cpu_cores"] = cpu_cores
    await settings.update_one(
        {"_id": "embedding_config"},
        {"$set": update},
        upsert=True,
    )


async def load_embedding_config() -> dict:
    """Load persisted embedding config. Returns {} if never saved."""
    settings = get_settings_collection()
    doc = await settings.find_one({"_id": "embedding_config"})
    if doc:
        doc.pop("_id", None)
    return doc or {}


# ─── Docx tag patterns ──────────────────────────────────────────

_DEFAULT_DOCX_PATTERNS = [
    {"match": r"^[a-z]+\.\d{4}\.docx$", "tag": "student-evaluation", "field": "filename", "regex": True},
    {"match": "agreement",               "tag": "legal",              "field": "filename", "regex": False},
    {"match": "authorization",           "tag": "legal",              "field": "filename", "regex": False},
    {"match": "portfolio",               "tag": "portfolio",          "field": "filename", "regex": False},
    {"match": "notes",                   "tag": "notes",              "field": "filename", "regex": False},
    {"match": "fasny",                   "tag": "school",             "field": "filename", "regex": False},
    {"match": "lfny",                    "tag": "school",             "field": "filename", "regex": False},
]


async def load_docx_tag_patterns() -> list[dict]:
    """Load docx tag patterns from DB. Seeds defaults on first call."""
    settings = get_settings_collection()
    doc = await settings.find_one({"_id": "docx_tag_patterns"})
    if doc:
        return doc.get("patterns", [])
    # First call — seed defaults
    await settings.update_one(
        {"_id": "docx_tag_patterns"},
        {"$set": {"patterns": _DEFAULT_DOCX_PATTERNS}},
        upsert=True,
    )
    return _DEFAULT_DOCX_PATTERNS


async def save_docx_tag_patterns(patterns: list[dict]):
    settings = get_settings_collection()
    await settings.update_one(
        {"_id": "docx_tag_patterns"},
        {"$set": {"patterns": patterns}},
        upsert=True,
    )


# ─── Vector Index ────────────────────────────────────────────────

def _vector_index_definition(dimensions: int) -> dict:
    return {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": dimensions,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "project"},
            {"type": "filter", "path": "instance"},
            {"type": "filter", "path": "tags"},
        ]
    }


async def ensure_vector_index():
    """Create the vector search index, or recreate it if dimensions changed.

    Uses pymongo's synchronous SearchIndexModel because motor does not
    yet support create_search_index. We get a sync pymongo collection
    reference from the motor client's delegate.

    Retries up to 30s — the Search Index Management service starts a few
    seconds after the replica set becomes healthy, so the first attempt
    may hit 'Error connecting to Search Index Management service.'
    """
    import time
    sync_collection = _episodes_collection.delegate                # pymongo.Collection

    # Wait for Search Index Management service to be ready
    for attempt in range(1, 13):
        try:
            existing_indexes = list(sync_collection.list_search_indexes())
            break
        except Exception as e:
            if "Search Index Management" in str(e) and attempt < 12:
                logger.warning(f"Search Index Management not ready (attempt {attempt}/12), retrying in 5s...")
                await asyncio.sleep(5)
            else:
                raise
    for idx in existing_indexes:
        if idx.get("name") != VECTOR_INDEX_NAME:
            continue

        # Index exists — check if dimensions match
        existing_dims = None
        for field in idx.get("latestDefinition", {}).get("fields", []):
            if field.get("type") == "vector":
                existing_dims = field.get("numDimensions")
                break

        if existing_dims == EMBEDDING_DIMENSIONS:
            logger.info(f"Vector index '{VECTOR_INDEX_NAME}' exists with correct dims ({EMBEDDING_DIMENSIONS}).")
            return

        # Dimension mismatch — drop and recreate
        logger.warning(
            f"Vector index dimension mismatch: index has {existing_dims}, "
            f"config wants {EMBEDDING_DIMENSIONS}. Dropping and recreating."
        )
        sync_collection.drop_search_index(VECTOR_INDEX_NAME)
        logger.info(f"Dropped vector index '{VECTOR_INDEX_NAME}'.")
        break

    logger.info(f"Creating vector search index '{VECTOR_INDEX_NAME}' ({EMBEDDING_DIMENSIONS} dims)...")
    search_index_model = SearchIndexModel(
        name=VECTOR_INDEX_NAME,
        type="vectorSearch",
        definition=_vector_index_definition(EMBEDDING_DIMENSIONS),
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

    # Oversample so priority reranking has enough candidates to work with
    fetch_limit = min(top_k * 4, 80)

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": top_k * 10,                      # oversample for better recall
                "limit": fetch_limit,                              # wider net; reranked below
            }
        },
        {
            "$addFields": {
                "similarity_score": {"$meta": "vectorSearchScore"},
            }
        },
        # Priority boost — episodes tagged 'critical' or 'correction' score higher
        {
            "$addFields": {
                "priority_multiplier": {
                    "$cond": {
                        "if": {
                            "$gt": [
                                {
                                    "$size": {
                                        "$setIntersection": [
                                            {"$ifNull": ["$tags", []]},
                                            ["critical", "correction"],
                                        ]
                                    }
                                },
                                0,
                            ]
                        },
                        "then": 1.5,
                        "else": 1.0,
                    }
                }
            }
        },
        {
            "$addFields": {
                "boosted_score": {"$multiply": ["$similarity_score", "$priority_multiplier"]}
            }
        },
        {"$sort": {"boosted_score": -1}},
        {"$limit": top_k},
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
