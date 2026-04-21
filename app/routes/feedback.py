"""Chat feedback — thumbs up/down for continuous training."""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from database import get_feedback_collection

logger = logging.getLogger("anamnesis.routes.feedback")

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    session_id: str
    user_message: str
    assistant_message: str
    rating: int = Field(..., ge=-1, le=1, description="-1 = bad, 1 = good")
    backend: str = "ollama"
    model: str = ""


class FeedbackStats(BaseModel):
    total: int = 0
    positive: int = 0
    negative: int = 0
    pending_export: int = 0


@router.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    col = get_feedback_collection()
    doc = {
        "session_id": req.session_id,
        "user_message": req.user_message,
        "assistant_message": req.assistant_message,
        "rating": req.rating,
        "backend": req.backend,
        "model": req.model,
        "exported": False,
        "timestamp": datetime.now(timezone.utc),
    }
    await col.insert_one(doc)
    label = "positive" if req.rating == 1 else "negative"
    logger.info("Feedback recorded: %s (%s)", label, req.session_id[:12])
    return {"ok": True, "rating": label}


@router.get("/api/feedback/stats")
async def feedback_stats():
    col = get_feedback_collection()
    total = await col.count_documents({})
    positive = await col.count_documents({"rating": 1})
    negative = await col.count_documents({"rating": -1})
    pending = await col.count_documents({"exported": False})
    return FeedbackStats(
        total=total, positive=positive, negative=negative, pending_export=pending,
    )


@router.get("/api/feedback/export")
async def export_training_data(
    min_count: int = 10,
    include_negative: bool = True,
):
    """Export unexported feedback as chat-format training JSONL lines.

    Positive feedback → direct SFT example.
    Negative feedback → tagged so the training script can use DPO or filter.
    """
    col = get_feedback_collection()

    query = {"exported": False}
    cursor = col.find(query).sort("timestamp", 1)

    CHAT_SYSTEM = (
        "You are AnamnesisGPT, a philosophical assistant trained on the writings "
        "of Elfege Leylavergne, particularly his doctoral dissertation on Hegel's "
        "Science of Logic. You answer questions about Hegel, dialectics, quantity, "
        "quality, and the Logic with precision and depth, grounded in the text. "
        "You can discuss in both French and English."
    )

    examples = []
    ids_to_mark = []

    async for doc in cursor:
        if doc["rating"] == -1 and not include_negative:
            continue

        example = {
            "messages": [
                {"role": "system", "content": CHAT_SYSTEM},
                {"role": "user", "content": doc["user_message"]},
                {"role": "assistant", "content": doc["assistant_message"]},
            ],
            "_feedback": "positive" if doc["rating"] == 1 else "negative",
            "_session_id": doc["session_id"],
            "_timestamp": doc["timestamp"].isoformat(),
        }
        examples.append(example)
        ids_to_mark.append(doc["_id"])

    if len(examples) < min_count:
        return {
            "exported": 0,
            "message": f"Only {len(examples)} examples available, need {min_count}. Waiting for more feedback.",
        }

    # Mark as exported
    if ids_to_mark:
        await col.update_many(
            {"_id": {"$in": ids_to_mark}},
            {"$set": {"exported": True}},
        )

    return {"exported": len(examples), "examples": examples}


@router.post("/api/training/run")
async def trigger_training_pipeline():
    """Manually trigger the training pipeline (export + retrain)."""
    from training_pipeline import run_training_pipeline
    result = await run_training_pipeline()
    return result


@router.get("/api/training/status")
async def training_status():
    """Check training pipeline and trainer container status."""
    import httpx
    from training_pipeline import TRAINER_ENDPOINTS

    col = get_feedback_collection()
    pending_feedback = await col.count_documents({"exported": False})

    from database import get_episodes_collection
    ep_col = get_episodes_collection()
    pending_episodes = await ep_col.count_documents({"training_exported": {"$ne": True}, "summary": {"$exists": True}})

    trainers = []
    for url in TRAINER_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/status")
                if resp.status_code == 200:
                    data = resp.json()
                    trainers.append({
                        "endpoint": url,
                        "running": data.get("running", False),
                        "machine": data.get("machine", "?"),
                        "gpu_type": data.get("gpu_type", "?"),
                        "progress": data.get("progress"),
                    })
        except Exception:
            trainers.append({"endpoint": url, "reachable": False})

    return {
        "pending_feedback": pending_feedback,
        "pending_episodes": pending_episodes,
        "trainers": trainers,
    }
