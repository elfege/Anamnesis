"""
Training pipeline — exports episodes + feedback to SFT training data,
appends to existing JSONL, and triggers retraining on GPU containers.

Designed to run nightly via the scheduler.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from database import get_feedback_collection, get_episodes_collection

logger = logging.getLogger("anamnesis.training_pipeline")

CHAT_SYSTEM = (
    "You are AnamnesisGPT, a philosophical assistant trained on the writings "
    "of Elfege Leylavergne, particularly his doctoral dissertation on Hegel's "
    "Science of Logic. You answer questions about Hegel, dialectics, quantity, "
    "quality, and the Logic with precision and depth, grounded in the text. "
    "You can discuss in both French and English."
)

# Where training data lives (mounted into trainer containers)
TRAIN_DIR = os.environ.get("TRAIN_HOST_DIR", "/home/elfege/0_LLM_finetune")

# Trainer endpoints (reuse NANOGPT_URLS)
_raw_urls = os.environ.get("NANOGPT_URLS", "http://localhost:3011")
TRAINER_ENDPOINTS = [u.strip().rstrip("/") for u in _raw_urls.split(",") if u.strip()]

# Min new examples before triggering retrain
MIN_NEW_EXAMPLES = int(os.environ.get("RETRAIN_MIN_EXAMPLES", "20"))


async def export_feedback_to_jsonl() -> list[dict]:
    """Export unexported feedback as chat-format training examples."""
    col = get_feedback_collection()
    cursor = col.find({"exported": False}).sort("timestamp", 1)

    examples = []
    ids = []

    async for doc in cursor:
        example = {
            "messages": [
                {"role": "system", "content": CHAT_SYSTEM},
                {"role": "user", "content": doc["user_message"]},
                {"role": "assistant", "content": doc["assistant_message"]},
            ],
            "_source": "feedback",
            "_feedback": "positive" if doc["rating"] == 1 else "negative",
            "_timestamp": doc["timestamp"].isoformat(),
        }
        examples.append(example)
        ids.append(doc["_id"])

    if ids:
        await col.update_many(
            {"_id": {"$in": ids}},
            {"$set": {"exported": True, "exported_at": datetime.now(timezone.utc)}},
        )

    return examples


async def export_episodes_to_jsonl(
    max_episodes: int = 500,
    min_summary_len: int = 100,
) -> list[dict]:
    """Export episodes that haven't been used for training yet.

    Converts episode summary + raw_exchange into chat-format Q&A pairs.
    Only exports episodes with substantive content.
    """
    col = get_episodes_collection()

    # Find episodes not yet exported for training
    cursor = col.find(
        {
            "training_exported": {"$ne": True},
            "summary": {"$exists": True},
        }
    ).sort("timestamp", -1).limit(max_episodes)

    examples = []
    ids = []

    async for doc in cursor:
        summary = (doc.get("summary") or "").strip()
        raw = (doc.get("raw_exchange") or "").strip()

        # Skip thin episodes
        if len(summary) < min_summary_len and len(raw) < min_summary_len:
            continue

        # Use raw_exchange as context, summary as the "answer"
        if raw and summary:
            # Format: user provides context/question, assistant provides insight
            example = {
                "messages": [
                    {"role": "system", "content": CHAT_SYSTEM},
                    {"role": "user", "content": raw[:2000]},
                    {"role": "assistant", "content": summary},
                ],
                "_source": "episode",
                "_episode_id": doc.get("episode_id", ""),
                "_project": doc.get("project", ""),
                "_tags": doc.get("tags", []),
            }
            examples.append(example)
        elif summary and len(summary) > 200:
            # Summary-only: split into a pseudo Q&A
            # Use first sentence as question context, rest as answer
            parts = summary.split(".", 1)
            if len(parts) == 2 and len(parts[1].strip()) > 50:
                example = {
                    "messages": [
                        {"role": "system", "content": CHAT_SYSTEM},
                        {"role": "user", "content": parts[0].strip() + "?"},
                        {"role": "assistant", "content": parts[1].strip()},
                    ],
                    "_source": "episode_summary",
                    "_episode_id": doc.get("episode_id", ""),
                }
                examples.append(example)

        ids.append(doc["_id"])

    # Mark as exported
    if ids:
        await col.update_many(
            {"_id": {"$in": ids}},
            {"$set": {"training_exported": True, "training_exported_at": datetime.now(timezone.utc)}},
        )

    return examples


def append_to_training_file(examples: list[dict], filepath: Optional[str] = None):
    """Append new examples to the training JSONL file."""
    if not examples:
        return 0

    filepath = filepath or os.path.join(TRAIN_DIR, "sft_train.jsonl")

    # Filter: only include positive feedback (negative tagged for future DPO)
    sft_examples = [
        ex for ex in examples
        if ex.get("_feedback") != "negative"
    ]

    if not sft_examples:
        return 0

    with open(filepath, "a") as f:
        for ex in sft_examples:
            # Strip metadata keys before writing
            clean = {"messages": ex["messages"]}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    logger.info("Appended %d examples to %s", len(sft_examples), filepath)
    return len(sft_examples)


def save_negative_examples(examples: list[dict]):
    """Save negative feedback separately for future DPO/RLHF training."""
    neg = [ex for ex in examples if ex.get("_feedback") == "negative"]
    if not neg:
        return 0

    filepath = os.path.join(TRAIN_DIR, "sft_negative.jsonl")
    with open(filepath, "a") as f:
        for ex in neg:
            f.write(json.dumps(ex, ensure_ascii=False, default=str) + "\n")

    logger.info("Saved %d negative examples to %s", len(neg), filepath)
    return len(neg)


async def trigger_retrain() -> dict:
    """Tell the first available trainer container to start a training run."""
    for url in TRAINER_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check if already running
                resp = await client.get(f"{url}/status")
                if resp.status_code == 200:
                    status = resp.json()
                    if status.get("running"):
                        logger.info("Trainer at %s already running, skipping", url)
                        return {"triggered": False, "reason": "already running", "endpoint": url}

                # Start training
                resp = await client.post(f"{url}/start")
                if resp.status_code == 200:
                    result = resp.json()
                    logger.info("Training started on %s: pid=%s", url, result.get("pid"))
                    return {"triggered": True, "endpoint": url, **result}

        except Exception as e:
            logger.debug("Trainer at %s unreachable: %s", url, e)
            continue

    return {"triggered": False, "reason": "no trainer available"}


async def run_training_pipeline() -> dict:
    """Full nightly pipeline: export → append → retrain."""
    logger.info("Training pipeline starting...")

    # 1. Export feedback
    feedback_examples = await export_feedback_to_jsonl()
    logger.info("Exported %d feedback examples", len(feedback_examples))

    # 2. Export episodes
    episode_examples = await export_episodes_to_jsonl()
    logger.info("Exported %d episode examples", len(episode_examples))

    all_examples = feedback_examples + episode_examples

    if not all_examples:
        logger.info("No new training data — skipping retrain")
        return {"new_examples": 0, "retrain": False}

    # 3. Append positive to training file, save negative separately
    appended = append_to_training_file(all_examples)
    neg_saved = save_negative_examples(feedback_examples)

    # 4. Trigger retrain if enough new data
    retrain_result = {"triggered": False, "reason": "below threshold"}
    if appended >= MIN_NEW_EXAMPLES:
        retrain_result = await trigger_retrain()
    else:
        logger.info(
            "Only %d new examples (need %d) — accumulating, not retraining yet",
            appended, MIN_NEW_EXAMPLES,
        )

    result = {
        "new_examples": len(all_examples),
        "feedback_exported": len(feedback_examples),
        "episodes_exported": len(episode_examples),
        "appended_to_train": appended,
        "negative_saved": neg_saved,
        "retrain": retrain_result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("Training pipeline complete: %s", json.dumps(result, default=str))
    return result
