"""
bassin.py — The Bassin de Tenseurs Potentiels (Tension Reservoir).

This file handles the STORAGE and CLASSIFICATION of accumulated tensions.
The optimizer (optimizer.py) computes the tension and stores it in memory.
This file handles:

1. CLASSIFYING tensions by negation type (Hegelian taxonomy)
2. STORING snapshots to MongoDB for persistence across restarts
3. RETRIEVING relevant tensions at inference time (when the model is in doubt)


THE NEGATION TAXONOMY:
=======================

When the model encounters friction (disagreement between what it predicts
and what the data says), that friction can be classified into four types:

| Type                   | What it means                                        | Tension |
|------------------------|------------------------------------------------------|---------|
| Inessential difference | Noise. Too small to matter. Below threshold.         | Low     |
| Essential difference   | Real but recoverable. The model could adjust.        | Medium  |
| Opposition             | Two real forces pulling in opposite directions.      | High    |
| Annihilation           | Total mutual cancellation. Both strong, net zero.    | Highest |

WHY classify? Because at inference time, when the model is uncertain,
we want to retrieve the MOST PRODUCTIVE tensions — the ones most likely
to help. Oppositions and annihilations are more informative than noise.


HOW RETRIEVAL WORKS AT INFERENCE TIME:
=======================================

1. Model generates tokens. We measure confidence (entropy of output
   probability distribution).

2. If confidence is LOW (high entropy = "I don't know"), we trigger
   bassin recall.

3. We query the bassin for tensions that are semantically related to
   the current context (using embeddings + cosine similarity).

4. The retrieved tensions are injected as contrastive context:
   "The model was uncertain about X. Past tensions suggest considering Y."

5. The model re-generates with this additional context.

This is structurally analogous to Plato's anamnesis: learning is
recollection of what was "forgotten" (discarded by standard training)
but retained in the reservoir.
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import torch

logger = logging.getLogger("anamnesis.d2.bassin")


# ============================================================================
# NEGATION TYPE CLASSIFICATION
# ============================================================================

class NegationType(Enum):
    """
    The four Hegelian types of negation, applied to gradient friction.

    Each type describes a different relationship between the retained
    gradient (what the optimizer used) and the discarded/opposing gradient
    (what was working against it).
    """
    INESSENTIAL = "inessential_difference"   # noise, too small to matter
    ESSENTIAL = "essential_difference"        # real difference, recoverable
    OPPOSITION = "opposition"                # two real forces, incompatible
    ANNIHILATION = "annihilation"             # mutual cancellation, both strong


def classify_negation(
    delta1: torch.Tensor,
    delta2: torch.Tensor,
    noise_threshold: float = 1e-6,
    opposition_threshold: float = -0.5,
    annihilation_threshold: float = 0.9,
) -> tuple[NegationType, float]:
    """
    Classify the relationship between logical and empirical friction.

    This is a simplified classification based on cosine similarity and
    magnitude. The formal definitions are in the addendum §5.

    Args:
        delta1: logical friction tensor (δ₁ = W - W̄)
        delta2: empirical friction tensor (δ₂ = ∇L, the gradient)
        noise_threshold: below this magnitude, it's inessential
        opposition_threshold: cosine sim below this = opposition
        annihilation_threshold: if sum magnitude is this fraction of
                                individual magnitudes, it's annihilation

    Returns:
        (negation_type, tension_score)
        tension_score is a float in [0, 1] indicating the strength
        of the negation. Higher = more productive for bassin recall.
    """
    # Flatten to 1D for global comparison
    d1_flat = delta1.flatten().float()
    d2_flat = delta2.flatten().float()

    mag1 = d1_flat.norm()
    mag2 = d2_flat.norm()

    # ── Case 1: Inessential difference ───────────────────────────────
    # Both frictions are tiny. Nothing interesting happening.
    if mag1 < noise_threshold and mag2 < noise_threshold:
        return NegationType.INESSENTIAL, 0.0

    # ── Compute cosine similarity ────────────────────────────────────
    # cos_sim = (δ₁ · δ₂) / (||δ₁|| × ||δ₂||)
    #
    # cos_sim ≈ +1: same direction (essential difference — they agree)
    # cos_sim ≈  0: orthogonal (independent, weak relationship)
    # cos_sim ≈ -1: opposite directions (opposition — they disagree)
    if mag1 < noise_threshold or mag2 < noise_threshold:
        # One is near zero, can't compute meaningful cosine similarity.
        # Classify as essential difference with low tension.
        return NegationType.ESSENTIAL, 0.1

    cos_sim = torch.dot(d1_flat, d2_flat) / (mag1 * mag2 + 1e-8)
    cos_sim_val = cos_sim.item()

    # ── Case 2: Annihilation ─────────────────────────────────────────
    # δ₁ + δ₂ ≈ 0: they cancel each other out. Both are strong,
    # but their sum is near zero. Maximum tension — two real forces
    # in total mutual cancellation.
    sum_mag = (d1_flat + d2_flat).norm()
    individual_mag = mag1 + mag2
    cancellation_ratio = 1.0 - (sum_mag / (individual_mag + 1e-8))

    if cancellation_ratio > annihilation_threshold and individual_mag > noise_threshold * 10:
        # High cancellation + both individually strong = annihilation
        tension = min(1.0, cancellation_ratio)
        return NegationType.ANNIHILATION, tension

    # ── Case 3: Opposition ───────────────────────────────────────────
    # Cosine similarity is strongly negative: the two frictions pull
    # in incompatible directions. Both are real; the system is in tension.
    if cos_sim_val < opposition_threshold:
        # Tension proportional to how negative the cosine similarity is
        tension = min(1.0, abs(cos_sim_val))
        return NegationType.OPPOSITION, tension

    # ── Case 4: Essential difference ─────────────────────────────────
    # They're not cancelling, not opposing — they're just different.
    # The model could accommodate both, but there's real information here.
    tension = min(1.0, (1.0 - cos_sim_val) / 2.0)  # 0 when identical, 0.5 when orthogonal
    return NegationType.ESSENTIAL, tension


# ============================================================================
# BASSIN SNAPSHOT STORAGE (MongoDB)
# ============================================================================

class BassinStore:
    """
    Persists bassin snapshots and negation classifications to MongoDB.

    This allows:
    - Bassin state to survive container restarts
    - Historical analysis of tension patterns
    - Inference-time retrieval of relevant past tensions

    Schema in MongoDB:
    {
        "step": int,                         # training step number
        "timestamp": datetime,               # when the snapshot was taken
        "experiment": str,                   # experiment name
        "negation_counts": {                 # how many of each type at this step
            "inessential_difference": int,
            "essential_difference": int,
            "opposition": int,
            "annihilation": int,
        },
        "tension_stats": {                   # summary statistics
            "mean": float,
            "max": float,
            "min": float,
            "abs_mean": float,
        },
        "high_tension_layers": [             # top-k layers by tension
            {"layer": str, "tension": float, "type": str},
            ...
        ],
    }

    Note: We do NOT store the full bassin tensors in MongoDB (they're huge —
    millions of floats). We store summary statistics and the full tensors
    only at checkpoint time (to disk, not MongoDB).
    """

    def __init__(self, mongo_uri: str, db_name: str, collection_name: str):
        """
        Args:
            mongo_uri:       MongoDB connection string
            db_name:         database name (e.g., "anamnesis")
            collection_name: collection name (e.g., "bassin_tensors")
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _get_collection(self):
        """Lazy connect to MongoDB."""
        if self._collection is None:
            try:
                from pymongo import MongoClient
                self._client = MongoClient(self.mongo_uri)
                db = self._client[self.db_name]
                self._collection = db[self.collection_name]
                logger.info(f"Connected to MongoDB: {self.db_name}.{self.collection_name}")
            except Exception as e:
                logger.warning(f"MongoDB connection failed: {e}. Bassin storage disabled.")
                return None
        return self._collection

    def save_snapshot(
        self,
        step: int,
        experiment: str,
        bassin_stats: dict,
        negation_summary: dict,
        high_tension_layers: Optional[list] = None,
    ):
        """
        Save a bassin snapshot to MongoDB.

        Args:
            step:                training step number
            experiment:          experiment name for grouping
            bassin_stats:        output of optimizer.get_bassin_stats()
            negation_summary:    counts of each negation type
            high_tension_layers: list of dicts with layer-level tension info
        """
        col = self._get_collection()
        if col is None:
            return

        doc = {
            "step": step,
            "timestamp": datetime.now(timezone.utc),
            "experiment": experiment,
            "tension_stats": bassin_stats,
            "negation_counts": negation_summary,
            "high_tension_layers": high_tension_layers or [],
        }

        try:
            col.insert_one(doc)
            logger.debug(f"Saved bassin snapshot at step {step}")
        except Exception as e:
            logger.warning(f"Failed to save bassin snapshot: {e}")

    def query_by_tension(
        self,
        experiment: str,
        min_tension: float = 0.5,
        limit: int = 10,
    ) -> list:
        """
        Retrieve high-tension snapshots for a given experiment.

        Used at inference time: when the model is uncertain, we look back
        at which training steps had the highest tension and what type
        of negation was dominant.

        Args:
            experiment:   experiment name to filter by
            min_tension:  minimum abs_mean tension to include
            limit:        max results to return

        Returns:
            list of snapshot dicts, sorted by tension (highest first)
        """
        col = self._get_collection()
        if col is None:
            return []

        try:
            cursor = col.find(
                {
                    "experiment": experiment,
                    "tension_stats.abs_mean": {"$gte": min_tension},
                },
                sort=[("tension_stats.abs_mean", -1)],
                limit=limit,
            )
            return list(cursor)
        except Exception as e:
            logger.warning(f"Bassin query failed: {e}")
            return []

    def get_tension_history(self, experiment: str) -> list:
        """
        Get the full tension history for plotting.

        Returns:
            list of (step, abs_mean_tension) tuples
        """
        col = self._get_collection()
        if col is None:
            return []

        try:
            cursor = col.find(
                {"experiment": experiment},
                {"step": 1, "tension_stats.abs_mean": 1, "_id": 0},
                sort=[("step", 1)],
            )
            return [(doc["step"], doc["tension_stats"]["abs_mean"]) for doc in cursor]
        except Exception as e:
            logger.warning(f"Tension history query failed: {e}")
            return []


# ============================================================================
# INFERENCE-TIME UNCERTAINTY DETECTION
# ============================================================================

def compute_entropy(logits: torch.Tensor) -> float:
    """
    Compute the entropy of the model's output distribution.

    High entropy = the model is uncertain (probability is spread across
    many tokens). Low entropy = the model is confident (probability is
    concentrated on a few tokens).

    This is the trigger for bassin recall: when entropy exceeds a
    threshold, the model is "in doubt" and should consult its reservoir
    of past tensions.

    Args:
        logits: (vocab_size,) tensor of raw output scores for one position

    Returns:
        entropy value (float). Higher = more uncertain.
        For reference:
            entropy of uniform distribution over 50k tokens ≈ 10.8
            entropy of confident prediction (one token dominates) ≈ 0.1
            typical threshold for "uncertain" ≈ 3.0 - 5.0
    """
    probs = torch.softmax(logits.float(), dim=-1)
    # Entropy = -Σ p(x) log p(x)
    # Add epsilon to avoid log(0)
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum()
    return entropy.item()
