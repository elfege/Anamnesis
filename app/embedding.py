import logging
import multiprocessing
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, EMBEDDING_CPU_PCT

logger = logging.getLogger("anamnesis.embedding")

_TOTAL_CORES = multiprocessing.cpu_count()

# Global model + pool — replaced atomically on hot-reload
_model: SentenceTransformer | None = None
_active_model_id: str | None = None
_active_dimensions: int = EMBEDDING_DIMENSIONS
_embedding_pool: ThreadPoolExecutor | None = None


# ─── CPU Affinity ────────────────────────────────────────────────

def _cores_from_pct(pct: int) -> list[int]:
    """Return the first N cores corresponding to pct% of total."""
    n = max(1, int(_TOTAL_CORES * pct / 100))
    return list(range(n))


def _thread_affinity_init(cores: frozenset[int]):
    """ThreadPoolExecutor initializer — pin this worker thread to specified cores.

    torch.set_num_threads(1): each worker uses exactly 1 torch thread.
    N workers pinned to N cores = N cores total, not N×N.
    """
    try:
        os.sched_setaffinity(0, cores)
    except Exception as e:
        logger.warning(f"Could not set CPU affinity for worker thread: {e}")

    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _build_pool(cores: list[int]) -> ThreadPoolExecutor:
    """Create a thread pool whose workers are pinned to the given cores."""
    core_set = frozenset(cores)
    pool = ThreadPoolExecutor(
        max_workers=len(cores),
        initializer=_thread_affinity_init,
        initargs=(core_set,),
    )
    logger.info(f"Embedding pool: {len(cores)} workers pinned to cores {sorted(core_set)}")
    return pool


# ─── Model Loading ───────────────────────────────────────────────

def load_embedding_model(
    model_id: str | None = None,
    cpu_pct: int | None = None,
    cpu_cores: list[int] | None = None,
) -> SentenceTransformer:
    """Load (or hot-reload) the sentence-transformers model.

    Args:
        model_id:   HuggingFace model ID. Defaults to EMBEDDING_MODEL env var.
        cpu_pct:    % of cores to use if cpu_cores not given. Defaults to EMBEDDING_CPU_PCT.
        cpu_cores:  Explicit list of core indices. Overrides cpu_pct if provided.
    """
    global _model, _active_model_id, _active_dimensions, _embedding_pool

    target_model = model_id or EMBEDDING_MODEL
    effective_pct = cpu_pct if cpu_pct is not None else EMBEDDING_CPU_PCT
    effective_cores = cpu_cores if cpu_cores is not None else _cores_from_pct(effective_pct)

    logger.info(f"Loading embedding model: {target_model}")
    start = time.time()

    new_model = SentenceTransformer(target_model)

    elapsed = time.time() - start
    actual_dims = new_model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded in {elapsed:.2f}s — dims: {actual_dims}")

    if actual_dims != EMBEDDING_DIMENSIONS and model_id is None:
        logger.warning(
            f"Dimension mismatch: model produces {actual_dims} but config says {EMBEDDING_DIMENSIONS}. "
            "Update EMBEDDING_DIMENSIONS or pass the correct value."
        )

    # Tear down old pool, build new one pinned to selected cores
    if _embedding_pool is not None:
        _embedding_pool.shutdown(wait=False)

    _model = new_model
    _active_model_id = target_model
    _active_dimensions = actual_dims
    _embedding_pool = _build_pool(effective_cores)

    return _model


def apply_cpu_config(cpu_pct: int | None = None, cpu_cores: list[int] | None = None):
    """Update CPU affinity without reloading the model.

    Called when the user changes CPU settings via the UI.
    """
    global _embedding_pool

    effective_cores = cpu_cores if cpu_cores is not None else _cores_from_pct(
        cpu_pct if cpu_pct is not None else EMBEDDING_CPU_PCT
    )

    if _embedding_pool is not None:
        _embedding_pool.shutdown(wait=False)

    _embedding_pool = _build_pool(effective_cores)
    logger.info(f"CPU config updated: cores {sorted(effective_cores)}")


def get_active_model_info() -> dict:
    """Return info about the currently loaded model."""
    return {
        "model_id": _active_model_id,
        "dimensions": _active_dimensions,
        "total_cores": _TOTAL_CORES,
        "pool_workers": _embedding_pool._max_workers if _embedding_pool else 0,
    }


# ─── Embedding ───────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def get_embedding(text: str) -> list[float]:
    """Embed a single text string. Raises RuntimeError if model not loaded."""
    if _model is None:
        raise RuntimeError("Embedding model not loaded. Call load_embedding_model() during startup.")

    normalized = _normalize_text(text)

    if not normalized:
        logger.warning("get_embedding() called with empty text — returning zero vector")
        return [0.0] * _active_dimensions

    embedding = _model.encode(normalized, normalize_embeddings=True)
    return embedding.tolist()


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed. More efficient than calling get_embedding() in a loop."""
    if _model is None:
        raise RuntimeError("Embedding model not loaded. Call load_embedding_model() during startup.")

    normalized = [_normalize_text(t) for t in texts]
    embeddings = _model.encode(normalized, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]


def get_embedding_pool() -> ThreadPoolExecutor | None:
    """Return the current embedding thread pool (for asyncio.to_thread usage)."""
    return _embedding_pool
