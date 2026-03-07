import logging
import re
import time

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS

logger = logging.getLogger("anamnesis.embedding")

# Global model reference — loaded once at startup via load_embedding_model()
_model: SentenceTransformer | None = None


def load_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformers model into memory.

    Called once during FastAPI lifespan startup. Subsequent calls
    to get_embedding() use the cached model.
    """
    global _model

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    start_time = time.time()

    _model = SentenceTransformer(EMBEDDING_MODEL)

    load_duration_seconds = time.time() - start_time
    actual_dimensions = _model.get_sentence_embedding_dimension()
    logger.info(
        f"Model loaded in {load_duration_seconds:.2f}s — "
        f"dimensions: {actual_dimensions} (expected: {EMBEDDING_DIMENSIONS})"
    )

    if actual_dimensions != EMBEDDING_DIMENSIONS:                  # sanity check
        logger.warning(
            f"Dimension mismatch! Model produces {actual_dimensions} dims "
            f"but config says {EMBEDDING_DIMENSIONS}. "
            f"Vector index will break if these don't match."
        )

    return _model


def _normalize_text(text: str) -> str:
    """Clean text before embedding: strip, collapse whitespace."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)                              # collapse runs of whitespace
    return text


def get_embedding(text: str) -> list[float]:
    """Convert a text string to a 384-dimensional embedding vector.

    Uses the globally loaded sentence-transformers model.
    Raises RuntimeError if the model has not been loaded yet.
    """
    if _model is None:
        raise RuntimeError(
            "Embedding model not loaded. "
            "Call load_embedding_model() during app startup."
        )

    normalized_text = _normalize_text(text)

    if not normalized_text:                                        # empty input guard
        logger.warning("get_embedding() called with empty text — returning zero vector")
        return [0.0] * EMBEDDING_DIMENSIONS

    embedding = _model.encode(normalized_text, normalize_embeddings=True)
    return embedding.tolist()


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch variant — embed multiple texts in one call.

    More efficient than calling get_embedding() in a loop because
    sentence-transformers batches the forward pass.
    """
    if _model is None:
        raise RuntimeError(
            "Embedding model not loaded. "
            "Call load_embedding_model() during app startup."
        )

    normalized_texts = [_normalize_text(t) for t in texts]
    embeddings = _model.encode(normalized_texts, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]
