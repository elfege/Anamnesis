import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import LOG_LEVEL
from embedding import load_embedding_model
from database import connect_to_mongo, close_mongo, ensure_vector_index
from routes.episodes import router as episodes_router
from routes.dashboard import router as dashboard_router
from routes.crawler import router as crawler_router
from routes.chat import router as chat_router
from crawler import start_crawler, stop_crawler

# ─── Logging ─────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("anamnesis")


# ─── Lifespan ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to MongoDB, load embedding model, create vector index.
    Shutdown: close MongoDB connection.
    """
    # --- Startup ---
    logger.info("Anamnesis starting...")

    logger.info("Connecting to MongoDB...")
    await connect_to_mongo()

    logger.info("Loading embedding model...")
    load_embedding_model()

    logger.info("Ensuring vector search index exists...")
    await ensure_vector_index()

    logger.info("Starting crawler...")
    start_crawler()

    logger.info("Anamnesis ready.")

    yield

    # --- Shutdown ---
    stop_crawler()
    close_mongo()
    logger.info("Anamnesis stopped.")


# ─── App ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Anamnesis",
    description=(
        "Embedding-based episodic memory for Claude instances. "
        "Part of the Genesis persistence project."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Static files (CSS, JS for dashboard)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Routes
app.include_router(episodes_router)
app.include_router(dashboard_router)
app.include_router(crawler_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "anamnesis",
        "version": "0.1.0",
    }
