import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import LOG_LEVEL
from embedding import load_embedding_model
from database import connect_to_mongo, close_mongo, ensure_vector_index, load_embedding_config
from routes.episodes import router as episodes_router, _reembed_state, reembed_auto_resume
from routes.dashboard import router as dashboard_router
from routes.crawler import router as crawler_router
from routes.chat import router as chat_router
from routes.jsonl import router as jsonl_router
from routes.files import router as files_router
from routes.bash import router as bash_router
from routes.embedding import router as embedding_router
from routes.anamnesis_gpt import router as anamnesis_gpt_router
from routes.feedback import router as feedback_router
from routes.context_index import router as context_index_router
from routes.avatar import router as avatar_router
from routes.workers import router as workers_router
from routes.restart import router as restart_router
from crawler import load_crawler_config, run_crawl_cycle
from jsonl_ingester import run_jsonl_ingestion, initialize_ingester, load_jsonl_source_roots
from scheduler import (
    start_crawler_scheduler, stop_crawler_scheduler,
    start_jsonl_scheduler, stop_jsonl_scheduler,
    start_training_scheduler, stop_training_scheduler,
)
from training_pipeline import run_training_pipeline
from models_registry import seed_models_registry

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
    saved_cfg = await load_embedding_config()
    if saved_cfg.get("model_id"):
        logger.info(f"Restoring saved embedding config: {saved_cfg}")
    load_embedding_model(
        model_id=saved_cfg.get("model_id") or None,
        cpu_pct=saved_cfg.get("cpu_pct") or None,
        cpu_cores=saved_cfg.get("cpu_cores") or None,
    )

    logger.info("Ensuring vector search index exists...")
    await ensure_vector_index()

    logger.info("Seeding models registry...")
    await seed_models_registry()

    logger.info("Loading crawler config from DB...")
    await load_crawler_config()

    logger.info("Loading JSONL source roots from DB...")
    await load_jsonl_source_roots()

    logger.info("Initializing JSONL ingester (loading state, reconciling orphans)...")
    await initialize_ingester()

    logger.info("Checking for reembed checkpoint...")
    await reembed_auto_resume()

    logger.info("Starting crawler scheduler...")
    start_crawler_scheduler(run_crawl_cycle)

    logger.info("Starting JSONL scheduler...")
    start_jsonl_scheduler(run_jsonl_ingestion)

    logger.info("Starting training pipeline scheduler...")
    start_training_scheduler(run_training_pipeline)

    logger.info("Anamnesis ready.")

    yield

    # --- Shutdown ---
    stop_training_scheduler()
    stop_jsonl_scheduler()
    stop_crawler_scheduler()

    # Save reembed checkpoint if running so it auto-resumes next start
    if _reembed_state.get("running") and _reembed_state.get("checkpoint_id"):
        from database import save_reembed_checkpoint
        logger.info("Saving reembed checkpoint before shutdown...")
        _reembed_state["pause_requested"] = True  # signal loop to stop
        await asyncio.sleep(0.5)  # brief grace period for loop to notice
        await save_reembed_checkpoint(
            _reembed_state["current_model"],
            max(0, _reembed_state["done"] - 1),  # minus 1 to be safe
            _reembed_state["total"],
            _reembed_state["checkpoint_id"],
        )
        logger.info(f"Reembed checkpoint saved at {_reembed_state['done'] - 1}/{_reembed_state['total']}")

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
app.include_router(jsonl_router)
app.include_router(files_router)
app.include_router(bash_router)
app.include_router(embedding_router)
app.include_router(anamnesis_gpt_router)
app.include_router(feedback_router)
app.include_router(context_index_router)
app.include_router(avatar_router)
app.include_router(workers_router)
app.include_router(restart_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "anamnesis",
        "version": "0.1.0",
    }
