import logging

from fastapi import APIRouter

from crawler import get_crawler_status, run_crawl_cycle

logger = logging.getLogger("anamnesis.routes.crawler")

router = APIRouter(prefix="/api/crawler", tags=["crawler"])


@router.get("/status")
async def crawler_status():
    """Return current crawler state — last run, episodes ingested, errors."""
    return get_crawler_status()


@router.post("/run")
async def trigger_crawl():
    """Manually trigger a crawl cycle (does not wait for the scheduled one)."""
    status = get_crawler_status()
    if status["running"]:
        return {"status": "already_running", "message": "A crawl cycle is already in progress."}

    logger.info("Manual crawl triggered via API")
    ingested = await run_crawl_cycle()
    return {
        "status": "completed",
        "episodes_ingested": ingested,
        "crawler_state": get_crawler_status(),
    }
