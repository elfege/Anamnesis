import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from crawler import get_crawler_status, run_crawl_cycle, get_crawler_config, save_crawler_config
from jsonl_ingester import (
    get_jsonl_source_roots_config,
    save_jsonl_source_roots,
)
from database import load_docx_tag_patterns, save_docx_tag_patterns

logger = logging.getLogger("anamnesis.routes.crawler")

router = APIRouter(prefix="/api/crawler", tags=["crawler"])


# ─── Pydantic models ─────────────────────────────────────────────

class MachineRootsUpdate(BaseModel):
    machine_roots: dict[str, str]


class SourceUpdate(BaseModel):
    name: str
    path: str
    instance: str
    project: str
    description: str = ""


class SourcesUpdate(BaseModel):
    sources: list[SourceUpdate]


class JsonlSourceRootsUpdate(BaseModel):
    roots: dict[str, str]


class DocxTagPattern(BaseModel):
    match: str
    tag: str
    field: str = "filename"
    regex: bool = False


class DocxTagPatternsUpdate(BaseModel):
    patterns: list[DocxTagPattern]


# ─── Crawler status / trigger ────────────────────────────────────

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


# ─── Crawler config (machine roots + named sources) ─────────────

@router.get("/config")
async def get_config():
    """Return current crawler configuration (machine roots + named sources)."""
    return await get_crawler_config()


@router.put("/config/machine-roots")
async def update_machine_roots(body: MachineRootsUpdate):
    """Update crawler machine roots. Replaces the entire dict."""
    await save_crawler_config(machine_roots=body.machine_roots)
    return {"status": "ok", "machine_roots": body.machine_roots}


@router.put("/config/sources")
async def update_sources(body: SourcesUpdate):
    """Update named crawler sources. Replaces the entire list."""
    sources = [s.model_dump() for s in body.sources]
    await save_crawler_config(sources=sources)
    return {"status": "ok", "sources": sources}


# ─── JSONL source roots ─────────────────────────────────────────

@router.get("/config/jsonl-roots")
async def get_jsonl_roots():
    """Return current JSONL ingester source roots."""
    return await get_jsonl_source_roots_config()


@router.put("/config/jsonl-roots")
async def update_jsonl_roots(body: JsonlSourceRootsUpdate):
    """Update JSONL ingester source roots. Replaces the entire dict."""
    await save_jsonl_source_roots(body.roots)
    return {"status": "ok", "roots": body.roots}


# ─── Docx tag patterns ───────────────────────────────────────────

@router.get("/config/docx-tag-patterns")
async def get_docx_tag_patterns():
    patterns = await load_docx_tag_patterns()
    return {"patterns": patterns}


@router.put("/config/docx-tag-patterns")
async def update_docx_tag_patterns(body: DocxTagPatternsUpdate):
    patterns = [p.model_dump() for p in body.patterns]
    await save_docx_tag_patterns(patterns)
    return {"status": "ok", "patterns": patterns}
