import logging
import os
import re
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from crawler import get_crawler_status, run_crawl_cycle, get_crawler_config, save_crawler_config
from jsonl_ingester import (
    get_jsonl_source_roots_config,
    save_jsonl_source_roots,
)
from database import load_doc_tag_patterns, save_doc_tag_patterns

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


class DocTagPattern(BaseModel):
    match: str
    tag: str
    field: str = "filename"
    regex: bool = False
    ignorecase: bool = False


class DocTagPatternsUpdate(BaseModel):
    patterns: list[DocTagPattern]


# ─── Available source mounts ─────────────────────────────────────

def _read_bind_sources() -> dict[str, str]:
    """Parse /proc/self/mountinfo and return {container_path: host_path} for /sources/* mounts.

    Bind mounts surface the host source path in column 4 of mountinfo. Used so the dashboard
    can show operators which host directory each /sources/<name> mount points at.
    """
    bind_sources: dict[str, str] = {}
    try:
        with open("/proc/self/mountinfo", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                host_path = parts[3]
                container_path = parts[4]
                if container_path.startswith("/sources/"):
                    bind_sources[container_path] = host_path
    except FileNotFoundError:
        pass
    return bind_sources


def _count_jsonl_files(root: str, max_walk: int = 500_000) -> int:
    """Count *.jsonl files anywhere under root. Capped at max_walk for safety on huge trees."""
    n = 0
    try:
        for _, _, files in os.walk(root):
            for fname in files:
                if fname.endswith(".jsonl"):
                    n += 1
                    if n >= max_walk:
                        return n
    except (OSError, PermissionError):
        pass
    return n


@router.get("/available-mounts")
async def list_available_mounts():
    """List directories under /sources/ that exist inside the container.

    Each mount entry includes the container path, the host bind-source path (parsed from
    /proc/self/mountinfo), and a count of JSONL files reachable under <mount>/.claude/projects/
    (the path the JSONL ingester actually scans). Mounts with zero JSONL files are flagged
    `jsonl_empty: true` so the UI can dim them.
    """
    sources_root = "/sources"
    bind_sources = _read_bind_sources()
    mounts = []
    try:
        for entry in sorted(os.listdir(sources_root)):
            full = os.path.join(sources_root, entry)
            if not os.path.isdir(full) or entry in ("teachings", "documents"):
                continue
            jsonl_root = os.path.join(full, ".claude", "projects")
            jsonl_count = _count_jsonl_files(jsonl_root) if os.path.isdir(jsonl_root) else 0
            mounts.append({
                "name": entry,
                "path": full,
                "host_path": bind_sources.get(full, "(unknown — not a bind mount)"),
                "jsonl_path": jsonl_root,
                "jsonl_file_count": jsonl_count,
                "jsonl_empty": jsonl_count == 0,
            })
    except FileNotFoundError:
        pass
    return {"mounts": mounts}


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


# ─── Document tag patterns ───────────────────────────────────────

@router.get("/config/doc-tag-patterns")
async def get_doc_tag_patterns():
    patterns = await load_doc_tag_patterns()
    return {"patterns": patterns}


@router.put("/config/doc-tag-patterns")
async def update_doc_tag_patterns(body: DocTagPatternsUpdate):
    patterns = [p.model_dump() for p in body.patterns]
    await save_doc_tag_patterns(patterns)
    return {"status": "ok", "patterns": patterns}


# Backward-compat aliases (old endpoint paths still work)
@router.get("/config/docx-tag-patterns")
async def get_docx_tag_patterns_compat():
    return await get_doc_tag_patterns()


@router.put("/config/docx-tag-patterns")
async def update_docx_tag_patterns_compat(body: DocTagPatternsUpdate):
    return await update_doc_tag_patterns(body)


# ─── Regex validation ───────────────────────────────────────────

class RegexValidateRequest(BaseModel):
    pattern: str


@router.post("/validate-regex")
async def validate_regex(body: RegexValidateRequest):
    """Check if a regex pattern compiles. Returns {valid, error}."""
    try:
        re.compile(body.pattern)
        return {"valid": True, "error": None}
    except re.error as e:
        return {"valid": False, "error": str(e)}
