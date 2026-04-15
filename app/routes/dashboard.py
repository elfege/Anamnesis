import logging
import os
import time

import httpx

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from crawler import SOURCES, MACHINE_ROOTS, TEACHINGS_DIR, DOCUMENTS_DIR, get_crawler_status
from jsonl_ingester import JSONL_SOURCE_ROOTS
from database import get_dashboard_stats
from embedding import get_active_model_info
from jsonl_ingester import get_ingester_status
from config import OLLAMA_URL, OLLAMA_ENDPOINTS, OLLAMA_DEFAULT_MODEL

_banner_cache = {"text": None, "at": 0}
_BANNER_TTL = 60  # seconds

logger = logging.getLogger("anamnesis.routes.dashboard")

templates = Jinja2Templates(directory="templates")

router = APIRouter(tags=["dashboard"])


@router.get("/chat", response_class=HTMLResponse)
async def chat_standalone(request: Request):
    """Standalone ANAMNESIS.CHAT full-page interface."""
    return templates.TemplateResponse("chat_standalone.html", {"request": request})


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the HTML dashboard with KPIs and episode browser."""

    stats = await get_dashboard_stats()

    # Build sources list dynamically from crawler config
    sources = [
        {"name": s["name"], "path": s["path"], "description": s.get("description", "")}
        for s in SOURCES
    ]
    sources.append({"name": "teachings", "path": TEACHINGS_DIR + "/", "description": "Teaching session files (directory)"})
    sources.append({"name": "documents", "path": DOCUMENTS_DIR + "/", "description": "OneDrive documents (.docx, .pdf, .odt, .pages, .md, .txt, .rtf)"})
    for machine_name, machine_root in MACHINE_ROOTS.items():
        sources.append({
            "name": f"{machine_name} projects + scripts",
            "path": machine_root + "/",
            "description": f"All projects with docker-compose.yml + 0_SCRIPTS/ on {machine_name}",
        })

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats,
        "sources": sources,
        "jsonl_source_roots": JSONL_SOURCE_ROOTS,
    })


@router.get("/api/dashboard/stats")
async def dashboard_stats_json():
    """Return dashboard stats as JSON (for AJAX refresh)."""

    stats = await get_dashboard_stats()
    return stats


@router.get("/api/status/summary")
async def status_summary():
    """Return a short LLM-generated status line for the dashboard banner.

    Aggregates system state, sends to Ollama, returns a ≤15-word sentence.
    Cached for 60s. Falls back to plain text if Ollama is unavailable.
    """
    now = time.time()
    if _banner_cache["text"] and now - _banner_cache["at"] < _BANNER_TTL:
        return {"text": _banner_cache["text"], "cached": True}

    # Gather state
    stats = await get_dashboard_stats()
    crawler = get_crawler_status()
    ingester = get_ingester_status()
    embed = get_active_model_info()

    from routes.episodes import _reembed_state
    reembed = _reembed_state

    state = {
        "episodes": stats.get("total_episodes", 0),
        "model": embed.get("model_id", "unknown"),
        "dims": embed.get("dimensions", 0),
        "cores": embed.get("pool_workers", 0),
        "crawler_running": crawler.get("running", False),
        "crawler_last": crawler.get("last_run", "never"),
        "ingestion_running": ingester.get("running", False),
        "ingestion_episodes": ingester.get("total_episodes_ingested", 0),
        "reembed_running": reembed.get("running", False),
        "reembed_done": reembed.get("done", 0),
        "reembed_total": reembed.get("total", 0),
    }

    prompt = (
        "You are Anamnesis, an episodic memory system. "
        "Write ONE sentence (max 15 words) describing your current state. "
        "Be terse and factual. No filler. No punctuation at end.\n\n"
        f"State: {state}"
    )

    text = None
    # Try Ollama endpoints in fallback order
    ollama_urls = [OLLAMA_URL] if OLLAMA_URL else [ep[0] for ep in OLLAMA_ENDPOINTS]
    for ollama_url in ollama_urls:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": OLLAMA_DEFAULT_MODEL,
                        "stream": False,
                        "options": {"num_predict": 40},
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                r.raise_for_status()
                text = r.json().get("message", {}).get("content", "").strip()
                if "." in text:
                    text = text.split(".")[0]
                text = text.strip('" \n')
                break
        except Exception:
            continue

    if not text:
        # Plain fallback
        parts = [f"{state['episodes']} episodes", f"{embed.get('model_id', '?')}"]
        if state["ingestion_running"]:
            parts.append("ingesting")
        elif state["crawler_running"]:
            parts.append("crawling")
        elif state["reembed_running"]:
            parts.append(f"re-embedding {state['reembed_done']}/{state['reembed_total']}")
        else:
            parts.append("idle")
        text = " · ".join(parts)

    _banner_cache["text"] = text
    _banner_cache["at"] = now
    return {"text": text, "cached": False}


@router.get("/api/config/trainers")
async def trainer_config():
    """Return trainer endpoint URLs from environment (keeps IPs out of public JS)."""
    raw = os.environ.get("TRAINER_URLS", "")
    # Format: "server-1:http://host:3011,server-2:http://host2:3011"
    trainers = []
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" not in entry:
            continue
        name, _, url = entry.partition(":")
        label = os.environ.get(f"TRAINER_LABEL_{name.upper()}", name)
        trainers.append({"name": name.strip(), "url": url.strip(), "label": label})
    return {"trainers": trainers}
