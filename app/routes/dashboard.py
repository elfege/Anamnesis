import logging
import os
import time

import httpx

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
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


@router.get("/", include_in_schema=False)
async def root_redirect():
    """Landing route — redirect to /dashboard.

    The dashboard is the canonical entry surface for the platform until a
    dedicated landing page is shipped (Phase B of the UI overhaul plan).
    A 307 preserves method semantics, but for a GET it is functionally a
    302 and crawlers / browsers handle it identically.
    """
    return RedirectResponse(url="/dashboard", status_code=307)


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

    # Cache-bust for /static/js/* — MAX mtime across all .js files in the bundle.
    # Stops browsers from serving stale JS when ANY file in the bundle changes
    # (previously only tracked dashboard.js, so edits to resource_status_panel.js
    # or future siblings wouldn't tick the version).
    js_dir = "/app/static/js"
    try:
        mtimes = [os.path.getmtime(os.path.join(js_dir, f))
                  for f in os.listdir(js_dir) if f.endswith(".js")]
        asset_version = str(int(max(mtimes))) if mtimes else "0"
    except OSError:
        asset_version = "0"
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats,
        "sources": sources,
        "jsonl_source_roots": JSONL_SOURCE_ROOTS,
        "asset_version": asset_version,
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
    """Return trainer endpoint URLs from environment (keeps IPs out of public JS).

    Supports two env shapes:
    - Legacy: TRAINER_URLS="server-1:http://host:3011,server-2:http://host2:3011"
    - Per-trainer: TRAINER_URL_SERVER1=http://host:3011 + TRAINER_URL_SERVER2=...
    The per-trainer shape is what AWS Secrets Manager stores; the legacy shape
    is preserved for fork-compat.
    """
    trainers = []

    # Legacy shape first (one env var, comma-sep)
    raw = os.environ.get("TRAINER_URLS", "")
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" not in entry:
            continue
        name, _, url = entry.partition(":")
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = url + (":" if False else "")
            # Re-join: the partition above split on the FIRST ':' which ate the http:
            # Actually re-parse: format is "name:URL", URL contains its own ':'
            # Detect by checking if the part after the first ':' starts with '/'
            continue
        label = os.environ.get(f"TRAINER_LABEL_{name.upper()}", name)
        trainers.append({"name": name.strip(), "url": url, "label": label})

    # Per-trainer shape: TRAINER_URL_SERVER1, TRAINER_URL_SERVER2, etc.
    # Add only if not already covered by legacy.
    seen_urls = {t["url"] for t in trainers}
    for key, url in os.environ.items():
        if not key.startswith("TRAINER_URL_") or not url:
            continue
        if not url.startswith(("http://", "https://")):
            continue
        if url in seen_urls:
            continue
        # Derive a name from the env-var suffix: TRAINER_URL_SERVER1 -> server1
        name = key[len("TRAINER_URL_"):].lower()
        label = os.environ.get(f"TRAINER_LABEL_{name.upper()}", name)
        trainers.append({"name": name, "url": url, "label": label})
        seen_urls.add(url)

    return {"trainers": trainers}
