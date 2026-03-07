import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from crawler import SOURCES, MACHINE_ROOTS, TEACHINGS_DIR, DOCUMENTS_DIR
from database import get_dashboard_stats

logger = logging.getLogger("anamnesis.routes.dashboard")

templates = Jinja2Templates(directory="templates")

router = APIRouter(tags=["dashboard"])


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
    sources.append({"name": "documents", "path": DOCUMENTS_DIR + "/", "description": "OneDrive .docx files"})
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
    })


@router.get("/api/dashboard/stats")
async def dashboard_stats_json():
    """Return dashboard stats as JSON (for AJAX refresh)."""

    stats = await get_dashboard_stats()
    return stats
