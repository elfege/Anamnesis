import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from database import get_dashboard_stats

logger = logging.getLogger("anamnesis.routes.dashboard")

templates = Jinja2Templates(directory="templates")

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the HTML dashboard with KPIs and episode browser."""

    stats = await get_dashboard_stats()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats,
    })


@router.get("/api/dashboard/stats")
async def dashboard_stats_json():
    """Return dashboard stats as JSON (for AJAX refresh)."""

    stats = await get_dashboard_stats()
    return stats
