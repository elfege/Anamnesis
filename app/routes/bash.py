import asyncio
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from bash_state import _pending_consents

logger = logging.getLogger("anamnesis.routes.bash")
router = APIRouter(tags=["bash"])


class BashRunRequest(BaseModel):
    command: str
    host: str = "local"
    timeout: int = 30


@router.post("/api/bash/consent/{consent_id}")
async def bash_consent(consent_id: str, approved: bool = True):
    """Approve or deny a pending bash execution request from the LLM."""
    entry = _pending_consents.get(consent_id)
    if not entry:
        raise HTTPException(404, "Consent request not found or already resolved")
    entry["approved"] = approved
    entry["event"].set()
    return {"status": "approved" if approved else "denied", "consent_id": consent_id}


@router.post("/api/bash/run")
async def bash_run_direct(req: BashRunRequest):
    """Direct (manual) bash execution from the terminal UI — no consent required."""
    from routes.chat import _run_bash
    output = await _run_bash(req.command, req.host)
    return {"output": output, "host": req.host}
