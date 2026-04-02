"""AnamnesisGPT — machine-gated access to the personal LLM.

Proxies generation requests to trainer containers running on GPU machines.
Supports multiple GPU endpoints with automatic failover.
"""

import os
import json
import logging

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("anamnesis.gpt")

router = APIRouter(prefix="/api/anamnesis-gpt", tags=["anamnesis-gpt"])

AUTHORIZED_MACHINE_ID = os.environ.get("AUTHORIZED_MACHINE_ID", "").strip()

# Comma-separated list of trainer URLs, tried in order.
# e.g. "http://gpu-server-1:3011,http://gpu-server-2:3011"
_raw_urls = os.environ.get("NANOGPT_URLS", os.environ.get("NANOGPT_URL", "http://localhost:3011"))
GPU_ENDPOINTS = [u.strip().rstrip("/") for u in _raw_urls.split(",") if u.strip()]


def _get_host_machine_id() -> str:
    """Read the host's machine-id (mounted read-only into the container)."""
    try:
        with open("/etc/host-machine-id") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def is_authorized() -> bool:
    if not AUTHORIZED_MACHINE_ID:
        return False
    return _get_host_machine_id() == AUTHORIZED_MACHINE_ID


@router.get("/status")
async def gpt_status():
    """Check whether AnamnesisGPT is available on this machine."""
    authorized = is_authorized()
    if authorized:
        return {
            "available": True,
            "message": "AnamnesisGPT is available.",
            "endpoints": GPU_ENDPOINTS,
        }
    return {
        "available": False,
        "message": (
            "AnamnesisGPT is only available on the authorized host. "
            "Custom training on your own data is coming soon — "
            "you'll be able to build and chat with your own model."
        ),
    }


class GenerateRequest(BaseModel):
    prompt: str = Field("\n", description="Text prompt")
    max_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.8, ge=0.01, le=2.0)
    top_k: int = Field(200, ge=1, le=1000)
    stream: bool = Field(True)


@router.post("/generate")
async def generate(req: GenerateRequest):
    """Proxy generation request to a trainer's inference endpoint.
    Tries each GPU endpoint in order until one responds."""
    if not is_authorized():
        raise HTTPException(403, "Not authorized on this host.")

    if req.stream:
        return StreamingResponse(
            _proxy_stream(req), media_type="text/event-stream"
        )

    # Non-streaming
    async with httpx.AsyncClient(timeout=60.0) as client:
        for url in GPU_ENDPOINTS:
            try:
                resp = await client.post(f"{url}/generate", json=req.model_dump())
                if resp.status_code == 200:
                    return resp.json()
            except httpx.ConnectError:
                continue
        raise HTTPException(503, "No GPU inference endpoint reachable")


async def _proxy_stream(req: GenerateRequest):
    """Stream tokens from a GPU trainer's inference endpoint as SSE.
    Tries each endpoint in order until one connects."""
    for url in GPU_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{url}/generate",
                    json=req.model_dump(),
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data:"):
                            yield line + "\n\n"
                return  # success — don't try next endpoint
        except httpx.ConnectError:
            logger.warning(f"GPU endpoint unreachable: {url}")
            continue
        except Exception as exc:
            logger.error(f"AnamnesisGPT stream error from {url}: {exc}")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return

    yield f"data: {json.dumps({'error': 'No GPU inference endpoint reachable. Tried: ' + ', '.join(GPU_ENDPOINTS)})}\n\n"
