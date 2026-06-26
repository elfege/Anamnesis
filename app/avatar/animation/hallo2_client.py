"""Hallo2 client — POSTs to the avatar_worker /hallo2/animate endpoint.

Hallo2 (Fudan U) is a higher-realism talking-head model. Heavier than
SadTalker/MuseTalk (~12GB VRAM, several seconds per output second), but
produces noticeably more natural motion — viable for portfolio-grade demos.

Install state (2026-06-08): NOT installed on any worker yet. This client
exists so the orchestrator can route to it; calls will fail with the clear
"No Hallo2 worker reachable" error until a worker exposes /hallo2/animate.
"""
import logging
from pathlib import Path
from typing import Optional

import httpx

import config
from avatar.workers import order_endpoints as _order_endpoints

logger = logging.getLogger("anamnesis.avatar.hallo2")


class Hallo2Client:

    @property
    def name(self) -> str:
        return "hallo2@worker"

    async def animate(
        self, audio_path: str, reference_image: str, output_path: str,
        preferred_worker: Optional[str] = None, no_fallback: bool = False,
    ) -> str:
        endpoints = _order_endpoints(config.AVATAR_WORKER_ENDPOINTS, preferred_worker, no_fallback)
        last_err = None
        for url, label in endpoints:
            try:
                async with httpx.AsyncClient(timeout=900) as client:
                    with open(audio_path, "rb") as a, open(reference_image, "rb") as i:
                        files = {
                            "audio": (Path(audio_path).name, a, "audio/wav"),
                            "image": (Path(reference_image).name, i, "image/png"),
                        }
                        r = await client.post(f"{url}/hallo2/animate", files=files)
                if r.status_code != 200:
                    raise RuntimeError(f"worker {label} returned {r.status_code}: {r.text[:200]}")
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as out:
                    out.write(r.content)
                logger.info(f"Hallo2 via {label} → {output_path} ({len(r.content)} bytes)")
                return output_path
            except Exception as e:
                logger.warning(f"Hallo2 worker {label} ({url}) failed: {e}")
                last_err = e
        raise RuntimeError(f"No Hallo2 worker reachable. Last error: {last_err}")
