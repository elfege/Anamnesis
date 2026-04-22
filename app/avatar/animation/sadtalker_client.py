"""SadTalker client — POSTs to the avatar_worker /sadtalker/animate endpoint."""
import logging
from pathlib import Path
from typing import Optional

import httpx

import config
from avatar.workers import order_endpoints as _order_endpoints

logger = logging.getLogger("anamnesis.avatar.sadtalker")


class SadTalkerClient:

    @property
    def name(self) -> str:
        return "sadtalker@worker"

    async def animate(
        self, audio_path: str, reference_image: str, output_path: str,
        preferred_worker: Optional[str] = None, no_fallback: bool = False,
    ) -> str:
        endpoints = _order_endpoints(config.AVATAR_WORKER_ENDPOINTS, preferred_worker, no_fallback)
        last_err = None
        for url, label in endpoints:
            try:
                async with httpx.AsyncClient(timeout=600) as client:
                    with open(audio_path, "rb") as a, open(reference_image, "rb") as i:
                        files = {
                            "audio": (Path(audio_path).name, a, "audio/wav"),
                            "image": (Path(reference_image).name, i, "image/png"),
                        }
                        r = await client.post(f"{url}/sadtalker/animate", files=files)
                if r.status_code != 200:
                    raise RuntimeError(f"worker {label} returned {r.status_code}: {r.text[:200]}")
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as out:
                    out.write(r.content)
                logger.info(f"SadTalker via {label} → {output_path} ({len(r.content)} bytes)")
                return output_path
            except Exception as e:
                logger.warning(f"SadTalker worker {label} ({url}) failed: {e}")
                last_err = e
        raise RuntimeError(f"No SadTalker worker reachable. Last error: {last_err}")
