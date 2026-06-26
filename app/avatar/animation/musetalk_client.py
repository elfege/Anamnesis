"""MuseTalk client — POSTs to the avatar_worker /musetalk/animate endpoint.

Mirrors sadtalker_client.py exactly. MuseTalk is Tencent's real-time lip-sync
model — drops the SadTalker 60-120s post-render to ~real-time on a decent GPU,
which lets the avatar UI play audio + video in sync without the deferred-audio
30-120s wait.

Install state (2026-06-08): worker-side install in progress via spawned Agent.
This client expects /musetalk/animate to exist on at least one avatar worker
with capability "musetalk" advertised in /health. Until that lands, callers
will see the "No MuseTalk worker reachable" error — by design, so the user
can see the picker option exists but the engine isn't ready yet.
"""
import logging
from pathlib import Path
from typing import Optional

import httpx

import config
from avatar.workers import order_endpoints as _order_endpoints

logger = logging.getLogger("anamnesis.avatar.musetalk")


class MuseTalkClient:

    @property
    def name(self) -> str:
        return "musetalk@worker"

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
                        r = await client.post(f"{url}/musetalk/animate", files=files)
                if r.status_code != 200:
                    # Tolerate 404 — that worker doesn't have the endpoint yet.
                    # Surface it as a normal failure so order_endpoints falls
                    # through to the next worker.
                    raise RuntimeError(f"worker {label} returned {r.status_code}: {r.text[:200]}")
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as out:
                    out.write(r.content)
                logger.info(f"MuseTalk via {label} → {output_path} ({len(r.content)} bytes)")
                return output_path
            except Exception as e:
                logger.warning(f"MuseTalk worker {label} ({url}) failed: {e}")
                last_err = e
        raise RuntimeError(f"No MuseTalk worker reachable. Last error: {last_err}")

    async def cancel_all(self) -> dict:
        """Fire-and-forget cancel to every configured worker. Matches
        SadTalkerClient — the worker's MuseTalk subprocess is detached from
        the HTTP request lifecycle and would otherwise keep occupying the
        GPU until it finishes."""
        results: dict[str, str] = {}
        for url, label in config.AVATAR_WORKER_ENDPOINTS:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    r = await client.post(f"{url}/musetalk/cancel")
                results[label] = (
                    f"killed={r.json().get('killed', 0)}"
                    if r.status_code == 200
                    else f"http {r.status_code}"
                )
            except Exception as e:
                results[label] = f"error: {e}"
        return results
