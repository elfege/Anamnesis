"""Demucs client — POSTs to the avatar_worker /demucs/extract endpoint."""
import logging
from pathlib import Path

import httpx

import config

logger = logging.getLogger("anamnesis.avatar.demucs")


async def extract_vocals_via_worker(
    song_path: str, out_path: str, trim_seconds: float = 15.0
) -> str:
    """Call the worker's /demucs/extract with the song file.

    Returns the local path (out_path) containing the vocals WAV written by the
    worker response. The trim/silence-remove is applied by the worker.
    """
    last_err = None
    for url, label in config.AVATAR_WORKER_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                with open(song_path, "rb") as f:
                    files = {"audio": (Path(song_path).name, f, "audio/mpeg")}
                    data = {"trim_seconds": str(trim_seconds)}
                    r = await client.post(f"{url}/demucs/extract", data=data, files=files)
            if r.status_code != 200:
                raise RuntimeError(f"worker {label} returned {r.status_code}: {r.text[:200]}")
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as out:
                out.write(r.content)
            logger.info(f"Demucs via {label} → {out_path} ({len(r.content)} bytes)")
            return out_path
        except Exception as e:
            logger.warning(f"Demucs worker {label} ({url}) failed: {e}")
            last_err = e
    raise RuntimeError(f"No Demucs worker reachable. Last error: {last_err}")
