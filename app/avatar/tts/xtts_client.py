"""XTTS client — POSTs to the avatar_worker /xtts/synthesize endpoint.

Iterates config.AVATAR_WORKER_ENDPOINTS (fallback chain, same pattern as
OLLAMA_ENDPOINTS). The first reachable worker wins.
"""
import logging
import subprocess
from pathlib import Path
from typing import Optional

import httpx

import config
from avatar.tts.base import TTSBackend

logger = logging.getLogger("anamnesis.avatar.tts.xtts")


class XTTSClient(TTSBackend):

    @property
    def name(self) -> str:
        return "xtts-v2@worker"

    @property
    def sample_rate(self) -> int:
        return 24000

    async def synthesize_to_file(
        self, text: str, output_path: str, voice_spec: Optional[dict] = None
    ) -> str:
        spec = voice_spec or {}
        speaker_wav = spec.get("speaker_wav")
        language = spec.get("language", "en")
        want_mp3 = output_path.lower().endswith(".mp3")

        if not speaker_wav or not Path(speaker_wav).exists():
            raise RuntimeError(f"XTTSClient needs an existing speaker_wav; got: {speaker_wav}")

        last_err = None
        for url, label in config.AVATAR_WORKER_ENDPOINTS:
            try:
                async with httpx.AsyncClient(timeout=300) as client:
                    with open(speaker_wav, "rb") as f:
                        files = {"speaker": (Path(speaker_wav).name, f, "audio/wav")}
                        data = {"text": text, "language": language, "want_mp3": str(want_mp3).lower()}
                        r = await client.post(f"{url}/xtts/synthesize", data=data, files=files)
                if r.status_code != 200:
                    raise RuntimeError(f"worker {label} returned {r.status_code}: {r.text[:200]}")
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as out:
                    out.write(r.content)
                logger.info(f"XTTS via {label} → {output_path} ({len(r.content)} bytes)")
                return output_path
            except Exception as e:
                logger.warning(f"XTTS worker {label} ({url}) failed: {e}")
                last_err = e
        raise RuntimeError(f"No XTTS worker reachable. Last error: {last_err}")
