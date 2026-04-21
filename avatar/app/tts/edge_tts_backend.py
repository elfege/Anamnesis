"""Edge TTS backend — Microsoft Edge's free TTS service.

No GPU, surprisingly high quality, ~400 voices.
`voice_spec` allows overriding the configured voice per request:
    {"voice": "en-US-AvaNeural", "rate": "+0%"}
"""
import logging
from typing import Optional

import edge_tts

import config
from tts.base import TTSBackend

logger = logging.getLogger("avatar.tts.edge")


class EdgeTTSBackend(TTSBackend):

    def __init__(self):
        self._default_voice = config.TTS_VOICE
        self._default_rate = config.TTS_RATE

    @property
    def name(self) -> str:
        return f"edge-tts/{self._default_voice}"

    @property
    def sample_rate(self) -> int:
        return 24000

    async def synthesize_to_file(
        self, text: str, output_path: str, voice_spec: Optional[dict] = None
    ) -> str:
        voice = (voice_spec or {}).get("voice") or self._default_voice
        rate = (voice_spec or {}).get("rate") or self._default_rate
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_path)
        logger.info(f"Edge TTS → {output_path} (voice={voice})")
        return output_path
