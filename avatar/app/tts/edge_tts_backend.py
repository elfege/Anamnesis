"""Edge TTS backend — uses Microsoft Edge's free TTS service.

No GPU required, surprisingly high quality, many voices available.
pip install edge-tts
"""
import io
import logging
import tempfile
from typing import Optional

import edge_tts

import config
from tts.base import TTSBackend

logger = logging.getLogger("avatar.tts.edge")


class EdgeTTSBackend(TTSBackend):

    def __init__(self):
        self._voice = config.TTS_VOICE
        self._rate = config.TTS_RATE

    @property
    def name(self) -> str:
        return f"edge-tts/{self._voice}"

    @property
    def sample_rate(self) -> int:
        return 24000  # Edge TTS outputs 24kHz audio

    async def synthesize(self, text: str) -> bytes:
        """Generate audio bytes from text."""
        communicate = edge_tts.Communicate(text, self._voice, rate=self._rate)
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)

    async def synthesize_to_file(self, text: str, output_path: str) -> str:
        """Generate audio and save to file."""
        communicate = edge_tts.Communicate(text, self._voice, rate=self._rate)
        await communicate.save(output_path)
        logger.info(f"TTS audio saved to {output_path}")
        return output_path
