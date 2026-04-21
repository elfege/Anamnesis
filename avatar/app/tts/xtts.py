"""XTTS v2 backend — voice cloning from a short audio sample (~6-10s).

Uses Coqui's XTTS v2 via the maintained `coqui-tts` fork.
Model downloads to HF cache on first use (~1.8GB).

voice_spec:
    {"speaker_wav": "/app/voices/belle.wav", "language": "en"}
"""
import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import config
from tts.base import TTSBackend

logger = logging.getLogger("avatar.tts.xtts")


class XTTSBackend(TTSBackend):

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._model_name = config.XTTS_MODEL_NAME
        # XTTS v2 outputs 24 kHz mono WAV
        self._sample_rate = 24000

    @property
    def name(self) -> str:
        return f"xtts-v2"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _ensure_loaded(self):
        """Lazy-load XTTS to avoid paying startup cost when only edge-tts is used."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            logger.info(f"Loading XTTS model: {self._model_name}")
            # Import here so the dep is optional when only edge-tts is used
            from TTS.api import TTS as CoquiTTS

            # License agreement is required for XTTS v2 — set env to auto-accept.
            os.environ.setdefault("COQUI_TOS_AGREED", "1")

            import torch
            gpu = torch.cuda.is_available()
            self._model = CoquiTTS(self._model_name).to("cuda" if gpu else "cpu")
            logger.info(f"XTTS loaded on {'GPU' if gpu else 'CPU'}")

    async def synthesize_to_file(
        self, text: str, output_path: str, voice_spec: Optional[dict] = None
    ) -> str:
        spec = voice_spec or {}
        speaker_wav = spec.get("speaker_wav")
        language = spec.get("language", "en")

        if not speaker_wav or not Path(speaker_wav).exists():
            raise RuntimeError(
                f"XTTSBackend requires a speaker_wav reference; got: {speaker_wav}"
            )

        # XTTS writes WAV; if caller wants .mp3 we convert after.
        wav_out = output_path
        want_mp3 = output_path.lower().endswith(".mp3")
        if want_mp3:
            wav_out = output_path.rsplit(".", 1)[0] + ".wav"

        # The TTS API is sync and heavy — run in a thread.
        def _run():
            self._ensure_loaded()
            self._model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=wav_out,
            )

        await asyncio.to_thread(_run)

        if want_mp3:
            await asyncio.to_thread(_wav_to_mp3, wav_out, output_path)
            Path(wav_out).unlink(missing_ok=True)

        logger.info(f"XTTS → {output_path} (speaker={Path(speaker_wav).name}, lang={language})")
        return output_path


def _wav_to_mp3(wav_path: str, mp3_path: str):
    """Convert WAV to MP3 via ffmpeg (already in the image)."""
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-q:a", "2", mp3_path],
        check=True,
        capture_output=True,
    )
