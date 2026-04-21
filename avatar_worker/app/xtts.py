"""XTTS v2 — voice cloning from a speaker_wav reference."""
import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger("worker.xtts")

_model = None
_lock = threading.Lock()


def _ensure_loaded():
    global _model
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        logger.info(f"Loading XTTS: {config.XTTS_MODEL_NAME}")
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS as CoquiTTS
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = CoquiTTS(config.XTTS_MODEL_NAME).to(device)
        logger.info(f"XTTS loaded on {device}")


async def synthesize(
    text: str,
    speaker_wav_path: str,
    output_wav_path: str,
    language: str = "en",
) -> str:
    def _run():
        _ensure_loaded()
        _model.tts_to_file(
            text=text,
            speaker_wav=speaker_wav_path,
            language=language,
            file_path=output_wav_path,
        )
    await asyncio.to_thread(_run)
    logger.info(f"XTTS → {output_wav_path} (lang={language}, spk={Path(speaker_wav_path).name})")
    return output_wav_path
