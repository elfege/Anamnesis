"""XTTS v2 — voice cloning from a speaker_wav reference.

Lazy-loaded, thread-safe, with explicit unload to free VRAM for SadTalker
on tight-VRAM hosts (e.g. GTX 1660 SUPER 6GB where XTTS + SadTalker don't
co-fit).
"""
import asyncio
import gc
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger("worker.xtts")

_model = None
_lock = threading.Lock()


def is_loaded() -> bool:
    return _model is not None


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


def unload() -> bool:
    """Drop the XTTS model from VRAM. Next synthesize() call pays the reload cost.

    Returns True if a model was unloaded, False if nothing was loaded.
    Thread-safe.
    """
    global _model
    if _model is None:
        return False
    with _lock:
        if _model is None:
            return False
        logger.info("Unloading XTTS to free VRAM")
        try:
            import torch
            _model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Partial unload (torch cleanup failed): {e}")
        return True


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
