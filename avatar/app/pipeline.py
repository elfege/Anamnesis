"""
Pipeline orchestrator — ties LLM, TTS, and animation together.

Flow:
  User message → LLM (streaming text) → collect full response
                                       → TTS (text → audio)
                                       → Animation (audio + face → video)
                                       → Return video + text to client
"""
import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import config
import llm
import tts
import animation

logger = logging.getLogger("avatar.pipeline")


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""
    text: str = ""
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    timings: dict = field(default_factory=dict)
    error: Optional[str] = None


class AvatarPipeline:
    """Orchestrates LLM → TTS → Animation."""

    def __init__(self):
        self._llm = llm.get_backend(config.LLM_BACKEND)
        self._tts = tts.get_backend(config.TTS_BACKEND)
        self._anim = animation.get_backend(config.ANIM_BACKEND)
        self._reference_image = config.ANIM_REFERENCE_IMAGE
        logger.info(
            f"Pipeline initialized: LLM={self._llm.name}, "
            f"TTS={self._tts.name}, Animation={self._anim.name}"
        )

    @property
    def backends(self) -> dict:
        return {
            "llm": self._llm.name,
            "tts": self._tts.name,
            "animation": self._anim.name,
        }

    async def process(
        self,
        user_message: str,
        on_token=None,
        on_audio=None,
        on_video=None,
    ) -> PipelineResult:
        """
        Run the full pipeline for a user message.

        Callbacks (optional):
            on_token(str): called for each LLM token (for streaming text to UI)
            on_audio(str): called when audio is ready (path)
            on_video(str): called when video is ready (path)
        """
        result = PipelineResult()
        t0 = time.monotonic()

        # ── Step 1: LLM generates text ──────────────────────────
        try:
            t_llm = time.monotonic()
            parts = []
            async for token in self._llm.generate(
                user_message,
                system=config.PERSONA_SYSTEM_PROMPT,
            ):
                parts.append(token)
                if on_token:
                    await _maybe_await(on_token, token)

            result.text = "".join(parts)
            result.timings["llm_ms"] = int((time.monotonic() - t_llm) * 1000)
            logger.info(f"LLM done: {len(result.text)} chars in {result.timings['llm_ms']}ms")

        except Exception as e:
            result.error = f"LLM error: {e}"
            logger.error(result.error)
            return result

        if not result.text.strip():
            result.error = "LLM returned empty response"
            return result

        # ── Step 2: TTS converts text to audio ──────────────────
        try:
            t_tts = time.monotonic()
            audio_dir = tempfile.mkdtemp(prefix="avatar_audio_")
            audio_path = os.path.join(audio_dir, "speech.mp3")
            result.audio_path = await self._tts.synthesize_to_file(result.text, audio_path)
            result.timings["tts_ms"] = int((time.monotonic() - t_tts) * 1000)
            logger.info(f"TTS done: {result.timings['tts_ms']}ms")

            if on_audio:
                await _maybe_await(on_audio, result.audio_path)

        except Exception as e:
            result.error = f"TTS error: {e}"
            logger.error(result.error)
            return result

        # ── Step 3: Animation (if enabled) ──────────────────────
        if config.ANIM_BACKEND != "none":
            try:
                t_anim = time.monotonic()
                result.video_path = await self._anim.animate(
                    result.audio_path, self._reference_image
                )
                result.timings["anim_ms"] = int((time.monotonic() - t_anim) * 1000)
                logger.info(f"Animation done: {result.timings['anim_ms']}ms")

                if on_video:
                    await _maybe_await(on_video, result.video_path)

            except Exception as e:
                # Animation failure is non-fatal — we still have audio
                logger.error(f"Animation error (non-fatal): {e}")
                result.timings["anim_error"] = str(e)

        result.timings["total_ms"] = int((time.monotonic() - t0) * 1000)
        return result


async def _maybe_await(fn, *args):
    """Call fn, awaiting if it's a coroutine."""
    ret = fn(*args)
    if asyncio.iscoroutine(ret):
        await ret
