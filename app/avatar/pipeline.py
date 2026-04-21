"""Avatar pipeline — LLM → TTS → Animation.

Runs in anamnesis-app on dellserver. Delegates GPU work to avatar_worker.
"""
import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import config
from avatar import voices as voices_module
from avatar import llm as avatar_llm
from avatar.tts.dispatch import synthesize_with_voice
from avatar.animation.sadtalker_client import SadTalkerClient

logger = logging.getLogger("anamnesis.avatar.pipeline")


@dataclass
class PipelineResult:
    text: str = ""
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    timings: dict = field(default_factory=dict)
    error: Optional[str] = None


class AvatarPipeline:

    def __init__(self):
        self._voices = voices_module.get_registry()
        self._sadtalker = SadTalkerClient()

    @property
    def info(self) -> dict:
        return {
            "persona": config.AVATAR_PERSONA_NAME,
            "llm": f"ollama/{config.OLLAMA_DEFAULT_MODEL}",
            "default_voice": config.DEFAULT_VOICE_ID,
            "worker_endpoints": [label for _, label in config.AVATAR_WORKER_ENDPOINTS],
            "animate_enabled": config.AVATAR_ANIMATE_DEFAULT,
        }

    async def process(
        self,
        user_message: str,
        voice_id: Optional[str] = None,
        animate: bool = None,
        reference_image: Optional[str] = None,
        on_token=None,
        on_audio=None,
        on_video=None,
    ) -> PipelineResult:
        result = PipelineResult()
        t0 = time.monotonic()
        animate = animate if animate is not None else config.AVATAR_ANIMATE_DEFAULT

        # ── Step 1: LLM ─────────────────────────────────────────
        t_llm = time.monotonic()
        parts: list[str] = []
        endpoint_label = None
        async for kind, text in avatar_llm.stream_reply(
            config.AVATAR_PERSONA_SYSTEM_PROMPT, user_message
        ):
            if kind == "token":
                parts.append(text)
                if on_token:
                    await _maybe_await(on_token, text)
            elif kind == "endpoint":
                endpoint_label = text
            elif kind == "error":
                result.error = f"LLM error: {text}"
                return result
        result.text = "".join(parts).strip()
        result.timings["llm_ms"] = int((time.monotonic() - t_llm) * 1000)
        if endpoint_label:
            result.timings["llm_endpoint"] = endpoint_label
        if not result.text:
            result.error = "LLM returned empty"
            return result
        logger.info(f"LLM done: {len(result.text)} chars in {result.timings['llm_ms']}ms")

        # Unload Ollama model so GPU is free for worker (XTTS + SadTalker)
        if animate or voice_id and voice_id.startswith("cloned:"):
            await avatar_llm.unload_model()

        # ── Step 2: TTS ────────────────────────────────────────
        t_tts = time.monotonic()
        audio_dir = tempfile.mkdtemp(prefix="anamnesis_avatar_")
        audio_path = os.path.join(audio_dir, "speech.mp3")
        voice_spec = self._voices.resolve(voice_id)
        try:
            result.audio_path = await synthesize_with_voice(
                voice_spec, result.text, audio_path
            )
            result.timings["tts_ms"] = int((time.monotonic() - t_tts) * 1000)
            result.timings["voice_id"] = voice_id or config.DEFAULT_VOICE_ID
            if on_audio:
                await _maybe_await(on_audio, result.audio_path)
        except Exception as e:
            logger.exception("TTS failed")
            result.error = f"TTS error: {e}"
            return result

        # ── Step 3: Animation ─────────────────────────────────
        if animate:
            ref = reference_image or config.AVATAR_REFERENCE_IMAGE
            if not ref or not os.path.exists(ref):
                logger.warning(f"Reference image missing: {ref} — skipping animation")
                result.timings["anim_skipped"] = "no_reference_image"
            else:
                t_anim = time.monotonic()
                video_path = os.path.join(audio_dir, "anim.mp4")
                try:
                    # SadTalker needs WAV, not MP3
                    wav_for_anim = os.path.join(audio_dir, "speech_for_anim.wav")
                    await asyncio.to_thread(_mp3_to_wav, result.audio_path, wav_for_anim)
                    result.video_path = await self._sadtalker.animate(
                        wav_for_anim, ref, video_path
                    )
                    result.timings["anim_ms"] = int((time.monotonic() - t_anim) * 1000)
                    if on_video:
                        await _maybe_await(on_video, result.video_path)
                except Exception as e:
                    logger.warning(f"Animation failed (non-fatal): {e}")
                    result.timings["anim_error"] = str(e)

        result.timings["total_ms"] = int((time.monotonic() - t0) * 1000)
        return result


async def _maybe_await(fn, *args):
    ret = fn(*args)
    if asyncio.iscoroutine(ret):
        await ret


def _mp3_to_wav(src_mp3: str, dst_wav: str, sample_rate: int = 16000):
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_mp3, "-ac", "1", "-ar", str(sample_rate), "-acodec", "pcm_s16le", dst_wav],
        check=True,
        capture_output=True,
    )


# Singleton
_pipeline: Optional[AvatarPipeline] = None


def get_pipeline() -> AvatarPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AvatarPipeline()
    return _pipeline
