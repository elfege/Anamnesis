"""No-op animation backend — returns audio only, no face animation.

Useful for testing the LLM + TTS pipeline without needing a GPU for animation.
"""
import logging

from animation.base import AnimationBackend

logger = logging.getLogger("avatar.animation.noop")


class NoopBackend(AnimationBackend):

    @property
    def name(self) -> str:
        return "none"

    async def is_ready(self) -> bool:
        return True

    async def animate(self, audio_path: str, reference_image: str) -> str:
        """No animation — just return the audio path as-is."""
        logger.info("NoopBackend: skipping animation, returning audio only")
        return audio_path
