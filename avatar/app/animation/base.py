"""Abstract base class for face animation backends."""
from abc import ABC, abstractmethod
from pathlib import Path


class AnimationBackend(ABC):
    """All animation backends implement this interface."""

    @abstractmethod
    async def animate(self, audio_path: str, reference_image: str) -> str:
        """
        Generate an animated video from audio + reference image.

        Args:
            audio_path: Path to the audio file (MP3/WAV)
            reference_image: Path to the reference face image

        Returns:
            Path to the generated video file (MP4)
        """
        ...

    @abstractmethod
    async def is_ready(self) -> bool:
        """Check if the animation backend is loaded and ready."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
