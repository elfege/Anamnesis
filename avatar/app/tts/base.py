"""Abstract base class for TTS backends."""
from abc import ABC, abstractmethod
from typing import Optional


class TTSBackend(ABC):
    """All TTS backends implement this interface."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (MP3 format)."""
        ...

    @abstractmethod
    async def synthesize_to_file(self, text: str, output_path: str) -> str:
        """Convert text to audio file. Returns the file path."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        ...
