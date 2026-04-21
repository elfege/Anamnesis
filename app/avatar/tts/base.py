"""Abstract base class for TTS backends."""
from abc import ABC, abstractmethod
from typing import Optional


class TTSBackend(ABC):
    """All TTS backends implement this interface.

    `voice_spec` is an optional dict with backend-specific fields. Each backend
    ignores fields it doesn't understand and falls back to its configured defaults.

    Edge TTS:   {"voice": "en-US-AvaNeural", "rate": "+0%"}
    XTTS v2:    {"speaker_wav": "/app/voices/belle.wav", "language": "en"}
    """

    @abstractmethod
    async def synthesize_to_file(
        self, text: str, output_path: str, voice_spec: Optional[dict] = None
    ) -> str:
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
