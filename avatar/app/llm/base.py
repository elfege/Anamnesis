"""Abstract base class for LLM backends."""
from abc import ABC, abstractmethod
from typing import AsyncGenerator


class LLMBackend(ABC):
    """All LLM backends implement this interface."""

    @abstractmethod
    async def generate(self, prompt: str, system: str = "") -> AsyncGenerator[str, None]:
        """Stream text tokens for the given prompt."""
        ...

    @abstractmethod
    async def generate_full(self, prompt: str, system: str = "") -> str:
        """Return the complete response as a single string."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...
