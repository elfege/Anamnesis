"""Claude API LLM backend — uses the Anthropic SDK."""
import logging
from typing import AsyncGenerator

import config
from llm.base import LLMBackend

logger = logging.getLogger("avatar.llm.claude")


class ClaudeBackend(LLMBackend):

    def __init__(self):
        self._model = config.CLAUDE_MODEL
        self._api_key = config.ANTHROPIC_API_KEY
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

    @property
    def name(self) -> str:
        return f"claude/{self._model}"

    async def generate(self, prompt: str, system: str = "") -> AsyncGenerator[str, None]:
        self._ensure_client()
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=512,
            system=system or config.PERSONA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def generate_full(self, prompt: str, system: str = "") -> str:
        self._ensure_client()
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system or config.PERSONA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
