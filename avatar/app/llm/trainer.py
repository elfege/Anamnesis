"""Trainer inference backend — uses the trainer /generate REST endpoint."""
import json
import logging
from typing import AsyncGenerator

import httpx

import config
from llm.base import LLMBackend

logger = logging.getLogger("avatar.llm.trainer")


class TrainerBackend(LLMBackend):

    def __init__(self):
        self._url = config.TRAINER_URL

    @property
    def name(self) -> str:
        return f"trainer@{self._url}"

    async def generate(self, prompt: str, system: str = "") -> AsyncGenerator[str, None]:
        payload = {
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.8,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{self._url}/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:].strip())
                    if data.get("done"):
                        break
                    token = data.get("token", "")
                    if token:
                        yield token

    async def generate_full(self, prompt: str, system: str = "") -> str:
        parts = []
        async for token in self.generate(prompt, system):
            parts.append(token)
        return "".join(parts)
