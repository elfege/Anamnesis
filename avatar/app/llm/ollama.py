"""Ollama LLM backend — uses the Ollama REST API."""
import json
import logging
from typing import AsyncGenerator

import httpx

import config
from llm.base import LLMBackend

logger = logging.getLogger("avatar.llm.ollama")


class OllamaBackend(LLMBackend):

    def __init__(self):
        self._url = config.OLLAMA_URL
        self._model = config.OLLAMA_MODEL

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    async def generate(self, prompt: str, system: str = "") -> AsyncGenerator[str, None]:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{self._url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

    async def generate_full(self, prompt: str, system: str = "") -> str:
        parts = []
        async for token in self.generate(prompt, system):
            parts.append(token)
        return "".join(parts)
