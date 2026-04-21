"""d² model backend — stub for when d² can do inference.

This will load the d² transformer directly and run local inference.
Requires: trained checkpoint, tokenizer, and the d2 neural_network module.
"""
import logging
from typing import AsyncGenerator

import config
from llm.base import LLMBackend

logger = logging.getLogger("avatar.llm.d2")


class D2Backend(LLMBackend):

    def __init__(self):
        self._model_path = config.D2_MODEL_PATH
        self._checkpoint = config.D2_CHECKPOINT
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return f"d2/{self._checkpoint or 'no-checkpoint'}"

    def _load(self):
        if self._model is not None:
            return
        if not self._model_path or not self._checkpoint:
            raise RuntimeError(
                "d² backend requires D2_MODEL_PATH and D2_CHECKPOINT env vars. "
                "The d² model must complete training before this backend is usable."
            )
        # TODO: import d2.neural_network, load checkpoint, set up tokenizer
        raise NotImplementedError(
            "d² inference not yet implemented — finish the d² training loop first"
        )

    async def generate(self, prompt: str, system: str = "") -> AsyncGenerator[str, None]:
        self._load()
        # Future: tokenize prompt, run model.generate(), yield tokens
        raise NotImplementedError

    async def generate_full(self, prompt: str, system: str = "") -> str:
        self._load()
        raise NotImplementedError
