"""LLM backend factory — returns the right backend based on config."""
from llm.base import LLMBackend


def get_backend(backend_name: str) -> LLMBackend:
    """Factory: return an LLM backend instance by name."""
    if backend_name == "ollama":
        from llm.ollama import OllamaBackend
        return OllamaBackend()
    elif backend_name == "claude":
        from llm.claude import ClaudeBackend
        return ClaudeBackend()
    elif backend_name == "trainer":
        from llm.trainer import TrainerBackend
        return TrainerBackend()
    elif backend_name == "d2":
        from llm.d2 import D2Backend
        return D2Backend()
    else:
        raise ValueError(f"Unknown LLM backend: {backend_name}")
