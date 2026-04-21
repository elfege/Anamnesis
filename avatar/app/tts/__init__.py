"""TTS backend factory + cross-backend synthesis helper."""
from typing import Optional

from tts.base import TTSBackend


def get_backend(backend_name: str) -> TTSBackend:
    if backend_name == "edge":
        from tts.edge_tts_backend import EdgeTTSBackend
        return EdgeTTSBackend()
    if backend_name == "xtts":
        from tts.xtts import XTTSBackend
        return XTTSBackend()
    raise ValueError(f"Unknown TTS backend: {backend_name}")


# ─── Backend cache — keep XTTS loaded across requests ────────────

_cache: dict = {}


def get_cached(backend_name: str) -> TTSBackend:
    if backend_name not in _cache:
        _cache[backend_name] = get_backend(backend_name)
    return _cache[backend_name]


async def synthesize_with_voice(
    voice_spec: dict, text: str, output_path: str
) -> str:
    """Dispatch to the right backend based on `voice_spec['backend']`."""
    backend_name = voice_spec.get("backend", "edge")
    backend = get_cached(backend_name)
    return await backend.synthesize_to_file(text, output_path, voice_spec)
