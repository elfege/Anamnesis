"""Dispatches a voice_spec to the right TTS backend (edge in-process or XTTS via worker)."""
from typing import Optional

from avatar.tts.base import TTSBackend
from avatar.tts.edge import EdgeTTSBackend
from avatar.tts.xtts_client import XTTSClient

_cache: dict[str, TTSBackend] = {}


def _get_backend(name: str) -> TTSBackend:
    if name not in _cache:
        if name == "edge":
            _cache[name] = EdgeTTSBackend()
        elif name == "xtts":
            _cache[name] = XTTSClient()
        else:
            raise ValueError(f"Unknown TTS backend: {name}")
    return _cache[name]


async def synthesize_with_voice(
    voice_spec: dict, text: str, output_path: str,
    preferred_worker: Optional[str] = None, no_fallback: bool = False,
) -> str:
    backend_name = voice_spec.get("backend", "edge")
    backend = _get_backend(backend_name)
    # Only worker-based backends (xtts) take the preferred_worker/no_fallback kwargs.
    # Edge TTS runs in-process and ignores them.
    if backend_name == "xtts":
        return await backend.synthesize_to_file(
            text, output_path, voice_spec,
            preferred_worker=preferred_worker, no_fallback=no_fallback,
        )
    return await backend.synthesize_to_file(text, output_path, voice_spec)
