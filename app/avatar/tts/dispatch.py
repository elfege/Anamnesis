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


async def synthesize_with_voice(voice_spec: dict, text: str, output_path: str) -> str:
    backend = _get_backend(voice_spec.get("backend", "edge"))
    return await backend.synthesize_to_file(text, output_path, voice_spec)
