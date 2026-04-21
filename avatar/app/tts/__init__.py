"""TTS backend factory."""
from tts.base import TTSBackend


def get_backend(backend_name: str) -> TTSBackend:
    if backend_name == "edge":
        from tts.edge_tts_backend import EdgeTTSBackend
        return EdgeTTSBackend()
    else:
        raise ValueError(f"Unknown TTS backend: {backend_name}")
