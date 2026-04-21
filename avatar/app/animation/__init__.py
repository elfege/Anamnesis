"""Animation backend factory."""
from animation.base import AnimationBackend


def get_backend(backend_name: str) -> AnimationBackend:
    if backend_name == "liveportrait":
        from animation.liveportrait import LivePortraitBackend
        return LivePortraitBackend()
    elif backend_name == "none":
        from animation.noop import NoopBackend
        return NoopBackend()
    else:
        raise ValueError(f"Unknown animation backend: {backend_name}")
