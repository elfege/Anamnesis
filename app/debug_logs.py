"""In-memory ring buffer + logging handler powering the Avatar Debug terminal.

Lets the browser tail what would otherwise live only in `docker logs anamnesis-app`
without paying for filesystem I/O or shelling out to the docker socket.

Sequence-numbered so the frontend can request only the deltas since its last poll.
"""
from collections import deque
from threading import Lock
import logging
import time

_MAX_LINES = 2000

_buffer: deque = deque(maxlen=_MAX_LINES)
_lock = Lock()
_next_seq = 0


# Noisy log sources that drown the Debug terminal but tell the operator nothing
# they can act on. Crawler PDF/docx parser warnings fire once per page, easily
# 1000/min during a crawl pass. Filter at the handler so they never enter the
# ring buffer in the first place.
_NAME_EXCLUDES = (
    "pdfminer.",          # CropBox / colorspace / font warnings — pure noise
    "anamnesis.crawler",  # corrupt-file warnings — operator can't fix the files
)


class RingBufferHandler(logging.Handler):
    """Append each formatted log record to the shared ring buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        global _next_seq
        if any(record.name.startswith(p) for p in _NAME_EXCLUDES):
            return
        try:
            msg = self.format(record)
        except Exception:
            return
        with _lock:
            seq = _next_seq
            _next_seq = seq + 1
            _buffer.append({
                "seq": seq,
                "ts": time.time(),
                "level": record.levelname,
                "name": record.name,
                "msg": msg,
            })


def install(level: int = logging.INFO, root: logging.Logger = None) -> None:
    """Attach the handler to the root logger so every logger we care about funnels in.

    Idempotent: re-running (e.g., after uvicorn --reload) won't double-attach.
    """
    root = root or logging.getLogger()
    for h in root.handlers:
        if isinstance(h, RingBufferHandler):
            return
    h = RingBufferHandler(level=level)
    h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    root.addHandler(h)


def fetch(since_seq: int = -1, limit: int = 500) -> dict:
    """Return entries with seq > since_seq, oldest first, up to `limit`.

    Frontend passes the seq of the last line it received; we return the next batch
    plus the new high-water mark so the next poll can resume.
    """
    with _lock:
        snapshot = list(_buffer)
        head = _next_seq
    if since_seq < 0:
        # Initial fetch — give the last `limit` lines so the panel boots populated.
        delta = snapshot[-limit:]
    else:
        delta = [e for e in snapshot if e["seq"] > since_seq][:limit]
    return {"head": head, "entries": delta}
