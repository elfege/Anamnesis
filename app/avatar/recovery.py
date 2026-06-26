"""Self-healing for known backend wedges.

When `_stream_ollama` detects an endpoint that accepts TCP but never sends
response headers (the classic ollama-runner-zombie pattern that hangs
`/api/chat` while `/api/tags` keeps returning 200), it fires
`try_heal(endpoint_url, reason)` fire-and-forget. The current user request
falls through to the next endpoint as usual — recovery doesn't block it. The
NEXT request gets the freshly-restarted endpoint.

Hardcoded plan registry: keyed by the host:port we'd see in OLLAMA_ENDPOINTS.
Each plan is the SSH target + identity file + service name. Identity file
paths assume the standard dellserver `~/.ssh` mount into the anamnesis-app
container (see docker-compose.yml `SSH_DIR`).

Cooldown prevents thrashing: a given endpoint can only be healed once per
`COOLDOWN_S`. If the wedge keeps recurring after a fresh restart, the cause
is GPU-level (ROCm contention from avatar worker, etc.) and a service-level
restart won't fix it — leave it to surface in the debug terminal.
"""
import asyncio
import logging
import time
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger("anamnesis.avatar.recovery")

COOLDOWN_S = 180  # 3 min — long enough that we don't restart on every poll round

# Map host:port → recovery plan. Keys must match what comes out of
# urlparse(OLLAMA_URL_N).netloc so lookup is exact.
RECOVERY_PLANS: dict[str, dict] = {
    "192.168.10.110:11434": {
        "label": "office ollama",
        "ssh_target": "elfege@192.168.10.110",
        "identity_file": "/root/.ssh/id_rsa_server_home_elfege",
        "remote_cmd": "sudo -n systemctl restart ollama",
    },
    "192.168.10.15:11434": {
        "label": "server ollama",
        "ssh_target": "elfege@192.168.10.15",
        "identity_file": "/root/.ssh/id_rsa_dellserver",
        "remote_cmd": "sudo -n systemctl restart ollama",
    },
}

# Per-endpoint last-attempt timestamp (monotonic seconds). Module-level so it
# survives across requests within the same anamnesis-app process.
_last_attempt: dict[str, float] = {}
_lock = asyncio.Lock()


def _endpoint_key(endpoint_url: str) -> str:
    """Pull host:port out of any URL form."""
    p = urlparse(endpoint_url if "://" in endpoint_url else f"http://{endpoint_url}")
    return p.netloc


async def try_heal(endpoint_url: str, reason: str) -> Optional[bool]:
    """Fire-and-forget; returns True/False/None.

    None     = no plan for this endpoint, or in cooldown — recovery NOT attempted.
    True     = SSH restart command exited 0.
    False    = SSH or systemctl returned non-zero — surfaced in logs.

    Always non-blocking from the caller's perspective when spawned via
    `asyncio.create_task(try_heal(...))`. Internal SSH is itself bounded by
    a 15 s timeout.
    """
    key = _endpoint_key(endpoint_url)
    plan = RECOVERY_PLANS.get(key)
    if not plan:
        logger.info(f"No recovery plan for {key} (reason: {reason}) — skipping")
        return None

    async with _lock:
        now = asyncio.get_running_loop().time()
        last = _last_attempt.get(key, 0)
        if now - last < COOLDOWN_S:
            wait = COOLDOWN_S - (now - last)
            logger.info(f"Recovery for {plan['label']} in cooldown ({wait:.0f}s left) — skipping (reason: {reason})")
            return None
        _last_attempt[key] = now

    logger.warning(f"AUTO-HEAL triggered for {plan['label']} — reason: {reason}")

    cmd = [
        "ssh",
        "-F", "/dev/null",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-i", plan["identity_file"],
        plan["ssh_target"],
        plan["remote_cmd"],
    ]

    t0 = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # ollama systemctl restart can take 30-60s because the daemon waits for
        # the runner subprocess to flush several GB of VRAM during shutdown.
        # 90s ceiling — generous but bounded.
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=90.0)
        except asyncio.TimeoutError:
            try: proc.kill()
            except Exception: pass
            logger.error(f"AUTO-HEAL {plan['label']}: ssh timed out after 90s")
            return False
        dt = time.monotonic() - t0
        if proc.returncode == 0:
            logger.warning(f"AUTO-HEAL {plan['label']}: restarted successfully in {dt:.1f}s")
            return True
        else:
            err = (stderr.decode(errors="replace") or stdout.decode(errors="replace")).strip()[:300]
            logger.error(f"AUTO-HEAL {plan['label']}: failed rc={proc.returncode} in {dt:.1f}s — {err}")
            return False
    except Exception as exc:
        logger.error(f"AUTO-HEAL {plan['label']}: exception {exc!r}")
        return False
