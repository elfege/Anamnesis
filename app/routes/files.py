import os
import stat
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("anamnesis.routes.files")
router = APIRouter(tags=["files"])

# ─── Mounted sources ──────────────────────────────────────────────
MOUNTED_SOURCES: dict[str, str] = {
    "dellserver": "/sources/dellserver",
    "server":     "/sources/server",
    "officewsl":  "/sources/officewsl",
    "hvtmc":      "/sources/hvtmc",
    "teachings":  "/sources/teachings",
    "documents":  "/sources/documents",
}

# ─── SSH hosts ────────────────────────────────────────────────────
# source id → (hostname_or_ip, username)
SSH_HOSTS: dict[str, tuple[str, str]] = {
    "ssh:server":    ("server",              "elfege"),
    "ssh:officewsl": ("officewsl",           "elfege"),
    "ssh:dellserver":("host.docker.internal","elfege"),
    "ssh:hvtmc":     ("10.200.50.76",        "elfege"),
}

SSH_CONFIG_PATH = "/root/.ssh/config"
SSH_KEY_CANDIDATES = [
    "/root/.ssh/id_ed25519",
    "/root/.ssh/id_rsa",
]

MAX_FILE_SIZE = 200 * 1024  # 200 KB


# ─── Path helpers ─────────────────────────────────────────────────

def _sanitize(path: str) -> str:
    """Collapse .. and ensure absolute."""
    parts: list[str] = []
    for p in ("/" + path.lstrip("/")).split("/"):
        if p == "..":
            if parts:
                parts.pop()
        elif p and p != ".":
            parts.append(p)
    return "/" + "/".join(parts)


def _find_key() -> Optional[str]:
    for k in SSH_KEY_CANDIDATES:
        if os.path.exists(k):
            return k
    return None


# ─── Local filesystem helpers ─────────────────────────────────────

def ls_local(base: str, path: str) -> list[dict]:
    full = base + _sanitize(path)
    try:
        entries = []
        with os.scandir(full) as it:
            for e in it:
                try:
                    s = e.stat(follow_symlinks=False)
                    entries.append({
                        "name": e.name,
                        "type": "dir" if e.is_dir(follow_symlinks=True) else "file",
                        "size": s.st_size,
                        "modified": datetime.fromtimestamp(s.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    })
                except (PermissionError, OSError):
                    continue
        entries.sort(key=lambda e: (e["type"] == "file", e["name"].lower()))
        return entries
    except PermissionError:
        raise HTTPException(403, "Permission denied")
    except FileNotFoundError:
        raise HTTPException(404, "Path not found")
    except NotADirectoryError:
        raise HTTPException(400, "Not a directory")


def cat_local(base: str, path: str) -> dict:
    full = base + _sanitize(path)
    try:
        size = os.path.getsize(full)
        truncated = size > MAX_FILE_SIZE
        with open(full, "r", errors="replace") as f:
            content = f.read(MAX_FILE_SIZE)
        return {"content": content, "size": size, "truncated": truncated, "path": path}
    except PermissionError:
        raise HTTPException(403, "Permission denied")
    except FileNotFoundError:
        raise HTTPException(404, "File not found")
    except IsADirectoryError:
        raise HTTPException(400, "Is a directory")


# ─── SSH helpers ──────────────────────────────────────────────────

def _ssh_client(host: str, user: str):
    try:
        import paramiko
    except ImportError:
        raise HTTPException(503, "paramiko not installed — rebuild container")

    # Honour ~/.ssh/config for HostName / Port / User / IdentityFile
    cfg = paramiko.SSHConfig()
    if os.path.exists(SSH_CONFIG_PATH):
        with open(SSH_CONFIG_PATH) as f:
            cfg.parse(f)
    hc = cfg.lookup(host)
    real_host = hc.get("hostname", host)
    real_port = int(hc.get("port", 22))
    real_user = hc.get("user", user)

    # Prefer IdentityFile from SSH config, fall back to generic candidates
    config_key = hc.get("identityfile", [])
    if isinstance(config_key, str):
        config_key = [config_key]
    # Expand ~ to /root (container root user)
    config_key = [k.replace("~", "/root") for k in config_key]
    key_path = next((k for k in config_key if os.path.exists(k)), None) or _find_key()
    if not key_path:
        raise HTTPException(503, "No SSH key found at /root/.ssh/")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            real_host, port=real_port, username=real_user,
            key_filename=key_path, timeout=8, banner_timeout=8,
        )
    except Exception as exc:
        raise HTTPException(503, f"SSH connect failed: {exc}")
    return client


def ls_ssh(host: str, user: str, path: str) -> list[dict]:
    path = _sanitize(path) or "/"
    client = _ssh_client(host, user)
    try:
        sftp = client.open_sftp()
        entries = []
        for attr in sftp.listdir_attr(path):
            is_dir = bool(attr.st_mode and stat.S_ISDIR(attr.st_mode))
            entries.append({
                "name": attr.filename,
                "type": "dir" if is_dir else "file",
                "size": attr.st_size or 0,
                "modified": datetime.fromtimestamp(attr.st_mtime).strftime("%Y-%m-%d %H:%M")
                            if attr.st_mtime else "",
            })
        sftp.close()
        entries.sort(key=lambda e: (e["type"] == "file", e["name"].lower()))
        return entries
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"SSH ls error: {exc}")
    finally:
        client.close()


def cat_ssh(host: str, user: str, path: str) -> dict:
    path = _sanitize(path)
    client = _ssh_client(host, user)
    try:
        sftp = client.open_sftp()
        attr = sftp.stat(path)
        size = attr.st_size or 0
        truncated = size > MAX_FILE_SIZE
        with sftp.open(path, "r") as f:
            raw = f.read(MAX_FILE_SIZE)
        content = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        sftp.close()
        return {"content": content, "size": size, "truncated": truncated, "path": path}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"SSH cat error: {exc}")
    finally:
        client.close()


# ─── Public dispatch ──────────────────────────────────────────────

def ls(path: str, source: str) -> list[dict]:
    if source in MOUNTED_SOURCES:
        return ls_local(MOUNTED_SOURCES[source], path)
    if source in SSH_HOSTS:
        host, user = SSH_HOSTS[source]
        return ls_ssh(host, user, path)
    raise HTTPException(400, f"Unknown source: {source!r}")


def cat(path: str, source: str) -> dict:
    if source in MOUNTED_SOURCES:
        return cat_local(MOUNTED_SOURCES[source], path)
    if source in SSH_HOSTS:
        host, user = SSH_HOSTS[source]
        return cat_ssh(host, user, path)
    raise HTTPException(400, f"Unknown source: {source!r}")


# ─── API endpoints ────────────────────────────────────────────────

@router.get("/api/files/sources")
async def list_sources():
    result = []
    for sid, base in MOUNTED_SOURCES.items():
        result.append({
            "id": sid, "label": sid, "type": "mounted",
            "available": os.path.exists(base),
        })
    for sid in SSH_HOSTS:
        result.append({"id": sid, "label": sid, "type": "ssh", "available": True})
    return result


@router.get("/api/files/ls")
async def api_ls(path: str = "/", source: str = "dellserver"):
    return ls(path, source)


@router.get("/api/files/cat")
async def api_cat(path: str, source: str = "dellserver"):
    return cat(path, source)
