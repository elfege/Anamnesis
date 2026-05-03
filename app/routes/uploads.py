"""Direct document/URL ingestion endpoints — bypass the 5-min crawler tick.

Endpoints:
    POST /api/episodes/upload      — multipart file (PDF, .docx, .md, .txt, .rtf, .odt)
    POST /api/episodes/ingest-url  — JSON {url, tags?, title_override?}

Both reuse the crawler's parsing + summarization + embedding pipeline
(crawler._DOC_PARSERS, crawler._content_hash, crawler._ingest_section).
The crawler itself is NOT modified — only imported.

Original files are stored under /app/uploads/ (writable in container) keyed
by SHA-256 to enable dedup against re-uploads. Because /app/uploads/ is NOT
under /sources/documents/, the crawler will never re-ingest them.
"""
import hashlib
import ipaddress
import logging
import mimetypes
import os
import re
import socket
import tempfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

# Reuse crawler internals — DO NOT modify crawler.py
from crawler import (
    _DOC_PARSERS,
    _content_hash,
    _ingest_section,
    _is_already_ingested,
    _parse_pdf,
    _parse_plain_text,
)

logger = logging.getLogger("anamnesis.routes.uploads")

router = APIRouter(prefix="/api/episodes", tags=["uploads"])

# ─── Constants ──────────────────────────────────────────────────

MAX_UPLOAD_BYTES = 50 * 1024 * 1024     # 50 MB
MAX_URL_BYTES = 10 * 1024 * 1024        # 10 MB
URL_TIMEOUT_SECONDS = 30.0
MIN_TEXT_BYTES = 30                     # below this we treat extraction as failed

UPLOAD_STORE_DIR = Path("/app/uploads")
UPLOAD_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Extensions we are willing to ingest from upload OR url.
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".rtf", ".odt"}

# MIME → extension fallback when the client doesn't give a usable filename.
MIME_TO_EXT = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.oasis.opendocument.text": ".odt",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/rtf": ".rtf",
    "text/rtf": ".rtf",
    "text/html": ".html",     # only used by the URL endpoint
}


# ─── Pydantic ───────────────────────────────────────────────────

class IngestUrlRequest(BaseModel):
    url: str
    tags: Optional[list[str]] = Field(default=None)
    title_override: Optional[str] = None
    instance: str = "office-genesis"
    project: str = "manual-upload"


# ─── Helpers ─────────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    """Strip path components + dangerous chars from a filename."""
    name = os.path.basename(name or "")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[:120] or "upload"


def _short(text: str, n: int = 200) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


def _split_tags(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_by_extension(filepath: str, ext: str) -> str:
    """Dispatch to the right crawler parser by extension."""
    parser = _DOC_PARSERS.get(ext)
    if parser is None:
        return ""
    return parser(filepath) or ""


# ─── HTML extraction (stdlib fallback) ──────────────────────────

class _HtmlTextExtractor(HTMLParser):
    """Minimal HTML→text extractor.

    Strips <script>, <style>, <noscript>, <head>, attributes; collapses
    whitespace.  Tries to grab <title> separately.  Used only when the
    optional `trafilatura` package is not installed.
    """

    _SKIP = {"script", "style", "noscript", "head", "svg", "iframe", "nav", "footer", "form"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._depth_skip = 0
        self._in_title = False
        self.title: str = ""

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self._SKIP:
            self._depth_skip += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self._SKIP and self._depth_skip > 0:
            self._depth_skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}:
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._depth_skip > 0:
            return
        if self._in_title:
            self.title += data
        else:
            self._chunks.append(data)

    def get_text(self) -> str:
        text = "".join(self._chunks)
        # Collapse runs of whitespace; preserve paragraph breaks.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
        return text.strip()


def _extract_html(html_bytes: bytes) -> tuple[str, str]:
    """Return (title, text) extracted from HTML bytes.

    Prefers trafilatura when present (better main-content extraction);
    falls back to the stdlib parser above otherwise.
    """
    try:
        import trafilatura                                         # noqa: WPS433
        text = trafilatura.extract(
            html_bytes.decode("utf-8", errors="replace"),
            include_comments=False,
            include_tables=True,
        ) or ""
        # Try to get title with trafilatura's metadata helper if available.
        title = ""
        try:
            meta = trafilatura.extract_metadata(html_bytes.decode("utf-8", errors="replace"))
            if meta and getattr(meta, "title", None):
                title = meta.title or ""
        except Exception:
            pass
        if text and len(text) >= MIN_TEXT_BYTES:
            return title.strip(), text.strip()
        # fall through to stdlib if trafilatura returned nothing useful
    except ImportError:
        pass

    parser = _HtmlTextExtractor()
    try:
        parser.feed(html_bytes.decode("utf-8", errors="replace"))
    except Exception as e:
        logger.warning(f"HTML parse failed: {e}")
    return parser.title.strip(), parser.get_text()


# ─── SSRF defense ────────────────────────────────────────────────

def _is_private_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_public_url(url: str) -> str:
    """Parse + validate a URL for SSRF safety.

    Raises HTTPException(400) on any rejection.  Returns the normalized URL.
    """
    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="Missing url")
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail=f"Unsupported scheme: {scheme!r} (only http/https)")
    host = (parsed.hostname or "").lower()
    if not host:
        raise HTTPException(status_code=400, detail="URL missing host")
    # Block obvious local names outright
    if host in {"localhost", "ip6-localhost", "ip6-loopback"} or host.endswith(".localhost"):
        raise HTTPException(status_code=400, detail="URL targets a local host (blocked)")
    # Resolve all addresses; reject if ANY is private
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if scheme == "https" else 80))
    except socket.gaierror as e:
        raise HTTPException(status_code=400, detail=f"DNS resolution failed: {e}")
    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        if _is_private_ip(ip_str):
            raise HTTPException(
                status_code=400,
                detail=f"URL resolves to a private/loopback address ({ip_str}) — blocked (SSRF defense)",
            )
    return parsed.geturl()


# ─── Upload pipeline (shared) ────────────────────────────────────

async def _ingest_text(
    *,
    text: str,
    title: str,
    instance: str,
    project: str,
    extra_tags: list[str],
    content_sha256: str,
    source_label: str,
) -> dict:
    """Run the crawler ingestion path on extracted text.

    Returns a dict suitable for HTTP response.  Raises HTTPException on
    pipeline failure.
    """
    if not text or len(text) < MIN_TEXT_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"Extracted text too short ({len(text)} bytes) — parser may not support this file",
        )

    section_content = f"[Document: {title}]\n\n{text}"
    content_hash = _content_hash(section_content)

    # Dedup against prior crawler/upload runs (SHA-256 over the same content).
    if await _is_already_ingested(content_hash):
        return {
            "episode_id": None,
            "title": title,
            "summary_chars": len(section_content),
            "char_count": len(text),
            "duplicate": True,
            "message": "Content already ingested (matched by SHA-256)",
        }

    source = {
        "name": source_label,
        "instance": instance,
        "project": project,
    }
    section = {
        "title": title,
        "content": section_content,
        "hash": content_hash,
    }
    tags = list(dict.fromkeys(["manual-upload"] + extra_tags))     # dedup, preserve order

    try:
        ingested = await _ingest_section(source, section, tags)
    except Exception as e:
        logger.exception("Ingestion failed for %s", title)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    if not ingested:
        return {
            "episode_id": None,
            "title": title,
            "summary_chars": len(section_content),
            "char_count": len(text),
            "duplicate": True,
            "message": "Ingestion skipped (already present)",
        }

    # Recompute episode_id the same way _ingest_section does (deterministic).
    from crawler import _make_episode_id
    episode_id = _make_episode_id(source_label, title, content_hash)

    return {
        "episode_id": episode_id,
        "title": title,
        "summary_chars": len(section_content),
        "char_count": len(text),
        "duplicate": False,
        "sha256": content_sha256,
    }


# ─── POST /api/episodes/upload ───────────────────────────────────

@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None),
    instance: str = Form("office-genesis"),
    project: str = Form("manual-upload"),
    title_override: Optional[str] = Form(None),
):
    """Ingest a single document file (PDF / .docx / .md / .txt / .rtf / .odt)."""
    # Enforce size limit at the route level using the request header.
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                )
        except ValueError:
            pass  # malformed header — fall through to read-time enforcement

    # Decide extension from filename, fall back to MIME mapping.
    raw_name = _sanitize_filename(file.filename or "")
    ext = os.path.splitext(raw_name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        guess = MIME_TO_EXT.get((file.content_type or "").lower())
        if guess in SUPPORTED_EXTENSIONS:
            ext = guess
        else:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {ext or file.content_type or 'unknown'}. "
                       f"Supported: {sorted(SUPPORTED_EXTENSIONS)}",
            )

    # Read in chunks so we can enforce the byte cap even when content-length lies.
    sha = hashlib.sha256()
    bytes_read = 0
    # Stage temp file inside the store dir so the final rename is on the same FS.
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=str(UPLOAD_STORE_DIR)) as tmp:
        tmp_path = tmp.name
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > MAX_UPLOAD_BYTES:
                tmp.close()
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                )
            sha.update(chunk)
            tmp.write(chunk)

    sha_hex = sha.hexdigest()
    logger.info(
        "Upload received: name=%s ext=%s mime=%s bytes=%d sha=%s",
        raw_name, ext, file.content_type, bytes_read, sha_hex[:12],
    )

    # Persist the original to a writable, NON-/sources/ path so the crawler
    # never picks it up.  Filename = sha256 + ext for natural dedup.
    stored_path = UPLOAD_STORE_DIR / f"{sha_hex}{ext}"
    if not stored_path.exists():
        try:
            os.replace(tmp_path, stored_path)
        except OSError as e:
            logger.warning("Could not move upload to store (%s); using temp path", e)
            stored_path = Path(tmp_path)
    else:
        # Already stored — drop the temp copy.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Parse to text via the crawler's parser table.
    try:
        text = _parse_by_extension(str(stored_path), ext)
    except Exception as e:
        logger.exception("Parser crashed for %s", stored_path)
        raise HTTPException(status_code=500, detail=f"Parser error: {e}")

    if not text:
        raise HTTPException(
            status_code=422,
            detail=f"Parser returned no text for {ext} file (corrupt or unsupported variant?)",
        )

    title = (title_override or "").strip() or os.path.splitext(raw_name)[0] or sha_hex[:12]
    extra_tags = _split_tags(tags) + [ext.lstrip("."), "uploaded"]

    logger.debug("Upload extract preview (%s): %s", title, _short(text))

    result = await _ingest_text(
        text=text,
        title=title,
        instance=instance,
        project=project,
        extra_tags=extra_tags,
        content_sha256=sha_hex,
        source_label=f"upload_{sha_hex[:12]}",
    )
    result["filename"] = raw_name
    result["mime"] = file.content_type
    result["bytes"] = bytes_read
    return result


# ─── POST /api/episodes/ingest-url ───────────────────────────────

async def _download_capped(url: str) -> tuple[bytes, str]:
    """Stream a URL, abort if it exceeds MAX_URL_BYTES.  Returns (bytes, content_type)."""
    timeout = httpx.Timeout(URL_TIMEOUT_SECONDS, connect=10.0)
    headers = {"User-Agent": "Anamnesis/0.1 (manual ingest)"}
    buf = bytearray()
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers=headers,
        max_redirects=5,
    ) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                buf.extend(chunk)
                if len(buf) > MAX_URL_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Remote content exceeds {MAX_URL_BYTES // (1024 * 1024)} MB limit",
                    )
    return bytes(buf), content_type


@router.post("/ingest-url")
async def ingest_url(req: IngestUrlRequest):
    """Fetch a public URL, extract text, ingest as one episode."""
    safe_url = _validate_public_url(req.url)

    # NOTE: validation re-resolves on the actual HTTP call — there is a small
    # TOCTOU window. Acceptable for a personal tool; harden later if exposed.
    try:
        body, content_type = await _download_capped(safe_url)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e.response.status_code}")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fetch failed: {e}")

    sha_hex = hashlib.sha256(body).hexdigest()

    parsed_url = urlparse(safe_url)
    url_basename = os.path.basename(parsed_url.path) or parsed_url.netloc
    derived_title = (req.title_override or "").strip() or url_basename or safe_url

    text = ""
    extracted_title = ""

    if content_type == "application/pdf" or safe_url.lower().endswith(".pdf"):
        # Save to temp, parse with crawler's pdfplumber wrapper.
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir="/tmp") as tmp:
            tmp.write(body)
            tmp_path = tmp.name
        try:
            text = _parse_pdf(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    elif content_type in {"text/html", "application/xhtml+xml"}:
        extracted_title, text = _extract_html(body)
    elif content_type in {"text/plain", "text/markdown", "text/x-markdown"} \
            or safe_url.lower().endswith((".md", ".txt", ".markdown")):
        text = body.decode("utf-8", errors="replace")
    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content-type: {content_type or 'unknown'}",
        )

    if not text or len(text) < MIN_TEXT_BYTES:
        raise HTTPException(
            status_code=422,
            detail="Could not extract enough text from URL",
        )

    title = (req.title_override or "").strip() or extracted_title.strip() or derived_title
    title = title[:200] or sha_hex[:12]

    extra_tags = list(req.tags or []) + ["url", parsed_url.netloc]

    logger.info(
        "URL ingest: url=%s ct=%s bytes=%d sha=%s title=%r",
        safe_url, content_type, len(body), sha_hex[:12], title[:80],
    )
    logger.debug("URL extract preview: %s", _short(text))

    result = await _ingest_text(
        text=text,
        title=title,
        instance=req.instance,
        project=req.project,
        extra_tags=extra_tags,
        content_sha256=sha_hex,
        source_label=f"url_{sha_hex[:12]}",
    )
    result["url"] = safe_url
    result["content_type"] = content_type
    result["bytes"] = len(body)
    return result
