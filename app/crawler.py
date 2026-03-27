"""
Anamnesis Crawler — Active source scanner and episode ingester.

Periodically scans known knowledge sources and ingests new content as episodes.
Sources are mounted read-only into the Docker container (or staged via rsync).
Deduplication is by SHA-256 hash of section content (crawl_state collection).

Source types:
  - Named SOURCES: specific files (genesis, handoff, history, intercom, user profile)
  - Project scanner: discovers projects by docker-compose.yml presence, ingests
    CLAUDE.md, README.md, docker-compose.yml, *.sh, *.py at project root
  - Scripts scanner: crawls 0_SCRIPTS/ for .sh files (bash style authority)
  - Teachings: each .md file in 0_TEACHINGS/ = one episode
  - Documents: .docx files from OneDrive

Machines: dellserver (local $HOME), server, officewsl, hvtmc (OHVD_APP_PROD).
Remote machines are staged via sync_sources.sh (runs hourly via cron).
"""

import asyncio
import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError

from database import get_episodes_collection
from embedding import get_embedding
from config import MONGO_DB

logger = logging.getLogger("anamnesis.crawler")

# ─── Source Definitions ──────────────────────────────────────────
# Paths are as seen inside the Docker container (via volume mounts)

SOURCES = [
    {
        "name": "genesis",
        "path": "/sources/dellserver/0_GENESIS_PROJECT/genesis.md",
        "instance": "office-genesis",
        "project": "0_GENESIS_PROJECT",
        "description": "Philosophical persistence insights — Hegel, quantity, AI growth",
    },
    {
        "name": "handoff",
        "path": "/sources/dellserver/README_handoff.md",
        "instance": "office",
        "project": "cross-project",
        "description": "Session handoff buffer — recent work, TODOs, file changes",
    },
    {
        "name": "project_history_officewsl",
        "path": "/sources/dellserver/README_project_history_officewsl.md",
        "instance": "office",
        "project": "cross-project",
        "description": "Long-term project history (officewsl)",
    },
    {
        "name": "project_history_server",
        "path": "/sources/dellserver/README_project_history_server.md",
        "instance": "office",
        "project": "cross-project",
        "description": "Long-term project history (server)",
    },
    {
        "name": "intercom",
        "path": "/sources/server/0_CLAUDE_IC/intercom.md",
        "instance": "cross-instance",
        "project": "0_CLAUDE_IC",
        "description": "Cross-instance intercom messages",
    },
    {
        "name": "user_profile",
        "path": "/sources/server/0_CLAUDE_IC/user_profile_elfege.md",
        "instance": "cross-instance",
        "project": "0_CLAUDE_IC",
        "description": "Elfege's persistent user profile — background, preferences, history",
    },
]

# Machine roots — paths as seen inside the container (via volume mounts / staging)
MACHINE_ROOTS = {
    "dellserver": "/sources/dellserver",
    "server":     "/sources/server",
    "officewsl":  "/sources/officewsl",
    "hvtmc":      "/sources/hvtmc",
    "jessica":    "/sources/jessica",
}

# File extensions treated as code (ingested as whole-file episodes, not section-split)
_CODE_EXTENSIONS = {".sh", ".py", ".yml", ".yaml"}

# Dir name fragments to skip when walking 0_SCRIPTS/
_SCRIPTS_SKIP = {"DEPRECATED", "TRASH", "ARCHIVE", "CONFLICTS"}

# Teachings directory — each .md file is a separate source
TEACHINGS_DIR = "/sources/teachings"

# Documents directory — .docx files from OneDrive
DOCUMENTS_DIR = "/sources/documents"

# ─── Crawler State ───────────────────────────────────────────────

_crawler_state = {
    "running": False,
    "last_run": None,
    "last_run_duration_seconds": 0,
    "episodes_ingested_last_run": 0,
    "total_episodes_ingested": 0,
    "errors": [],
    "interval_seconds": 300,                                       # 5 minutes default
}

_crawler_task: Optional[asyncio.Task] = None


def get_crawler_status() -> dict:
    """Return current crawler state for the dashboard/API."""
    return {**_crawler_state}


# ─── Parsing ─────────────────────────────────────────────────────

def _content_hash(text: str) -> str:
    """SHA-256 hash of text content for deduplication."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _parse_markdown_sections(text: str) -> list[dict]:
    """Split markdown text into sections by ### headers.

    Returns list of dicts with 'title', 'content', 'hash'.
    Skips sections that are too short to be meaningful (< 50 chars).
    """
    sections = []
    # Split on ### headers (keep the header as part of the section)
    parts = re.split(r"(?=^### )", text, flags=re.MULTILINE)

    for part in parts:
        part = part.strip()
        if not part or len(part) < 50:                             # skip tiny fragments
            continue

        # Extract title from ### header if present
        title_match = re.match(r"^###\s+(.+?)$", part, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "untitled"

        sections.append({
            "title": title,
            "content": part,
            "hash": _content_hash(part),
        })

    return sections


def _parse_intercom_messages(text: str) -> list[dict]:
    """Parse intercom.md into individual messages by ### MSG-* headers."""
    sections = []
    parts = re.split(r"(?=^### MSG-)", text, flags=re.MULTILINE)

    for part in parts:
        part = part.strip()
        if not part or not part.startswith("### MSG-"):
            continue
        if len(part) < 30:                                         # skip ACK-only fragments
            continue

        # Extract MSG ID
        msg_match = re.match(r"^### (MSG-\d+)", part)
        msg_id = msg_match.group(1) if msg_match else "unknown"

        sections.append({
            "title": msg_id,
            "content": part,
            "hash": _content_hash(part),
        })

    return sections


def _parse_docx(filepath: str) -> str:
    """Extract all text from a .docx file. Returns full text as a string."""
    try:
        doc = DocxDocument(filepath)
    except PackageNotFoundError:
        logger.warning(f"Corrupt or unreadable docx: {filepath}")
        return ""
    except Exception as e:
        logger.warning(f"Failed to parse docx {filepath}: {e}")
        return ""

    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)

    return "\n".join(paragraphs)


def _categorize_docx(filename: str, content: str) -> list[str]:
    """Infer tags for a docx file from filename patterns and content."""
    tags = ["document", "onedrive"]
    name_lower = filename.lower()

    # Student evaluation pattern: lastname.year.docx
    if re.match(r"^[a-z]+\.\d{4}\.docx$", name_lower):
        tags.append("student-evaluation")

    # Named document patterns
    if "agreement" in name_lower or "authorization" in name_lower:
        tags.append("legal")
    if "portfolio" in name_lower:
        tags.append("portfolio")
    if "notes" in name_lower:
        tags.append("notes")
    if "fasny" in name_lower or "lfny" in name_lower:
        tags.append("school")

    return tags


def _parse_code_file(filepath: str) -> list[dict]:
    """Read a code/config file as a single section (no header splitting)."""
    try:
        content = Path(filepath).read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return []
    if len(content) < 20:
        return []
    return [{"title": Path(filepath).name, "content": content, "hash": _content_hash(content)}]


def _make_episode_id(source_name: str, section_title: str, content_hash: str) -> str:
    """Generate a deterministic episode_id from source + content."""
    # Clean title for use in ID
    clean_title = re.sub(r"[^a-zA-Z0-9_-]", "_", section_title)[:60]
    short_hash = content_hash[:8]
    return f"crawl_{source_name}_{clean_title}_{short_hash}"


# ─── Ingestion ───────────────────────────────────────────────────

async def _get_crawl_state_collection():
    """Get the crawl_state collection for tracking what's been ingested."""
    from database import _db
    return _db["crawl_state"]


async def _is_already_ingested(content_hash: str) -> bool:
    """Check if a section with this content hash has already been ingested."""
    crawl_state = await _get_crawl_state_collection()
    existing = await crawl_state.find_one({"content_hash": content_hash})
    return existing is not None


async def _mark_as_ingested(content_hash: str, episode_id: str, source_name: str):
    """Record that this content hash has been ingested."""
    crawl_state = await _get_crawl_state_collection()
    await crawl_state.update_one(
        {"content_hash": content_hash},
        {
            "$set": {
                "content_hash": content_hash,
                "episode_id": episode_id,
                "source_name": source_name,
                "ingested_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )


async def _ingest_section(
    source: dict,
    section: dict,
    tags: list[str],
) -> bool:
    """Embed and store a single section as an episode. Returns True if ingested."""

    content_hash = section["hash"]

    # Dedup check
    if await _is_already_ingested(content_hash):
        return False

    episode_id = _make_episode_id(source["name"], section["title"], content_hash)
    collection = get_episodes_collection()

    # Also check if episode_id already exists (belt and suspenders)
    existing = await collection.find_one({"episode_id": episode_id})
    if existing:
        await _mark_as_ingested(content_hash, episode_id, source["name"])
        return False

    # Generate embedding from the section content
    summary = section["content"]
    try:
        embedding_vector = get_embedding(summary)
    except Exception as e:
        logger.error(f"Embedding failed for {episode_id}: {e}")
        return False

    now = datetime.now(timezone.utc)
    document = {
        "episode_id": episode_id,
        "instance": source["instance"],
        "project": source["project"],
        "summary": summary,
        "raw_exchange": None,
        "tags": tags + ["crawled", f"source:{source['name']}"],
        "embedding": embedding_vector,
        "timestamp": now,
        "retrieval_count": 0,
        "last_retrieved": None,
    }

    await collection.insert_one(document)
    await _mark_as_ingested(content_hash, episode_id, source["name"])

    logger.info(f"Ingested: {episode_id}")
    return True


# ─── Source Scanning ─────────────────────────────────────────────

async def _scan_source(source: dict) -> int:
    """Scan a single source file and ingest new sections. Returns count ingested."""
    path = source["path"]

    if not os.path.exists(path):
        logger.debug(f"Source not found (skipping): {path}")
        return 0

    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return 0

    if not content.strip():
        return 0

    # Choose parser based on source type
    if source["name"] == "intercom":
        sections = _parse_intercom_messages(content)
        default_tags = ["intercom"]
    else:
        sections = _parse_markdown_sections(content)
        default_tags = [source["name"]]

    ingested_count = 0
    for section in sections:
        try:
            if await _ingest_section(source, section, default_tags):
                ingested_count += 1
        except Exception as e:
            logger.error(f"Failed to ingest section '{section['title']}' from {source['name']}: {e}")

    return ingested_count


async def _scan_teachings_dir() -> int:
    """Scan the teachings directory for .md files. Each file = one episode."""
    if not os.path.isdir(TEACHINGS_DIR):
        logger.debug(f"Teachings dir not found (skipping): {TEACHINGS_DIR}")
        return 0

    ingested_count = 0
    for root, _dirs, files in os.walk(TEACHINGS_DIR):
        for filename in files:
            if not filename.endswith(".md"):
                continue

            filepath = os.path.join(root, filename)
            try:
                content = Path(filepath).read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.error(f"Failed to read teaching file {filepath}: {e}")
                continue

            if not content.strip() or len(content) < 50:
                continue

            content_hash = _content_hash(content)
            if await _is_already_ingested(content_hash):
                continue

            # Treat the whole file as one episode
            relative_path = os.path.relpath(filepath, TEACHINGS_DIR)
            source = {
                "name": f"teaching_{relative_path}",
                "instance": "office",
                "project": "0_TEACHINGS",
            }
            section = {
                "title": filename.replace(".md", ""),
                "content": content,
                "hash": content_hash,
            }

            try:
                if await _ingest_section(source, section, ["teaching", "learning"]):
                    ingested_count += 1
            except Exception as e:
                logger.error(f"Failed to ingest teaching {filepath}: {e}")

    return ingested_count


async def _scan_documents_dir() -> int:
    """Scan the documents directory for .docx files. Each file = one episode."""
    if not os.path.isdir(DOCUMENTS_DIR):
        logger.debug(f"Documents dir not found (skipping): {DOCUMENTS_DIR}")
        return 0

    ingested_count = 0
    skipped_empty = 0
    for root, _dirs, files in os.walk(DOCUMENTS_DIR):
        for filename in files:
            if not filename.lower().endswith(".docx"):
                continue

            filepath = os.path.join(root, filename)

            # Extract text from docx
            content = _parse_docx(filepath)
            if not content or len(content) < 30:
                skipped_empty += 1
                continue

            content_hash = _content_hash(content)
            if await _is_already_ingested(content_hash):
                continue

            # Build a summary: filename + first ~500 chars of content
            doc_title = filename.replace(".docx", "")
            summary = f"[Document: {doc_title}]\n\n{content}"

            tags = _categorize_docx(filename, content)
            source = {
                "name": f"doc_{doc_title}",
                "instance": "office",
                "project": "personal_documents",
            }
            section = {
                "title": doc_title,
                "content": summary,
                "hash": content_hash,
            }

            try:
                if await _ingest_section(source, section, tags):
                    ingested_count += 1
            except Exception as e:
                logger.error(f"Failed to ingest document {filepath}: {e}")

    if skipped_empty:
        logger.debug(f"  documents: skipped {skipped_empty} empty/tiny docx files")
    return ingested_count


async def _scan_projects_dir(machine_root: str, machine_name: str) -> int:
    """Discover projects by docker-compose.yml presence and ingest key files.

    Per project: CLAUDE.md, README.md, docker-compose.yml/yaml, *.sh, *.py
    (all at project root only — no deep recursion into project subdirs).
    """
    root_path = Path(machine_root)
    if not root_path.is_dir():
        logger.debug(f"Machine root not found (skipping): {machine_root}")
        return 0

    # Collect unique project dirs
    project_dirs: set[Path] = set()
    for pattern in ["*/docker-compose.yml", "*/docker-compose.yaml"]:
        for p in root_path.glob(pattern):
            project_dirs.add(p.parent)

    ingested_count = 0
    for project_dir in sorted(project_dirs):
        project_name = project_dir.name

        # Named files to always attempt
        candidates: list[Path] = [
            project_dir / "CLAUDE.md",
            project_dir / "README.md",
            project_dir / "docker-compose.yml",
            project_dir / "docker-compose.yaml",
        ]
        # Code files at project root
        candidates += sorted(project_dir.glob("*.sh"))
        candidates += sorted(project_dir.glob("*.py"))

        for filepath in candidates:
            if not filepath.exists():
                continue

            ext = filepath.suffix.lower()
            source_name = f"{machine_name}_{project_name}_{filepath.name.replace('.', '_')}"
            source = {"name": source_name, "instance": machine_name, "project": project_name}

            if ext == ".md":
                content_text = filepath.read_text(encoding="utf-8", errors="replace")
                sections = _parse_markdown_sections(content_text)
                tags = ["project", machine_name, project_name, "markdown"]
            else:
                sections = _parse_code_file(str(filepath))
                lang = "bash" if ext == ".sh" else ("python" if ext == ".py" else "compose")
                tags = ["project", machine_name, project_name, "code", lang]

            for section in sections:
                try:
                    if await _ingest_section(source, section, tags):
                        ingested_count += 1
                except Exception as e:
                    logger.error(f"Failed to ingest {filepath}: {e}")

    return ingested_count


async def _scan_scripts_dir(machine_root: str, machine_name: str) -> int:
    """Scan 0_SCRIPTS/ for .sh files (bash style authority).

    Skips dirs containing DEPRECATED, TRASH, ARCHIVE, CONFLICTS.
    Each .sh file = one episode (whole file, no section splitting).
    """
    scripts_dir = Path(machine_root) / "0_SCRIPTS"
    if not scripts_dir.is_dir():
        logger.debug(f"Scripts dir not found (skipping): {scripts_dir}")
        return 0

    ingested_count = 0
    for root, dirs, files in os.walk(scripts_dir):
        # Prune skip dirs in-place
        dirs[:] = [
            d for d in dirs
            if not any(skip in d.upper() for skip in _SCRIPTS_SKIP)
        ]
        for filename in files:
            if not filename.endswith(".sh"):
                continue
            filepath = os.path.join(root, filename)
            sections = _parse_code_file(filepath)
            if not sections:
                continue
            rel_path = os.path.relpath(filepath, str(scripts_dir))
            source_name = f"{machine_name}_scripts_{re.sub(r'[^a-zA-Z0-9_-]', '_', rel_path)}"
            source = {"name": source_name, "instance": machine_name, "project": "0_SCRIPTS"}
            tags = ["script", "bash", "code-style", machine_name]
            for section in sections:
                try:
                    if await _ingest_section(source, section, tags):
                        ingested_count += 1
                except Exception as e:
                    logger.error(f"Failed to ingest script {filepath}: {e}")

    return ingested_count


# ─── Main Crawl Loop ────────────────────────────────────────────

async def run_crawl_cycle():
    """Execute one full crawl cycle across all sources."""
    _crawler_state["running"] = True
    start_time = datetime.now(timezone.utc)
    total_ingested = 0
    errors = []

    logger.info("Crawl cycle starting...")

    # Scan each defined source
    for source in SOURCES:
        try:
            count = await _scan_source(source)
            total_ingested += count
            if count > 0:
                logger.info(f"  {source['name']}: {count} new episodes")
        except Exception as e:
            error_msg = f"{source['name']}: {e}"
            logger.error(f"  {error_msg}")
            errors.append(error_msg)

    # Scan each machine's projects and scripts
    for machine_name, machine_root in MACHINE_ROOTS.items():
        try:
            count = await _scan_projects_dir(machine_root, machine_name)
            total_ingested += count
            if count > 0:
                logger.info(f"  {machine_name} projects: {count} new episodes")
        except Exception as e:
            error_msg = f"{machine_name}_projects: {e}"
            logger.error(f"  {error_msg}")
            errors.append(error_msg)

        try:
            count = await _scan_scripts_dir(machine_root, machine_name)
            total_ingested += count
            if count > 0:
                logger.info(f"  {machine_name} scripts: {count} new episodes")
        except Exception as e:
            error_msg = f"{machine_name}_scripts: {e}"
            logger.error(f"  {error_msg}")
            errors.append(error_msg)

    # Scan teachings directory
    try:
        teaching_count = await _scan_teachings_dir()
        total_ingested += teaching_count
        if teaching_count > 0:
            logger.info(f"  teachings: {teaching_count} new episodes")
    except Exception as e:
        error_msg = f"teachings: {e}"
        logger.error(f"  {error_msg}")
        errors.append(error_msg)

    # Scan documents directory (.docx files)
    try:
        doc_count = await _scan_documents_dir()
        total_ingested += doc_count
        if doc_count > 0:
            logger.info(f"  documents: {doc_count} new episodes")
    except Exception as e:
        error_msg = f"documents: {e}"
        logger.error(f"  {error_msg}")
        errors.append(error_msg)

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()

    _crawler_state["last_run"] = start_time.isoformat()
    _crawler_state["last_run_duration_seconds"] = round(duration, 2)
    _crawler_state["episodes_ingested_last_run"] = total_ingested
    _crawler_state["total_episodes_ingested"] += total_ingested
    _crawler_state["errors"] = errors
    _crawler_state["running"] = False

    logger.info(
        f"Crawl cycle complete: {total_ingested} episodes ingested "
        f"in {duration:.1f}s ({len(errors)} errors)"
    )

    return total_ingested


async def _crawler_loop():
    """Background loop — runs crawl cycles on a schedule."""
    # Wait a bit after startup to let everything settle
    await asyncio.sleep(10)

    while True:
        try:
            await run_crawl_cycle()
        except Exception as e:
            logger.error(f"Crawler loop error: {e}")
            _crawler_state["errors"].append(str(e))
            _crawler_state["running"] = False

        await asyncio.sleep(_crawler_state["interval_seconds"])


def start_crawler():
    """Start the background crawler task. Called from main.py lifespan."""
    global _crawler_task
    _crawler_task = asyncio.create_task(_crawler_loop())
    logger.info(
        f"Crawler started (interval: {_crawler_state['interval_seconds']}s)"
    )


def stop_crawler():
    """Stop the background crawler task."""
    global _crawler_task
    if _crawler_task and not _crawler_task.done():
        _crawler_task.cancel()
        logger.info("Crawler stopped.")
    _crawler_task = None
