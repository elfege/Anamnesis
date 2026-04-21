"""
Voice registry — resolves voice IDs to TTS backends and speaker samples.

Voice ID format:
    edge:en-US-AvaNeural           → Edge TTS with preset voice
    cloned:belle-singer            → XTTS with /app/voices/belle-singer.wav

Cloned voices are persisted to VOICES_DIR (WAV + metadata JSON sidecar).
Voice metadata: name, slug, backend, source (file|song|record), created_at, language.
"""
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import config

logger = logging.getLogger("avatar.voices")

# ─── Edge TTS preset list (curated — Microsoft has 400+, we list the good ones) ─

EDGE_PRESETS = [
    # English (US/GB) — mature young-adult female first
    {"id": "edge:en-US-AvaNeural",      "name": "Ava (EN-US, warm)",        "lang": "en", "gender": "female"},
    {"id": "edge:en-US-AriaNeural",     "name": "Aria (EN-US, confident)",  "lang": "en", "gender": "female"},
    {"id": "edge:en-US-EmmaNeural",     "name": "Emma (EN-US, soft)",       "lang": "en", "gender": "female"},
    {"id": "edge:en-US-JennyNeural",    "name": "Jenny (EN-US, friendly)",  "lang": "en", "gender": "female"},
    {"id": "edge:en-US-MichelleNeural", "name": "Michelle (EN-US, low)",    "lang": "en", "gender": "female"},
    {"id": "edge:en-GB-SoniaNeural",    "name": "Sonia (EN-GB, mature)",    "lang": "en", "gender": "female"},
    {"id": "edge:en-GB-LibbyNeural",    "name": "Libby (EN-GB, youthful)",  "lang": "en", "gender": "female"},
    # English male
    {"id": "edge:en-US-GuyNeural",      "name": "Guy (EN-US)",              "lang": "en", "gender": "male"},
    {"id": "edge:en-US-AndrewNeural",   "name": "Andrew (EN-US)",           "lang": "en", "gender": "male"},
    # French
    {"id": "edge:fr-FR-DeniseNeural",   "name": "Denise (FR-FR)",           "lang": "fr", "gender": "female"},
    {"id": "edge:fr-FR-VivienneNeural", "name": "Vivienne (FR-FR)",         "lang": "fr", "gender": "female"},
    {"id": "edge:fr-FR-HenriNeural",    "name": "Henri (FR-FR)",            "lang": "fr", "gender": "male"},
    # Keep "Ana" at the bottom — child voice, labeled
    {"id": "edge:en-US-AnaNeural",      "name": "Ana (child voice)",        "lang": "en", "gender": "child"},
]


@dataclass
class ClonedVoice:
    slug: str
    name: str
    source: str              # "file" | "song" | "record"
    wav_path: str            # absolute path inside the container
    language: str = "en"
    created_at: float = field(default_factory=time.time)
    original_filename: Optional[str] = None
    notes: Optional[str] = None

    @property
    def id(self) -> str:
        return f"cloned:{self.slug}"

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "source": self.source,
            "lang": self.language,
            "gender": "cloned",
            "created_at": self.created_at,
            "notes": self.notes,
        }


class VoiceRegistry:

    def __init__(self, voices_dir: str):
        self.dir = Path(voices_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ── Listing ─────────────────────────────────────────────────

    def list_all(self) -> dict:
        """Return {presets: [...edge...], cloned: [...custom...]}."""
        return {
            "presets": EDGE_PRESETS,
            "cloned": [v.to_public() for v in self._list_cloned()],
        }

    def _list_cloned(self) -> List[ClonedVoice]:
        voices = []
        for meta_path in sorted(self.dir.glob("*.json")):
            try:
                data = json.loads(meta_path.read_text())
                voices.append(ClonedVoice(**data))
            except Exception as e:
                logger.warning(f"Skipping malformed voice metadata {meta_path}: {e}")
        return voices

    def get_cloned(self, slug: str) -> Optional[ClonedVoice]:
        meta_path = self.dir / f"{slug}.json"
        if not meta_path.exists():
            return None
        try:
            return ClonedVoice(**json.loads(meta_path.read_text()))
        except Exception as e:
            logger.error(f"Bad metadata for {slug}: {e}")
            return None

    # ── Resolution ──────────────────────────────────────────────

    def resolve(self, voice_id: Optional[str]) -> dict:
        """
        Resolve a voice_id to a backend spec.
        Returns: {backend: "edge"|"xtts", **params}
        Falls back to config default on miss.
        """
        if not voice_id:
            voice_id = config.DEFAULT_VOICE_ID

        if voice_id.startswith("edge:"):
            edge_name = voice_id.split(":", 1)[1]
            return {"backend": "edge", "voice": edge_name}

        if voice_id.startswith("cloned:"):
            slug = voice_id.split(":", 1)[1]
            voice = self.get_cloned(slug)
            if voice is None:
                logger.warning(f"Cloned voice not found: {slug}; falling back to default")
                return self.resolve(config.DEFAULT_VOICE_ID)
            return {
                "backend": "xtts",
                "speaker_wav": voice.wav_path,
                "language": voice.language,
                "slug": slug,
            }

        logger.warning(f"Unknown voice_id format: {voice_id}")
        return self.resolve(config.DEFAULT_VOICE_ID)

    # ── Creation ────────────────────────────────────────────────

    def add(
        self,
        name: str,
        source_wav_path: str,
        source: str,
        language: str = "en",
        original_filename: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> ClonedVoice:
        """
        Register a new cloned voice. Copies source_wav_path into the registry.
        Returns the stored ClonedVoice.
        """
        slug = self._make_slug(name)
        target_wav = self.dir / f"{slug}.wav"

        # Copy (don't move — source may be a temp file the caller cleans up)
        import shutil
        shutil.copy2(source_wav_path, target_wav)

        voice = ClonedVoice(
            slug=slug,
            name=name,
            source=source,
            wav_path=str(target_wav),
            language=language,
            original_filename=original_filename,
            notes=notes,
        )
        meta_path = self.dir / f"{slug}.json"
        meta_path.write_text(json.dumps(asdict(voice), indent=2))
        logger.info(f"Voice registered: {voice.id} ({target_wav})")
        return voice

    def delete(self, slug: str) -> bool:
        wav = self.dir / f"{slug}.wav"
        meta = self.dir / f"{slug}.json"
        removed = False
        for p in (wav, meta):
            if p.exists():
                p.unlink()
                removed = True
        if removed:
            logger.info(f"Voice deleted: cloned:{slug}")
        return removed

    # ── Helpers ─────────────────────────────────────────────────

    def _make_slug(self, name: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "voice"
        candidate = base
        i = 2
        while (self.dir / f"{candidate}.wav").exists():
            candidate = f"{base}-{i}"
            i += 1
        return candidate


# Singleton
_registry: Optional[VoiceRegistry] = None


def get_registry() -> VoiceRegistry:
    global _registry
    if _registry is None:
        _registry = VoiceRegistry(config.VOICES_DIR)
    return _registry
