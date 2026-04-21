"""Vocal extraction from a song file using Demucs.

Returns a path to a clean vocals WAV suitable as XTTS speaker reference.
Demucs produces 44.1kHz stereo; XTTS accepts it.

First use downloads ~2.5GB of model weights to torch hub cache.
"""
import asyncio
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger("avatar.audio.demucs")


async def extract_vocals(
    source_audio_path: str,
    out_dir: Optional[str] = None,
    trim_seconds: Optional[float] = 15.0,
) -> str:
    """
    Run Demucs on `source_audio_path` and return the path to the extracted vocals WAV.

    If `trim_seconds` is set, the returned file is trimmed to that length from
    the first detected onset (avoids leading silence / intro noise).
    """
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="demucs_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    def _run():
        # `python -m demucs --two-stems=vocals -n <model> -o OUT SOURCE`
        cmd = [
            "python", "-m", "demucs",
            "--two-stems=vocals",
            "-n", config.DEMUCS_MODEL,
            "-o", str(out_dir),
            source_audio_path,
        ]
        logger.info(f"Running demucs: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"demucs failed: {res.stderr}")
        return res.stdout

    await asyncio.to_thread(_run)

    # Demucs writes to OUT/<model>/<track_name>/vocals.wav
    track_stem = Path(source_audio_path).stem
    vocals_path = out_dir / config.DEMUCS_MODEL / track_stem / "vocals.wav"
    if not vocals_path.exists():
        # Fallback: glob in case naming differs
        matches = list(out_dir.rglob("vocals.wav"))
        if not matches:
            raise RuntimeError(f"demucs did not produce vocals.wav under {out_dir}")
        vocals_path = matches[0]

    if trim_seconds:
        trimmed = vocals_path.parent / "vocals_trim.wav"
        await asyncio.to_thread(_trim_to_length, str(vocals_path), str(trimmed), trim_seconds)
        return str(trimmed)

    return str(vocals_path)


def _trim_to_length(src: str, dst: str, seconds: float):
    """Use ffmpeg silenceremove to skip leading silence, then trim to `seconds`."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", src,
            "-af", "silenceremove=start_periods=1:start_duration=0.2:start_threshold=-40dB",
            "-t", str(seconds),
            dst,
        ],
        check=True,
        capture_output=True,
    )
