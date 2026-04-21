"""Demucs vocal extraction."""
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger("worker.demucs")


async def extract_vocals(
    source_audio_path: str,
    out_dir: str,
    trim_seconds: Optional[float] = 15.0,
) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def _run():
        cmd = [
            "python", "-m", "demucs",
            "--two-stems=vocals",
            "-n", config.DEMUCS_MODEL,
            "-o", out_dir,
            source_audio_path,
        ]
        logger.info(f"demucs: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"demucs failed: {r.stderr[-500:]}")

    await asyncio.to_thread(_run)

    stem = Path(source_audio_path).stem
    vocals = Path(out_dir) / config.DEMUCS_MODEL / stem / "vocals.wav"
    if not vocals.exists():
        matches = list(Path(out_dir).rglob("vocals.wav"))
        if not matches:
            raise RuntimeError(f"demucs produced no vocals.wav in {out_dir}")
        vocals = matches[0]

    if trim_seconds:
        trimmed = vocals.parent / "vocals_trim.wav"
        await asyncio.to_thread(_trim, str(vocals), str(trimmed), trim_seconds)
        return str(trimmed)
    return str(vocals)


def _trim(src: str, dst: str, seconds: float):
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
