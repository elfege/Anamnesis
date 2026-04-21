"""SadTalker audio-driven animation."""
import asyncio
import glob
import logging
import os
from pathlib import Path

import config

logger = logging.getLogger("worker.sadtalker")


async def is_ready() -> bool:
    return Path(config.SADTALKER_DIR).exists() and Path(config.SADTALKER_CHECKPOINT_DIR).exists()


async def animate(audio_path: str, image_path: str, output_dir: str) -> str:
    if not await is_ready():
        raise RuntimeError(
            f"SadTalker not ready (dir={config.SADTALKER_DIR}, ckpt={config.SADTALKER_CHECKPOINT_DIR})"
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3", os.path.join(config.SADTALKER_DIR, "inference.py"),
        "--source_image", image_path,
        "--driven_audio", audio_path,
        "--result_dir", output_dir,
        "--checkpoint_dir", config.SADTALKER_CHECKPOINT_DIR,
        "--still",
        "--enhancer", "gfpgan",
        "--size", "512",
    ]

    env = {**os.environ}
    if config.GPU_TYPE == "rocm":
        env.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

    logger.info("Running SadTalker…")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=config.SADTALKER_DIR,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        tail = stderr.decode()[-500:]
        raise RuntimeError(f"SadTalker failed (rc={proc.returncode}): {tail}")

    mp4s = sorted(
        glob.glob(os.path.join(output_dir, "**/*.mp4"), recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    if not mp4s:
        raise RuntimeError(f"SadTalker produced no .mp4 in {output_dir}")
    logger.info(f"SadTalker → {mp4s[0]}")
    return mp4s[0]
