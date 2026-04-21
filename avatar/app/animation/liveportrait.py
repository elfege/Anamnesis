"""LivePortrait animation backend.

Uses LivePortrait to animate a reference face image driven by audio.
Requires NVIDIA GPU (CUDA) or AMD GPU (ROCm) with sufficient VRAM.

LivePortrait repo: https://github.com/KwaiVGI/LivePortrait
Pipeline: audio → motion coefficients → face animation → video
"""
import asyncio
import logging
import os
import tempfile
import subprocess
from pathlib import Path

import config
from animation.base import AnimationBackend

logger = logging.getLogger("avatar.animation.liveportrait")

# LivePortrait install location (cloned into container or mounted)
LIVEPORTRAIT_DIR = os.environ.get("LIVEPORTRAIT_DIR", "/opt/LivePortrait")


class LivePortraitBackend(AnimationBackend):

    def __init__(self):
        self._checkpoint_dir = config.LIVEPORTRAIT_CHECKPOINT_DIR
        self._ready = False

    @property
    def name(self) -> str:
        return "liveportrait"

    async def is_ready(self) -> bool:
        if self._ready:
            return True
        # Check if LivePortrait is installed and checkpoints exist
        lp_path = Path(LIVEPORTRAIT_DIR)
        ckpt_path = Path(self._checkpoint_dir)
        self._ready = lp_path.exists() and ckpt_path.exists()
        if not self._ready:
            logger.warning(
                f"LivePortrait not ready. Dir: {lp_path.exists()}, "
                f"Checkpoints: {ckpt_path.exists()}"
            )
        return self._ready

    async def animate(self, audio_path: str, reference_image: str) -> str:
        """
        Run LivePortrait audio-driven animation.

        Uses LivePortrait's audio2motion pipeline:
        1. Extract audio features (mel spectrogram)
        2. Predict facial motion coefficients from audio
        3. Apply motion to reference image
        4. Render video frames
        5. Encode to MP4

        Returns path to the output video.
        """
        if not await self.is_ready():
            raise RuntimeError("LivePortrait not ready — check installation and checkpoints")

        output_dir = tempfile.mkdtemp(prefix="avatar_")
        output_path = os.path.join(output_dir, "output.mp4")

        # LivePortrait inference command
        cmd = [
            "python3", os.path.join(LIVEPORTRAIT_DIR, "inference.py"),
            "--source_image", reference_image,
            "--driving_audio", audio_path,
            "--output_path", output_path,
            "--checkpoint_dir", self._checkpoint_dir,
        ]

        # GPU environment
        env = {**os.environ}
        if config.GPU_TYPE == "rocm":
            env.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

        logger.info(f"Running LivePortrait: {' '.join(cmd)}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=LIVEPORTRAIT_DIR,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode()[-500:]  # Last 500 chars of error
            logger.error(f"LivePortrait failed (rc={proc.returncode}): {error_msg}")
            raise RuntimeError(f"LivePortrait inference failed: {error_msg}")

        if not os.path.exists(output_path):
            raise RuntimeError(f"LivePortrait produced no output at {output_path}")

        logger.info(f"Animation complete: {output_path}")
        return output_path
