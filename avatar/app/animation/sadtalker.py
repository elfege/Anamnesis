"""SadTalker animation backend — audio-driven face animation.

Takes a single reference image + audio file and produces a talking-head video.
This is the recommended default for the avatar prototype: single-stage,
audio-driven, ~4GB VRAM, works on both CUDA and ROCm.

Repo: https://github.com/OpenTalker/SadTalker
"""
import asyncio
import glob
import logging
import os
import tempfile
from pathlib import Path

import config
from animation.base import AnimationBackend

logger = logging.getLogger("avatar.animation.sadtalker")


class SadTalkerBackend(AnimationBackend):

    def __init__(self):
        self._sadtalker_dir = config.SADTALKER_DIR
        self._checkpoint_dir = config.SADTALKER_CHECKPOINT_DIR
        self._ready = False

    @property
    def name(self) -> str:
        return "sadtalker"

    async def is_ready(self) -> bool:
        if self._ready:
            return True
        st_path = Path(self._sadtalker_dir)
        ckpt_path = Path(self._checkpoint_dir)
        self._ready = st_path.exists() and ckpt_path.exists()
        if not self._ready:
            logger.warning(
                f"SadTalker not ready. Dir: {st_path.exists()}, "
                f"Checkpoints: {ckpt_path.exists()}"
            )
        return self._ready

    async def animate(self, audio_path: str, reference_image: str) -> str:
        """
        Run SadTalker: reference image + audio → talking head video.

        SadTalker pipeline (single stage):
        1. Extract 3DMM coefficients from reference image
        2. Predict facial motion from audio (ExpNet + PoseVAE)
        3. Render face with predicted motion via face renderer
        4. Optionally enhance with GFPGAN
        5. Output MP4 video

        Returns path to the generated video file.
        """
        if not await self.is_ready():
            raise RuntimeError("SadTalker not ready — check installation and checkpoints")

        output_dir = tempfile.mkdtemp(prefix="avatar_sadtalker_")

        cmd = [
            "python3", os.path.join(self._sadtalker_dir, "inference.py"),
            "--source_image", reference_image,
            "--driven_audio", audio_path,
            "--result_dir", output_dir,
            "--checkpoint_dir", self._checkpoint_dir,
            "--still",           # No head motion (keeps face centered)
            "--enhancer", "gfpgan",  # Face enhancement for quality
            "--size", "512",     # Output resolution
        ]

        env = {**os.environ}
        if config.GPU_TYPE == "rocm":
            env.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

        logger.info(f"Running SadTalker inference")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self._sadtalker_dir,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode()[-500:]
            logger.error(f"SadTalker failed (rc={proc.returncode}): {error_msg}")
            raise RuntimeError(f"SadTalker inference failed: {error_msg}")

        # SadTalker outputs to result_dir with auto-generated filenames
        # Find the most recent .mp4 in the output directory
        mp4_files = sorted(
            glob.glob(os.path.join(output_dir, "**/*.mp4"), recursive=True),
            key=os.path.getmtime,
            reverse=True,
        )

        if not mp4_files:
            raise RuntimeError(f"SadTalker produced no video output in {output_dir}")

        output_path = mp4_files[0]
        logger.info(f"Animation complete: {output_path}")
        return output_path
