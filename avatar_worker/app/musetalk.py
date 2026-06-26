"""MuseTalk audio-driven lip-sync animation.

MuseTalk (TMElyralab) is a diffusion-based lip-sync model that runs ~real-time
on a 16GB ROCm card (vs. SadTalker's 30-120s on the same hardware). API mirrors
sadtalker.py so the orchestrator can drop one in for the other.

The MuseTalk inference script is invoked as a subprocess against its own pinned
deps (diffusers 0.30.2, accelerate 0.28.0, decord, face_alignment) which live
in a separate PYTHONUSERBASE overlay (/opt/MuseTalk_pyenv) so they don't clash
with XTTS's pinned transformers 5.x.

A YAML inference config is generated per call; MuseTalk reads {video_path,
audio_path[, bbox_shift]} per task and emits ``<video_basename>_<audio_basename>.mp4``
into the result_dir.
"""
import asyncio
import glob
import logging
import os
import signal
import tempfile
import time
from pathlib import Path

import config

logger = logging.getLogger("worker.musetalk")

_running: set[asyncio.subprocess.Process] = set()


async def is_ready() -> bool:
    repo = Path(config.MUSETALK_DIR)
    if not repo.exists():
        return False
    if config.MUSETALK_VERSION == "v15":
        unet = repo / "models" / "musetalkV15" / "unet.pth"
        cfg = repo / "models" / "musetalkV15" / "musetalk.json"
    else:
        unet = repo / "models" / "musetalk" / "pytorch_model.bin"
        cfg = repo / "models" / "musetalk" / "musetalk.json"
    return unet.exists() and cfg.exists()


async def cancel_all() -> int:
    procs = list(_running)
    killed = 0
    for proc in procs:
        if proc.returncode is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.warning(f"SIGTERM pid={proc.pid} failed: {e}")
    if killed:
        await asyncio.sleep(0.5)
        for proc in procs:
            if proc.returncode is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
    return killed


def _write_inference_config(yaml_path: str, video_path: str, audio_path: str,
                            bbox_shift: int = 0) -> None:
    """MuseTalk reads task entries keyed by arbitrary names. The YAML is dead
    simple — we author it directly to avoid pulling pyyaml into the worker
    just for this. Quote paths defensively in case temp dirs contain spaces."""
    with open(yaml_path, "w") as f:
        f.write("task_0:\n")
        f.write(f'  video_path: "{video_path}"\n')
        f.write(f'  audio_path: "{audio_path}"\n')
        if bbox_shift:
            f.write(f"  bbox_shift: {bbox_shift}\n")


async def animate(audio_path: str, image_path: str, output_dir: str) -> str:
    """Lip-sync `image_path` (a single face) to `audio_path`. Returns the
    path to the produced .mp4. MuseTalk treats a single image as a 1-frame
    "video" and loops it for the duration of the audio."""
    if not await is_ready():
        raise RuntimeError(
            f"MuseTalk not ready (dir={config.MUSETALK_DIR}, "
            f"version={config.MUSETALK_VERSION})"
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # MuseTalk reads its config relative to its repo cwd. We stash the
    # per-request YAML inside the output_dir so it gets garbage-collected
    # naturally with the rest of the request artifacts.
    yaml_path = os.path.join(output_dir, "inference.yaml")
    _write_inference_config(yaml_path, image_path, audio_path)

    if config.MUSETALK_VERSION == "v15":
        unet_model = "models/musetalkV15/unet.pth"
        unet_config = "models/musetalkV15/musetalk.json"
        version_arg = "v15"
    else:
        unet_model = "models/musetalk/pytorch_model.bin"
        unet_config = "models/musetalk/musetalk.json"
        version_arg = "v1"

    cmd = [
        "python3", "-m", "scripts.inference",
        "--inference_config", yaml_path,
        "--result_dir", output_dir,
        "--unet_model_path", unet_model,
        "--unet_config", unet_config,
        "--version", version_arg,
    ]
    if config.MUSETALK_USE_FLOAT16:
        cmd.append("--use_float16")

    env = {**os.environ}
    if config.GPU_TYPE == "rocm":
        env.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    # MuseTalk's pinned deps (diffusers 0.30.2, decord, face_alignment) live
    # in a PYTHONUSERBASE overlay so they don't clobber the transformers 5.x
    # the rest of the worker relies on. Compose injects this; reassert here
    # to survive direct-uvicorn dev invocations.
    env.setdefault("PYTHONUSERBASE", config.MUSETALK_PYENV)

    logger.info(f"Running MuseTalk ({version_arg}, float16={config.MUSETALK_USE_FLOAT16})…")
    t0 = time.time()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=config.MUSETALK_DIR,
        start_new_session=True,
    )
    _running.add(proc)
    try:
        stdout, stderr = await proc.communicate()
    except asyncio.CancelledError:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        raise
    finally:
        _running.discard(proc)

    if proc.returncode != 0:
        tail = stderr.decode()[-800:]
        if proc.returncode in (-signal.SIGTERM, -signal.SIGKILL):
            raise RuntimeError(f"MuseTalk cancelled (signal {-proc.returncode})")
        raise RuntimeError(f"MuseTalk failed (rc={proc.returncode}): {tail}")

    elapsed = time.time() - t0
    mp4s = sorted(
        glob.glob(os.path.join(output_dir, "**/*.mp4"), recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    if not mp4s:
        raise RuntimeError(f"MuseTalk produced no .mp4 in {output_dir}")
    logger.info(f"MuseTalk → {mp4s[0]} ({elapsed:.1f}s)")
    return mp4s[0]
