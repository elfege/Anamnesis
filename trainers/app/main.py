from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import aiofiles

import config
import gpu
import trainer

app = FastAPI(title="Anamnesis Trainer API")


class StartBody(BaseModel):
    resume: Optional[str] = None


@app.get("/health")
async def health():
    return {
        "ok": True,
        "machine": config.MACHINE_NAME,
        "gpu_type": config.GPU_TYPE,
    }


@app.get("/status")
async def status():
    running = trainer.is_running()
    log_data = trainer.parse_log()
    gpu_stats = await gpu.get_gpu_stats(config.GPU_TYPE)
    exit_code = trainer.get_exit_code()

    done = not running and trainer._state["proc"] is not None

    return {
        "running": running,
        "done": done,
        "exit_code": exit_code,
        "pid": trainer._state["pid"],
        "started_at": trainer._state["started_at"],
        "progress": log_data["progress"],
        "latest_metrics": log_data["latest_metrics"],
        "history": log_data["history"],
        "gpu": gpu_stats,
        "machine": config.MACHINE_NAME,
        "gpu_type": config.GPU_TYPE,
    }


@app.post("/start")
async def start(body: StartBody = StartBody()):
    result = trainer.start_training(resume=body.resume)
    return result


@app.post("/stop")
async def stop():
    return trainer.stop_training()


@app.get("/log/tail")
async def log_tail(lines: int = Query(default=50, ge=1, le=2000)):
    try:
        async with aiofiles.open(config.LOG_FILE, "r", errors="replace") as f:
            content = await f.read()
        tail = content.splitlines()[-lines:]
        return {"lines": tail, "total_shown": len(tail)}
    except FileNotFoundError:
        return {"lines": [], "total_shown": 0, "error": "log file not found"}
