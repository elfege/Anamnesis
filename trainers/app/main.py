import os

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import aiofiles

import config
import gpu
import trainer
import inference

app = FastAPI(title="Anamnesis Trainer API")

AUTO_LOAD_MODEL = os.environ.get("AUTO_LOAD_MODEL", "true").lower() == "true"


@app.on_event("startup")
async def _startup():
    if AUTO_LOAD_MODEL:
        import threading
        threading.Thread(target=inference.load_model, daemon=True).start()


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


@app.get("/gpu")
async def gpu_stats():
    """Lightweight GPU stats only — no log parsing. Safe to poll at 500ms."""
    return await gpu.get_gpu_stats(config.GPU_TYPE)


@app.get("/log/tail")
async def log_tail(lines: int = Query(default=50, ge=1, le=2000)):
    try:
        async with aiofiles.open(config.LOG_FILE, "r", errors="replace") as f:
            content = await f.read()
        tail = content.splitlines()[-lines:]
        return {"lines": tail, "total_shown": len(tail)}
    except FileNotFoundError:
        return {"lines": [], "total_shown": 0, "error": "log file not found"}


# ─── Inference endpoints ─────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    max_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.8, ge=0.01, le=2.0)
    top_k: int = Field(200, ge=1, le=1000)
    stream: bool = Field(True)


@app.get("/inference/status")
async def inference_status():
    return inference.get_status()


@app.post("/inference/load")
async def inference_load():
    """Load the fine-tuned model into GPU memory."""
    if inference.is_loaded():
        return {"ok": True, "message": "Model already loaded"}
    success = inference.load_model()
    if success:
        return {"ok": True, "message": "Model loaded"}
    return {"ok": False, "error": inference.get_status()["error"]}


@app.post("/inference/unload")
async def inference_unload():
    """Unload model and free GPU memory."""
    inference.unload_model()
    return {"ok": True, "message": "Model unloaded"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate text from the fine-tuned model. Streams SSE by default."""
    if req.stream:
        return StreamingResponse(
            inference.generate_stream(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
            ),
            media_type="text/event-stream",
        )
    # Non-streaming: collect all tokens
    tokens = []
    async for event in inference.generate_stream(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    ):
        import json as _json
        if event.startswith("data:"):
            try:
                d = _json.loads(event[5:].strip())
                if d.get("token"):
                    tokens.append(d["token"])
                if d.get("error"):
                    return {"error": d["error"]}
            except Exception:
                pass
    return {"text": "".join(tokens)}
