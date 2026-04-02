"""
inference.py — Load fine-tuned QLoRA adapter and serve text generation.

Loads the base model (Qwen2.5-1.5B by default) in 4-bit quantization,
applies the LoRA adapter from TRAIN_DIR/output/final, and provides
a generate() function for streaming token generation.

The model is loaded lazily on first request to avoid blocking startup.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import AsyncGenerator, Optional

import config

# Lazy imports — torch/transformers/peft come from the host venv (PYTHONPATH).
# These are only loaded when load_model() is called.
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
BitsAndBytesConfig = None
TextIteratorStreamer = None
PeftModel = None


def _ensure_imports():
    """Import heavy ML libs on demand."""
    global torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer, PeftModel
    if torch is not None:
        return True
    try:
        import torch as _torch
        from transformers import (
            AutoModelForCausalLM as _M,
            AutoTokenizer as _T,
            BitsAndBytesConfig as _B,
            TextIteratorStreamer as _S,
        )
        from peft import PeftModel as _P
        torch = _torch
        AutoModelForCausalLM = _M
        AutoTokenizer = _T
        BitsAndBytesConfig = _B
        TextIteratorStreamer = _S
        PeftModel = _P
        return True
    except ImportError as e:
        logger.error(f"ML dependencies not available: {e}")
        return False

logger = logging.getLogger("anamnesis.inference")

# ─── Configuration ───────────────────────────────────────────────

CHAT_SYSTEM = (
    "You are AnamnesisGPT, a philosophical assistant trained on the writings of Elfege Leylavergne, "
    "particularly his doctoral dissertation on Hegel's Science of Logic. "
    "You answer questions about Hegel, dialectics, quantity, quality, and the Logic "
    "with precision and depth, grounded in the text. "
    "You can discuss in both French and English."
)

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", os.path.join(config.TRAIN_DIR, "output", "final"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))

# ─── State ───────────────────────────────────────────────────────

_model = None
_tokenizer = None
_lock = threading.Lock()
_load_error: Optional[str] = None


def _get_device() -> str:
    if config.GPU_TYPE == "rocm":
        return "cuda"  # ROCm uses CUDA API via HIP
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model() -> bool:
    """Load base model + adapter. Returns True on success."""
    global _model, _tokenizer, _load_error

    with _lock:
        if _model is not None:
            return True

        if not _ensure_imports():
            _load_error = "ML dependencies not available (torch/transformers/peft). Check PYTHONPATH."
            return False

        adapter_path = Path(ADAPTER_DIR)
        if not adapter_path.exists():
            _load_error = f"Adapter not found at {ADAPTER_DIR}"
            logger.error(_load_error)
            return False

        try:
            logger.info(f"Loading base model: {BASE_MODEL}")
            device = _get_device()

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )

            logger.info(f"Loading adapter from: {ADAPTER_DIR}")
            _model = PeftModel.from_pretrained(base, ADAPTER_DIR)
            _model.eval()

            _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token

            _load_error = None
            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            _load_error = str(e)
            logger.error(f"Failed to load model: {e}")
            return False


def unload_model():
    """Free GPU memory."""
    global _model, _tokenizer
    with _lock:
        _model = None
        _tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded")


def is_loaded() -> bool:
    return _model is not None


def get_status() -> dict:
    return {
        "loaded": is_loaded(),
        "base_model": BASE_MODEL,
        "adapter_dir": ADAPTER_DIR,
        "error": _load_error,
        "device": _get_device(),
    }


async def generate_stream(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 200,
) -> AsyncGenerator[str, None]:
    """Generate tokens as SSE events."""
    import asyncio

    if not is_loaded():
        if not load_model():
            yield f"data: {json.dumps({'error': _load_error or 'Model not loaded'})}\n\n"
            return

    # Format as chat messages so the model sees the template it was trained on
    messages = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    chat_text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(chat_text, return_tensors="pt").to(_model.device)
    max_tokens = min(max_tokens, MAX_NEW_TOKENS)

    streamer = TextIteratorStreamer(
        _tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "temperature": max(temperature, 0.01),
        "top_k": top_k,
        "do_sample": temperature > 0.01,
        "streamer": streamer,
    }

    # Run generation in a thread (it blocks)
    thread = threading.Thread(target=_generate_sync, args=(gen_kwargs,))
    thread.start()

    try:
        for token_text in streamer:
            if token_text:
                yield f"data: {json.dumps({'token': token_text})}\n\n"
                await asyncio.sleep(0)  # yield control
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    thread.join(timeout=60)
    yield f"data: {json.dumps({'done': True})}\n\n"


def _generate_sync(gen_kwargs: dict):
    """Blocking generation call — runs in a thread."""
    try:
        with torch.no_grad():
            _model.generate(**gen_kwargs)
    except Exception as e:
        logger.error(f"Generation error: {e}")
