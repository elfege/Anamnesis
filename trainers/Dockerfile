FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends procps && rm -rf /var/lib/apt/lists/*

# Base requirements (FastAPI, uvicorn, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ML/inference deps — torch index URL varies by GPU backend.
# CUDA:  https://download.pytorch.org/whl/cu121
# ROCm:  https://download.pytorch.org/whl/rocm6.3
# CPU:   https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# Install torch from the GPU-specific index (--index-url so ROCm isn't shadowed by PyPI CUDA)
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch
# Then install the rest from PyPI (they don't need the GPU index)
RUN pip install --no-cache-dir transformers peft bitsandbytes accelerate trl datasets

WORKDIR /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3011", "--reload"]
