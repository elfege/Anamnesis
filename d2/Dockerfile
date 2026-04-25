# Dockerfile for the δ² engine service.
#
# Builds a container that runs d2/server.py (FastAPI) on port 3015.
# The container has GPU access via NVIDIA Container Toolkit (or ROCm
# devices for AMD). Hardware backend is selected at build time via the
# TORCH_INDEX_URL build arg.
#
# BUILD (CUDA — for server, RunPod, etc.):
#   docker build -t anamnesis-d2:cuda \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
#     -f d2/Dockerfile .
#
# BUILD (ROCm — for office RX 6800):
#   docker build -t anamnesis-d2:rocm \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.2 \
#     -f d2/Dockerfile .
#
# RUN:
#   docker run --gpus all -p 3015:3015 \
#     -v ./d2_runs:/workspace/runs \
#     anamnesis-d2:cuda
#
# OR via compose (preferred):
#   docker compose -f d2/docker-compose.yml up -d

FROM python:3.12-slim

# ── Build args ──────────────────────────────────────────────────────────────
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

# ── System deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ─────────────────────────────────────────────────────────────
# We install torch separately because the index URL varies by hardware.
RUN pip install --no-cache-dir --upgrade pip wheel
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX_URL}

# Lightweight inference + serving deps.
# transformers/datasets are pulled in for tokenizer + dataset prep.
RUN pip install --no-cache-dir \
        fastapi==0.115.0 \
        uvicorn==0.30.6 \
        pydantic==2.9.2 \
        httpx==0.27.2 \
        numpy \
        tiktoken \
        datasets

# ── Copy d2 module ──────────────────────────────────────────────────────────
WORKDIR /app
COPY d2/ /app/d2/

# Make /app the package root so `python -m d2.server` works
ENV PYTHONPATH=/app
ENV D2_PROJECT_DIR=/app
ENV D2_SERVICE_PORT=3015
# Where benchmark run outputs are written (mount this as a volume to persist)
ENV D2_RUNS_DIR=/workspace/runs

# ── Healthcheck (used by docker-compose's depends_on) ──────────────────────
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:3015/health || exit 1

EXPOSE 3015

# Run the FastAPI service
CMD ["python", "-m", "d2.server"]
