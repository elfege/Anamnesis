FROM python:3.12-slim

# Build arg: PyTorch index URL varies by GPU backend
# cuda:  https://download.pytorch.org/whl/cu121
# rocm:  https://download.pytorch.org/whl/rocm6.3
# cpu:   https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps (base — no torch yet for CPU/edge-tts-only mode)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch only if LivePortrait animation is needed
# (controlled by ANIM_BACKEND env at runtime, but torch must be present at build time)
ARG INSTALL_TORCH=false
RUN if [ "$INSTALL_TORCH" = "true" ]; then \
      pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch torchvision; \
    fi

# Clone LivePortrait if building with animation support
ARG INSTALL_LIVEPORTRAIT=false
RUN if [ "$INSTALL_LIVEPORTRAIT" = "true" ]; then \
      git clone https://github.com/KwaiVGI/LivePortrait.git /opt/LivePortrait && \
      cd /opt/LivePortrait && \
      pip install --no-cache-dir -r requirements.txt; \
    fi

# App code
COPY app/ /app/

EXPOSE 3012

CMD ["python", "main.py"]
