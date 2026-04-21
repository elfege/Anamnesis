FROM python:3.12-slim

# Build arg: PyTorch index URL varies by GPU backend
# cuda:  https://download.pytorch.org/whl/cu121
# rocm:  https://download.pytorch.org/whl/rocm6.3
# cpu:   https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# System deps (cmake + build-essential needed for dlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps (base — no torch yet for CPU/edge-tts-only mode)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch when GPU animation is needed
ARG INSTALL_TORCH=false
RUN if [ "$INSTALL_TORCH" = "true" ]; then \
      pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch torchvision torchaudio; \
    fi

# SadTalker dependencies (installed when torch is present)
# SadTalker itself is mounted as a volume at /opt/SadTalker
RUN if [ "$INSTALL_TORCH" = "true" ]; then \
      pip install --no-cache-dir \
        face_alignment \
        dlib \
        imageio \
        imageio-ffmpeg \
        librosa \
        numba \
        resampy \
        pydub \
        scipy \
        kornia \
        yacs \
        gfpgan \
        facexlib \
        basicsr \
        safetensors; \
    fi

# App code
COPY app/ /app/

EXPOSE 3012

CMD ["python", "main.py"]
