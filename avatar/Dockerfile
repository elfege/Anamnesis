FROM python:3.12-slim

# Build arg: PyTorch index URL varies by GPU backend
# cuda:  https://download.pytorch.org/whl/cu121
# rocm:  https://download.pytorch.org/whl/rocm6.3
# cpu:   https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# System deps (cmake + build-essential for dlib; libsndfile1 for audio I/O)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Base Python deps (no torch — fits CPU + GPU flows)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch (with matching backend) when animation/XTTS/demucs is wanted
ARG INSTALL_TORCH=false
RUN if [ "$INSTALL_TORCH" = "true" ]; then \
      pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch torchvision torchaudio; \
    fi

# SadTalker + voice cloning deps (only when torch is present)
# SadTalker itself is bind-mounted at /opt/SadTalker.
# numpy<2 is required because SadTalker references np.VisibleDeprecationWarning.
RUN if [ "$INSTALL_TORCH" = "true" ]; then \
      pip install --no-cache-dir "numpy<2" && \
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
        safetensors \
        coqui-tts \
        demucs \
        soundfile; \
    fi

# Persistent voice storage (bind-mounted by compose)
RUN mkdir -p /app/voices

# App code
COPY app/ /app/

EXPOSE 3012

CMD ["python", "main.py"]
