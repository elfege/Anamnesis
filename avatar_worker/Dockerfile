FROM python:3.12-slim

# Torch index varies by backend (set per compose file)
#  cuda: https://download.pytorch.org/whl/cu121
#  rocm: https://download.pytorch.org/whl/rocm6.3
#  cpu:  https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch (backend-specific)
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch torchvision torchaudio

# numpy<2 required by SadTalker (uses removed np.VisibleDeprecationWarning)
RUN pip install --no-cache-dir "numpy<2"

# Heavy deps — SadTalker, XTTS, Demucs
RUN pip install --no-cache-dir \
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
      "transformers<5" \
      "coqui-tts[codec]" \
      demucs \
      soundfile

# Accept Coqui TTS terms (XTTS v2)
ENV COQUI_TOS_AGREED=1

COPY app/ /app/

EXPOSE 3013

CMD ["python", "main.py"]
