import os

GPU_TYPE = os.environ.get("GPU_TYPE", "cuda")  # rocm | cuda
MACHINE_NAME = os.environ.get("MACHINE_NAME", "unknown")
TRAIN_DIR = os.environ.get("TRAIN_DIR", "/train")
VENV_PYTHON = os.environ.get("VENV_PYTHON", "/train/venv/bin/python")
TRAIN_SCRIPT = os.environ.get("TRAIN_SCRIPT", "/train/qlora_train.py")
LOG_FILE = os.path.join(TRAIN_DIR, "train.log")
PORT = int(os.environ.get("PORT", 3011))
