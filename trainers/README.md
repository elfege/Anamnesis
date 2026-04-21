# Anamnesis Trainer

GPU-based training and inference server for **AnamnesisGPT** — a LoRA fine-tuned Qwen2.5-1.5B trained on Elfege Leylavergne's doctoral dissertation on Hegel's Science of Logic.

## Architecture

```
dellserver:3010 (Anamnesis main app, CPU-only)
│
│  POST /api/anamnesis-gpt/generate  (proxy)
│  GET  /api/anamnesis-gpt/status
↓
office:3011 (anamnesis-trainer container, AMD RX 6800 16GB)
│
├── /generate          → streaming inference (SSE)
├── /inference/load    → load model into GPU
├── /inference/unload  → free GPU memory
├── /start             → start training run
├── /stop              → stop training
├── /status            → GPU stats + training progress
└── /gpu               → lightweight GPU stats (500ms polling)
```

## Quick Start

### Deploy on office (AMD RX 6800, ROCm)

```bash
cd ~/0_GENESIS_PROJECT/0_ANAMNESIS/trainers

# First time: build + start
./deploy.sh --host office

# Subsequent starts (no rebuild)
./start.sh --host office

# Stop
./stop.sh
```

### Deploy on server (NVIDIA GTX 1660 SUPER, CUDA)

```bash
./deploy.sh --host server
```

### Systemd (auto-start on boot)

```bash
sudo cp anamnesis-trainer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable anamnesis-trainer
sudo systemctl start anamnesis-trainer
```

## Training Data Pipeline

Training data is prepared in 3 steps, all in `tools/`:

### Step 1: Extract text from dissertation PDFs

```bash
python tools/extract_pdf.py \
  --input "/path/to/dissertation.pdf" \
  --output train_data/chunks.jsonl
```

### Step 2: Generate Q&A pairs (requires Claude API key)

```bash
export ANTHROPIC_API_KEY=sk-...
python tools/generate_qa.py \
  --input train_data/chunks.jsonl \
  --output train_data/sft_chat_full.jsonl \
  --model claude-opus-4-6
```

### Step 3: Train/val split

```bash
python tools/split_data.py \
  --input train_data/sft_chat_full.jsonl \
  --train train_data/sft_train.jsonl \
  --val train_data/sft_val.jsonl \
  --val-ratio 0.1
```

## Training

Training data lives on the host at `$TRAIN_HOST_DIR` (default: `~/0_LLM_finetune`), mounted as `/train` inside the container.

### Start a training run

```bash
# Via API
curl -X POST http://localhost:3011/start

# Resume from checkpoint
curl -X POST http://localhost:3011/start \
  -H "Content-Type: application/json" \
  -d '{"resume": "output/checkpoint-500"}'
```

### Monitor training

```bash
# Full status (GPU + progress + metrics)
curl http://localhost:3011/status

# GPU stats only (fast, safe for polling)
curl http://localhost:3011/gpu

# Training log tail
curl 'http://localhost:3011/log/tail?lines=50'
```

### Training configuration

The training script (`qlora_train.py`) uses these defaults:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Qwen/Qwen2.5-1.5B | |
| LoRA rank | 16 | |
| LoRA alpha | 32 | |
| Target modules | q/k/v/o/gate/up/down_proj | All attention + MLP |
| Batch size | 1 | Limited by VRAM |
| Gradient accumulation | 16 | Effective batch = 16 |
| Learning rate | 2e-4 | Cosine schedule |
| Epochs | 3 | |
| Max seq length | 1024 | |
| Checkpoints | Every 30 steps | Keep last 3 |

### Output

```
~/0_LLM_finetune/output/
├── checkpoint-*/          # Intermediate checkpoints
└── final/                 # Final adapter (auto-loaded at inference)
    ├── adapter_config.json
    ├── adapter_model.safetensors  (~74 MB)
    ├── chat_template.jinja
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── training_args.bin
```

## Inference

The trainer auto-loads the adapter from `/train/output/final` on startup (`AUTO_LOAD_MODEL=true`).

### Generate text

```bash
# Streaming (SSE)
curl -N -X POST http://localhost:3011/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the role of quantity in Hegel'\''s Science of Logic?",
    "max_tokens": 512,
    "temperature": 0.7,
    "stream": true
  }'

# Non-streaming
curl -X POST http://localhost:3011/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain dialectics", "max_tokens": 256, "stream": false}'
```

### Model management

```bash
curl http://localhost:3011/inference/status   # Check if loaded
curl -X POST http://localhost:3011/inference/load     # Load into GPU
curl -X POST http://localhost:3011/inference/unload   # Free GPU memory
```

## ROCm vs CUDA

| Feature | ROCm (office, RX 6800) | CUDA (server, GTX 1660) |
|---------|------------------------|-------------------------|
| Compose file | `docker-compose.office.yml` | `docker-compose.server.yml` |
| Torch index | `rocm6.3` | `cu121` |
| Quantization | fp16 (no BitsAndBytes) | 4-bit BitsAndBytes |
| Inference VRAM | ~3 GB (1.5B fp16) | ~1.5 GB (1.5B 4-bit) |
| Training VRAM | ~10-12 GB (LoRA fp16) | ~4-6 GB (QLoRA 4-bit) |
| Device passthrough | `/dev/kfd` + `/dev/dri` | NVIDIA runtime |

**Note**: BitsAndBytes 4-bit silently falls back to CPU on ROCm, causing ~0.07 tok/s inference. The inference module detects ROCm and loads in fp16 instead.

## Docker Compose Files

| File | GPU | Machine |
|------|-----|---------|
| `docker-compose.office.yml` | AMD RX 6800 (ROCm) | office (192.168.10.110) |
| `docker-compose.server.yml` | NVIDIA GTX 1660 (CUDA) | server (192.168.10.15) |
| `docker-compose.trainer1.yml` | ROCm (generic) | Any ROCm host |
| `docker-compose.trainer2.yml` | CUDA (generic) | Any CUDA host |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_TYPE` | `cuda` | `rocm` or `cuda` |
| `MACHINE_NAME` | `unknown` | Label for status endpoint |
| `TRAIN_DIR` | `/train` | Container path to training data |
| `TRAIN_SCRIPT` | `/train/qlora_train.py` | Training script path |
| `VENV_PYTHON` | `/train/venv/bin/python` | Python binary for training |
| `BASE_MODEL` | `Qwen/Qwen2.5-1.5B` | HuggingFace model ID |
| `ADAPTER_DIR` | `$TRAIN_DIR/output/final` | Path to LoRA adapter |
| `AUTO_LOAD_MODEL` | `true` | Load model on container startup |
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | ROCm GFX version override for RX 6800 |

## File Structure

```
trainers/
├── Dockerfile                     # Trainer image (Python 3.12 + torch + peft)
├── requirements.txt               # FastAPI, uvicorn, psutil
├── docker-compose.office.yml      # Office RX 6800 (ROCm)
├── docker-compose.server.yml      # Server GTX 1660 (CUDA)
├── docker-compose.trainer1.yml    # Generic ROCm
├── docker-compose.trainer2.yml    # Generic CUDA
├── deploy.sh                      # Build + start
├── start.sh                       # Start (no build)
├── stop.sh                        # Stop
├── anamnesis-trainer.service      # Systemd unit file
├── app/
│   ├── main.py                    # FastAPI app (training + inference endpoints)
│   ├── config.py                  # Environment config
│   ├── trainer.py                 # Training subprocess management
│   ├── inference.py               # Model loading + streaming generation
│   └── gpu.py                     # GPU stats (rocm-smi / nvidia-smi)
└── tools/
    ├── extract_pdf.py             # PDF → text chunks
    ├── generate_qa.py             # Chunks → Q&A pairs (Claude API)
    ├── split_data.py              # Train/val split
    └── train_status.sh            # Quick status check script
```
