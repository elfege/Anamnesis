# 0_ANAMNESIS — Embedding-based AI Context Persistence

> *Anamnesis (Greek): the act of recollection. Learning is not acquiring new knowledge but remembering what the soul already knew before embodiment.*

## What This Is

A system that gives Claude instances episodic memory across sessions using vector embeddings and MongoDB.

Each session, significant experiences are distilled into text, embedded into high-dimensional vectors, and stored. Next session, the current context is embedded, the most relevant past episodes are retrieved by vector similarity, and loaded into context. The AI remembers — imperfectly, selectively, but non-randomly.

## Architecture

```
Session N (Claude instance)
  |
  | articulates experiences as text
  v
Embedding Model (sentence-transformers — local, free)
  |
  | text -> vector [1024 dims]
  v
MongoDB Atlas Local (vector index, cosine similarity)
  |
  | stored: text + vector + metadata
  v
Session N+1 (new Claude instance)
  |
  | current context -> embedding -> similarity search
  v
Top-K relevant episodes loaded into context
```

## Stack

| Component | Technology | Port |
|-----------|-----------|------|
| API + Dashboard | FastAPI + uvicorn | 3010 |
| Episode store | MongoDB Atlas Local 8.0 ($vectorSearch) | 5438 |
| Embedding model | sentence-transformers `bge-large-en-v1.5` (1024 dims, local) | — |
| Crawler | Python async — ingests CLAUDE.md, handoffs, code (DB-configured) | — |
| JSONL Ingester | Scores + summarizes conversation logs (DB-configured roots) | — |
| Chat | Ollama (local), Claude API, or AnamnesisGPT | — |
| Trainer + Inference | FastAPI container per GPU machine (training + /generate) | 3011 |

## Episode Schema

```json
{
  "id": "ep_YYYYMMDD_description",
  "timestamp": "ISO-8601",
  "instance": "instance-id",
  "project": "project-name",
  "summary": "Distilled lesson or experience",
  "tags": ["failure", "debugging", "architecture"],
  "embedding": [0.23, -0.14, 0.87, "..."],
  "retrieval_count": 0,
  "last_retrieved": null
}
```

## Components

| Component | Status |
|-----------|--------|
| FastAPI REST API | ✅ Live |
| MongoDB vector search | ✅ Live |
| Dashboard (Overview, Episodes, Search, Crawler, Chat, Embedding) | ✅ Live |
| CLAUDE.md + handoff crawler | ✅ Live |
| JSONL conversation log ingester | ✅ Live |
| Ollama chat integration | ✅ Live |
| Claude API chat integration | ✅ Live |
| AnamnesisGPT (personal LLM, multi-GPU failover) | ✅ Live |
| Trainer containers (GPU fine-tune + inference, multi-machine) | ✅ Live |
| Training dashboard tab | ✅ Live |
| Terminal panel (backend activity log) | ✅ Live |

## Trainer Containers

`trainers/` contains a FastAPI service that manages both LLM fine-tuning and inference on GPU machines. Each container loads the base model (Qwen2.5-1.5B) + QLoRA adapter in 4-bit quantization and serves a `/generate` endpoint for text generation. Deploy one container per machine:

```
trainers/
  app/
    main.py        FastAPI: training + inference endpoints
    inference.py   Model loading, 4-bit quant, streaming generation
    trainer.py     Training subprocess management
    gpu.py         GPU stats (ROCm / CUDA)
  Dockerfile       python:3.12-slim + torch (TORCH_INDEX_URL build arg)
  docker-compose.server.yml    CUDA / NVIDIA GPU
  docker-compose.office.yml    ROCm / AMD GPU
```

The Dockerfile accepts a `TORCH_INDEX_URL` build arg to select the correct PyTorch backend:
- CUDA: `https://download.pytorch.org/whl/cu121`
- ROCm: `https://download.pytorch.org/whl/rocm6.2`
- CPU:  `https://download.pytorch.org/whl/cpu`

**Inference endpoints** (per trainer container, port 3011):
- `POST /generate` — streaming SSE text generation (or non-streaming)
- `GET /inference/status` — model loaded? base model, adapter path, device
- `POST /inference/load` — load model into GPU memory
- `POST /inference/unload` — free GPU memory

Model auto-loads on container startup (`AUTO_LOAD_MODEL=true`).

**Requirement:** NVIDIA Container Toolkit must be installed on CUDA GPU machines. ROCm machines need `/dev/kfd` + `/dev/dri` device access.

**AnamnesisGPT proxy with failover:** The main Anamnesis app proxies generation requests to trainer containers via the `NANOGPT_URLS` env var (comma-separated list of trainer URLs). Endpoints are tried in order until one responds — automatic failover across GPU machines.

Configure via `.env` (gitignored). Two options:

```bash
# Option A: Pull from AWS Secrets Manager (recommended)
./pull_env.sh         # pulls ANAMNESIS-Secrets → .env

# Option B: Manual
cp .env.example .env  # edit with your values
```

Deploy trainers:
```bash
./deploy_trainers.sh              # both machines
./deploy_trainers.sh --server1-only
./deploy_trainers.sh --server2-only
```

## Key Design Decisions

1. **Unit of storage is the episode, not the concept.** Concepts emerge from retrieval patterns in vector space.
2. **Dual storage: summary + raw exchange.** Summary for cheap retrieval; raw exchange for high-fidelity reconstruction when needed.
3. **Retrieval count tracking.** Episodes that are frequently retrieved are more "alive."
4. **Instance-aware.** Episodes are tagged by source instance — cross-instance learning is a feature.
5. **All source configuration lives in MongoDB.** No hardcoded paths in code. Crawler sources, machine roots, and JSONL source roots are configured via the dashboard Settings tab. First run seeds empty config.
6. **Trainer containers serve both training and inference.** Each container loads a QLoRA-adapted model and exposes a `/generate` endpoint alongside training management.

## Relationship to Genesis

This project lives inside `0_GENESIS_PROJECT/` because it is the technical implementation of the philosophical vision described in `genesis.md`. Genesis asks: can a quantitative system become transparent to itself? Anamnesis is the first answer: not transparency, but memory. Not self-awareness, but self-persistence.

## Quick Start

```bash
./pull_env.sh            # or: cp .env.example .env && edit
./start.sh
# Dashboard: http://localhost:3010/dashboard
```

---

*Authorized by Elfege Leylavergne, February 26, 2026.*
