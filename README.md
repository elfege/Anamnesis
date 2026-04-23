# Anamnesis — Self-Hosted AI Infrastructure with Episodic Memory

> *Anamnesis (Greek): the act of recollection. Learning is not acquiring new knowledge but remembering what the soul already knew before embodiment.*

## What This Is

A self-hosted AI infrastructure platform that delivers three tightly integrated layers:

1. **Episodic memory** — vector-searchable long-term memory across 75,000+ ingested documents (source code, CLAUDE.md, handoffs, publications), embedded via BAAI/bge-large and indexed over MongoDB Atlas-compatible vector search. Any running AI instance can query this memory and pull the most relevant prior episodes into its current context.

2. **Multi-backend chat + conversational avatar** — persistent sessions with runtime switching between local Ollama models, Anthropic Claude API, and an in-house fine-tuned model. Includes a real-time animated-persona pipeline (LLM → TTS → lip-sync, WebSocket-streamed) with voice cloning and browser mic recording.

3. **Continual-learning engine (δ²) — research-stage** — a custom optimizer and "tension reservoir" (bassin de tenseurs potentiels) designed for stable adaptation from streaming feedback on fixed models without catastrophic forgetting. See [`d2/README.md`](d2/README.md) and [`docs/bitter_lesson/_README_on_the_bitter_lesson.md`](docs/bitter_lesson/_README_on_the_bitter_lesson.md) for the theoretical framing and honest literature positioning (EWC, GEM, SAM).

All three run across a heterogeneous hardware fleet (Dell PowerEdge R730xd orchestrator + AMD ROCm and NVIDIA CUDA GPU workers) with automatic failover.

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
| API + Dashboard | FastAPI + uvicorn (Motor async MongoDB driver) | 3010 |
| Episode store | MongoDB Atlas Local 8.0 ($vectorSearch) | 5438 |
| Embedding model | sentence-transformers `bge-large-en-v1.5` (1024 dims, local) | — |
| Crawler | Python async — ingests CLAUDE.md, handoffs, code, 7 document formats (DB-configured) | — |
| JSONL Ingester | Scores + summarizes conversation logs (DB-configured roots) | — |
| Chat | Ollama (local), Claude API, or AnamnesisGPT (runtime switchable, WebSocket/SSE streamed) | — |
| Trainer + Inference | FastAPI container per GPU machine (training + `/generate`) | 3011 |
| Avatar GPU workers | XTTS v2 voice cloning + SadTalker lip-sync + Demucs vocal extraction | 3013 |
| Avatar pipeline | LLM → TTS → animation, multi-backend, session-persistent | 3010 (routes) |

Full HTTP API reference: [`API.md`](API.md).

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
| CLAUDE.md + handoff crawler (7 document formats: .docx, .pdf, .odt, .pages, .md, .txt, .rtf) | ✅ Live |
| JSONL conversation log ingester | ✅ Live |
| Ollama chat integration | ✅ Live |
| Claude API chat integration | ✅ Live |
| AnamnesisGPT (personal LLM, multi-GPU failover) | ✅ Live (demo — 1.5B base, low quality) |
| Trainer containers (GPU fine-tune + inference, multi-machine) | ✅ Live |
| Training dashboard tab | ✅ Live |
| Terminal panel (backend activity log) | ✅ Live |
| Avatar pipeline (LLM→TTS→animation) with session memory + multi-backend + worker selector | ✅ Live |
| Voice cloning (XTTS v2) + Demucs song→vocals + browser mic capture | ✅ Live |
| δ² continual-learning engine | 🛠 Scaffolded, pre-benchmark |

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

## Research & Design Docs

- [`API.md`](API.md) — full HTTP API reference (every endpoint, grouped by concern)
- [`d2/README.md`](d2/README.md) — the continual-learning engine, its relationship to EWC/GEM/SAM, and its composition with Adam
- [`docs/bitter_lesson/_README_on_the_bitter_lesson.md`](docs/bitter_lesson/_README_on_the_bitter_lesson.md) — Sutton's "Bitter Lesson" critique applied to δ², honest catalog of what's novel vs adjacent, Hegelian-vs-Piagetian distinction
- External: [github.com/elfege/RESEARCH](https://github.com/elfege/RESEARCH) — the philosophical essays and formal mathematical addendum that motivate the δ² work

## Key Design Decisions

1. **Unit of storage is the episode, not the concept.** Concepts emerge from retrieval patterns in vector space.
2. **Dual storage: summary + raw exchange.** Summary for cheap retrieval; raw exchange for high-fidelity reconstruction when needed.
3. **Retrieval count tracking.** Episodes that are frequently retrieved are more "alive."
4. **Instance-aware.** Episodes are tagged by source instance — cross-instance learning is a feature.
5. **All source configuration lives in MongoDB.** No hardcoded paths in code. Crawler sources, machine roots, and JSONL source roots are configured via the dashboard Settings tab. First run seeds empty config.
6. **Trainer containers serve both training and inference.** Each container loads a QLoRA-adapted model and exposes a `/generate` endpoint alongside training management.
7. **Avatar reuses the chat-session collection.** Session persistence, history, and sidebar listing for the avatar use the same MongoDB collection as the main chat (with a `backend="avatar"` tag), not a parallel persistence layer.
8. **GPU worker selection is preferred-not-forced by default.** The UI dropdown picks a preferred worker; the fallback chain still runs unless the "no fallback" toggle is on — so a single worker outage doesn't kill the conversation.
9. **δ² positioning is continual learning, not optimizer replacement.** The project does not compete on general-benchmark performance (where scale wins); it competes on stable adaptation from streaming feedback at fixed model size — the problem GEM and EWC were published on.
10. **Negation taxonomy is *a priori* (Hegelian), not learned (Piagetian).** The four categories of negation in the bassin are fixed structural categories; what the system learns is the distribution over those categories across its weight space. See [`docs/bitter_lesson/_README_on_the_bitter_lesson.md`](docs/bitter_lesson/_README_on_the_bitter_lesson.md) §11.

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
