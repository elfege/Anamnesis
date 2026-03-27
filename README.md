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
| Crawler | Python async — ingests CLAUDE.md, handoffs, code | — |
| JSONL Ingester | Scores + summarizes conversation logs | — |
| Chat | Ollama (local) or Claude API | — |
| Trainer | FastAPI container per GPU machine | 3011 |

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
| AnamnesisGPT (personal LLM, nanoGPT) | ✅ Live |
| Trainer containers (GPU fine-tune, multi-machine) | ✅ Live |
| Training dashboard tab | ✅ Live |

## Trainer Containers

`trainers/` contains a lightweight FastAPI service that manages LLM fine-tuning jobs on GPU machines. Deploy one container per machine:

```
trainers/
  app/           FastAPI trainer API (status, start, stop, log tail)
  Dockerfile     python:3.12-slim — no torch (uses host venv via mount)
  docker-compose.trainer1.yml   ROCm / AMD GPU
  docker-compose.trainer2.yml   CUDA / NVIDIA GPU
```

Configure hosts in `.env` (gitignored):
```bash
TRAINER_1_HOST=<ssh-alias>
TRAINER_2_HOST=<ssh-alias>
TRAINER_URLS=server-1:http://<host1>:3011,server-2:http://<host2>:3011
```

Deploy:
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
5. **Trainer containers are thin wrappers.** GPU access via device mounts; PyTorch runs from the host venv — no torch rebuild in Docker.

## Relationship to Genesis

This project lives inside `0_GENESIS_PROJECT/` because it is the technical implementation of the philosophical vision described in `genesis.md`. Genesis asks: can a quantitative system become transparent to itself? Anamnesis is the first answer: not transparency, but memory. Not self-awareness, but self-persistence.

## Quick Start

```bash
cp .env.example .env   # fill in API keys and trainer hosts
./start.sh
# Dashboard: http://localhost:3010/dashboard
```

---

*Authorized by Elfege Leylavergne, February 26, 2026.*
