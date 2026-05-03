# QUICKSTART — stand up Anamnesis in 30 minutes

A self-hosted continual-learning + embodied-AI platform. Episodic memory is one piece; alongside it run the **δ² (delta-squared) continual-learning optimizer** with its bassin reservoir, a **multi-backend chat surface** that routes to up to eight inference resources (three local Ollamas, Claude API, Claude CLI, Together.ai, RunPod, and the in-house δ² engine), the **Belle avatar pipeline** (LLM → TTS → SadTalker animation, in development), and a **two-track experimentation surface** that strictly separates paper-eligible bench work from private personal-corpus work.

This file is the fast path from `git clone` to a running dashboard. Aimed at a developer with general ML/Docker familiarity.

---

## 1. Prerequisites

Required:

- **Docker** 24+ and **Docker Compose** v2 (`docker compose ...`, not the old `docker-compose`).
- ~10 GB free disk for images + MongoDB volume.

Optional but recommended:

- **NVIDIA GPU** + `nvidia-container-toolkit` if you want to run δ² training or local LoRA fine-tunes. Inference-only usage works on CPU (slow Ollama, fine for chat).
- **AWS account + Secrets Manager** if you want to use the encrypted vault flow (`mobius-vault`) instead of plaintext `.env`. Skippable.
- **HuggingFace account + token** for gated models (Llama-3.x family). Not needed for `gpt2-medium` or open Qwen models.
- **RunPod account + API key** if you want to spin a cloud GPU on demand.
- **Together.ai account + API key** if you want stock 70B inference without renting your own GPU.
- **Anthropic API key** if you want direct Claude API (the dashboard also supports Claude CLI, which uses your existing subscription with no API key).

---

## 2. Five-minute path

```bash
git clone https://github.com/elfege/anamnesis.git
cd anamnesis
cp .env.example .env

# Optional: edit .env to add keys you want enabled. Everything is optional —
# unset keys just disable the corresponding backend.
docker compose up -d

# Wait ~30 s for MongoDB + the embedding model to load, then:
xdg-open http://localhost:3010/   # Linux
open http://localhost:3010/       # macOS
```

Browse to <http://localhost:3010/>. The root redirects to `/dashboard`. You will see:

- **Overview tab** — episode counts, project breakdown.
- **δ² tab** — continual-learning research surface, leaderboard, bassin, two-tracks card, **Personal Benchmarks** panel (yellow border — empty until you fire a personal run; see §6).
- **Architecture tab** — Mermaid platform diagram + episodic-memory pipeline diagram.
- **Resource Status mini-panel** (top-right of every page) — Ollama / Together.ai / Anthropic / RunPod / δ² engine / hosts / GPU memory at a glance, polled every 30 s.
- **/chat** — multi-backend chat with a Bench / Personal track toggle.

What you get **with no keys at all**:

- Dashboard fully functional.
- Episode storage + semantic search (uses BAAI/bge-large-en-v1.5 on CPU; ~2 GB download on first run).
- δ² tab UI (status panel will show "δ² engine unreachable" until you wire one — see §4).
- Chat backed by Ollama if you have one running on the host (`http://host.docker.internal:11434`).

---

## 3. The `.env` file — what is optional

Open `.env.example` and copy. Every field is optional unless noted.

| Field | Required? | What it does |
|---|---|---|
| `MONGO_USER`, `MONGO_PASSWORD` | yes | MongoDB credentials. Defaults are dev-only — change for any networked deployment. |
| `EMBEDDING_MODEL` | no | sentence-transformer model for episode embedding. Default `BAAI/bge-large-en-v1.5`. |
| `OLLAMA_URL_1/2/3` | no | Ollama endpoints (one per `_N`). Each gets a `_LABEL_N` and `_GPU_N`. |
| `ANTHROPIC_API_KEY` | no | Direct Claude API. CLI is separate (no key needed). |
| `TOGETHER_AI_KEY` | no | Together.ai key. Enables 80+ stock models in the chat resource picker. |
| `RUNPOD_API_KEY` | no | Enables on-demand pod creation from the chat UI. |
| `RUNPOD_REGISTRY_AUTH_ID` | no | RunPod-side ID for pulling private ghcr.io images. See `README_morning_2026-05-04.md`. |
| `D2_ENDPOINT_URL` | no | URL of the δ² engine container (e.g. `http://192.168.10.15:3015`). Empty = δ² features disabled in dashboard. |

---

## 4. Adding inference backends

### 4.1 Ollama (local GPU, free)

On any GPU host:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

In `.env`:

```bash
OLLAMA_URL_1=http://192.168.10.15:11434
OLLAMA_LABEL_1="server (GPU)"
OLLAMA_GPU_1=true
```

Restart: `docker compose up -d --no-deps anamnesis-app`.

Verify in **Resource Status** panel — green dot next to "server (GPU)".

### 4.2 Claude API or Claude CLI

- **API**: `ANTHROPIC_API_KEY=sk-ant-...` in `.env`. Restart app.
- **CLI**: install `claude` on the host. The container reaches it via SSH. Configure `CLAUDE_CLI_HOST` and `CLAUDE_CLI_PATH` if not at default.

### 4.3 Together.ai

```bash
# .env
TOGETHER_AI_KEY=tgp_v1_...
TOGETHER_AI_ID=key_...
```

Restart app. Together.ai now appears in the chat resource picker. The model dropdown is populated live from `/v1/models`.

### 4.4 RunPod (on-demand cloud GPU)

```bash
# .env
RUNPOD_API_KEY=rpa_...
# Optional: pre-baked private-image auth (one-time setup, see deploy_runpod.sh)
RUNPOD_REGISTRY_AUTH_ID=cmop...
```

Restart app. From the chat UI, click "RunPod" → "Start pod" — creates a pod, polls until live, returns the OpenAI-compatible endpoint. Cost meter polls every 5 s.

### 4.5 δ² engine (the in-house GPU service)

Build + run on a GPU host:

```bash
cd d2/
docker build -t anamnesis-d2:cuda -f Dockerfile.example .
docker run -d --gpus all -p 3015:3015 \
    -v $(pwd)/d2_checkpoints_bench:/workspace/checkpoints \
    -v $(pwd)/d2_checkpoints_personal:/workspace/checkpoints_personal \
    -v $(pwd)/data:/app/d2/data \
    --name anamnesis-d2 anamnesis-d2:cuda
```

Then on the dashboard host's `.env`:

```bash
D2_ENDPOINT_URL=http://<gpu-host-ip>:3015
```

Restart app. δ² tab now shows live status; **Resource Status** shows green dot next to "δ² engine".

---

## 5. Reproducing the bench experiments

The bench track is what goes in the paper. Public datasets, public code, fully reproducible.

### 5.1 Data prep

```bash
# Inside the d² container
docker exec anamnesis-d2 python /app/d2/data/prepare_wikitext.py
# Produces /app/d2/data/wikitext/{train.bin, val.bin}
```

For permuted-MNIST + split-MNIST: data is downloaded automatically by `d2/experiments/continual.py` on first run (via `torchvision.datasets`). Requires `torchvision`.

### 5.2 Fire a sweep (permuted-MNIST, all baselines + δ²)

```bash
# From the dashboard's δ² tab → "Train start" form, OR directly:
curl -X POST http://localhost:3010/api/d2/train/start -H 'Content-Type: application/json' \
    -d '{"optimizer":"delta2_additive","benchmark":"permuted_mnist","tasks":5,"epochs":1,"seed":0}'

# Or at the d² engine directly:
curl -X POST http://<gpu-host>:3015/train/start -H 'Content-Type: application/json' \
    -d '{"optimizer":"delta2_additive","benchmark":"permuted_mnist","tasks":5}'
```

Polling: `GET /api/d2/train/status`. Results stored in `d2/d2_checkpoints_bench/runs/`. Leaderboard auto-refreshes.

### 5.3 Pareto / multi-seed sweeps

See `d2/scripts/overnight_2026_05_04.sh` and `d2/scripts/pareto_meaningful_eta_2026_05_04.sh` for canonical recipes.

---

## 6. Reproducing the personal-track experiment (the "lived" side)

> **PRIVATE TRACK.** You supply your **own corpus**, not Elfege's. Outputs stay local. Never push to a public registry.

The personal track LoRA-fine-tunes a base model (gpt2-medium today; Llama-3.2-3B-Instruct or Qwen-2.5-1.5B-Instruct in the queue) on a chronologically-ordered private corpus. The point: build a chat partner that talks more like you, with structural memory across sessions thanks to δ²'s bassin.

### 6.1 Provide your own corpus

The data pipeline expects an Anamnesis-shaped JSON of episodes (one document per session/exchange, with `summary`, `text`, `timestamp`). The simplest way:

- **Already running Anamnesis with episodes**: nothing to do, the pipeline reads from `/api/episodes`.
- **Bring your own JSON**: write a one-off importer that POSTs to `/api/episodes`. See `app/routes/episodes.py` for the schema.

### 6.2 Tokenize

```bash
docker exec anamnesis-d2 python /app/d2/scripts/anamnesis_to_tokens.py \
    --api http://<dashboard-host>:3010 \
    --out-dir /workspace/data_personal \
    --split-mode by_month \
    --min-summary-chars 50
```

Output: `/workspace/data_personal/anamnesis_chronological/{task_NN.bin, val.bin, manifest.json}` — tokenized with **tiktoken GPT-2 BPE** (so the base model must be a GPT-2-family vocab today; extend with `--tokenizer` flag for Qwen/Llama).

### 6.3 Fire the run

Canonical recipe: `d2/scripts/personal_arms_only_2026_05_04.sh` runs δ² + Adam control arms back-to-back.

```bash
# Run on the GPU host
docker exec -d anamnesis-d2 bash /app/d2/scripts/personal_arms_only_2026_05_04.sh
```

Wall time: ~10–20 min for 600 steps × 2 tasks on a GTX 1660 SUPER. Larger models / more steps scale linearly.

### 6.4 Inspect

- Dashboard: **δ² tab → Personal Benchmarks** panel. Row per personal_* run.
- Disk: `d2/d2_checkpoints_personal/personal_*/` (config.json, metrics.jsonl, lora_adapter_final/).
- API: `curl http://localhost:3010/api/d2/personal-runs`.

---

## 7. Two-track separation rule

Read `README_canonical_two_tracks.md`. TL;DR:

- **Bench track** writes to `d2/d2_checkpoints_bench/`. WikiText, permuted-MNIST, split-MNIST. Reproducible. In the paper.
- **Personal track** writes to `d2/d2_checkpoints_personal/`. Your conversations. Private. Never published.
- Personal experiment names MUST start with `personal_` — enforced by the chat resource probe.
- **Never mix `--output-dir`** between tracks.
- **Never ingest Anamnesis episodes into a bench training script.**
- The Personal Benchmarks panel is read-only and never logs to MongoDB or any shared collection. It scans disk only.

The privacy posture is documented in `~/.claude/.../memory/project_two_chatrooms_purpose.md`.

---

## 8. Where to look next

- **Dashboard tour**: open `/dashboard`, click each tab in order. The opening paragraph and Architecture tab orient you.
- **Resource Status panel**: top-right, every page. Help icon (`?`) explains every row + color.
- **Canonical READMEs**:
  - `README_canonical_today_2026-05-02.md` — what works now, button-by-button.
  - `README_canonical_two_tracks.md` — bench vs personal architecture.
  - `README_morning_2026-05-04.md` — last overnight run results (RunPod private-image auth, Pareto sweep, personal arms).
  - `README_isolation_rule_office_GPU_must_not_run_anything_that_could_crash_it.md` — operational rule for the office GPU.
- **Plans in flight**: `docs/plans/` — `ui_overhaul_toward_user_friendly_out_of_the_box.md`, `avatar_refactor_plan.md`. (Untracked per Rule 19.1.3 — these are working docs.)
- **Morning reports**: `README_morning_*.md` — daily session summaries (untracked, gitignored).
- **API surface**: `API.md` for the full endpoint catalog.

If something here doesn't match what you observe in the running app, the running app is the source of truth — this file is an index, written for the day it was committed.
