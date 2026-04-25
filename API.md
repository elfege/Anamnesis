# Anamnesis HTTP API

Base URL (local): `http://dellserver:3010`

All endpoints return JSON unless otherwise noted. Streaming endpoints use
Server-Sent Events (SSE) over HTTP or a WebSocket. Timestamps are ISO 8601
UTC. MongoDB `_id` fields are serialized as strings.

---

## Table of Contents

- [Episodic Memory](#episodic-memory)
- [Chat (general)](#chat-general)
- [Avatar](#avatar)
- [AnamnesisGPT (custom LLM proxy)](#anamnesisgpt-custom-llm-proxy)
- [Crawler](#crawler)
- [JSONL ingestion](#jsonl-ingestion)
- [Feedback & Training](#feedback--training)
- [Embedding model](#embedding-model)
- [Context index](#context-index)
- [Files browser](#files-browser)
- [Dashboard & Status](#dashboard--status)
- [Bash (sandboxed)](#bash-sandboxed)
- [Worker registry (ephemeral GPUs)](#worker-registry-ephemeral-gpus)

---

## Episodic Memory

Prefix: `/api/episodes` · tag: `episodes`

Vector-searchable long-term memory. Episodes are embedded via the configured
model (default: `BAAI/bge-large-en-v1.5`) and searched over a MongoDB
Atlas-compatible vector index.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/episodes` | Create an episode. Body: `{instance, project, summary, raw_exchange, tags?}`. Server computes the embedding. Returns the persisted `EpisodeOut`. |
| `POST` | `/api/episodes/search` | Vector similarity search. Body: `{query_text, top_k?}`. Returns up to `top_k` episodes ranked by cosine similarity, each with a `score`. |
| `GET` | `/api/episodes` | List episodes (paginated). Query: `limit`, `skip`, `project`, `instance`, `tag`. |
| `GET` | `/api/episodes/recent` | Stream all episodes from the last N days as NDJSON (no page cap). Query: `days` (required, fractional OK), optional `project`, `instance`, `tag`, `limit`. |
| `GET` | `/api/episodes/{episode_id}` | Fetch a single episode by its ID. |
| `DELETE` | `/api/episodes/{episode_id}` | Delete an episode. |
| `POST` | `/api/episodes/reembed` | Re-embed all episodes with the current embedding model. Runs in background; poll `/reembed/status` for progress. |
| `POST` | `/api/episodes/reembed/pause` | Pause the current re-embed run. |
| `POST` | `/api/episodes/reembed/resume` | Resume a paused re-embed run. |
| `GET` | `/api/episodes/reembed/status` | `{running, paused, done, total, last_id, model_id}`. |

---

## Chat (general)

No prefix · tag: `chat`

Multi-backend chat with MongoDB session persistence. Supports Ollama
(local), Anthropic Claude API (if `ANTHROPIC_API_KEY` is set), and Claude
CLI over SSH (if configured).

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat/stream` | Streaming chat endpoint (SSE). Query: `session_id`. Body: `{message, backend, model}`. Yields `data: {token: "..."}` events, then `data: {done: true}`. Each round-trip is persisted as a chat session and as an embedded episode (tags: `chat`, `<backend>`, `<session_id[:8]>`). |
| `GET` | `/api/chat/models` | Available Ollama models on the first reachable endpoint. Returns `{models, default, endpoint}`. |
| `GET` | `/api/chat/balance` | Anthropic API credits/spend snapshot (if configured). |
| `GET` | `/api/chat/sessions` | List persisted chat sessions (metadata only, excludes `messages` field). |
| `GET` | `/api/chat/sessions/{session_id}` | Load a full session with messages. Also populates the in-memory cache. |
| `PATCH` | `/api/chat/sessions/{session_id}/title` | Rename a session. Body: `{title}`. |
| `DELETE` | `/api/chat/sessions/{session_id}/delete` | Delete a persisted session (permanent). |
| `DELETE` | `/api/chat/session/{session_id}` | Clear only the in-memory session state (non-destructive to MongoDB). |

---

## Avatar

No prefix · tag: `avatar`

Conversational avatar (Belle) pipeline: LLM → TTS → lip-sync. Sessions
reuse the `chat_sessions` collection with `backend="avatar"`.

### Page

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/avatar` | HTML page for the avatar UI. |

### Info

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/avatar/info` | `{status, persona, llm, default_voice, worker_endpoints, animate_enabled, reference_image}`. |

### Backends & Workers (Phase 7.2)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/avatar/models` | Available LLM backends and their models. Shape: `{default_backend, default_model, backends: {ollama, claude, anamnesis_gpt}}`. Each backend reports `{available, models, endpoint?}`. |
| `GET` | `/api/avatar/workers` | Parallel `/health` probes of every GPU worker. Shape: `{workers: [{url, label, reachable, worker_id, gpu_type, capabilities, vram_total_mb?, vram_free_mb?, error}]}`. |

### Voices

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/avatar/voices` | `{presets: [...edge...], cloned: [...], default_voice_id}`. |
| `POST` | `/api/avatar/voices` | Upload a voice sample. Multipart: `file`, `name`, `kind=file\|song\|record`, `language`, `notes?`. `kind=song` runs Demucs vocal extraction on the worker first. Returns `{ok, voice}`. |
| `DELETE` | `/api/avatar/voices/{slug}` | Remove a cloned voice. |
| `POST` | `/api/avatar/voices/{slug}/preview` | Short audio preview. Body: `{text?}`. Returns `{audio_url}`. |
| `POST` | `/api/avatar/preview-edge` | Preview an edge TTS preset. Body: `{voice_id: "edge:...", text?}`. Returns `{audio_url}`. |

### Chat (REST + WebSocket)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/avatar/chat` | Blocking REST chat. Body: `{message, session_id?, voice_id?, animate?, backend?, model?, preferred_worker?, no_fallback?}`. Returns `{text, timings, session_id, audio_url?, video_url?, error?}`. |
| `WS` | `/api/avatar/ws` | Streaming chat. Incoming frame: same shape as REST body, or `{type: "stop"}` to cancel in-flight generation. Outgoing frames: `{type: "token\|audio\|video\|done\|stopped\|error", ...}`. |

### Sessions (Phase 7.1)

Reuses `chat_sessions` with `backend="avatar"`. All endpoints reject sessions whose `backend` is not `avatar`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/avatar/sessions` | List avatar sessions (metadata only). Query: `limit`. |
| `GET` | `/api/avatar/sessions/{session_id}` | Load a full avatar session with messages. |
| `DELETE` | `/api/avatar/sessions/{session_id}` | Delete an avatar session. |
| `PATCH` | `/api/avatar/sessions/{session_id}/title` | Rename. Body: `{title}`. |

### Media

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/avatar/media/{filename}` | Serve a generated `.mp3` or `.mp4` from the short-lived media directory. |

---

## AnamnesisGPT (custom LLM proxy)

Prefix: `/api/anamnesis-gpt` · tag: `anamnesis-gpt`

Machine-gated proxy to trainer `/generate` endpoints (`NANOGPT_URLS`).
Currently serves the Qwen2.5-1.5B QLoRA adapter (proof-of-concept; low
quality). Will route to the δ² engine once trained.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/anamnesis-gpt/status` | `{available, training, message, progress?, endpoints}`. Returns `available=false, training=true` while a training run is active. |
| `POST` | `/api/anamnesis-gpt/generate` | Streaming SSE generation. Body: `{prompt, max_tokens?, temperature?, top_k?, stream?}`. Yields `data: {token: "..."}` events, then `data: {done: true}`. Tries each endpoint in `NANOGPT_URLS` with failover. |

---

## Crawler

Prefix: `/api/crawler` · tag: `crawler`

Multi-machine source code / document crawler. Ingests into `episodes`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/crawler/available-mounts` | Mounted source directories the crawler can reach. |
| `GET` | `/api/crawler/status` | `{running, current_activity, total_episodes_ingested, last_run, last_run_duration_seconds, episodes_ingested_last_run}`. |
| `POST` | `/api/crawler/run` | Trigger an ad-hoc crawl cycle outside the scheduler. |
| `GET` | `/api/crawler/config` | Full crawler configuration. |
| `PUT` | `/api/crawler/config/machine-roots` | Update which root directories per machine are crawled. |
| `PUT` | `/api/crawler/config/sources` | Update named sources. |
| `GET` / `PUT` | `/api/crawler/config/jsonl-roots` | JSONL source roots. |
| `GET` / `PUT` | `/api/crawler/config/doc-tag-patterns` | Document tag extraction regexes. Accepts `ignorecase` flag. |
| `GET` / `PUT` | `/api/crawler/config/docx-tag-patterns` | Legacy alias (reads old key for backward-compat). |
| `POST` | `/api/crawler/validate-regex` | Validate a regex string server-side. Body: `{pattern, flags?}`. |

---

## JSONL ingestion

Prefix: `/api/jsonl` · tag: `jsonl`

Nightly-scheduled ingestion of JSONL episode exports.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/jsonl/status` | Current ingester state. |
| `POST` | `/api/jsonl/ingest` | Trigger an ad-hoc ingest. Body: `{path?}`. |
| `POST` | `/api/jsonl/stop` | Stop an in-progress ingest. |
| `GET` / `PUT` | `/api/jsonl/settings` | JSONL ingester settings. |
| `GET` / `PUT` | `/api/jsonl/schedule` | Cron-like schedule configuration. |
| `GET` | `/api/jsonl/models` | Embedding models visible to the ingester. |

---

## Feedback & Training

No prefix · tag: `feedback`

Collects user feedback (thumbs up/down on chat replies) and exports it
as SFT training data for the trainer.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/feedback` | Record feedback on a chat message. Body: `{session_id, message_index, rating, comment?}`. |
| `GET` | `/api/feedback/stats` | Aggregate counts of up/down feedback across sessions. |
| `GET` | `/api/feedback/export` | Export unexported feedback as SFT JSONL. |
| `POST` | `/api/training/run` | Trigger a training run on a trainer endpoint (if reachable). |
| `GET` | `/api/training/status` | Current training status (running / progress / last metrics). |

---

## Embedding model

Prefix: `/api/embedding` · tag: `embedding`

Select and configure the embedding model used by the crawler, JSONL
ingester, and chat episode storage.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/embedding/config` | Current embedding model ID, dimension, and registry of available models. |
| `PUT` | `/api/embedding/config` | Switch embedding models. Body: `{model_id}`. New embeddings use the new model; existing episodes need re-embedding via `/api/episodes/reembed`. |

---

## Context index

No prefix · tag: `context-index`

Keyword-based reference index for terms, abbreviations, and
project-specific vocabulary injected into LLM context on demand.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/context-index` | List all entries. |
| `PUT` | `/api/context-index` | Upsert an entry. Body: `{key, value, tags?}`. |
| `DELETE` | `/api/context-index` | Remove an entry. Query: `key`. |

---

## Files browser

No prefix · tag: `files`

Read-only browser of mounted source directories (via SSH or bind mounts).

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/files/sources` | List available source mounts. |
| `GET` | `/api/files/ls` | Directory listing. Query: `source`, `path`. |
| `GET` | `/api/files/cat` | File contents (text only). Query: `source`, `path`. |

---

## Dashboard & Status

No prefix · tag: `dashboard`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/chat` | HTML page: standalone chat UI (same backends as `/api/chat/stream`). |
| `GET` | `/dashboard` | HTML page: overview, crawler status, episode counts, model selectors. |
| `GET` | `/api/dashboard/stats` | `{total_episodes, episodes_by_project, ...}`. |
| `GET` | `/api/status/summary` | Combined status across crawler, jsonl, training, workers. |
| `GET` | `/api/config/trainers` | Trainer endpoints and their reachability. |

---

## Bash (sandboxed)

No prefix · tag: `bash`

Limited shell access for admin operations. All commands require a per-command
consent token (issued by `/api/bash/consent/{consent_id}`) to execute.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/bash/consent/{consent_id}` | Issue a consent token for a pending command. |
| `POST` | `/api/bash/run` | Execute a command. Body: `{command, consent_token}`. |

---

## Worker registry (ephemeral GPUs)

Prefix: `/api/workers` · tag: `workers`

Dynamic registry for ephemeral GPU workers (RunPod, EC2, etc.) whose URLs
change between sessions. Static workers continue to be configured via
`.env` (`NANOGPT_URLS`, `AVATAR_WORKER_URL_*`); this registry adds
hot-swap support without an app restart.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/workers/register` | Register / refresh a worker. Body: `{url, label, kind="manual"\|"runpod"\|"ec2"}`. Upsert by `label`. |
| `DELETE` | `/api/workers/register/{worker_id}` | Unregister a worker. 404 if not found. |
| `GET` | `/api/workers/list` | List registered workers. Query: `kind` (optional filter). |
| `POST` | `/api/workers/{worker_id}/heartbeat` | Update `last_seen` timestamp. |

Schema: `{worker_id, url, kind, registered_at, last_seen}` in MongoDB
collection `worker_registry`. Used by `deploy_runpod.sh` and similar
scripts to register/unregister cloud GPU pods without editing `.env`.

---

## Conventions

- **Session IDs** are client-generated UUIDs (v4). Clients should generate on first message and reuse across turns.
- **SSE events** use the standard `data: <json>\n\n` framing. Terminate with `data: {"done": true}\n\n`.
- **WebSocket frames** are JSON objects with a `type` discriminator.
- **Errors** return `{error: "..."}` in the response body with an appropriate HTTP status (400 for bad input, 404 for not-found, 500 for server-side faults).
- **Timeouts**: streaming endpoints use `httpx` client timeouts of 120–300s; clients should match or exceed.
- **Authorization**: `AnamnesisGPT` and some admin endpoints gate on `AUTHORIZED_MACHINE_ID`, compared against `/etc/machine-id` mounted read-only into the container.
