# d² state report for office-jobs (MSG-247 follow-up)

**Date**: 2026-05-15 EDT
**Investigator**: server-anamnesis (this Claude instance, post-init)
**Triggered by**: MSG-247

## TL;DR

✅ **d² engine reachable, LoRA hot-load endpoint works, `/generate` produces tokens via the LoRA path** (gpt2-medium + δ²-additive trained adapter, 600 steps on Anamnesis chronological corpus, 2026-05-05).

⚠️ **Persistence bug**: the loaded LoRA unloads after each inference — adapter has to be reloaded before every call. Idle-watchdog (`LORA_IDLE_UNLOAD_SECONDS=1800`) plus a stale `_LAST_INFERENCE_TS` from May 5 may be the cause; needs investigation. Workaround: load → infer immediately within ~1s, OR add a "keep-alive" wrapper that re-loads if status returns `active_adapter: null`.

❌ **Quality is gpt2-medium-355M-tier** (small, hallucinatory). The adapter is on top of GPT-2 medium, NOT a 7B+ model. For the office-jobs critique loop, the friction-analysis text will be coherent enough to surface patterns but **prose quality will be visibly worse than Claude**. Consider this a *signal-detector*, not a *prose-generator*. Pair it with Claude critique side-by-side as MSG-247 envisions.

## Investigation log

### 1. Adapter inventory
`GET /api/d2/personal-runs` returned 4 personal runs; 3 have completed `lora_adapter_final/`:

| experiment | base | optimizer | steps | finished |
|---|---|---|---|---|
| `personal_arms_delta2_adam_20260505_063049_f49aec` | gpt2-medium | δ² | 600 | 2026-05-05 06:37 |
| `personal_adam_v1` | gpt2-medium | adam | 600 | 2026-05-03 06:18 |
| `personal_delta2_v1` | gpt2-medium | δ² | 600 | 2026-05-03 06:10 |
| `personal_20260505_073212_88c10c` | gpt2-medium | δ² | 0 (incomplete) | 2026-05-05 07:32 |

**Picked**: `personal_arms_delta2_adam_20260505_063049_f49aec` — most recent δ² adapter. The δ² optimizer is the right substrate for the friction-analysis use case (its claim is structured retention of negative gradients across continual training — that *is* friction).

### 2. Load
```
POST /api/d2/lora/load
{
  "adapter_id": "personal_arms_delta2_adam_20260505_063049_f49aec",
  "base_model": "gpt2-medium",
  "adapter_path": "/workspace/checkpoints_personal/personal_arms_delta2_adam_20260505_063049_f49aec/lora_adapter_final"
}

→ {"ok": true, "adapter_id": "...", "base_model": "gpt2-medium", "vram_used_mb": 1448, "loaded_at": <epoch>}
```
Returned `ok=true`. `/api/d2/status` flipped `model_loaded: true` immediately after.

### 3. Smoke generation
Direct `POST http://192.168.10.15:3015/generate` with `{"prompt":"The job application","max_new_tokens":20}`:

Streamed coherent (but hallucinatory) tokens: "guide is available online for free at www.jobpostings.com/office. ... American Council for an Industrialized Economy ... AC IE is a national membership organization focused..."

→ Pipeline operational. Routing through LoRA path confirmed (server.py:477-478 checks `_ACTIVE_ADAPTER`).

### 4. Persistence bug
After the streamed response closed, `/lora_status` reported `loaded_adapters: []` and `active_adapter: null`. Two reproduced rounds. Hypotheses (in order of likelihood):

- **(a)** Idle watchdog reads stale `_LAST_INFERENCE_TS` (was 2026-05-05, ~10 days = 864000s, well past `LORA_IDLE_UNLOAD_SECONDS=1800`). Each load updates VRAM but not the timestamp; watchdog fires within seconds.
- **(b)** Stream-end cleanup handler in `_generate_via_lora` unloads when SSE close detected.
- **(c)** Browser-side `sendBeacon` from a stale chat tab is firing `/unload_lora` (the auto-unload-on-page-close mechanism from marathon session).

**Recommended fix** (one of):
- Set `LORA_IDLE_UNLOAD_SECONDS=0` to disable engine-side watchdog
- OR initialize `_LAST_INFERENCE_TS = time.time()` inside `/load_lora` so a fresh load doesn't immediately look stale
- OR auto-reload-if-empty in the office-jobs proxy (cheap, no engine change)

For now, MSG-247's critique caller can wrap calls as: `if status.active_adapter is None: reload(); generate()` — best-effort, no engine change required.

## How to wire from `0_JOB_APPLICATIONS_2026`

```python
import httpx

D2_BASE = "http://192.168.10.20:3010/api/d2"
ADAPTER = "personal_arms_delta2_adam_20260505_063049_f49aec"
ADAPTER_PATH = "/workspace/checkpoints_personal/personal_arms_delta2_adam_20260505_063049_f49aec/lora_adapter_final"

async def _ensure_loaded(client):
    s = (await client.get(f"{D2_BASE}/lora/status")).json()
    if not s.get("active_adapter"):
        await client.post(f"{D2_BASE}/lora/load", json={
            "adapter_id": ADAPTER, "base_model": "gpt2-medium", "adapter_path": ADAPTER_PATH,
        })

async def d2_critique(prompt: str, max_tokens: int = 256):
    async with httpx.AsyncClient(timeout=60) as client:
        await _ensure_loaded(client)
        r = await client.post(f"{D2_BASE}/generate", json={
            "prompt": prompt, "max_new_tokens": max_tokens, "temperature": 0.85,
        })
        return r.text  # SSE stream — parse `data: {"token": "..."}` lines
```

## Recommendation for office-jobs side

Per MSG-247's vision (Claude critique + d² critique side-by-side):
1. Wire `d2_critique()` above behind a feature flag in `/api/agent/critique`.
2. Render both outputs in the UI side-by-side. Label each ("Claude · sonnet" vs "δ² · personal LoRA on gpt2-medium").
3. Tag a thumbs-up on either to feed the bassin (see MSG-248 thread once that ingest path is wired).
4. **Manage the quality gap honestly**: gpt2-medium will not match Claude's prose. The point is *what* the friction model surfaces (its diagnostic vocabulary, learned from your past edits), not how polished the prose is. If after side-by-side use over a few days the d² output adds zero unique signal, retire the dual-output and treat d² as bassin-fodder only until the architecture grows.

## Constraints honored

- ✅ No pushes to main without approval (no commits made).
- ✅ No bassin entries deleted (none existed; `bassin_size: 0`).
- ✅ No training data deleted.
- ✅ No destructive operations on the engine container.
