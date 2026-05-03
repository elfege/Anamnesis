# What happened today (2026-05-02), in plain English

**Skip to:** [What works now](#what-works-now-buttons-you-can-press) · [How to actually fire training](#how-to-actually-fire-the-personal-fine-tune) · [What still needs your decision](#what-still-needs-your-decision) · [Glossary for the dummy in your head](#glossary)

---

## TL;DR

Today the project crossed a big architectural line. **δ² stopped being just an abstract experiment on Wikipedia text and got plumbed into the actual scientific use case: continually fine-tune an LLM on YOUR conversations, in chronological order, with the bassin retaining the structural memory of your back-and-forths.**

The pipeline that does this is now built end-to-end:

```
Anamnesis MongoDB (77K episodes)
    ↓  scripts/anamnesis_to_tokens.py     (you ran this; it works — 213M tokens written)
chronological .bin files
    ↓  d2/finetune_lora.py                 (built today; not yet fired)
δ²-fine-tuned LoRA adapter on a Llama-3.2-3B base
    ↓  load + chat
a chat model that talks more like you, with structural memory across months
```

Plus several pieces of plumbing/safety so this doesn't accidentally leak personal data through the wrong channel.

---

## What works now (buttons you can press)

1. **`/chat` with 8 backends in the resource selector.** Together.ai (80 models including Llama-3.1-70B-Instruct, DeepSeek-V4-Pro), Claude API, Claude CLI, three local Ollamas, RunPod (when a pod is up), and the δ² engine. Click resource → model dropdown auto-fills → chat.

2. **`/dashboard → δ² tab` with a "Two tracks" section.** Bench (paper-eligible, public datasets, reproducible) vs Personal (private corpus, never published). Help icons on every panel. Explain buttons that call Claude CLI for plain-English interpretations of any KPI/section.

3. **`/dashboard → Chat ↗`** — clicks through to `/chat` (single source of truth — no more two chat UIs to keep in sync).

4. **`./start.sh → option 6 → option 3`** spins a RunPod pod for δ² training (idempotent, cost-confirmed). Now correctly prompts for the MOBIUS.VAULT passphrase up front instead of silently falling back to .env.

5. **The pre-push security hook** at `.git/hooks/pre-push` will refuse any `git push` that contains `COPY .env`, `COPY *.pem`, `COPY d2/d2_checkpoints/`, an AWS access key, a GitHub token, a `.pt` model weight, etc. Run `./scripts/security_check.sh` manually any time to audit.

---

## How to actually fire the personal fine-tune

Once you decide on the ghcr.io image visibility (see below), and assuming the d² container has the HF stack (it does — installed today), this works end-to-end on the **server's GTX 1660 SUPER** for the Tier 1 3B-base run:

### Step 1 — fresh data export (re-run any time)

```bash
ssh server 'docker exec anamnesis-d2 python /app/d2/scripts/anamnesis_to_tokens.py \
    --api http://192.168.10.20:3010 \
    --out-dir /workspace/data_personal \
    --split-mode by_month \
    --min-summary-chars 50'
```

Output: `/workspace/data_personal/anamnesis_chronological/{task_NN.bin, val.bin, manifest.json}`. **Already done once today** — 77K episodes, 213M tokens.

### Step 2 — fire the δ² fine-tune (the experiment arm)

```bash
ssh server 'docker exec -d anamnesis-d2 python -m d2.finetune_lora \
    --base-model meta-llama/Llama-3.2-3B-Instruct \
    --data-dir /workspace/data_personal/anamnesis_chronological \
    --output-dir /workspace/checkpoints_personal \
    --experiment personal_delta2_v1 \
    --optimizer delta2 \
    --steps-per-task 200 \
    --block-size 512 \
    --batch-size 2 \
    --d2-eta 1e-6'
```

First run downloads Llama-3.2-3B-Instruct from HuggingFace (~6GB, 5-15 min). Then trains in 4-bit with LoRA-on-attention. Approximate wall time on the 1660 SUPER: 1-3 hours per task × 2 tasks = 2-6h total.

### Step 3 — fire the AdamW control arm (same flags, different optimizer)

```bash
# Same command, but: --experiment personal_adam_v1 --optimizer adam
```

The two checkpoints can then be compared on:
- **val.bin perplexity** (does δ² preserve general ability across the chronological tail better than Adam?)
- **Held-out WikiText perplexity** (does δ² preserve "general English" better after personal fine-tuning?)
- **Chat with both via /chat** (qualitative — does δ²-fine-tuned Belle sound more like you than Adam-fine-tuned Belle?)

---

## What still needs your decision

**One blocker — corrected guidance**:

### Should the d² Docker image on ghcr.io be PUBLIC at this stage?

**No. Keep it private.** (My earlier recommendation to make it public was wrong-headed for the current phase. Updated reasoning below.)

The image stays private during experimentation. The image only becomes public at **publication / peer-review time**, and then with a **frozen, immutable tag** (e.g. `:bench-paper-v1`) referenced in the manuscript — never the rolling `:cuda-runpod` dev tag.

**Reasoning** (the right reasoning, which the user articulated and I should have arrived at first):

1. We're still experimenting. Anything published at this stage is a moving target — anyone "reproducing" a rolling tag gets different output every week.
2. Versioning at publication time only. A specific tag is frozen, documented, paired with the manuscript — that's how reproducibility actually works in published work, not by exposing the dev branch.
3. Premature exposure invites scrutiny on incomplete intermediates and bug-ridden iterations — not what you want as a preview of your work.

**To unblock RunPod without making the image public** (when needed):

Configure RunPod's "Container Registry Credentials" feature in their web console with your GitHub PAT. ~15-30 min one-time setup. Then RunPod can pull the private ghcr.io image like any other authenticated puller. `deploy_runpod.sh` would need one new line passing `containerRegistryAuthId` in the GraphQL pod-creation mutation.

(For tonight's local-1660-SUPER personal-corpus run: nothing to do; the image visibility doesn't enter the picture at all.)

---

## Why the two tracks matter (the architecture point)

If you train δ² on personal episodes and then say "δ² beats Adam at continual learning," reviewers will rightly demand to see the same numbers on a **public dataset they can reproduce**. So:

```
┌─────────────────────────────┐    ┌──────────────────────────────────┐
│  d2_checkpoints_bench/      │    │  d2_checkpoints_personal/        │
│  ─────────────────────      │    │  ───────────────────────         │
│  Public datasets only:      │    │  Anamnesis episodes only:        │
│   • WikiText-103            │    │   • Your conversations           │
│   • permuted-MNIST          │    │   • Chronological tasks (months) │
│   • split-MNIST             │    │                                  │
│                             │    │                                  │
│  → Numbers in the paper     │    │  → For your own use ONLY         │
│  → Reproducible             │    │  → NEVER published, anywhere     │
│  → Image can be public      │    │  → Different bind-mount, never   │
│                             │    │    ever mixed with bench.        │
└─────────────────────────────┘    └──────────────────────────────────┘
```

The two are kept apart at every level: separate directories on host, separate bind mounts in compose, separate naming convention enforced by `finetune_lora.py` (will refuse to write a `personal_*` experiment to `_bench/`, or vice versa). Read `README_canonical_two_tracks.md` for the full architecture and the protections.

---

## Glossary

For when you re-read this in 3 weeks and forget half the terms.

| Term | What it actually means |
|---|---|
| **bassin** | A reservoir of "tensions" (squared frictions). δ² accumulates them across training and injects `tanh(bassin) × η` into each weight update. The whole point of δ². |
| **δ² additive (path B)** | The form of δ² that works: `W_next = W_now − base_lr·∇L + η·tanh(B)`. Standard gradient descent PLUS a δ² nudge. |
| **δ² standalone (path A)** | The form that doesn't work, twice falsified: `W_next = W_now + η·tanh(B)`. No gradient term. Model never learns. |
| **LoRA** | Low-Rank Adaptation. Instead of fine-tuning the full base model (3B params), you train tiny "adapter" matrices on top (~1M params). The base stays frozen. Quick to train, cheap to store. |
| **QLoRA** | LoRA + 4-bit quantization of the base model. Lets you fine-tune big models on small GPUs. |
| **Anamnesis episode** | A distilled record of a Claude session — summary + raw exchange + tags + timestamp. Lives in MongoDB, accessed via `/api/episodes`. |
| **Bench track** | δ² training runs on public datasets only. Reproducible. Goes in the paper. |
| **Personal track** | δ² training runs on your Anamnesis episodes. Private. NEVER published. The actual demonstration of δ²'s use case. |
| **MOBIUS.VAULT** | Your encrypted credential vault on dellserver:8450. Decrypted via passphrase, holds AWS keys. `start.sh` now prompts for the passphrase up front (canonical VP rule). |
| **ghcr.io** | GitHub Container Registry. We pushed our slim d² image there. It's currently PRIVATE (RunPod can't pull). One command flips it to public. |

---

## Where things live (so future-you can find them)

| Thing | Path |
|---|---|
| Data pipeline | `scripts/anamnesis_to_tokens.py` |
| LoRA fine-tune script | `d2/finetune_lora.py` |
| δ² optimizer (with `additive_mode=True` default) | `d2/optimizer.py` |
| d² engine (FastAPI) | `d2/server.py` |
| Two-track architecture explanation | `README_canonical_two_tracks.md` |
| Today's session, in plain English (this file) | `README_canonical_today_2026-05-02.md` |
| Bench checkpoints | `d2/d2_checkpoints_bench/` |
| Personal checkpoints | `d2/d2_checkpoints_personal/` |
| Personal corpus .bin files | `d2/d2_data/personal/` (gitignored, never published) |
| Security check script | `scripts/security_check.sh` (untracked) |
| Pre-push hook | `.git/hooks/pre-push` |
| Dockerfile templates | `*.example` (tracked); actual `Dockerfile` files are untracked, build locally only |

---

*Generated automatically at the end of a long session. If something here doesn't match what you remember, the actual file/command is the source of truth — this is just an index.*
