# UI overhaul — toward "clone the repo, run start.sh, try the experiment" out of the box

> Author: office (Claude). Created 2026-05-03. Status: in progress.
>
> Mission: evolve the Anamnesis UI from "engineer-oriented dashboard" to "a stranger
> can clone the repo, run start.sh, open localhost:3010, and immediately understand
> (a) what δ² is, (b) what they can try, (c) where the bench/personal split lives,
> and (d) what each metric means."
>
> This file is the canonical multi-pass roadmap. Each pass appends a results
> section at the bottom and updates the phase-by-phase status header.

---

## North star (for the eventual unfamiliar user)

After `git clone … && cd Anamnesis && ./start.sh`, the user lands on `localhost:3010` and sees:

- A **landing/menu page** that explains δ² in three sentences and offers four entry points: *Try the chat*, *See the benchmarks*, *Train your own model*, *Read the theory*.
- **Two chatrooms** with explicit purpose split:
  - **Chatroom 1 — Bench/Scientific**: backed by either user infrastructure or a public 3B-fine-tuned d² model, reproducible, outputs OK to share.
  - **Chatroom 2 — Personal/Belle**: backed by Together.ai stock 70B (or, eventually, the user's personal-fine-tuned LoRA). Outputs strictly private — banner, no shared logging.
- A **training launcher** per track (Bench: WikiText/permuted-MNIST/split-CIFAR; Personal: anamnesis chronological corpus).
- A **live benchmark dashboard** per track (KPIs, leaderboard, comparison plots).
- A **resource status mini-panel** (which GPUs reachable, RunPod up + burn, Together.ai key configured, last training run).
- Every panel has an SVG help icon with opaque-background tooltip; every metric has an "Explain this" button backed by Claude CLI through MongoDB cache (route `/api/d2/explain` already exists).

This document plans the path from today's state to that end state, in passes that are each individually safe to ship.

---

## Current state inventory (2026-05-03)

### Pages

| Path | Template | Purpose | Notable surfaces |
|---|---|---|---|
| `/dashboard` | `app/templates/dashboard.html` (1811 lines) | Engineer dashboard — 10 tabs | Overview, Episodes, Search, Crawler, JSONL Ingester, Chat (link), Embedding, Training, **δ²**, Architecture, Avatar (link) |
| `/chat` | `app/templates/chat_standalone.html` (1295 lines) | Standalone chat surface (single source of truth) | Resource selector → 8 backends (Ollama×3, Claude API, Claude CLI, RunPod, Together.ai, δ² engine), session sidebar, terminal heartbeat panel, RunPod lifecycle widget |
| `/avatar` | `app/templates/avatar.html` | Avatar prototype (Belle) | (separate refactor in `docs/plans/avatar_refactor_plan.md`) |
| `/`     | n/a | **Missing** — currently 404 | Should be the landing/menu page |

### Help-icon coverage audit

| Tab / surface | Help-icon style | Verdict |
|---|---|---|
| Dashboard → Crawler | Literal `?` text in `.btn-help` button | **Inconsistent — fix to SVG** |
| Dashboard → δ² | SVG `.d2-help` with opaque tooltip (`#0d1117`) | Canonical pattern, keep |
| Dashboard → Training | `title="…"` HTML attributes only — no icons, native browser tooltip (translucent) | Acceptable for compactness; can enhance later |
| Dashboard → JSONL Ingester | None | Has `.settings-hint` text; could add SVG icons on heading rows |
| Dashboard → Embedding | None | Same as above |
| Chat (`/chat`) | None on heading; some `title=` attrs | Should add an SVG help icon to the terminal-panel header at minimum |

### Two-track surfacing audit

| Surface | Bench/Personal split visible? | Notes |
|---|---|---|
| Dashboard δ² tab — "Two tracks" section | YES — explanatory card | Read-only explanation; no track-aware controls |
| Chat header / resource selector | NO — flat list of backends | **Needs explicit track toggle** |
| Privacy banner on Chatroom 2 | NO | **Needs adding** |
| Training launcher per track | NO | Future pass |

### Backend routes touched by this plan

- `app/routes/chat.py` — DO NOT MODIFY in track-toggle pass; the toggle is purely UI/localStorage. Backend routing already supports per-resource selection.
- `app/routes/anamnesis_d2.py` — read-only by this UI pass.
- New static asset directory: none required this pass; SVGs inline.

### Privacy constraints (from `~/.claude/.../project_two_chatrooms_purpose.md`)

- Anything generated in Chatroom 2 must NOT be logged to shared collections, crawled into Anamnesis-public, or used in any paper.
- Together.ai usage from Chatroom 2 = personal inference. No public dashboards, no shared MongoDB collections that get crawled.
- Default: when in doubt, keep them isolated.

---

## Information-architecture proposal

```
/                       — Landing / menu (NEW; future pass)
├─ /dashboard           — Engineer/research surface (existing; 10 tabs)
│   ├─ Overview         — KPIs about the memory system
│   ├─ Episodes         — Browse / search Anamnesis episodes
│   ├─ Search           — Semantic search
│   ├─ Crawler          — Source crawler config
│   ├─ JSONL Ingester   — Claude Code JSONL ingestion
│   ├─ Embedding        — Embedding model config
│   ├─ Training         — Per-machine trainer cards
│   ├─ δ²               — Continual-learning research
│   │   ├─ Hero KPIs
│   │   ├─ Two-tracks card  (existing — explanatory only)
│   │   ├─ Leaderboard
│   │   ├─ Bassin
│   │   ├─ Trainer status
│   │   ├─ Historical runs
│   │   ├─ TRAINING LAUNCHER (NEW; future pass)
│   │   └─ RESOURCE STATUS (NEW; future pass)
│   └─ Architecture     — Diagrams
│
├─ /chat                — Conversational surface
│   ├─ TRACK TOGGLE          (NEW this pass — Bench / Personal)
│   ├─ Resource selector    (existing — auto-filters by track in future)
│   ├─ PRIVACY BANNER       (NEW this pass — visible on Personal track)
│   ├─ Sessions sidebar
│   ├─ Messages
│   └─ Terminal heartbeat panel  (existing; gain a "?" icon this pass)
│
└─ /avatar              — Avatar prototype (separate plan)
```

---

## Phase breakdown

### Phase A — safe, ship now (this pass)

1. **Help-icon consistency in Crawler tab.** Convert the four literal `?` `.btn-help` buttons to SVG circles matching the `.d2-help` pattern. Same `data-help` data attribute, same handler. Cosmetic only, zero JS change.
2. **Track toggle in chat header.** Two-button segmented control: `Bench` / `Personal`. Stored in `localStorage["anamnesis.chat.track"]`. Visual: blue accent border for Bench, amber for Personal. No backend change — when the user switches to Personal we just (a) show the privacy banner, (b) update header color cue, (c) (future) suggest a default resource.
3. **Privacy banner on Personal track.** Yellow-bordered card just above the message stream: "This conversation is private — outputs are not logged to shared collections, not crawled into Anamnesis-public, not used in any paper." Dismissible (state persisted in localStorage). Always recallable via a small lock icon next to the track toggle.
4. **Terminal-panel "what is this" help.** Add a small SVG `?` next to the "terminal" header in the chat right panel. Tooltip explains the heartbeat lines: backend, model, worker, latency.
5. **Landing route placeholder.** The `/` route currently 404s. **Defer** — adding even a placeholder requires a route change and a new template. Plan it for Phase B.

### Phase B — needs user review before ship (next pass)

6. **Landing/menu page at `/`** with the four entry-point cards.
7. **Training Launcher card per track** in δ² tab (Bench, Personal). Shows next available run + the docker-exec command (display-only; user fires manually). Pulls cost/wall-time estimates from a small JSON config.
8. **Resource Status mini-panel** in δ² tab. Aggregates GPU reachability (server, office), RunPod pod state + burn rate, Together.ai key presence, last training run status. Replaces current silent fallback chain with explicit status icons.
9. **Track-aware resource filtering in chat selector.** When track=Bench, hide Together.ai (personal-only) from the dropdown; when track=Personal, surface it first. Requires a metadata flag on each resource probe.

### Phase C — bigger lifts, after Phase B reviewed

10. Help-icon SVGs on JSONL Ingester / Embedding / Crawler-modal headings (full coverage audit).
11. Per-tab "Explain this tab" button hooked up to `/api/d2/explain`.
12. Move all inline `<style>` blocks to `static/css/` — currently dashboard.html has ~1100 lines of inline CSS, chat_standalone.html ~440 lines.
13. Per-track training-history view (split leaderboard into Bench-only and Personal-only when personal runs exist).

---

## File-by-file change list (this pass — Phase A)

| File | Change | Rationale | Risk |
|---|---|---|---|
| `app/templates/dashboard.html` | Lines 157, 180, 197, 213: replace `<button class="btn-help" …>?</button>` with SVG variants | Match canonical SVG pattern from δ² tab | Low — same class, same JS handler binds on `.btn-help` click |
| `app/templates/chat_standalone.html` | Add CSS block for `.track-toggle`, `.track-toggle button`, `.privacy-banner`, `.term-help`. Add HTML in `header > .header-controls` (track toggle + lock icon). Add HTML before `.messages` (privacy banner). Add SVG `?` in `#sc-term-header`. Add JS to handle toggle clicks, persist to localStorage, show/hide banner, color-cue header | Surface track explicitly; warn user about Chatroom 2 privacy posture | Low — pure additive; no existing element removed; no backend route touched |

No backend Python changes this pass.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Help-icon SVG conversion breaks existing `.btn-help` click handler | Keep `.btn-help` class + `data-help` attribute identical; only swap the inner content from `?` to `<svg>` |
| Track toggle confuses users who don't know what "bench" vs "personal" means | Each button has a tooltip; the privacy banner explicates the personal posture; heading-line label "Track" precedes the buttons |
| Privacy banner is too visually loud or too quiet | Use the established amber accent (already used in two-tracks card on dashboard) — known-quantity color; dismissible per-session |
| Backend behavior accidentally diverges by track | Out of scope this pass — the toggle is UI-only. Future Phase B will introduce track-aware backend filtering with explicit user review |
| Inline CSS bloats chat_standalone.html further | Acknowledged; Phase C item to migrate to `static/css/` |
| User can't recall the privacy banner after dismissing | A persistent lock icon in the track-toggle area re-shows it on click |

### Test plan for this pass

- `curl -s -o /dev/null -w '%{http_code}\n' http://localhost:3010/dashboard` → expect 200
- `curl -s -o /dev/null -w '%{http_code}\n' http://localhost:3010/chat` → expect 200
- `docker logs anamnesis-app --tail 50` → no Python tracebacks after the curls
- Visual: open `/dashboard → Crawler` and confirm SVG `?` icons render, tooltip still pops on click
- Visual: open `/chat`, click `Personal`, confirm header takes amber accent and privacy banner appears; reload → state persists; click `Bench`, banner disappears, accent reverts to blue

---

## Out-of-scope this pass (deferred)

- Anything that adds a new backend route or modifies `app/routes/chat.py`
- Landing page at `/` (Phase B; needs route + template + content review)
- Resource Status panel (Phase B; needs probing logic for GPU/RunPod/Together)
- Training Launcher cards (Phase B; needs estimate-config + display-only UX confirmation)
- Track-aware backend filtering (Phase B; needs metadata audit on every resource probe)
- Inline-CSS extraction (Phase C)
- Avatar refactor (covered by `docs/plans/avatar_refactor_plan.md`)

---

## Pass log

### 2026-05-03 — initial pass (Phase A items 1–4)

Shipped:

- **A1 done.** Crawler-tab `?` literal-text help icons → SVG (4 buttons). Added `.btn-help svg { display:block; width:11px; height:11px }` to `app/static/css/style.css`. Click handler unchanged (`data-help` still keys into `_help_map`).
- **A2 done.** Track toggle (Bench / Personal) added to chat header. Persisted in `localStorage["anamnesis.chat.track"]`. Visual: blue accent on Bench, amber on Personal. Header bottom-border tints amber when track=personal (via `body.track-personal`).
- **A3 done.** Privacy banner shown above chat messages when track=personal. Dismissible per-session (`sessionStorage["anamnesis.chat.privacyBannerDismissed"]`). Lock icon next to track toggle re-shows the banner on click. Switching back to Personal clears the dismiss flag so the user is re-warned.
- **A4 done.** SVG `?` icon next to the chat terminal-panel header. Hover/focus shows opaque-bg tooltip explaining heartbeat lines. Click swallowed so it doesn't toggle panel collapse.

Verified:

- `curl /dashboard` → 200, `curl /chat` → 200.
- `docker logs anamnesis-app --tail 25` → no tracebacks; only 200 OK lines for `/dashboard` and `/chat` after edits.
- Rendered HTML contains expected new fragments: 11 matches for `sc-track-toggle|sc-privacy-banner|sc-term-help` in `/chat`; 4 SVG-bearing `.btn-help` in `/dashboard`.

Not shipped (skipped this pass; deferred to Phase B):

- Training Launcher cards (need cost/wall-time JSON + display-only UX confirmation).
- Resource Status mini-panel (needs probing logic for GPU/RunPod/Together; touches resource probe flow).
- Landing page at `/` (needs route + content review).
- Track-aware backend filtering in chat (Phase B; needs metadata flag on every resource probe).

User decisions punted:

- Whether the Personal track should default to selecting Together.ai automatically (vs. just hint at it). Defaults to no auto-selection this pass.
- Exact wording for the privacy banner (current text is conservative; can be tuned).
- Color tokens for the amber accent on Personal (currently `--warning` = `#d29922` matches dashboard convention).
