# Rename: `~/0_GENESIS_PROJECT/0_ANAMNESIS` → `~/0_MOBIUS.ANAMNESIS` (top-level) + nest genesis + migrate `.claude` data

- **Status:** **POSTPONED** (operator's call) — ready-when-you-want artifact; nothing executes until the operator chooses to.
- **Author:** `dellserver-anamnesis:assistant` — for the operator
- **Date:** 2026-06-26
- **⚠ DO NOT run any step from a Claude session whose CWD is inside the dir being renamed — it closes mid-operation. Execute from `~` (operator shell) or a neutral-dir session.**

---

## 0. Context & goal (the inversion)

Anamnesis is nested at `~/0_GENESIS_PROJECT/0_ANAMNESIS`, so it's invisible in the operator's top-level `~/0_*` list and off-pattern vs the other `0_MOBIUS.*` projects (NVR, SMART_HOME, TILES, PROXY, JIRA…). Promote it to its own top-level project `~/0_MOBIUS.ANAMNESIS`, with the genesis material (the philosophical-persistence experiment Anamnesis serves) **inverted to be the nested child**.

```
BEFORE                                   AFTER
~/0_GENESIS_PROJECT/                      ~/0_MOBIUS.ANAMNESIS/            ← promoted, top-level
├── 0_ANAMNESIS/        ← the app         ├── (app/ d2/ docs/ docker-compose.yml …)
├── genesis.md                            └── 0_GENESIS_PROJECT/          ← nested child
├── CLAUDE.md                                 ├── genesis.md
├── claudia-final-note-20260429.md            ├── CLAUDE.md
├── user_profile_elfege.md                    ├── claudia-final-note-20260429.md
└── README_handoff.md                         └── user_profile_elfege.md
```

The risk isn't the `mv` — it's that the rename (a) **closes every Claude session inside the dir** (both `:assistant` and `:d2`), and (b) **can orphan `mongo_data`** (83,788 episodes) via the unpinned compose-project-name → volume-prefix coupling. So all reversible prep happens first; the irreversible, session-killing `mv` is last, run by the operator from a neutral shell.

---

## 1. Decision points — multiple suggestions where hesitant (operator picks)

**D1. `mongo_data` / volume safety** *(must resolve before any `mv`)*
- **(a) Pin only** — `COMPOSE_PROJECT_NAME=0_anamnesis` in `.env`. *Recommended: minimal, fully reversible, zero data movement.* Volume prefix stays `0_anamnesis_*` regardless of dir name. Trade-off: volumes remain Docker-managed (still outside the backup sweep).
- **(b) Fold in named-volume → bind-mount migration** (the standing CRITICAL TODO). Move `mongo_data` / `anamnesis_voices` / `model_cache` into the renamed tree as gitignored bind mounts → path-portable, backup-sweepable, trap killed permanently. Trade-off: bigger/riskier (must copy volume data with the stack stopped), but done once.
- **(c) Hybrid** — pin now to unblock the rename; schedule (b) as a separate follow-up pass.

**D2. Genesis nesting depth**
- **(a) Fully nest** `0_GENESIS_PROJECT/` under Anamnesis — matches the stated intent.
- **(b) Thin pointer** — keep a minimal top-level genesis (or a symlink) so ecosystem habits/paths assuming `~/0_GENESIS_PROJECT` still resolve on dellserver.

**D3. Who executes** *(neither `:assistant` nor `:d2` can — both die on the `mv`)*
- **(a) Operator from a `~` shell** — recommended.
- **(b) A throwaway Claude session in a neutral dir** (e.g. `~`) that survives the rename.

**D4. Instance ID**
- **(a) Keep `dellserver-anamnesis`** — recommended; MOBIUS projects keep short IDs (cf. `dellserver-nvr`).
- **(b) Adopt a new ID** — registry + host_project_relations churn.

**D5. New `.claude` slug** — confirm slugification (`/`,`_`,`.` → `-`; expected `-home-elfege-0-MOBIUS-ANAMNESIS`) against a live `~/.claude/projects/*MOBIUS*` entry **before** copying. Wrong slug ⇒ the new session starts memory-less.

**D6. Coordination mechanism** — MSG-350 announces **`MOBIUS.COMM`** will replace the `.md` intercom, but per operator it is **not officially up yet** → the `.md` intercom **remains the live mechanism for now**. Keep using it; move the d2 handoff to MOBIUS.COMM only once it's officially live.

---

## 2. Data-loss landmines (neutralize regardless of options)

### 🔴 A. `mongo_data` named-volume orphaning — THE critical one
`COMPOSE_PROJECT_NAME` is **unpinned**, so named volumes are prefixed by the dir basename → currently `0_anamnesis_*`: `0_anamnesis_mongo_data` (**83,788 episodes**), `0_anamnesis_mongo_configdb`, `0_anamnesis_model_cache`, `0_anamnesis_anamnesis_voices`. Rename the dir → compose project name changes → `docker compose up` looks for `0_mobius.anamnesis_*` → not found → **creates NEW EMPTY volumes → memory orphaned.** Neutralize via **D1** before any `mv`; verify `docker compose config` still resolves `0_anamnesis_*`.

### 🟠 B. `.claude` session data (memory + transcripts)
Slug `-home-elfege-0-GENESIS-PROJECT-0-ANAMNESIS` holds the **`memory/` dir** (MEMORY.md + all feedback/project memories) and **2 transcripts**. New CWD → new slug → next session starts memory-less unless migrated. **`cp -a` (copy, keep original as fallback)** old slug → new slug **before** the `mv`.

### 🟡 C. genesis cross-references — mostly a non-issue
`genesis.md`'s **canonical copy lives on `server` (192.168.10.15)** and is cross-referenced ecosystem-wide as `server:~/0_GENESIS_PROJECT/genesis.md` — that copy and those refs are **untouched** (server's dir isn't renamed). This rename is **dellserver-local**; only dellserver path refs to the nested genesis change. Don't touch server's canonical.

---

## 3. Reference inventory to update

| Ref | Location | Change |
|-----|----------|--------|
| Compose project name | `.env` | pin `COMPOSE_PROJECT_NAME=0_anamnesis` (D1) |
| `.claude` slug | `~/.claude/projects/<old> → <new>` | `cp -a` (memory + transcripts) |
| Registry path | `server:~/0_CLAUDE_IC/CLAUDE.md.registry.md` | `dellserver-anamnesis` → `~/0_MOBIUS.ANAMNESIS/CLAUDE.md`; genesis entry → nested path |
| host↔project map | `~/0_SCRIPTS/0_SYNC/0_ENVIRONMENTS/.env.host_project_relations` (+ `.env.sync_arrays`) | add `["0_MOBIUS.ANAMNESIS"]="dellserver"`; resolve nested `0_GENESIS_PROJECT` |
| Project CLAUDE.md | `0_ANAMNESIS/CLAUDE.md` | self-refs `~/0_GENESIS_PROJECT/0_ANAMNESIS` → `~/0_MOBIUS.ANAMNESIS` (project-home, §6 mounts) |
| Crawler/JSONL roots | Mongo `settings` (`jsonl_source_roots`, crawler config) | any old absolute paths |
| cron/scripts | sweep `~/0_SCRIPTS`, `~/0_CRON` for `0_GENESIS_PROJECT/0_ANAMNESIS` | update |

**Unaffected (verify, don't change):** MOBIUS.PROXY `mobius.anamnesis` (routes to `192.168.10.20:3010` by IP); `/dev/shm/anamnesis-restart` (absolute tmpfs); compose relative binds (`./app`, `./samples`, `voices_data` — move with the dir); git remotes; already-ingested episodes whose ids embed old paths (historical — leave).

---

## 4. Execution order (operator, from `~` — NOT inside the dir)

**Pre-flight (reversible):** 1. Confirm new slug (D5). 2. Resolve D1 (pin and/or bind-mount); `docker compose config` shows `0_anamnesis_*`. 3. `cp -a` `.claude` slug → new (verify `memory/` + transcripts present). 4. Record `total_episodes` (83788).
**Config (reversible):** 5. Update registry / host_project_relations / sync arrays / CLAUDE.md / Mongo source roots.
**Cutover:** 6. `cd …/0_ANAMNESIS && ./stop.sh`. 7. `cd ~ && mv ~/0_GENESIS_PROJECT/0_ANAMNESIS ~/0_MOBIUS.ANAMNESIS`. 8. `mv ~/0_GENESIS_PROJECT ~/0_MOBIUS.ANAMNESIS/0_GENESIS_PROJECT` *(any anamnesis-dir Claude session closes here)*. 9. `cd ~/0_MOBIUS.ANAMNESIS && ./start.sh`.

---

## 5. Verification & rollback

- `curl http://localhost:3010/api/dashboard/stats` → **`total_episodes == 83788`** (mongo_data reattached).
- `https://mobius.anamnesis` reachable; crawler + jsonl schedulers healthy in logs; a fresh Claude session in `~/0_MOBIUS.ANAMNESIS` loads memory (MEMORY.md present from the new slug).
- **Rollback:** `.claude` was *copied* (not moved) and the volume prefix is *pinned* → `mv` dirs back + `./start.sh` from the old path loses **zero episodes** in any branch.

---

## 6. After approval (deferred — postponed)

When the operator chooses to act: hand this to `:d2` via the **`.md` intercom** (MOBIUS.COMM not live yet — switch when it is); operator-directed coordination; executor = neutral-dir/operator shell. Until then, nothing runs.
