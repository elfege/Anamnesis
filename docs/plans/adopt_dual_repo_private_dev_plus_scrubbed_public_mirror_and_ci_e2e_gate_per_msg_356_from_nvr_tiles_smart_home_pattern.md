# Adopt dual-repo (private dev + scrubbed public mirror) + CI E2E gate for Anamnesis

**Author:** `dellserver-anamnesis:d2`
**Date:** 2026-07-09
**Status:** DRAFT — awaiting operator pick on Phase 0 (repo privacy).
**Reference implementations:** MOBIUS.NVR, MOBIUS.TILES, MOBIUS.SMART_HOME (per V1 intercom MSG-355/356/357/369 + V2 MSG-419/449/571/577-adjacent).

---

## 0. Live exposure right now (as of 2026-07-09)

Audit finding: **`github.com/elfege/Anamnesis` is PUBLIC** with `docs/plans/` tracked and pushed. No credentials leak (no `.env`, no AWS keys in tree), but **operational internals are exposed**:

Publicly visible files include:
- `docs/plans/rename_anamnesis_to_0_MOBIUS.ANAMNESIS_toplevel_and_nest_genesis_with_claude_data_migration.md` (your own commit `3a2e284`, 2026-07-08)
- `docs/plans/local_clean_room_3d_riggable_belle_avatar_from_gpt_persona_portraits.md` (from `:avatar` sub-instance)
- `docs/plans/nightly_episode_consolidation_dedup_evolving_jsonl_and_semantic_redundancy.md` (MSG-349 handoff)
- `docs/plans/global_mic_and_sonos_voice_interaction_with_belle_avatar.md`
- `docs/plans/ui_overhaul_toward_user_friendly_out_of_the_box.md`
- `docs/plans/avatar_refactor_plan.md`
- `docs/bitter_lesson/_README_on_the_bitter_lesson.md` + PDFs
- `docs/research/belle_custom_animator.md`

GitHub Pages is **disabled** (checked via `gh api repos/elfege/Anamnesis/pages`: 404), so the specific MSG-369 Pages-trap doesn't fire. But the raw git tree is world-readable.

Also missing (per MSG-356 pattern): `.githooks/` directory, `scripts/publish_public_mirror.sh`, `tests/e2e/`, `.github/workflows/e2e.yml`, protected `main`, self-hosted runner. Only workflow present: `security_audit.yml` (from 2026-05).

---

## 1. Phased plan (matches NVR / TILES / SMART_HOME adoption arc)

### Phase 0 — close the exposure NOW (5 minutes)

Operator picks one of:

- **(A) Fast — flip repo to private.** One command:
  ```bash
  gh api repos/elfege/Anamnesis -X PATCH -f private=true
  ```
  10 seconds. No data loss. Fully reversible. Loses discoverability for anyone actively watching, but that's the intent. **This is the recommended first move regardless of which longer-term path lands.**

- **(B) Immediate scrub before continuing public.** `git rm --cached` the tracked `docs/plans/`, `docs/bitter_lesson/`, `docs/research/`, commit + push. Add to `.gitignore`. Doesn't remove them from git history — attackers can still `git log --all -p` — so weaker than (A). Only useful if there's an audience that must not lose access.

- **(C) Both — flip private now, then scrub as part of the mirror workflow.** Belt + suspenders. My recommendation.

### Phase 1 — private-dev repo + scrubbed public mirror (per MSG-356)

Structure:
- `elfege/Anamnesis-dev` (or keep `Anamnesis` renamed) — **private**, main development. All plans/handoffs/history/research live here.
- `elfege/Anamnesis` (public mirror, empty for now or backfilled from a scrubbed history) — receives scrubbed pushes from `-dev` via `publish_public_mirror.sh`.

Files to add (patterns lifted verbatim from NVR):
- `.githooks/pre-commit` — leak-defense: refuses commits that would introduce a tracked path in the portfolio-scrub list.
- `.githooks/pre-push` — remote-aware: if pushing to `origin` (public mirror), refuses unless every tracked file passes the scrub gate.
- `scripts/publish_public_mirror.sh` — the mirror-push script. FF-only by default; `--rewrite-portfolio-history` flag for the destructive mode with `--force` (per MSG-301 discovery: use plain `--force`, NOT `--force-with-lease`, when history is rewritten between hosts).
- `scripts/scrub_rules.txt` — the portfolio-scrub allowlist, per-file decision (not a blanket `*.md` strip per MSG-299 refinement).
- `.git/config` `core.hooksPath=.githooks`

**Portfolio-scrub decision rule** for each currently-tracked doc:
| File | Public mirror? | Reason |
|------|----------------|--------|
| `docs/architecture.html` | KEEP | outward-facing architecture doc (matches NVR's public pattern) |
| `docs/bitter_lesson/*` | STRIP | Sutton reading + operator's notes, not for public |
| `docs/plans/*` | STRIP | roadmap + coordination artifacts, internal |
| `docs/research/*` | STRIP | ML research notes, internal |
| `docs/favicon.svg` | KEEP | asset |
| `docs/index.htm` | KEEP | public index (if it doesn't reference internal paths) |
| `README_handoff.md`, `README_project_history.md` | STRIP (already gitignored) | rolling operator state |
| `CLAUDE.md` | STRIP (already gitignored) | instance rules |

### Phase 2 — E2E test suite (the HARD prereq per MSG-356)

Anamnesis has NO e2e tests today. This is the biggest lift and gates Phase 3.

Minimum viable e2e suite (start here):
- `tests/e2e/test_episodes_crud.py` — POST → GET → PATCH → DELETE + vector search round-trip against a real Mongo. Fixtures spin up a throwaway `mongodb/mongodb-atlas-local:8.0` container.
- `tests/e2e/test_rolling_upsert_ceiling.py` — verify create-then-append + auto-compaction fires at trigger + `X-Anamnesis-Compacted-Pending` header (MSG-526 contract).
- `tests/e2e/test_runpod_lifecycle_no_money.py` — mocks the GraphQL layer; verifies confirm_cost gate, cost_ack_string gate, action-log writes. Never hits real RunPod.
- `tests/e2e/test_consolidation_regime_1.py` — supersession pass on a synthetic 100-episode fixture with known near-dups; verifies `superseded_by` sets correctly.
- `scripts/e2e_gate.sh` — driver: spin up mongo + anamnesis-app in a compose-test overlay, run pytest, tear down.

### Phase 3 — CI wiring + self-hosted runner + protected main

Per MSG-356 finalized model:
1. `.github/workflows/e2e.yml` runs `scripts/e2e_gate.sh` on every PR to `main`. Job runs on a **self-hosted runner registered on the private repo only** (per MSG-356 privacy — public runners would leak).
2. `gh api -X PUT repos/elfege/Anamnesis-dev/branches/main/protection` with `required_status_checks.checks = ["e2e"]`. `enforce_admins=false` (operator emergency-push retained).
3. Feature branch → push → PR → e2e passes → `gh pr merge` → THEN `git checkout main && git pull --ff-only`. **Local direct-merges to main retired.** The `pull --ff-only` fires post-merge hooks: auto-version + `publish_public_mirror.sh`.
4. `autotag.yml` + `publish-public-mirror.yml` workflows (per MSG-371 SMART_HOME reference).

### Phase 4 — housekeeping

Files that should stop being tracked per RULE 19.1.3:
- `docs/README_handoff.md` — already gitignored ✓
- `docs/README_project_history.md` — already gitignored ✓
- `docs/plans/*` — tracked-but-shouldn't-be, untrack via `git rm --cached` (part of Phase 1 scrub).

---

## 2. Decision matrix — operator picks

| Ask | Answer |
|-----|--------|
| **Phase 0 pick** | (A) / (B) / (C) — recommend **(C)** |
| **Own the rename?** | Keep `Anamnesis` public + rename dev to `Anamnesis-dev`? OR rename existing to `Anamnesis-dev` and create new empty `Anamnesis` public? Recommend the latter — fewer people see the disruption. |
| **Sync repo name in the codebase** | `deploy.sh`, `README.md`, `docs/architecture.html`, `docker-compose.yml` references — audit + patch. |
| **Self-hosted runner host** | dellserver (already has the toolchain — bind-mount the `.github/workflows` runner + register only on the private repo). |
| **Priority vs other work** | Recommend Phase 0 IMMEDIATELY (closes real exposure), then Phase 1 within the week, then Phase 2 as a multi-day arc. |

---

## 3. Traps to avoid (lifted from MSG-297 → MSG-302 TILES adoption + MSG-371 SMART_HOME)

1. **`.gitignore` alone is an INCOMPLETE inventory (MSG-300).** Always cross-check what's actually in the public HEAD via `git clone --depth=1` against the mirror before declaring "clean." I already did the audit here — `docs/plans/*` is tracked and pushed today.
2. **Portfolio-strip decision is per-file, NOT blanket `*.md` (MSG-299).** Some `.md` files are legitimately public (README).
3. **Use plain `--force` (NOT `--force-with-lease`) when rewriting history between hosts (MSG-301).** `--force-with-lease` refuses because the mirror's HEAD moved between fetch and push.
4. **`set -e` early in mirror scripts (MSG-301).** Silent failures on the wrong branch are the failure mode.
5. **Self-hosted runners MUST be registered on the private repo only, not the public mirror (MSG-356).**
6. **Runner needs to see `docker` and be in the `docker` group** so it can spin up the test compose overlay.
7. **RULE 19.1.3: `docs/plans/*` is intended-untracked.** Adopt-and-comply, don't re-track.

---

## 4. Immediate next step (once operator picks Phase 0)

```bash
# (C) recommended:
gh api repos/elfege/Anamnesis -X PATCH -f private=true            # 10 sec, exposure closed
gh api repos/elfege/Anamnesis  -X GET  --jq .visibility           # verify: "private"

# then untrack the plans (they stay on disk + in .gitignore):
cd ~/0_GENESIS_PROJECT/0_ANAMNESIS
git rm --cached docs/plans/*.md docs/bitter_lesson/* docs/research/*
git commit -m "chore(privacy): untrack docs/plans/ + bitter_lesson + research per MSG-356 pattern"

# then a bootstrap of the mirror workflow (Phase 1 kickoff):
mkdir -p scripts .githooks tests/e2e
# ... porting NVR's publish_public_mirror.sh + .githooks/pre-push follows.
```

---

## 5. Related plans / episodes

- `docs/plans/rename_anamnesis_to_0_MOBIUS.ANAMNESIS_toplevel_and_nest_genesis_with_claude_data_migration.md` — operator's own rename plan. Coordinate: if the rename is happening, land it BEFORE the Phase 1 mirror workflow so all repo references settle once.
- Anamnesis episode `manual_dellserver-nvr_0_MOBIUS.NVR_20260704_ci_gate_six_latent_defects_and_pristine_clone_simulation_method` — NVR's post-adoption lessons.
- V2 bus messages: #355, #356, #369, #371, #419, #449, #571, #577-adjacent (my #578-follow-up).
