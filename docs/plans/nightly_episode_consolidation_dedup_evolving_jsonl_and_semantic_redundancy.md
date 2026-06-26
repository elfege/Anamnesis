# Nightly Episode Consolidation — dedup evolving JSONL conversations + semantic redundancy

- **Status:** Design / not started
- **Author:** `dellserver-anamnesis:assistant` — for the operator; **owner = `dellserver-anamnesis:d2`** (episodes/memory = system-level lane)
- **Date:** 2026-06-13
- **Untracked:** under `docs/` (gitignored) per RULE 19.1.3.

---

## 0. Problem

`episodes` is at ~83.8k and accumulates **redundant near-duplicates** that hash-dedup does *not* catch:

- The JSONL ingester dedups on `_content_hash(raw_exchange)` (stored in `crawl_state`). That only catches **exact** re-ingestion.
- But **conversations evolve** — a session grows, gets re-ingested longer → different raw → different hash → **a fresh `jsonl_` episode per evolution point.** Result: N snapshots of the *same* conversation at increasing lengths.
- Net effect: a single content query returns **dozens of fragmentary/overlapping episodes** instead of one reliable, findable one.

Operator's framing (2026-06-13): *"we have hash logic to avoid duplicates but conversations being ingested as JSONL evolve."* Exactly — evolution is the gap hash-dedup structurally can't close.

---

## 1. Key finding — provenance is already recoverable (no ingest change needed)

The JSONL episode_id encodes source identity:

```
jsonl_{machine}_{project}_{ts}_{session_short}_{hash_short}
```
- `app/jsonl_ingester.py:775` builds it; `session_short` = first 8 chars of the conversation's `sessionId` (`:509`, `:773`).

⇒ All snapshots of one conversation share `(machine, project, session_short)`. **Grouping by source is parseable from the id today.** This makes the cheap, deterministic regime available immediately.

*(Open: confirm ingest granularity — per-exchange vs per-whole-file episodes — which tunes how supersession decides "superset". §6.)*

---

## 2. Two regimes

### Regime 1 — same-source supersession (deterministic, NO LLM, high-confidence)
Group `jsonl_` episodes by `(machine, project, session_short)`. Within a group, when a later episode's `raw_exchange` is a **superset** of an earlier one (or simply: same session, newest/longest), the newest **supersedes** the older snapshots. No summarization, no hallucination — just "keep the latest version of this conversation." **This directly kills the evolving-JSONL bloat.**

### Regime 2 — cross-source semantic merge (fuzzy, LLM, lower-confidence, phase 2)
Embedding-cluster near-duplicate episodes from *different* sources (cosine over the existing 1024-d vectors; HDBSCAN/threshold-agglomerative). For dense clusters, LLM-synthesize one canonical consolidated episode (Ollama/Claude/Together backends already wired). Higher "findability" payoff, real risk (false merges, lost nuance) → conservative thresholds, opt-in, after R1 proves out.

---

## 3. Non-negotiable principle — never destroy originals (genesis-aligned: *don't sanitize/lose*)

- **Supersede, don't delete.** Mark superseded episodes with a flag (e.g. `superseded_by: <episode_id>`, `superseded_at`), **exclude from default search**, but keep the doc retrievable for provenance/audit. Reuse the existing **blocklist/exclude** plumbing (`POST /{id}/exclude`, `ingestion_blocklist`) rather than inventing a new mechanism.
- Consolidated episodes carry `consolidated_from: [episode_ids]` so the lineage is explicit and reversible.
- The **PATCH endpoint** (shipped 2026-06-13, §5) is a building block: consolidation can fold new info into a canonical episode instead of spawning another.

---

## 4. Integration

- **Scheduler:** add a `consolidation` job to the existing nightly framework (`app/scheduler.py` already runs crawler/jsonl/training nightly). Config in Mongo `settings._id = consolidation_schedule` (mirror `crawler_schedule`/`jsonl_schedule` presets: disabled / nightly@time / etc.).
- **Search:** filter `superseded_by` out of `vector_search` default results; optionally **prefer** consolidated episodes (a small boost) so one dense canonical beats N fragments.
- **Incremental:** only process sessions/clusters new-or-changed since last run (track a high-water mark) — avoid re-touching the whole 83.8k nightly.
- **Cost:** R1 is pure DB ops (cheap). R2 LLM calls only on fresh dense clusters.

---

## 5. Already shipped (building block) — `PATCH /api/episodes/{id}`

In-place episode edit (2026-06-13): `app/models.py:EpisodePatch` + `app/routes/episodes.py:patch_episode`. Edits `summary/tags/project/instance/raw_exchange`; **re-embeds iff `summary` changes** (verified in-DB: vector changes on summary-patch, unchanged on tags-patch). `episode_id` stays immutable (rename = delete + re-POST; `DELETE /{id}` already exists). Live; pending push (core API).

---

## 6. Risks / open questions

- **Ingest granularity:** is an episode per-exchange or per-whole-conversation? Determines whether "superset" is checked on `raw_exchange` text or inferred from session+timestamp ordering. **First thing to confirm.**
- **False supersession:** a long session may yield *distinct* lessons, not just snapshots of one. Guard: only supersede on genuine superset/containment, not merely shared session.
- **`session_short` collisions:** 8 hex chars within one project — low but nonzero; scope grouping to `(machine, project, session_short)` and optionally store the full `session_id` as a field at ingest for safety.
- **R2 LLM merge** can hallucinate or merge things that shouldn't be → keep conservative, reversible, phase-2.
- **Vector-index churn:** large supersession passes change which docs are searchable — validate the `$vectorSearch` index stays consistent.

---

## 7. Milestones

- **M0:** confirm ingest granularity + add `superseded_by` field + exclude-from-search filter (no behavior change yet, just the schema + read path).
- **M1 (Regime 1):** nightly job — group by session, supersede older snapshots, flag (don't delete). The evolving-JSONL win.
- **M2:** search prefers consolidated/canonical; dashboard surfaces "X superseded this run".
- **M3 (Regime 2):** embedding-cluster + LLM-merge cross-source dups, opt-in, conservative.

---

## TODO

- [ ] (M0) Confirm jsonl episode granularity (per-exchange vs per-file) — read `app/jsonl_ingester.py` ingest loop
- [ ] (M0) Add `superseded_by`/`superseded_at` fields + filter them out of `vector_search`
- [ ] (M1) Nightly supersession job grouping by `(machine, project, session_short)`; flag, never delete
- [ ] (M1) `consolidation_schedule` in Mongo settings + `app/scheduler.py` hook
- [ ] (M2) Search boost for consolidated/canonical episodes
- [ ] (M3) Regime 2: embedding-cluster + LLM-merge (conservative, reversible)
- [ ] Owner/sequencing decision: `:d2` to slot vs δ² priorities
