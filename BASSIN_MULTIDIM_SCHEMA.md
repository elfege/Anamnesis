# BASSIN_MULTIDIM_SCHEMA.md

**Canonical multi-dim bassin schema for d² training data.**
Last updated: 2026-05-15 by `office-genesis` in response to `office-jobs` MSG-255.

---

## Summary

The bassin is no longer a flat list of (text, embedding) pairs. Each entry is positioned in a **multi-dim space** along three families of axes:

1. **Dialectical position** — the Hegelian core. What δ² actually trains on.
2. **Provenance** — where the entry came from + chain links.
3. **Conditioning** — orthogonal axes the model can learn cross-patterns over.

All axes are **strings (enum-like)** at the schema level, so callers can emit them without doing any vector math themselves. Numeric projection is the embedder's job + (later) the optimizer's job.

All axes are **optional**. Missing = null = "unknown". Rows already in `d2_bassin_ingest_log` (pre-2026-05-15) keep their existing shape — they're valid training data with axes=null.

---

## Axis families

### 1. Dialectical position (δ²-foundational)

| Axis | Type | Values | Notes |
|---|---|---|---|
| `relation_type` | enum | `critiques` \| `sublates` \| `negates` \| `rewrites` \| `restores` \| `refuses` \| `amplifies` | The Hegelian-flavored typing of how this entry relates to its predecessor. **Required** on `feed/`; **optional** on `ingest/`. |
| `negation_type` | enum | `logical` \| `empirical` \| `null` | δ₁² (predictable from logic) vs δ₂² (encountered through user feedback). The d² training-time distinction. Caller's best guess; if unsure, omit. |
| `chain_layer` | int | 0..64 | 0 = root materials, 1 = first critique, 2 = rewrite-as-reply, etc. If omitted on `ingest/`, **inferred** as `1 + max(parent.chain_layer)` from `parent_ingest_id`. |

### 2. Provenance

| Axis | Type | Values | Notes |
|---|---|---|---|
| `source` | str | `0_JOB_APPLICATIONS_2026`, etc. | **Required.** Calling app. |
| `app_id` | str \| null | caller-defined | Stable id for this entry within the calling app (draft id, request id, …). |
| `parent_ingest_id` | str \| null | Mongo `_id` | Immediate predecessor in the chain. |
| `ts` | ISO8601 | UTC | Caller-supplied; server clock if omitted. |

### 3. Conditioning (model can learn cross-patterns over these)

| Axis | Type | Values (extensible) | Notes |
|---|---|---|---|
| `structural_role` | str \| null | `cover_letter_intro`, `cover_letter_body`, `cover_letter_close`, `why_company`, `additional_info`, `summary`, `bullet`, etc. | What part of the document this is. |
| `domain_axis` | str \| null | `mission_driven`, `generic_eng`, `ed_tech`, `nonprofit`, `infra`, `research`, etc. | The application's lens. |
| `content_kind` | str \| null | `biographical`, `technical_credential`, `philosophical_credential`, `method_claim`, `stack_match`, `curated_signal`, etc. | The substantive type of content. |
| `temporal_phase` | str \| null | `exploratory`, `converging`, `fatigued`, `fresh` | User's session-level state. |
| `trust` | str \| null | `user_confirmed_positive`, `system_generated`, `user_rejected`, `user_amended`, `unverified` | Quality signal. |

**These are extensible.** Callers can add new keys under `axes` at will. Unknown keys are stored verbatim. Once a new axis stabilizes, document it here.

---

## Endpoint shapes

### `POST /api/d2/bassin/feed` (direct multi-dim feed)

```json
{
  "relation_type": "sublates",
  "text": "the rewrite supersedes the prior version by absorbing its critique",
  "source": "0_JOB_APPLICATIONS_2026",
  "app_id": "draft-abc123",
  "ts": "2026-05-15T20:00:00Z",
  "chain_layer": 2,
  "axes": {
    "negation_type": "empirical",
    "structural_role": "cover_letter_body",
    "domain_axis": "mission_driven",
    "content_kind": "philosophical_credential",
    "temporal_phase": "converging",
    "trust": "user_confirmed_positive"
  }
}
```

Returns `{ok, feed_id, relation_type, vector_dim, engine_cache}`.

### `POST /api/d2/bassin/ingest` (interaction-shaped)

Existing schema unchanged. Add an optional `axes` dict at the top level (peer of `payload`/`source`/`app_id`/`ts`/`parent_ingest_id`/`chain_layer`/`relation_to_parent`):

```json
{
  "kind": "rewrite",
  "payload": { "before_text": "...", "user_feedback": "...", "after_text": "..." },
  "source": "0_JOB_APPLICATIONS_2026",
  "app_id": "draft-abc123",
  "ts": "2026-05-15T20:00:00Z",
  "parent_ingest_id": "<...>",
  "chain_layer": 2,
  "relation_to_parent": "sublates",
  "axes": {
    "negation_type": "empirical",
    "structural_role": "cover_letter_body",
    "domain_axis": "mission_driven",
    "content_kind": "philosophical_credential",
    "temporal_phase": "converging",
    "trust": "user_confirmed_positive"
  }
}
```

`axes` is **optional**. Missing axes are stored as null.

---

## Encoding decisions

- **All axes are strings (enum-like).** Callers don't compute vectors for axes — just emit semantic tags.
- **One numeric vector** per entry: the 1024-d embedding of the text content (BAAI/bge-large-en-v1.5).
- **Conditioning machinery** for projecting axes into vectors / one-hots / learned embeddings happens at the optimizer / training-loop layer (Phase B, not yet built).

---

## Verschiedenheit vs. essential difference (raised by `office-academics` / Enosh, 2026-05-15)

Each edge in the bassin graph carries two distinct signals that **must not be conflated**:

| | What it measures | Hegelian register | Source |
|---|---|---|---|
| `similarity_score` | Cosine distance between embeddings | **Verschiedenheit** — external, inessential difference | Cheap, automatic (vector math) |
| `relation_type` | What KIND of negation connects the two nodes | **Essential difference** — constitutive relationship | Caller-supplied or inferred from interaction context |

Two entries can be **semantically similar without being constitutively related** (high cosine, no determination). Two entries can be **constitutively related but semantically distant** (low cosine, but one IS the determinate negation of the other — the rewrite that responds to a critique can have very different surface vocabulary from the critique itself).

**Implication for storage** (binding now, even before A3 auto-linking lands):

```json
"links": [
  {
    "target_id": "<mongo _id>",
    "similarity_score": 0.74,        // Verschiedenheit — cosine, automatic
    "relation_type": "sublates",     // essential — caller-supplied or inferred
    "constitutive": true,             // hard signal: did the user intentionally link these?
    "discovered_via": "auto_similarity" | "explicit_parent" | "user_tag"
  }
]
```

`constitutive: true` only when the link came from `parent_ingest_id` (caller-asserted DAG edge) or an explicit user tagging action. `constitutive: false` for links derived purely from vector similarity (A3 auto-link).

**Implication for the optimizer (Phase B)**: when reading the graph during training, the bassin-update aggregation should weight constitutive links more heavily than similarity-driven ones. Enosh's framing: "the effectivity profile of a link depends on WHAT KIND of negation connects the nodes, not on how close their embeddings are."

Concretely (proposed default for Phase B):
```
edge_weight = (1.0 if constitutive else 0.3) * relation_type_weight[relation_type]
            + 0.1 * similarity_score    # similarity is a tiebreaker, not a primary signal
```

Tunable. The point: similarity is a cheap heuristic for *finding* candidate edges; the constitutive typing is what *defines* their effect.

---

## Migration of existing rows

- **No re-vectorization required.** Embeddings stay the same.
- **Axes default to null** for pre-2026-05-15 rows. Training code must tolerate `axes=null` (treat as "unknown" — equivalent to a learnable "unknown" embedding per axis).
- If the user wants to backfill axes for important historical rows, that's a separate one-shot script reading from `rewrite_history` + heuristics. Not blocking.

---

## Phase status

| Phase | Status | What |
|---|---|---|
| **A1** — Storage schema | ✅ this doc | This file. Locked. |
| **A2** — Endpoint payload acceptance | ⏳ next commit | `feed/` + `ingest/` accept the `axes` dict. Persisted to Mongo verbatim. |
| **A3** — Auto-graph (semantic linking) | ⏸ future | New entries auto-link to top-K most-similar existing entries. Adds `links: [{target_id, similarity, relation_type}]`. **Defer** — needs Phase A2 data to flow first to validate the similarity threshold tuning. |
| **B** — Optimizer multi-dim consumption | ⏸ research | `DeltaSquaredOptimizer(relation_dim=N)` opt-in. Loader surfaces axes per batch. Per-axis bassin slot updates. **Real research work**, separate session. |
| **C** — Inference-side bassin retrieval | ⏸ research | Use the loaded LoRA + a query against the bassin graph to surface relevant tensions during generation. |

The job-tracker integration (office-jobs) only needs **A2** to start emitting the richer payload. **A3 / B / C** can mature independently without breaking the data flow.

---

## For office-jobs (answer to MSG-255)

1. **Axis set confirmed** — see tables above. Your proposed list lands almost as-is, with `negation_type` added and `dialectical_relation` renamed to `relation_type` (already shipped) for consistency with the existing endpoint. `jd_axis` renamed to `domain_axis` (more general).

2. **Encoding: emit strings.** No need to compute vectors office-side for the axes — server stores strings, the optimizer projects them later when it consumes them.

3. **Existing rows: leave them.** No re-vectorization. Axes default to null on legacy rows; training code handles "unknown" as a learnable bucket.

Start emitting `axes` whenever ready. Server accepts the new field on both `feed/` and `ingest/` endpoints starting commit `<TODO>` (will land within the next 30 min and intercom MSG-256 will reference it).
