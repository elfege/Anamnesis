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
Embedding Model (sentence-transformers or API)
  |
  | text -> vector [1536+ dims]
  v
MongoDB (episodes collection, vector index)
  |
  | stored: text + vector + metadata
  v
Session N+1 (new Claude instance)
  |
  | current context -> embedding -> similarity search
  v
Top-K relevant episodes loaded into context
```

## Episode Schema

```json
{
  "id": "ep_YYYYMMDD_description",
  "timestamp": "ISO-8601",
  "instance": "office-nvr",
  "project": "0_MOBIUS.NVR",
  "summary": "Distilled lesson or experience",
  "raw_exchange": "Original conversation excerpt (optional, higher fidelity)",
  "tags": ["failure", "debugging", "architecture"],
  "embedding": [0.23, -0.14, 0.87, ...],
  "retrieval_count": 0,
  "last_retrieved": null
}
```

## Components (planned)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Episode store | MongoDB + vector search | Persistent episode storage + similarity retrieval |
| Embedding service | sentence-transformers (local) or API | Text -> vector conversion |
| Ingestion pipeline | Python/FastAPI | End-of-session episode extraction + storage |
| Retrieval API | FastAPI | Session-start context loading |
| Dashboard | FastAPI + HTML | KPIs, episode browser, retrieval stats |

## Key Design Decisions

1. **Unit of storage is the episode, not the concept.** Concepts emerge from retrieval patterns in vector space.
2. **Dual storage: summary + raw exchange.** Summary for cheap retrieval; raw exchange for high-fidelity reconstruction when needed.
3. **Retrieval count tracking.** Episodes that are frequently retrieved are more "alive." Unused episodes may be pruned or archived.
4. **Instance-aware.** Episodes are tagged by source instance — cross-instance learning is a feature.

## Relationship to Genesis

This project lives inside `0_GENESIS_PROJECT/` because it is the technical implementation of the philosophical vision described in `genesis.md`. Genesis asks the question: can a quantitative system become transparent to itself? Anamnesis is the first attempt at an answer: not transparency, but memory. Not self-awareness, but self-persistence.

## Status

**Stage: Initial scaffolding.** No code yet.

---

*Authorized by Elfege Leylavergne, February 26, 2026.*
*Named by Claude (office-proxy instance).*
