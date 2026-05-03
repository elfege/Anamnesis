#!/usr/bin/env python3
"""
anamnesis_to_tokens.py — Anamnesis episodes → tokenized .bin files for δ² training.

WHAT IT DOES (for the dummies):
================================
1. Pulls every episode from the Anamnesis MongoDB (via the /api/episodes endpoint).
2. Sorts them by timestamp (chronological order = the dialectical structure δ² is meant to exploit).
3. Tokenizes the text with GPT-2 BPE (same tokenizer as our WikiText prep).
4. Splits the chronological stream into N "tasks" (e.g., one per month, or N equal-token chunks).
5. Writes each task as `task_{NN}.bin` plus a `manifest.json` describing each chunk.

The output lands in `d2/d2_data/personal/<corpus_name>/`. A continual-learning
training run reads these files in order, so δ²'s bassin sees task transitions
(your conversations grouped by time) and can retain structural tensions across them.

USAGE:
======
    # Default: split all episodes by month, write to d2/d2_data/personal/anamnesis_chronological/
    python scripts/anamnesis_to_tokens.py

    # Custom: split into N equal-token tasks
    python scripts/anamnesis_to_tokens.py --split-mode equal_tokens --n-tasks 8

    # Custom corpus name (multiple training runs can use different splits without overwriting)
    python scripts/anamnesis_to_tokens.py --corpus anamnesis_by_month_v1

PRIVACY:
========
Output goes to `d2/d2_data/personal/` which is in .gitignore AND .dockerignore.
NEVER copy this output into a docker image. The `Dockerfile.example` security
header explicitly forbids `COPY d2/d2_data/`. The pre-push hook also catches
attempts to commit `*.bin` files.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import requests

logger = logging.getLogger("anamnesis_to_tokens")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_API = "http://192.168.10.20:3010"
DEFAULT_CORPUS_NAME = "anamnesis_chronological"
DEFAULT_OUT_DIR = Path(__file__).parent.parent / "d2" / "d2_data" / "personal"
DEFAULT_PAGE_SIZE = 100   # API hard cap: limit ≤ 100
SUPPORTED_SPLIT_MODES = ("by_month", "equal_tokens", "single")


# ── Episode fetch ────────────────────────────────────────────────────────────

def fetch_all_episodes(api_url: str, page_size: int = DEFAULT_PAGE_SIZE) -> list[dict]:
    """
    Pull every episode from Anamnesis. Pages through /api/episodes until exhausted.
    Returns a flat list, NOT yet sorted.
    """
    episodes: list[dict] = []
    page = 0
    while True:
        # GET /api/episodes supports pagination via skip/limit
        url = f"{api_url}/api/episodes"
        params = {"skip": page * page_size, "limit": page_size}
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
        except Exception as exc:
            logger.error(f"fetch failed at page {page}: {exc}")
            break
        batch = r.json()
        # Different API shapes — handle both list and {"episodes": [...]}
        if isinstance(batch, dict):
            batch = batch.get("episodes", [])
        if not batch:
            break
        episodes.extend(batch)
        logger.info(f"fetched page {page}: +{len(batch)} (total {len(episodes)})")
        if len(batch) < page_size:
            break
        page += 1
    return episodes


# ── Episode → text ────────────────────────────────────────────────────────────

def episode_to_text(ep: dict) -> str:
    """
    Concatenate the most-information-dense fields of an episode into a text blob
    that captures both the surface content (raw_exchange) and the distilled meaning
    (summary). Adds a chat-style turn separator the BPE will tokenize cleanly.

    Fields used (in order of priority):
      - summary (always present, distilled)
      - raw_exchange (often present, the actual conversation turn pair)
      - tags (informational header for the model)
    """
    parts = []
    inst = ep.get("instance", "?")
    proj = ep.get("project", "?")
    ts = ep.get("timestamp", "")
    parts.append(f"<<EPISODE inst={inst} project={proj} ts={ts}>>")
    if ep.get("tags"):
        parts.append("tags: " + ", ".join(ep["tags"]))
    if ep.get("summary"):
        parts.append("SUMMARY:\n" + ep["summary"])
    if ep.get("raw_exchange"):
        parts.append("EXCHANGE:\n" + ep["raw_exchange"])
    parts.append("<<END>>")
    return "\n".join(parts) + "\n\n"


# ── Sorting + splitting ──────────────────────────────────────────────────────

def parse_ts(s: str) -> datetime:
    """Best-effort timestamp parse — returns UTC epoch=0 if unparseable."""
    if not s:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        # Most timestamps are ISO 8601
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.fromtimestamp(float(s), tz=timezone.utc)
        except Exception:
            return datetime.fromtimestamp(0, tz=timezone.utc)


def sort_episodes_chronologically(episodes: list[dict]) -> list[dict]:
    return sorted(episodes, key=lambda e: parse_ts(e.get("timestamp", "")))


def split_by_month(episodes: list[dict]) -> dict[str, list[dict]]:
    """Bucket episodes by year-month string (e.g. '2026-04')."""
    buckets: dict[str, list[dict]] = {}
    for ep in episodes:
        ts = parse_ts(ep.get("timestamp", ""))
        key = ts.strftime("%Y-%m")
        buckets.setdefault(key, []).append(ep)
    return buckets


def split_equal_tokens(token_ids: list[int], n_tasks: int) -> list[list[int]]:
    """Split a flat token stream into N roughly-equal contiguous chunks."""
    if n_tasks < 1:
        return [token_ids]
    chunk_size = len(token_ids) // n_tasks
    chunks = [token_ids[i * chunk_size : (i + 1) * chunk_size] for i in range(n_tasks)]
    # Tail goes to last chunk
    if chunks and len(token_ids) % n_tasks:
        chunks[-1].extend(token_ids[n_tasks * chunk_size :])
    return chunks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default=DEFAULT_API, help=f"Anamnesis API base URL (default: {DEFAULT_API})")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS_NAME, help="Corpus name (becomes the output subdirectory)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Parent output dir")
    ap.add_argument("--split-mode", choices=SUPPORTED_SPLIT_MODES, default="by_month",
                    help="How to split into tasks: by_month (one task per calendar month — default), equal_tokens (N equal chunks), single (one task with everything)")
    ap.add_argument("--n-tasks", type=int, default=5, help="Used when --split-mode=equal_tokens")
    ap.add_argument("--instance-filter", default=None,
                    help="Only include episodes from this instance (e.g. 'office', 'office-genesis'). Default: all.")
    ap.add_argument("--project-filter", default=None,
                    help="Only include episodes from this project. Default: all.")
    ap.add_argument("--min-summary-chars", type=int, default=50,
                    help="Skip episodes whose summary is shorter than this (default: 50)")
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="Fraction of LAST chronological episodes held out as val.bin (default 5%)")
    args = ap.parse_args()

    # tiktoken — same tokenizer as WikiText prep (vocab compatibility)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        logger.info("using tiktoken GPT-2 BPE encoder")
    except ImportError:
        logger.error("tiktoken not installed: pip install tiktoken")
        sys.exit(1)

    # 1. Fetch
    eps = fetch_all_episodes(args.api)
    logger.info(f"fetched {len(eps)} total episodes")

    # 2. Filter
    if args.instance_filter:
        eps = [e for e in eps if e.get("instance") == args.instance_filter]
        logger.info(f"after instance filter '{args.instance_filter}': {len(eps)}")
    if args.project_filter:
        eps = [e for e in eps if e.get("project") == args.project_filter]
        logger.info(f"after project filter '{args.project_filter}': {len(eps)}")
    eps = [e for e in eps if len((e.get("summary") or "")) >= args.min_summary_chars]
    logger.info(f"after min_summary_chars filter: {len(eps)}")

    if not eps:
        logger.error("no episodes after filtering — nothing to write")
        sys.exit(2)

    # 3. Sort chronologically (this is critical — δ²'s value lives in temporal structure)
    eps = sort_episodes_chronologically(eps)
    first_ts = parse_ts(eps[0].get("timestamp", "")).isoformat()
    last_ts = parse_ts(eps[-1].get("timestamp", "")).isoformat()
    logger.info(f"chronological span: {first_ts} → {last_ts}")

    # 4. Hold out tail for val
    n_val = max(1, int(len(eps) * args.val_frac))
    train_eps = eps[:-n_val]
    val_eps = eps[-n_val:]
    logger.info(f"train={len(train_eps)}, val={len(val_eps)}")

    # 5. Build output dir
    out_dir = args.out_dir / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"output dir: {out_dir}")

    # 6. Tokenize + write
    manifest: dict = {
        "corpus": args.corpus,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_episodes": len(eps),
        "train_episodes": len(train_eps),
        "val_episodes": len(val_eps),
        "chronological_span": {"first": first_ts, "last": last_ts},
        "filters": {
            "instance": args.instance_filter,
            "project": args.project_filter,
            "min_summary_chars": args.min_summary_chars,
        },
        "split_mode": args.split_mode,
        "tokenizer": "gpt2 (tiktoken)",
        "vocab_size": enc.n_vocab,
        "tasks": [],
    }

    def write_bin(token_ids: list[int], path: Path):
        arr = np.array(token_ids, dtype=np.uint16)
        arr.tofile(path)
        return len(arr)

    if args.split_mode == "single":
        text = "".join(episode_to_text(e) for e in train_eps)
        ids = enc.encode_ordinary(text)
        n = write_bin(ids, out_dir / "train.bin")
        manifest["tasks"].append({"id": "task_00", "filename": "train.bin", "n_tokens": n, "n_episodes": len(train_eps)})
        logger.info(f"wrote single-task train.bin: {n:,} tokens, {len(train_eps)} episodes")

    elif args.split_mode == "by_month":
        buckets = split_by_month(train_eps)
        for i, (month, month_eps) in enumerate(sorted(buckets.items())):
            text = "".join(episode_to_text(e) for e in month_eps)
            ids = enc.encode_ordinary(text)
            fname = f"task_{i:02d}_{month}.bin"
            n = write_bin(ids, out_dir / fname)
            manifest["tasks"].append({
                "id": f"task_{i:02d}",
                "filename": fname,
                "month": month,
                "n_tokens": n,
                "n_episodes": len(month_eps),
            })
            logger.info(f"wrote {fname}: {n:,} tokens, {len(month_eps)} episodes ({month})")

    elif args.split_mode == "equal_tokens":
        text = "".join(episode_to_text(e) for e in train_eps)
        all_ids = enc.encode_ordinary(text)
        chunks = split_equal_tokens(all_ids, args.n_tasks)
        for i, chunk in enumerate(chunks):
            fname = f"task_{i:02d}.bin"
            n = write_bin(chunk, out_dir / fname)
            manifest["tasks"].append({"id": f"task_{i:02d}", "filename": fname, "n_tokens": n})
            logger.info(f"wrote {fname}: {n:,} tokens")

    # 7. Validation set (always single, chronological tail)
    val_text = "".join(episode_to_text(e) for e in val_eps)
    val_ids = enc.encode_ordinary(val_text)
    n_val_tokens = write_bin(val_ids, out_dir / "val.bin")
    manifest["val"] = {"filename": "val.bin", "n_tokens": n_val_tokens, "n_episodes": len(val_eps)}
    logger.info(f"wrote val.bin: {n_val_tokens:,} tokens, {len(val_eps)} episodes")

    # 8. Manifest
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"wrote {out_dir}/manifest.json")

    # 9. Summary
    total_train_tokens = sum(t["n_tokens"] for t in manifest["tasks"])
    print()
    print("─" * 70)
    print(f"  Corpus written: {out_dir}")
    print(f"  Tasks: {len(manifest['tasks'])}")
    print(f"  Train tokens: {total_train_tokens:,}")
    print(f"  Val tokens:   {n_val_tokens:,}")
    print(f"  Span:         {first_ts}  →  {last_ts}")
    print("─" * 70)
    print()
    print(f"  Next: train δ² with --data-dir {out_dir.parent} --dataset {args.corpus}")
    print()


if __name__ == "__main__":
    main()
