#!/usr/bin/env python3
"""
anachat_render.py — Render Anamnesis search results from JSON on stdin.
Called by anachat bash script: curl ... | python3 anachat_render.py
"""
import sys
import json
import os
import textwrap

raw = sys.stdin.read()
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print("  [error parsing response]")
    sys.exit(1)

results = data if isinstance(data, list) else data.get("results", [])

RESET  = "\033[0m"
YELLOW = "\033[1;33m"
TEAL   = "\033[1;36m"
GREEN  = "\033[0;32m"
DIM    = "\033[2m"
WHITE  = "\033[0;37m"

if not results:
    print(f"  {DIM}No episodes found.{RESET}\n")
    sys.exit(0)

try:
    width = min(os.get_terminal_size().columns, 100)
except OSError:
    width = 80

for i, ep in enumerate(results, 1):
    score   = ep.get("boosted_score") or ep.get("similarity_score", 0)
    boosted = (ep.get("priority_multiplier") or 1.0) > 1.0
    star    = " ★" if boosted else ""
    score_color = TEAL if boosted else YELLOW
    tags    = " · ".join(ep.get("tags") or [])
    ts      = (ep.get("timestamp") or "")[:10]
    project  = ep.get("project", "")
    instance = ep.get("instance", "")
    summary  = ep.get("summary", "")

    print(
        f"  [{i}] {score_color}{score:.3f}{star}{RESET}  "
        f"{DIM}{project}  │  {instance}{RESET}"
        + (f"  {DIM}{ts}{RESET}" if ts else "")
    )

    for line in textwrap.wrap(summary, width - 8):
        print(f"      {WHITE}{line}{RESET}")

    if tags:
        print(f"      {GREEN}{tags}{RESET}")

    print()
