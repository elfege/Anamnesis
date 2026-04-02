"""
generate_qa.py — Generate instruction-tuned Q&A pairs from text chunks using Claude API.

Reads chunks from extract_pdf.py output, sends each to Claude Opus,
collects chat-format training pairs.

Usage:
    python generate_qa.py corpus_chunks.jsonl -o sft_chat.jsonl
    python generate_qa.py corpus_chunks.jsonl -o sft_chat.jsonl --pairs 5 --resume
    python generate_qa.py corpus_chunks.jsonl -o sft_chat.jsonl --dry-run

Requires: ANTHROPIC_API_KEY env var
Cost estimate: ~$0.01-0.02 per chunk with Opus → ~$5-10 for 500 chunks
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic

SYSTEM_PROMPT = """\
You are a teaching assistant for a doctoral dissertation on Hegel's Science of Logic.
The dissertation is "Une critique hégélienne de Hegel" by Elfege Leylavergne (2014, Université de Nantes).
Its central thesis: quantity is the hinge and transversal of Hegel's entire system, yet Hegel himself dismisses it as "speculatively unfruitful" — a self-undermining move.

Given a passage from this dissertation, generate exactly {n_pairs} question-answer pairs.

Rules:
- Questions should be what a philosophy student, a curious reader, or a colleague would ask
- Mix difficulty levels: some introductory ("What does Hegel mean by..."), some advanced ("How does this relate to...")
- Answers must be grounded in the passage — cite or paraphrase specific claims
- Answers should be 2-6 sentences, conversational but precise
- Write in the same language as the passage (French or English)
- If the passage contains both languages, generate pairs in both
- Do NOT invent claims not supported by the passage
- Format the output as a JSON array of objects with "question" and "answer" keys
- Return ONLY the JSON array, no other text"""

USER_TEMPLATE = """\
Here is a passage from the dissertation:

---
{chunk}
---

Generate {n_pairs} question-answer pairs as a JSON array."""


def generate_pairs(client: anthropic.Anthropic, chunk: str, n_pairs: int, model: str) -> list[dict]:
    """Call Claude API to generate Q&A pairs from a text chunk."""
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT.format(n_pairs=n_pairs),
        messages=[{
            "role": "user",
            "content": USER_TEMPLATE.format(chunk=chunk, n_pairs=n_pairs),
        }],
    )

    text = response.content[0].text.strip()

    # Extract JSON array from response (handle markdown code blocks)
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove ```json line
        text = text.rsplit("```", 1)[0]  # remove closing ```
    text = text.strip()

    try:
        pairs = json.loads(text)
        if isinstance(pairs, list):
            return pairs
    except json.JSONDecodeError:
        print(f"    [warn] Failed to parse JSON response, skipping", file=sys.stderr)

    return []


def format_chat_messages(question: str, answer: str, system: str) -> dict:
    """Format a Q&A pair as chat-format training data."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


CHAT_SYSTEM = (
    "You are AnamnesisGPT, a philosophical assistant trained on the writings of Elfege Leylavergne, "
    "particularly his doctoral dissertation on Hegel's Science of Logic. "
    "You answer questions about Hegel, dialectics, quantity, quality, and the Logic "
    "with precision and depth, grounded in the text. "
    "You can discuss in both French and English."
)


def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from text chunks")
    parser.add_argument("input", help="Input JSONL (from extract_pdf.py)")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL (chat format)")
    parser.add_argument("--pairs", type=int, default=5, help="Q&A pairs per chunk")
    parser.add_argument("--model", default="claude-opus-4-6", help="Claude model to use")
    parser.add_argument("--resume", action="store_true", help="Skip chunks already processed")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--limit", type=int, default=0, help="Max chunks to process (0=all)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Load input chunks
    chunks = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks from {args.input}", file=sys.stderr)

    # Load existing output for resume
    existing = set()
    if args.resume and Path(args.output).exists():
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                existing.add(d.get("_source_chunk", -1))
        print(f"Resuming: {len(existing)} chunks already done", file=sys.stderr)

    if args.dry_run:
        remaining = [i for i in range(len(chunks)) if i not in existing]
        if args.limit:
            remaining = remaining[:args.limit]
        cost_est = len(remaining) * 0.015  # rough estimate
        print(f"Would process {len(remaining)} chunks", file=sys.stderr)
        print(f"Estimated cost: ~${cost_est:.2f} with {args.model}", file=sys.stderr)
        return

    client = anthropic.Anthropic()
    output_path = Path(args.output)
    mode = "a" if args.resume else "w"

    total_pairs = 0
    processed = 0

    with open(output_path, mode, encoding="utf-8") as out:
        for i, chunk_data in enumerate(chunks):
            if i in existing:
                continue
            if args.limit and processed >= args.limit:
                break

            text = chunk_data["text"]
            source = chunk_data.get("source", "unknown")

            print(f"  [{i+1}/{len(chunks)}] {source} chunk {chunk_data.get('chunk_index', '?')} "
                  f"({len(text)} chars)...", file=sys.stderr, end=" ", flush=True)

            try:
                pairs = generate_pairs(client, text, args.pairs, args.model)
            except anthropic.RateLimitError:
                print("rate limited, waiting 60s...", file=sys.stderr)
                time.sleep(60)
                pairs = generate_pairs(client, text, args.pairs, args.model)
            except Exception as e:
                print(f"error: {e}", file=sys.stderr)
                continue

            for pair in pairs:
                q = pair.get("question", "")
                a = pair.get("answer", "")
                if q and a:
                    record = format_chat_messages(q, a, CHAT_SYSTEM)
                    record["_source_chunk"] = i
                    record["_source_file"] = source
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_pairs += 1

            print(f"→ {len(pairs)} pairs", file=sys.stderr)
            processed += 1
            out.flush()

            if args.delay > 0:
                time.sleep(args.delay)

    print(f"\nDone: {processed} chunks → {total_pairs} Q&A pairs → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
