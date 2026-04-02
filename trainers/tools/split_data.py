"""
split_data.py — Split chat-format JSONL into train/val sets.

Usage:
    python split_data.py sft_chat.jsonl -o /path/to/output/ --val-ratio 0.1
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split JSONL into train/val")
    parser.add_argument("input", help="Input JSONL (chat format)")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Remove metadata fields before writing training data
    for r in records:
        r.pop("_source_chunk", None)
        r.pop("_source_file", None)

    random.seed(args.seed)
    random.shuffle(records)

    n_val = max(1, int(len(records) * args.val_ratio))
    val_records = records[:n_val]
    train_records = records[n_val:]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, data in [("sft_train.jsonl", train_records), ("sft_val.jsonl", val_records)]:
        with open(out / name, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Train: {len(train_records)} | Val: {len(val_records)} → {out}")


if __name__ == "__main__":
    main()
