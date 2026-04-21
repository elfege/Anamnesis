"""
prepare_wikitext.py — Download and tokenize WikiText-103 for training.

WikiText-103 is a standard language modeling benchmark:
- ~100M tokens from Wikipedia
- Public, reproducible, well-understood
- Used by virtually every LM paper for comparison

We use GPT-2's tokenizer (tiktoken) to convert text → token IDs,
then save as binary files (train.bin, val.bin) that the training
loop can memory-map efficiently.

Usage:
    python d2/data/prepare_wikitext.py

Output:
    d2/data/wikitext/train.bin  (~200 MB)
    d2/data/wikitext/val.bin    (~500 KB)
"""

import os
import numpy as np

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "wikitext")
    os.makedirs(out_dir, exist_ok=True)

    # Check if already prepared
    if os.path.exists(os.path.join(out_dir, "train.bin")):
        print("WikiText data already prepared. Delete d2/data/wikitext/ to re-prepare.")
        return

    print("Downloading WikiText-103...")

    # Use HuggingFace datasets to download
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("  pip install datasets")
        return

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    print(f"Train: {len(dataset['train'])} articles")
    print(f"Val:   {len(dataset['validation'])} articles")
    print(f"Test:  {len(dataset['test'])} articles")

    # Use GPT-2 tokenizer via tiktoken (fast, no HF dependency at runtime)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda text: enc.encode_ordinary(text)
        print("Using tiktoken (GPT-2 BPE tokenizer)")
    except ImportError:
        print("ERROR: 'tiktoken' not installed.")
        print("  pip install tiktoken")
        return

    # Tokenize
    for split_name, split_key in [("train", "train"), ("val", "validation")]:
        print(f"Tokenizing {split_name}...")
        all_ids = []
        for example in dataset[split_key]:
            text = example['text']
            if text.strip():  # skip empty lines
                ids = encode(text)
                all_ids.extend(ids)

        ids_array = np.array(all_ids, dtype=np.uint16)
        out_path = os.path.join(out_dir, f"{split_name}.bin")
        ids_array.tofile(out_path)
        print(f"  {split_name}: {len(ids_array):,} tokens → {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
