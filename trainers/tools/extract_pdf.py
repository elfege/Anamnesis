"""
extract_pdf.py — Extract text from dissertation PDFs into clean chunks.

Usage:
    python extract_pdf.py /path/to/thesis.pdf -o corpus_chunks.jsonl
    python extract_pdf.py /path/to/*.pdf -o corpus_chunks.jsonl  # multiple PDFs
"""

import argparse
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader


def extract_pages(pdf_path: str) -> list[str]:
    """Extract text from each page of a PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if len(text) > 50:  # skip near-empty pages (cover, blank, etc.)
            pages.append(text)
    return pages


def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    # Fix broken hyphenation at line breaks
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove page numbers alone on a line
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    # Remove excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Clean up line breaks within paragraphs (keep double newlines as paragraph breaks)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned.append(line)
        else:
            cleaned.append('')
    text = '\n'.join(cleaned)
    # Collapse runs of empty lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 6000, overlap: int = 500) -> list[str]:
    """Split text into overlapping chunks, breaking at paragraph boundaries."""
    paragraphs = text.split('\n\n')
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > max_chars and current:
            chunk = '\n\n'.join(current)
            if len(chunk) >= 200:
                chunks.append(chunk)
            # Keep last paragraph for overlap
            current = [current[-1]] if current else []
            current_len = len(current[0]) if current else 0
        current.append(para)
        current_len += para_len

    # Last chunk
    if current:
        chunk = '\n\n'.join(current)
        if len(chunk) >= 200:
            chunks.append(chunk)

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Extract and chunk PDF text")
    parser.add_argument("pdfs", nargs="+", help="PDF files to process")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")
    parser.add_argument("--max-chars", type=int, default=6000, help="Max chars per chunk")
    args = parser.parse_args()

    all_chunks = []
    for pdf_path in args.pdfs:
        path = Path(pdf_path)
        if not path.exists():
            print(f"  [skip] {path} not found", file=sys.stderr)
            continue

        print(f"  Processing: {path.name}", file=sys.stderr)
        pages = extract_pages(str(path))
        full_text = '\n\n'.join(pages)
        full_text = clean_text(full_text)
        chunks = chunk_text(full_text, max_chars=args.max_chars)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": path.name,
                "chunk_index": i,
                "text": chunk,
            })

        print(f"    {len(pages)} pages → {len(chunks)} chunks", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_chunks)} chunks → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
