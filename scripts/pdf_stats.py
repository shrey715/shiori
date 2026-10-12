"""Compare candidate PDFs for the RAG corpus.

For every PDF under a directory, report page count, word count, a rough chunk
estimate (~150 words per chunk), pages with no extractable text (scanned), and
how often exception/negation phrases appear. This is how the ICC T20 World Cup
2026 Playing Conditions were chosen over the other candidates.

Usage: python scripts/pdf_stats.py <dir-of-pdfs>
"""
import pathlib
import sys

import pymupdf

TRAP_PHRASES = ["not ", "shall not", "cannot", "prohibit", "unless", "except", "only if", "provided that"]


def stats(path):
    doc = pymupdf.open(path)
    pages = [page.get_text() for page in doc]
    words = sum(len(t.split()) for t in pages)
    return {
        "document": path.name[:52],
        "pages": doc.page_count,
        "words": words,
        "chunks": words // 150,
        "img_pages": sum(len(t.strip()) < 50 for t in pages),   # likely scanned, needs OCR
        "neg_exc": sum(t.lower().count(k) for t in pages for k in TRAP_PHRASES),
        "modified": doc.metadata.get("modDate", "")[2:10],
    }


def main(root):
    rows = []
    for path in sorted(pathlib.Path(root).rglob("*.pdf")):
        try:
            rows.append(stats(path))
        except Exception as e:  # broken or non-PDF download
            print(f"ERR {path.name}: {e}")
    print(f"{'document':52} {'pages':>5} {'words':>6} {'~chunks':>7} {'imgpgs':>6} {'neg/exc':>7} modified")
    for r in sorted(rows, key=lambda r: -r["words"]):
        print(f"{r['document']:52} {r['pages']:5} {r['words']:6} {r['chunks']:7} {r['img_pages']:6} {r['neg_exc']:7} {r['modified']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
