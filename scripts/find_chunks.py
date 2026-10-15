"""Search data/chunks.jsonl for a phrase, to find or check the gold chunk for a question.

Usage: python scripts/find_chunks.py "Super Over" "ineligible" [--full]
Every phrase must appear in the chunk (case-insensitive).
"""
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("phrases", nargs="+")
parser.add_argument("--full", action="store_true", help="print the whole chunk text")
args = parser.parse_args()

chunks = pd.read_json("data/chunks.jsonl", lines=True)
mask = pd.Series(True, index=chunks.index)
for p in args.phrases:
    mask &= chunks.text.str.contains(p, case=False, regex=False)

for _, r in chunks[mask].iterrows():
    text = r.text if args.full else r.text[:200] + ("…" if len(r.text) > 200 else "")
    print(f"{r.id} | {r.section_path} | {r.clauses} | p.{r.page}\n  {text}\n")
print(f"{mask.sum()} match(es)")
