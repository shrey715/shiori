"""Print the raw text pymupdf extracts from selected PDF pages.

Shows tabs, line breaks and odd characters (repr), which is what the chunker's
regexes have to handle: running headers, page numbers, "9.1<TAB>" clause
numbers, CAPS section titles and "APPENDIX X:" headings.

Usage: python scripts/inspect_pdf.py data/icc_t20wc_2026_playing_conditions.pdf 3 16 73 [--chars 900]
"""
import argparse

import pymupdf

parser = argparse.ArgumentParser()
parser.add_argument("pdf")
parser.add_argument("pages", nargs="+", type=int, help="1-based page numbers")
parser.add_argument("--chars", type=int, default=900, help="characters to show per page")
args = parser.parse_args()

doc = pymupdf.open(args.pdf)
for pno in args.pages:
    print(f"=================== PAGE {pno}")
    print(repr(doc[pno - 1].get_text()[: args.chars]))
