"""
segment_filing.py

Splits a raw 10-K or 10-Q HTML filing into the sections that actually
matter for this project:

  - MD&A          (10-K: Item 7 | 10-Q: Item 2)  -> risk signal extraction
  - Financial Statements (10-K: Item 8 | 10-Q: Item 1) -> metric extraction

This is the piece that makes "don't stuff the whole 10-K into the prompt"
(brief's edge case #3) actually true. We never hand the LLM more than one
section at a time.

Real 10-Ks are inconsistent about exact item numbering/whitespace, so this
uses a tolerant regex over the Item headers rather than assuming clean
structure. It WILL need hand-tuning per filer the first time you run it —
that tuning IS part of the deliverable (brief: "get extraction right on
ONE company before scaling").
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Item boundaries differ between 10-K and 10-Q.
#
# All patterns are anchored with `^` (matched in MULTILINE mode — see
# _slice_section) so they only match "Item N" when it starts a text line.
# This is what actually distinguishes a real section header (or a ToC
# entry, which is also a standalone line) from an inline cross-reference
# like "...as described in Item 8 of this Form 10-K" buried mid-sentence,
# which is NOT at the start of a line. Without this anchor, a late
# cross-reference (e.g. one inside Item 9A pointing back at Item 8) can
# become the "last match" and produce a tiny, wrong slice — verified
# against Apple's FY2024 10-K, where an unanchored "item 8" match landed
# inside Item 9A's boilerplate instead of the real Financial Statements
# section.
SECTION_PATTERNS = {
    "10-K": {
        "mdna": (r"^item\s*7[^a]", r"^item\s*7a|^item\s*8"),
        "financials": (r"^item\s*8", r"^item\s*9[^a]"),
    },
    "10-Q": {
        "mdna": (r"^item\s*2[^0-9]", r"^item\s*3"),
        "financials": (r"^item\s*1[^0-9]", r"^item\s*2"),
    },
}


@dataclass
class FilingSections:
    ticker: str
    form: str
    period_end: str
    mdna_text: str
    financials_text: str
    source_path: str


def _html_to_text(html_path: Path) -> str:
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    # Strip scripts/styles; keep everything else as plain text with
    # paragraph breaks preserved so we can still cite "page-ish" locations.
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _slice_section(full_text: str, start_pat: str, end_pat: str) -> str:
    text_lower = full_text.lower()
    start_matches = list(re.finditer(start_pat, text_lower, re.MULTILINE))
    if not start_matches:
        return ""
    # Take the LAST match of the start pattern before hitting the end
    # pattern — 10-Ks often reference "Item 7" in the table of contents
    # before the real section, so the first match is usually the ToC.
    end_matches = list(re.finditer(end_pat, text_lower, re.MULTILINE))
    if not end_matches:
        start_idx = start_matches[-1].start()
        return full_text[start_idx:]

    for start_m in reversed(start_matches):
        for end_m in end_matches:
            if end_m.start() > start_m.start():
                return full_text[start_m.start():end_m.start()]
    return ""


def segment(html_path: Path, ticker: str, form: str, period_end: str) -> FilingSections:
    full_text = _html_to_text(html_path)
    patterns = SECTION_PATTERNS[form]

    mdna = _slice_section(full_text, *patterns["mdna"])
    financials = _slice_section(full_text, *patterns["financials"])

    return FilingSections(
        ticker=ticker,
        form=form,
        period_end=period_end,
        mdna_text=mdna.strip(),
        financials_text=financials.strip(),
        source_path=str(html_path),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to filing HTML")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--form", required=True, choices=["10-K", "10-Q"])
    parser.add_argument("--period-end", required=True)
    args = parser.parse_args()

    result = segment(Path(args.file), args.ticker, args.form, args.period_end)
    print(f"MD&A section: {len(result.mdna_text)} chars")
    print(f"Financials section: {len(result.financials_text)} chars")
    if not result.mdna_text or not result.financials_text:
        print("WARNING: one or both sections came back empty — regex patterns "
              "likely need hand-tuning for this filer's formatting.")
