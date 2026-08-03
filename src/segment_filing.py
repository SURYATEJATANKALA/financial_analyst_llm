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
# _find_validated_matches) so they only match "Item N" when it starts a
# text line. That alone is enough to distinguish a real header from an
# inline cross-reference like "...as described in Item 8 of this Form
# 10-K" buried mid-sentence (verified against Apple's FY2024 10-K, where
# an unanchored "item 8" match landed inside Item 9A's boilerplate instead
# of the real Financial Statements section) — but it is NOT enough on its
# own for every filer. Microsoft's 10-K repeats a bare "Item 7" / "Item 8"
# as a running page-header at the top of literally every page within those
# sections, so line-start anchoring alone still picks up dozens of matches
# scattered through the body text, and "last match" lands on some
# arbitrary page instead of the real header.
#
# The fix: each boundary also carries one or more required title phrases.
# A line-start match only counts as a real section boundary if one of
# these phrases appears within TITLE_WINDOW characters after it — real
# headers and ToC entries both spell out the full title ("Item 8.
# Financial Statements and Supplementary Data"), running-header repeats
# never do (they're just "Item 8" followed by whatever body text happens
# to be on that page). Title phrases deliberately avoid apostrophes
# (curly vs straight quote encoding varies) and get whitespace-stripped
# before comparison (see _normalize_ws) since HTML->text extraction
# sometimes breaks a word mid-token across a line ("FINANCIAL STATE\nMENTS").
TITLE_WINDOW = 250

SECTION_BOUNDARIES = {
    "10-K": {
        "mdna": {
            "start": [(r"^item\s*7[^a]", ["discussion and analysis of financial condition"])],
            "end": [
                (r"^item\s*7a", ["quantitative and qualitative disclosures about market risk"]),
                (r"^item\s*8", ["financial statements and supplementary data"]),
            ],
        },
        "financials": {
            "start": [(r"^item\s*8", ["financial statements and supplementary data"])],
            "end": [(r"^item\s*9[^a]", ["changes in and disagreements with accountants"])],
        },
    },
    "10-Q": {
        "mdna": {
            "start": [(r"^item\s*2[^0-9]", ["discussion and analysis of financial condition"])],
            "end": [(r"^item\s*3", ["quantitative and qualitative disclosures about market risk"])],
        },
        "financials": {
            "start": [(r"^item\s*1[^0-9]", ["financial statements"])],
            "end": [(r"^item\s*2[^0-9]", ["discussion and analysis of financial condition"])],
        },
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


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _find_validated_matches(text_lower: str, alternatives: list[tuple[str, list[str]]]) -> list[re.Match]:
    matches = []
    for pattern, title_hints in alternatives:
        normalized_hints = [_normalize_ws(h) for h in title_hints]
        for m in re.finditer(pattern, text_lower, re.MULTILINE):
            window = _normalize_ws(text_lower[m.start(): m.start() + TITLE_WINDOW])
            if any(hint in window for hint in normalized_hints):
                matches.append(m)
    matches.sort(key=lambda m: m.start())
    return matches


def _slice_section(full_text: str, boundary_cfg: dict) -> str:
    text_lower = full_text.lower()
    start_matches = _find_validated_matches(text_lower, boundary_cfg["start"])
    if not start_matches:
        return ""
    # Take the LAST valid start match before hitting a valid end match —
    # 10-Ks reference the section in the table of contents (also a valid,
    # title-carrying match) before the real section, so the first valid
    # match is usually the ToC, not the section itself.
    end_matches = _find_validated_matches(text_lower, boundary_cfg["end"])
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
    boundaries = SECTION_BOUNDARIES[form]

    mdna = _slice_section(full_text, boundaries["mdna"])
    financials = _slice_section(full_text, boundaries["financials"])

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
