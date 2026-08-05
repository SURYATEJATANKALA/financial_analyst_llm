"""
extract_metrics.py

Core extraction module. Given the Financial Statements section text for one
filing, asks GPT-4o to extract a fixed set of metrics as structured JSON,
with a short verbatim source snippet for each value so extraction can be
traced back to the filing (brief's #1 grading criterion, 25% weight).

Design choices driven directly by the brief's "points of failure" list:
  - Hallucination is the worst failure mode -> the prompt explicitly
    requires "null" over a guessed number, and every value must carry
    a literal source_snippet we can grep for in the original text as a
    sanity check (see verify_snippet() below).
  - Units (thousands vs millions) -> model is required to report both
    the raw value AND the unit it saw, never to silently convert.
  - Exact-match formatting -> normalization happens in evaluate.py, NOT
    here. This module reports numbers as it found them.
"""

import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from llm_usage import (
    MODEL, CHUNK_THRESHOLD_CHARS, ExtractionMeta, Timer,
    build_meta, combine_meta, save_meta, split_into_chunks,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

# The ~8 metrics ceiling the brief specifies. Keep this list short and
# fight the urge to add "just one more" metric — scope creep here is what
# blows up the manual ground-truth-building step later.
METRICS = [
    "Total Revenue",
    "Net Income",
    "Diluted EPS",
    "Gross Profit",
    "Operating Income",
    "Total Assets",
    "Total Liabilities",
    "Operating Cash Flow",
]

SYSTEM_PROMPT = """You are a financial data extraction engine. You will be given \
raw text extracted from the Financial Statements section of an SEC filing (10-K or 10-Q).

Extract ONLY the following metrics: {metrics}

Rules (violating these is worse than reporting a metric as not found):
1. NEVER estimate, calculate, or infer a number that is not explicitly stated in the text.
   If a metric is not present verbatim, set "value": null and "source_snippet": null.
2. For every non-null value, include a "source_snippet": a short, EXACT, verbatim excerpt \
   (under 15 words) copied character-for-character from the provided text that contains \
   that number, starting at or immediately before the number itself. Never skip past other \
   numbers on the same line to reach a different, later one — the snippet must be the text \
   that actually precedes and contains the value you report, not a different column's value. \
   Do not add a "$" or any other character to the snippet unless it is literally adjacent to \
   the number in the source text — financial statement tables frequently put the label and \
   the number on separate lines with no currency symbol between them at all; do not insert \
   one from memory of how such figures are typically formatted elsewhere.
3. Report "value" as the digits exactly as printed, with only thousands-separator commas \
   removed and any decimal point preserved. NEVER perform arithmetic on it for any reason — \
   do not divide, multiply, or rescale the number, even if you believe a different unit \
   label would look cleaner or more consistent with other figures. Report "unit" exactly as \
   presented in the filing (e.g. "thousands", "millions", "actual"); do not convert units \
   yourself, and do not infer a unit that isn't stated just because the number "looks like" \
   it needs rescaling.
4. If a line item shows more than one reporting period side by side — for example a 10-Q \
   presenting both a single quarter and a year-to-date column, or a current-year vs. \
   prior-year column — extract the value for the single most recent reporting period only: \
   the first number immediately following the label. Never report a year-to-date, \
   cumulative, or prior-period figure even if it appears on the same line.
5. If a figure appears to be a restated or revised prior-period number (common in 10-Ks that \
   show current + prior year side by side), extract the value for the period explicitly \
   requested and note "restated": true if the text flags it as such (e.g. "as restated", \
   "as revised").
6. Respond with ONLY valid JSON, no markdown fences, no commentary.

Output schema:
{{
  "metrics": [
    {{
      "metric": "<name from the list above>",
      "value": <number or null>,
      "unit": "<string or null>",
      "restated": <true/false>,
      "source_snippet": "<verbatim excerpt or null>"
    }}
  ]
}}
"""


@dataclass
class ExtractedMetric:
    metric: str
    value: float | None
    unit: str | None
    restated: bool
    source_snippet: str | None
    verified_in_source: bool = False


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _call_llm(client: OpenAI, financials_text: str):
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=2000,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(metrics=", ".join(METRICS))},
            {"role": "user", "content": financials_text},
        ],
    )
    return resp.choices[0].message.content, resp.usage


_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers_in(text: str) -> list[float]:
    numbers = []
    for raw in _NUMBER_RE.findall(text):
        try:
            numbers.append(float(raw.replace(",", "")))
        except ValueError:
            pass
    return numbers


def verify_snippet(source_snippet: str | None, full_text: str, value: float | None = None) -> bool:
    """Confirm the model's claimed source_snippet actually appears in the
    filing text, AND — if a value was supplied — that the reported value
    itself is one of the raw numbers literally present in that snippet.

    The second check exists because a real, verbatim quote is not enough
    on its own: found in practice (Etsy's 10-K, which reports in
    thousands) the model can quote a real "2,883,501" from the filing but
    report value=2883.501, unit="millions" — silently dividing by 1,000
    and relabeling the unit, exactly the conversion rule 3 of the prompt
    forbids. The quote alone passes the old verbatim-only check every
    time, because the text really is there; only checking that the
    reported value matches a number actually written in that quote
    catches the rescaling.
    """
    if not source_snippet:
        return False
    # Normalize whitespace since HTML->text extraction can mangle spacing
    normalized_source = " ".join(full_text.split())
    normalized_snippet = " ".join(source_snippet.split())
    if normalized_snippet not in normalized_source:
        return False
    if value is not None:
        return any(abs(n - value) < 0.005 for n in _numbers_in(source_snippet))
    return True


def _coerce_value(raw_value) -> float | None:
    """The schema asks for a JSON number, but the model occasionally returns
    the same digits as a JSON string (e.g. "94193" instead of 94193) despite
    response_format enforcing valid JSON — it doesn't enforce our specific
    field types. Coerce defensively rather than letting a str reach
    downstream arithmetic (verify_snippet, evaluate.py's normalize()) and
    crash with a TypeError.
    """
    if raw_value is None:
        return None
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    if isinstance(raw_value, str):
        try:
            return float(raw_value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _extract_one_call(client: OpenAI, chunk_text: str, verify_against: str) -> tuple[list[ExtractedMetric], ExtractionMeta]:
    with Timer() as timer:
        raw, usage = _call_llm(client, chunk_text)
    meta = build_meta(usage, timer.elapsed_seconds)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {raw[:500]}") from e

    results = []
    for m in parsed.get("metrics", []):
        value = _coerce_value(m.get("value"))
        verified = verify_snippet(m.get("source_snippet"), verify_against, value=value)
        results.append(
            ExtractedMetric(
                metric=m["metric"],
                value=value,
                unit=m.get("unit"),
                restated=m.get("restated", False),
                source_snippet=m.get("source_snippet"),
                verified_in_source=verified,
            )
        )
    return results, meta


def extract(financials_text: str, api_key: str | None = None) -> tuple[list[ExtractedMetric], ExtractionMeta]:
    if len(financials_text) > 300_000:
        # Sanity ceiling, not a tight guess — real Item 8 sections vary a
        # lot by filer (Apple ~62K chars, Microsoft ~161K chars with its
        # 18 numbered notes). If we hit this, segment_filing.py likely
        # over-captured. Fail loudly rather than silently truncating and
        # losing a table.
        raise ValueError(
            f"Financials text is {len(financials_text)} chars — suspiciously large. "
            "Check segment_filing.py boundaries before proceeding."
        )

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    if len(financials_text) <= CHUNK_THRESHOLD_CHARS:
        return _extract_one_call(client, financials_text, financials_text)

    # Oversized section: the account's per-request token rate limit (see
    # llm_usage.py) rejects a single call outright for most large-cap
    # filers, so split into chunks small enough to each clear it and run
    # extraction on every chunk independently. The prompt's own "null if
    # not explicitly stated" rule means a chunk that doesn't contain a
    # given metric simply reports it null — no special-casing needed here
    # beyond merging each metric's best answer across chunks afterward.
    chunks = split_into_chunks(financials_text)
    per_metric_candidates: dict[str, list[ExtractedMetric]] = {m: [] for m in METRICS}
    metas = []
    for chunk in chunks:
        chunk_results, chunk_meta = _extract_one_call(client, chunk, financials_text)
        metas.append(chunk_meta)
        for r in chunk_results:
            if r.metric in per_metric_candidates:
                per_metric_candidates[r.metric].append(r)

    results = []
    for metric_name in METRICS:
        candidates = per_metric_candidates[metric_name]
        # Prefer a verified non-null answer; fall back to any non-null
        # answer (still useful, just flagged unverified downstream); only
        # report null if every chunk agreed the metric wasn't there.
        chosen = next((c for c in candidates if c.value is not None and c.verified_in_source), None)
        if chosen is None:
            chosen = next((c for c in candidates if c.value is not None), None)
        if chosen is None:
            chosen = ExtractedMetric(
                metric=metric_name, value=None, unit=None,
                restated=False, source_snippet=None, verified_in_source=False,
            )
        results.append(chosen)

    return results, combine_meta(metas)


def save_results(results: list[ExtractedMetric], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a .txt file of the Financials section")
    parser.add_argument("--output", required=True, help="Where to write extracted_metrics.json")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    results, meta = extract(text)
    save_results(results, Path(args.output))
    save_meta(meta, Path(args.output).with_name(Path(args.output).stem + "_meta.json"))
    print(f"Model {meta.model}: {meta.total_tokens} tokens, {meta.latency_seconds}s, "
          f"~${meta.estimated_cost_usd}")

    unverified = [r for r in results if r.value is not None and not r.verified_in_source]
    print(f"Extracted {len(results)} metrics -> {args.output}")
    if unverified:
        print(f"WARNING: {len(unverified)} metric(s) had a source_snippet that "
              f"could NOT be found verbatim in the source text. Flag these for "
              f"manual review — likely hallucinations:")
        for r in unverified:
            print(f"  - {r.metric}: {r.value} (claimed snippet: {r.source_snippet!r})")
