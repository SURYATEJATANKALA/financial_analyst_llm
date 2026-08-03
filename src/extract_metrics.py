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
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

MODEL = "gpt-4o"

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
   that number. This is used to programmatically verify you did not hallucinate.
3. Report "unit" exactly as presented in the filing (e.g. "thousands", "millions", "actual"). \
   Do not convert units yourself.
4. If a figure appears to be a restated or revised prior-period number (common in 10-Ks that \
   show current + prior year side by side), extract the value for the period explicitly \
   requested and note "restated": true if the text flags it as such (e.g. "as restated", \
   "as revised").
5. Respond with ONLY valid JSON, no markdown fences, no commentary.

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
def _call_llm(client: OpenAI, financials_text: str) -> str:
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
    return resp.choices[0].message.content


def verify_snippet(source_snippet: str | None, full_text: str) -> bool:
    """Confirm the model's claimed source_snippet actually appears in the
    filing text. This is a cheap, mechanical hallucination check — not a
    substitute for human ground-truth comparison, but catches the worst
    cases (invented numbers with invented quotes) for free.
    """
    if not source_snippet:
        return False
    # Normalize whitespace since HTML->text extraction can mangle spacing
    normalized_source = " ".join(full_text.split())
    normalized_snippet = " ".join(source_snippet.split())
    return normalized_snippet in normalized_source


def extract(financials_text: str, api_key: str | None = None) -> list[ExtractedMetric]:
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
    raw = _call_llm(client, financials_text)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {raw[:500]}") from e

    results = []
    for m in parsed.get("metrics", []):
        verified = verify_snippet(m.get("source_snippet"), financials_text)
        results.append(
            ExtractedMetric(
                metric=m["metric"],
                value=m.get("value"),
                unit=m.get("unit"),
                restated=m.get("restated", False),
                source_snippet=m.get("source_snippet"),
                verified_in_source=verified,
            )
        )
    return results


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
    results = extract(text)
    save_results(results, Path(args.output))

    unverified = [r for r in results if r.value is not None and not r.verified_in_source]
    print(f"Extracted {len(results)} metrics -> {args.output}")
    if unverified:
        print(f"WARNING: {len(unverified)} metric(s) had a source_snippet that "
              f"could NOT be found verbatim in the source text. Flag these for "
              f"manual review — likely hallucinations:")
        for r in unverified:
            print(f"  - {r.metric}: {r.value} (claimed snippet: {r.source_snippet!r})")
