"""
extract_risks.py

Extracts and categorizes risk signals from the MD&A section text.
Weighted 20% in the grading breakdown, scored on precision/recall of
risk-mention frequency (used later for the heatmap: rows=companies,
columns=risk categories).

MD&A risk language is subtle by design (brief's edge case: "easy to
over/under-extract") — companies hedge, soften, and bury risk statements
in forward-looking-statement boilerplate. We ask the model to quote the
actual sentence, not paraphrase, so a human reviewer can quickly agree or
disagree with the categorization during ground-truth validation.
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

# Fixed category list keeps the heatmap columns stable across companies.
# Expand only if you find a recurring category that doesn't fit — don't
# let this balloon past ~8 or the heatmap gets unreadable.
RISK_CATEGORIES = [
    "Macroeconomic / Market Conditions",
    "Supply Chain",
    "Competition",
    "Regulatory / Legal",
    "Cybersecurity / Data Privacy",
    "Foreign Exchange / Currency",
    "Litigation",
    "Labor / Talent",
]

SYSTEM_PROMPT = """You are analyzing the MD&A (Management Discussion & Analysis) section \
of an SEC filing to identify risk signals the company itself discusses.

Categories to use (use ONLY these, do not invent new ones): {categories}

For each distinct risk statement you find:
1. Quote the actual sentence (or a tight excerpt, under 30 words) — do not paraphrase.
2. Assign exactly one category from the list above (pick the closest fit).
3. Rate severity_language as "explicit" (company states clear negative impact/concern) \
   or "hedged" (forward-looking boilerplate, conditional language like "could" / "may").

Do not invent risks that aren't discussed. If the MD&A section barely discusses a category, \
that's a valid finding — do not pad the list to cover every category.

Respond with ONLY valid JSON:
{{
  "risk_mentions": [
    {{"category": "<category>", "excerpt": "<quote>", "severity_language": "explicit|hedged"}}
  ]
}}
"""


@dataclass
class RiskMention:
    category: str
    excerpt: str
    severity_language: str


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _call_llm(client: OpenAI, mdna_text: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=3000,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(categories=", ".join(RISK_CATEGORIES))},
            {"role": "user", "content": mdna_text},
        ],
    )
    return resp.choices[0].message.content


def extract(mdna_text: str, api_key: str | None = None) -> list[RiskMention]:
    if len(mdna_text) > 150_000:
        raise ValueError(
            f"MD&A text is {len(mdna_text)} chars — suspiciously large; "
            "check segment_filing.py boundaries."
        )

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    raw = _call_llm(client, mdna_text)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {raw[:500]}") from e

    return [
        RiskMention(
            category=m["category"],
            excerpt=m["excerpt"],
            severity_language=m.get("severity_language", "hedged"),
        )
        for m in parsed.get("risk_mentions", [])
    ]


def to_frequency_table(mentions: list[RiskMention]) -> dict[str, int]:
    """Collapse mentions into category -> count, for the heatmap."""
    counts = {c: 0 for c in RISK_CATEGORIES}
    for m in mentions:
        counts[m.category] = counts.get(m.category, 0) + 1
    return counts


def save_results(mentions: list[RiskMention], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(m) for m in mentions], f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a .txt file of the MD&A section")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    mentions = extract(text)
    save_results(mentions, Path(args.output))
    print(f"Found {len(mentions)} risk mentions -> {args.output}")
    print(to_frequency_table(mentions))
