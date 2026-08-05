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

from llm_usage import (
    MODEL, CHUNK_THRESHOLD_CHARS, ExtractionMeta, Timer,
    build_meta, combine_meta, save_meta, split_into_chunks,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

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
def _call_llm(client: OpenAI, mdna_text: str):
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
    return resp.choices[0].message.content, resp.usage


def _extract_one_call(client: OpenAI, chunk_text: str) -> tuple[list[RiskMention], ExtractionMeta]:
    with Timer() as timer:
        raw, usage = _call_llm(client, chunk_text)
    meta = build_meta(usage, timer.elapsed_seconds)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {raw[:500]}") from e

    mentions = [
        RiskMention(
            category=m["category"],
            excerpt=m["excerpt"],
            severity_language=m.get("severity_language", "hedged"),
        )
        for m in parsed.get("risk_mentions", [])
    ]
    return mentions, meta


def extract(mdna_text: str, api_key: str | None = None) -> tuple[list[RiskMention], ExtractionMeta]:
    if len(mdna_text) > 150_000:
        raise ValueError(
            f"MD&A text is {len(mdna_text)} chars — suspiciously large; "
            "check segment_filing.py boundaries."
        )

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    if len(mdna_text) <= CHUNK_THRESHOLD_CHARS:
        return _extract_one_call(client, mdna_text)

    # Oversized MD&A: same per-request token rate limit as extract_metrics.py
    # (see llm_usage.py). Unlike the fixed 8-metric schema, risk mentions are
    # an open list, so chunks merge by concatenation rather than by picking
    # one best candidate per slot — deduplicated on normalized excerpt text
    # in case the same boilerplate risk sentence happens to appear in more
    # than one chunk.
    chunks = split_into_chunks(mdna_text)
    all_mentions: list[RiskMention] = []
    seen_excerpts: set[str] = set()
    metas = []
    for chunk in chunks:
        chunk_mentions, chunk_meta = _extract_one_call(client, chunk)
        metas.append(chunk_meta)
        for m in chunk_mentions:
            key = " ".join(m.excerpt.split()).lower()
            if key not in seen_excerpts:
                seen_excerpts.add(key)
                all_mentions.append(m)

    return all_mentions, combine_meta(metas)


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
    mentions, meta = extract(text)
    save_results(mentions, Path(args.output))
    save_meta(meta, Path(args.output).with_name(Path(args.output).stem + "_meta.json"))
    print(f"Found {len(mentions)} risk mentions -> {args.output}")
    print(to_frequency_table(mentions))
    print(f"Model {meta.model}: {meta.total_tokens} tokens, {meta.latency_seconds}s, "
          f"~${meta.estimated_cost_usd}")
