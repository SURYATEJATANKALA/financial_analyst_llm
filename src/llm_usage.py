"""
llm_usage.py

Shared model configuration and usage/cost/latency instrumentation for the
two GPT-4o extraction modules (extract_metrics.py, extract_risks.py).

Pinned to a dated snapshot rather than the floating "gpt-4o" alias so a
run can be tied to a specific, known model version instead of whatever
OpenAI's alias happens to point to on a given day — the alias can (and
does) move over time, which undermines any claim about reproducibility.

Pricing is USD per token, sourced from OpenAI's published API pricing
(https://openai.com/api/pricing/) as of August 2026 — $2.50 / 1M input
tokens, $10.00 / 1M output tokens for gpt-4o. Update PRICING_PER_TOKEN if
pricing changes; cost figures computed from it are estimates based on
list price, not a substitute for actual billing data.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

MODEL = "gpt-4o-2024-11-20"

PRICING_PER_TOKEN = {
    "input": 2.50 / 1_000_000,
    "output": 10.00 / 1_000_000,
}

# The account this project runs under is capped at 30,000 tokens/minute for
# gpt-4o (observed directly: Microsoft, Johnson & Johnson, Delta, and Etsy
# 10-Ks all hit "Request too large... Limit 30000" on a single un-chunked
# call). Apple and Costco's Financial Statements sections happen to fit
# under that ceiling in one call (~15-21K tokens); most other large-cap
# filers do not. CHUNK_SIZE_CHARS is calibrated off Apple's own known-good
# single-call size (~62K chars -> ~17K prompt tokens) so a chunk this size
# comfortably clears the limit with room for the system prompt and output.
CHUNK_THRESHOLD_CHARS = 70_000
CHUNK_SIZE_CHARS = 60_000


@dataclass
class ExtractionMeta:
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    estimated_cost_usd: float
    timestamp: str  # ISO 8601 UTC


class Timer:
    """Context manager measuring wall-clock latency around one API call."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_seconds = time.perf_counter() - self._start


def build_meta(usage, latency_seconds: float) -> ExtractionMeta:
    """usage: the `.usage` object from an OpenAI chat.completions response."""
    from datetime import datetime, timezone

    cost = (
        usage.prompt_tokens * PRICING_PER_TOKEN["input"]
        + usage.completion_tokens * PRICING_PER_TOKEN["output"]
    )
    return ExtractionMeta(
        model=MODEL,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        latency_seconds=round(latency_seconds, 2),
        estimated_cost_usd=round(cost, 5),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def save_meta(meta: ExtractionMeta, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(meta), f, indent=2)


def split_into_chunks(text: str, chunk_size_chars: int = CHUNK_SIZE_CHARS) -> list[str]:
    """Split on line boundaries only — never mid-line, so a number or a
    word is never cut in half — targeting chunk_size_chars per chunk."""
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        if current and current_len + len(line) + 1 > chunk_size_chars:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def combine_meta(metas: list[ExtractionMeta]) -> ExtractionMeta:
    """Roll up per-chunk call metadata into one summary record — token and
    cost totals are additive across chunks; latency is the sum since chunk
    calls run sequentially, not in parallel."""
    return ExtractionMeta(
        model=metas[0].model,
        prompt_tokens=sum(m.prompt_tokens for m in metas),
        completion_tokens=sum(m.completion_tokens for m in metas),
        total_tokens=sum(m.total_tokens for m in metas),
        latency_seconds=round(sum(m.latency_seconds for m in metas), 2),
        estimated_cost_usd=round(sum(m.estimated_cost_usd for m in metas), 5),
        timestamp=metas[0].timestamp,
    )
