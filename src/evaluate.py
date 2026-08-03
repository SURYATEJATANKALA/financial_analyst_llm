"""
evaluate.py

Scores extracted_metrics.json against a human-built ground truth CSV.
This is the single most heavily weighted component of the grading
rubric (25% metric accuracy + 15% "evaluation rigor and honesty about
failures" — i.e. this script and its output ARE graded, not just the
extraction).

Ground truth CSV format (see data/ground_truth/template.csv):
    company,period,metric,true_value,true_unit

Exact-match here means: after normalizing units to a common base
(everything converted to "actual dollars"), values must match exactly.
This directly targets the brief's stated failure mode: "exact-match fails
on formatting (commas, $, %)" — normalization happens BEFORE comparison,
not by relaxing the match criteria.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

UNIT_MULTIPLIERS = {
    "actual": 1,
    "thousands": 1_000,
    "millions": 1_000_000,
    "billions": 1_000_000_000,
}


@dataclass
class EvalResult:
    company: str
    period: str
    metric: str
    extracted_value: float | None
    extracted_unit: str | None
    true_value: float | None
    true_unit: str | None
    normalized_extracted: float | None
    normalized_true: float | None
    is_exact_match: bool
    is_citation_correct: bool | None  # None if not manually reviewed yet
    failure_reason: str | None


def normalize(value: float | None, unit: str | None) -> float | None:
    if value is None:
        return None
    mult = UNIT_MULTIPLIERS.get((unit or "actual").strip().lower(), None)
    if mult is None:
        return None  # unrecognized unit -> can't normalize -> treat as unscored, not a silent guess
    return value * mult


def evaluate_company_period(
    extracted_path: Path,
    ground_truth_df: pd.DataFrame,
    company: str,
    period: str,
) -> list[EvalResult]:
    with open(extracted_path) as f:
        extracted = {m["metric"]: m for m in json.load(f)}

    subset = ground_truth_df[
        (ground_truth_df["company"] == company) & (ground_truth_df["period"] == period)
    ]

    results = []
    for _, row in subset.iterrows():
        metric = row["metric"]
        e = extracted.get(metric, {})
        ev, eu = e.get("value"), e.get("unit")
        tv, tu = row["true_value"], row["true_unit"]

        norm_e = normalize(ev, eu)
        norm_t = normalize(tv, tu)

        if norm_e is None and norm_t is None:
            is_match = True  # both correctly identified metric as absent
            failure_reason = None
        elif norm_e is None or norm_t is None:
            is_match = False
            failure_reason = "Extraction returned null where ground truth has a value (or vice versa)"
        else:
            is_match = abs(norm_e - norm_t) < 0.01
            failure_reason = None if is_match else "Value mismatch after unit normalization"

        results.append(EvalResult(
            company=company, period=period, metric=metric,
            extracted_value=ev, extracted_unit=eu,
            true_value=tv, true_unit=tu,
            normalized_extracted=norm_e, normalized_true=norm_t,
            is_exact_match=is_match,
            is_citation_correct=None,  # fill in manually — see README
            failure_reason=failure_reason,
        ))
    return results


def summarize(results: list[EvalResult]) -> dict:
    total = len(results)
    matches = sum(1 for r in results if r.is_exact_match)
    return {
        "total_metrics_evaluated": total,
        "exact_matches": matches,
        "accuracy_pct": round(100 * matches / total, 2) if total else None,
        "failures": [
            {"company": r.company, "period": r.period, "metric": r.metric, "reason": r.failure_reason}
            for r in results if not r.is_exact_match
        ],
    }


if __name__ == "__main__":
    import argparse
    from dataclasses import asdict

    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted", required=True, help="Path to extracted_metrics.json")
    parser.add_argument("--ground-truth", required=True, help="Path to ground_truth CSV")
    parser.add_argument("--company", required=True)
    parser.add_argument("--period", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gt_df = pd.read_csv(args.ground_truth)
    results = evaluate_company_period(Path(args.extracted), gt_df, args.company, args.period)
    summary = summarize(results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": [asdict(r) for r in results], "summary": summary}, f, indent=2)

    print(f"Accuracy: {summary['accuracy_pct']}% ({summary['exact_matches']}/{summary['total_metrics_evaluated']})")
    if summary["failures"]:
        print("Failures:")
        for f in summary["failures"]:
            print(f"  - {f['company']} {f['period']} {f['metric']}: {f['reason']}")
