"""
run_pipeline.py

End-to-end runner for ONE company: download -> segment -> extract metrics
-> extract risks -> save everything to outputs/{ticker}/.

Deliberately does NOT do period comparison or cross-company steps — per
the brief's own guidance, get one company's single-period extraction
right first. Run this once per period, then use compare_periods.py once
you have two periods' outputs.

Usage:
    python src/run_pipeline.py --ticker AAPL --form 10-K --period-end 2024-09-28
"""

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    # Windows consoles default to a cp1252-family codepage, which can't
    # encode characters like "—" or "⚠" used in status messages below and
    # crashes with UnicodeEncodeError mid-run.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import download_filings
import segment_filing
import extract_metrics
import extract_risks

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"


def run(ticker: str, form: str, period_end: str) -> None:
    print(f"[1/4] Downloading {form} for {ticker} (period end {period_end})...")
    filing_dirs = download_filings.download(ticker, form, limit=1, period_end=period_end)
    if not filing_dirs:
        print("No filings downloaded — check EDGAR_CONTACT_EMAIL in .env and network access.")
        return
    filing_dir = filing_dirs[0]
    print(f"  -> accession number: {filing_dir.name}")

    # sec-edgar-downloader saves the primary document; find the .htm file
    html_candidates = list(filing_dir.glob("*.htm")) + list(filing_dir.glob("*.html"))
    if not html_candidates:
        print(f"No HTML file found in {filing_dir} — check download output manually.")
        return
    html_path = html_candidates[0]

    print(f"[2/4] Segmenting filing at {html_path}...")
    sections = segment_filing.segment(html_path, ticker, form, period_end)
    if not sections.mdna_text or not sections.financials_text:
        print("WARNING: segmentation returned an empty section. Inspect the raw HTML "
              "and hand-tune the regex patterns in segment_filing.py for this filer "
              "before trusting downstream results.")

    out_dir = OUTPUT_ROOT / ticker / period_end
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mdna_raw.txt").write_text(sections.mdna_text, encoding="utf-8")
    (out_dir / "financials_raw.txt").write_text(sections.financials_text, encoding="utf-8")

    print("[3/4] Extracting financial metrics via GPT-4o...")
    metrics = extract_metrics.extract(sections.financials_text)
    extract_metrics.save_results(metrics, out_dir / "extracted_metrics.json")
    unverified = [m for m in metrics if m.value is not None and not m.verified_in_source]
    if unverified:
        print(f"  ⚠ {len(unverified)} metric(s) failed the snippet-verification check — review before trusting.")

    print("[4/4] Extracting risk signals from MD&A via GPT-4o...")
    risks = extract_risks.extract(sections.mdna_text)
    extract_risks.save_results(risks, out_dir / "risk_mentions.json")

    print(f"\nDone. Outputs written to {out_dir}")
    print("Next step: manually build ground truth for this company/period in "
          "data/ground_truth/template.csv, then run src/evaluate.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--form", required=True, choices=["10-K", "10-Q"])
    parser.add_argument("--period-end", required=True, help="e.g. 2024-09-28, used as output folder label")
    args = parser.parse_args()

    run(args.ticker, args.form, args.period_end)
