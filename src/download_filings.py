"""
download_filings.py

Downloads 10-K / 10-Q filings from SEC EDGAR for a given ticker using the
sec-edgar-downloader package, which wraps the official EDGAR full-text
search + submissions APIs.

IMPORTANT (per project brief edge cases):
- SEC requires a descriptive User-Agent with a real contact email on every
  request, or it will block you. Set EDGAR_CONTACT_EMAIL in your .env file.
- We download raw filing HTML/txt here. Segmentation into Financial
  Statements vs MD&A happens in segment_filing.py, NOT here — keep this
  script dumb and reliable.

Usage:
    python src/download_filings.py --ticker AAPL --form 10-K --limit 2
    python src/download_filings.py --ticker AAPL --form 10-Q --limit 2
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "filings"


def _headers(contact_email: str, company_name: str) -> dict:
    return {"User-Agent": f"{company_name} {contact_email}"}


def _get_cik(ticker: str, contact_email: str, company_name: str) -> str:
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=_headers(contact_email, company_name),
        timeout=30,
    )
    r.raise_for_status()
    for entry in r.json().values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Could not find CIK for ticker {ticker!r} in SEC's ticker list.")


def find_filing_for_period(
    ticker: str, form: str, period_end: str, contact_email: str, company_name: str
) -> tuple[str, str]:
    """Look up the exact accession number + filed date for a filing whose
    reportDate (fiscal period end) matches `period_end` (YYYY-MM-DD).

    This exists because sec-edgar-downloader's `.get(..., limit=1)` only
    returns the MOST RECENT filing of a form type — it has no concept of
    "the filing for fiscal year X". Blindly using limit=1 for two different
    calls (meant to be FY2024 and FY2023) would silently download the same
    (latest) filing twice. We look up EDGAR's submissions API first so we
    know precisely which accession number we're targeting before we ever
    call the downloader.
    """
    cik = _get_cik(ticker, contact_email, company_name)
    r = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=_headers(contact_email, company_name),
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    recent = data["filings"]["recent"]
    for i, f in enumerate(recent["form"]):
        if f == form and recent["reportDate"][i] == period_end:
            return recent["accessionNumber"][i], recent["filingDate"][i]

    # Older filings are paginated into separate files, not included in
    # "recent" — check those too if we didn't find it above.
    for older in data["filings"].get("files", []):
        r2 = requests.get(
            f"https://data.sec.gov/submissions/{older['name']}",
            headers=_headers(contact_email, company_name),
            timeout=30,
        )
        r2.raise_for_status()
        older_data = r2.json()
        for i, f in enumerate(older_data["form"]):
            if f == form and older_data["reportDate"][i] == period_end:
                return older_data["accessionNumber"][i], older_data["filingDate"][i]

    raise ValueError(
        f"No {form} filing found for {ticker} with fiscal period end {period_end}. "
        "Check the period-end date is correct (it must match EDGAR's reportDate)."
    )


def list_filings(ticker: str, form: str, limit: int = 8) -> list[dict]:
    """Look up the most recent `limit` filings of `form` type for `ticker`
    via EDGAR's submissions API, without downloading anything.

    Used by the Streamlit app's live "analyze a company" flow so a user can
    pick a real fiscal period from a dropdown instead of guessing a date.
    Returns most-recent-first: [{"period_end", "filed_date", "accession"}].
    """
    contact_email = os.getenv("EDGAR_CONTACT_EMAIL")
    if not contact_email:
        raise RuntimeError(
            "Set EDGAR_CONTACT_EMAIL in your .env file — SEC EDGAR requires "
            "a real contact email in the User-Agent header for every request."
        )
    company_name = os.getenv("EDGAR_COMPANY_NAME", "Independent Research")

    cik = _get_cik(ticker, contact_email, company_name)
    r = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=_headers(contact_email, company_name),
        timeout=30,
    )
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]

    results = []
    for i, f in enumerate(recent["form"]):
        if f == form:
            results.append({
                "period_end": recent["reportDate"][i],
                "filed_date": recent["filingDate"][i],
                "accession": recent["accessionNumber"][i],
            })
            if len(results) >= limit:
                break
    return results


def download(
    ticker: str, form: str, limit: int = 1, period_end: str | None = None
) -> list[Path]:
    """Download filing(s) of `form` type for `ticker`.

    If `period_end` (YYYY-MM-DD, the fiscal period end date) is given, we
    first resolve the exact accession number via EDGAR's submissions API,
    then constrain sec-edgar-downloader's date window tightly around that
    filing's actual filed date so we download precisely that filing and
    nothing else — not just "whatever is most recent".

    Returns list of paths to the downloaded filing directories so the
    caller (or a pipeline runner) knows exactly what landed on disk.
    """
    contact_email = os.getenv("EDGAR_CONTACT_EMAIL")
    if not contact_email:
        raise RuntimeError(
            "Set EDGAR_CONTACT_EMAIL in your .env file — SEC EDGAR requires "
            "a real contact email in the User-Agent header for every request."
        )

    company_name = os.getenv("EDGAR_COMPANY_NAME", "Independent Research")

    dl = Downloader(company_name, contact_email, download_folder=str(DATA_DIR.parent.parent))

    expected_accession = None
    if period_end:
        expected_accession, filed_date_str = find_filing_for_period(
            ticker, form, period_end, contact_email, company_name
        )
        filed_date = datetime.strptime(filed_date_str, "%Y-%m-%d").date()
        dl.get(
            form,
            ticker,
            limit=1,
            after=(filed_date - timedelta(days=1)).isoformat(),
            before=(filed_date + timedelta(days=1)).isoformat(),
            download_details=True,
        )
    else:
        dl.get(form, ticker, limit=limit, download_details=True)

    # sec-edgar-downloader lays files out as:
    # sec-edgar-filings/{ticker}/{form}/{accession_number}/
    filings_root = DATA_DIR.parent.parent / "sec-edgar-filings" / ticker / form
    if not filings_root.exists():
        return []
    dirs = sorted(filings_root.iterdir())

    if expected_accession:
        # sec-edgar-downloader names folders with dashes stripped sometimes
        # differ in formatting; compare on digits only to be safe.
        wanted = expected_accession.replace("-", "")
        matches = [d for d in dirs if d.name.replace("-", "") == wanted]
        if not matches:
            raise RuntimeError(
                f"Downloaded filing(s) {[d.name for d in dirs]} do not match the "
                f"expected accession number {expected_accession} for period_end="
                f"{period_end}. Aborting rather than proceeding with the wrong filing."
            )
        return matches

    return dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SEC filings for a ticker")
    parser.add_argument("--ticker", required=True, help="e.g. AAPL")
    parser.add_argument("--form", required=True, choices=["10-K", "10-Q"])
    parser.add_argument("--limit", type=int, default=2, help="Number of periods to pull")
    parser.add_argument("--period-end", default=None, help="Fiscal period end YYYY-MM-DD to target exactly")
    args = parser.parse_args()

    paths = download(args.ticker, args.form, args.limit, args.period_end)
    print(f"Downloaded {len(paths)} {args.form} filing(s) for {args.ticker}:")
    for p in paths:
        print(f"  - {p}")
