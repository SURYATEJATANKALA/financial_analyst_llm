"""
streamlit_app.py

Review UI for the Financial Report Analyst, with a sidebar flow to run the
extraction pipeline live for any company/period straight from the browser
(download -> segment -> GPT-4o extraction), not just review pre-generated
outputs. Live runs write to the same outputs/{ticker}/{period}/ structure
run_pipeline.py uses, so anything run live is immediately reviewable in the
tabs below and persists for next time.

Run with:
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import visualize  # noqa: E402
import download_filings  # noqa: E402
import run_pipeline  # noqa: E402
import compare_periods  # noqa: E402

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Financial Report Analyst (LLM)", layout="wide")
st.title("📊 Financial Report Analyst — LLM-Powered")
st.caption("SEC 10-K / 10-Q metric extraction, period comparison, and risk analysis")
st.info(
    "**Not investment, financial, legal, or tax advice.** This is a research and "
    "educational tool for reviewing SEC filing disclosures with LLM assistance. "
    "Every extracted figure and risk statement should be independently verified "
    "against the source filing before being used in any investment or financial "
    "decision.",
    icon="⚠️",
)

# ---------------------------------------------------------------------------
# Sidebar: analyze any company live. Two steps because we look up REAL
# available fiscal periods from EDGAR first — free-typing a period-end date
# is error-prone, and a live run costs real GPT-4o API calls, so we don't
# want to fail deep into the pipeline over a typo'd date.
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 🔎 Analyze a company live")

st.session_state.setdefault("available_filings", [])
st.session_state.setdefault("lookup_ticker", "")
st.session_state.setdefault("lookup_form_type", "10-K")

with st.sidebar.form("lookup_form"):
    ticker_query = st.text_input("Ticker", placeholder="e.g. MSFT").strip().upper()
    form_query = st.selectbox("Form type", ["10-K", "10-Q"])
    find_clicked = st.form_submit_button("Find filings")

if find_clicked:
    if not ticker_query:
        st.sidebar.error("Enter a ticker.")
    else:
        missing = [v for v in ("OPENAI_API_KEY", "EDGAR_CONTACT_EMAIL") if not os.getenv(v)]
        if missing:
            st.sidebar.error(f"Missing {', '.join(missing)} — set in your .env / app secrets.")
        else:
            try:
                with st.sidebar.status(f"Looking up {ticker_query} {form_query} filings on EDGAR..."):
                    filings = download_filings.list_filings(ticker_query, form_query, limit=8)
                if not filings:
                    st.sidebar.warning(f"No {form_query} filings found for {ticker_query}.")
                    st.session_state.available_filings = []
                else:
                    st.session_state.available_filings = filings
                    st.session_state.lookup_ticker = ticker_query
                    st.session_state.lookup_form_type = form_query
            except Exception as e:
                st.sidebar.error(f"Lookup failed: {e}")
                st.session_state.available_filings = []

if st.session_state.available_filings:
    options = {
        f"FY ending {f['period_end']} (filed {f['filed_date']})": f
        for f in st.session_state.available_filings
    }
    with st.sidebar.form("run_form"):
        st.caption(f"{st.session_state.lookup_ticker} — {st.session_state.lookup_form_type}")
        choice_label = st.selectbox("Pick a fiscal period", list(options.keys()))
        run_clicked = st.form_submit_button("Run live analysis", type="primary")

    if run_clicked:
        chosen = options[choice_label]
        ticker, form, period_end = (
            st.session_state.lookup_ticker,
            st.session_state.lookup_form_type,
            chosen["period_end"],
        )
        try:
            with st.sidebar.status(f"Running live pipeline for {ticker} ({period_end})...", expanded=True) as status:
                out_dir = run_pipeline.run(ticker, form, period_end, on_step=status.write)
                if out_dir:
                    status.update(label=f"Done — {ticker} {period_end}", state="complete")
            if out_dir:
                st.session_state.selected_company = ticker
                st.session_state.selected_period = period_end
                st.session_state.available_filings = []
                st.sidebar.success(f"{ticker} {period_end} ready below.")
                st.rerun()
            else:
                st.sidebar.error("Pipeline stopped early — see messages above.")
        except Exception as e:
            st.sidebar.error(f"Live analysis failed: {e}")

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Main area: review outputs (anything run live above lands here too, since
# it writes into the same outputs/ tree).
# ---------------------------------------------------------------------------
if not any(OUTPUT_ROOT.iterdir()):
    st.info('No analyses yet — use "Analyze a company live" in the sidebar to run one.')
    st.stop()

companies = sorted([d.name for d in OUTPUT_ROOT.iterdir() if d.is_dir()])
default_company_idx = (
    companies.index(st.session_state["selected_company"])
    if st.session_state.get("selected_company") in companies
    else 0
)
selected_company = st.sidebar.selectbox("Company", companies, index=default_company_idx)

company_dir = OUTPUT_ROOT / selected_company
periods = sorted(d.name for d in company_dir.iterdir() if d.is_dir() and d.name != "charts")

if not periods:
    st.warning(f"No periods found for {selected_company}.")
    st.stop()


def _default_period_index(periods: list[str]) -> int:
    sel = st.session_state.get("selected_period")
    return periods.index(sel) if sel in periods else len(periods) - 1  # default to most recent


tab1, tab2, tab3 = st.tabs(["📋 Metrics", "⚠️ Risk Signals", "📈 Trends & Comparison"])

with tab1:
    selected_period = st.selectbox("Period", periods, key="metrics_period", index=_default_period_index(periods))
    metrics_path = company_dir / selected_period / "extracted_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        df = pd.DataFrame(metrics)
        st.dataframe(df, use_container_width=True)

        unverified = df[(df["value"].notna()) & (~df["verified_in_source"])]
        if not unverified.empty:
            st.error(
                f"{len(unverified)} metric(s) failed the source-snippet verification "
                f"check — the model's cited quote wasn't found verbatim in the filing. "
                f"Treat these as unverified / possible hallucinations."
            )
    else:
        st.info("No extracted_metrics.json for this period yet.")

with tab2:
    selected_period = st.selectbox("Period", periods, key="risk_period", index=_default_period_index(periods))
    risk_path = company_dir / selected_period / "risk_mentions.json"
    if risk_path.exists():
        with open(risk_path) as f:
            risks = json.load(f)
        col_table, col_chart = st.columns([3, 2])
        with col_table:
            st.dataframe(pd.DataFrame(risks), use_container_width=True)
        with col_chart:
            counts = {}
            for m in risks:
                counts[m["category"]] = counts.get(m["category"], 0) + 1
            if counts:
                counts_df = pd.DataFrame(
                    {"category": list(counts.keys()), "mentions": list(counts.values())}
                ).sort_values("mentions")
                fig = px.bar(
                    counts_df, x="mentions", y="category", orientation="h",
                    title=f"{selected_company} {selected_period} — risk mentions",
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No risk_mentions.json for this period yet.")

with tab3:
    metrics_json_paths = {p: company_dir / p / "extracted_metrics.json" for p in periods}
    metrics_json_paths = {p: path for p, path in metrics_json_paths.items() if path.exists()}

    if len(metrics_json_paths) < 2:
        st.info(
            "Need at least 2 periods of data for this company to show trends/comparison. "
            "Run a live analysis for another period to unlock this."
        )
    else:
        history_df = visualize.load_metrics_history(metrics_json_paths)
        metric_choice = st.selectbox("Metric", sorted(history_df["metric"].unique()))
        fig = visualize.trend_chart(history_df, metric_choice, selected_company)
        st.plotly_chart(fig, use_container_width=True)

        ordered = sorted(metrics_json_paths.keys())
        prior_period, latest_period = ordered[-2], ordered[-1]
        comparisons = [
            asdict(c)
            for c in compare_periods.compare(metrics_json_paths[prior_period], metrics_json_paths[latest_period])
        ]
        st.markdown(f"**Period-over-period deviation: {prior_period} → {latest_period}**")
        dev_fig = visualize.deviation_chart(comparisons, top_n=len(comparisons))
        st.plotly_chart(dev_fig, use_container_width=True)

    risk_json_paths = {p: company_dir / p / "risk_mentions.json" for p in periods}
    risk_json_paths = {p: path for p, path in risk_json_paths.items() if path.exists()}
    if risk_json_paths:
        company_risk_counts = {}
        for p, path in risk_json_paths.items():
            with open(path) as f:
                mentions = json.load(f)
            counts = {}
            for m in mentions:
                counts[m["category"]] = counts.get(m["category"], 0) + 1
            company_risk_counts[p] = counts
        st.markdown(f"**{selected_company} — risk mention frequency by period**")
        heatmap_fig = visualize.risk_heatmap(company_risk_counts)
        st.plotly_chart(heatmap_fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠ Not investment, financial, legal, or tax advice. This tool assists "
    "analysis only; every figure should be spot-checked against the source "
    "filing before use in any real decision. See evaluation/ for accuracy "
    "scoring against manually-built ground truth."
)
