"""
streamlit_app.py

Demo/review UI for the Financial Report Analyst. Reads already-generated
outputs (from run_pipeline.py) rather than running extraction live on
every page load — LLM calls are slow/costly and this is a review tool,
not a real-time service.

Run with:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import visualize  # noqa: E402

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"

st.set_page_config(page_title="Financial Report Analyst (LLM)", layout="wide")
st.title("📊 Financial Report Analyst — LLM-Powered")
st.caption("SEC 10-K / 10-Q metric extraction, period comparison, and risk analysis")

if not OUTPUT_ROOT.exists() or not any(OUTPUT_ROOT.iterdir()):
    st.warning(
        "No outputs found yet. Run `python src/run_pipeline.py --ticker AAPL "
        "--form 10-K --period-end 2024-09-28` first, then refresh this page."
    )
    st.stop()

companies = sorted([d.name for d in OUTPUT_ROOT.iterdir() if d.is_dir()])
selected_company = st.sidebar.selectbox("Company", companies)

company_dir = OUTPUT_ROOT / selected_company
periods = sorted([d.name for d in company_dir.iterdir() if d.is_dir()])

if not periods:
    st.warning(f"No periods found for {selected_company}.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📋 Metrics", "⚠️ Risk Signals", "📈 Trends & Comparison"])

with tab1:
    selected_period = st.selectbox("Period", periods, key="metrics_period")
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
    selected_period = st.selectbox("Period", periods, key="risk_period")
    risk_path = company_dir / selected_period / "risk_mentions.json"
    if risk_path.exists():
        with open(risk_path) as f:
            risks = json.load(f)
        st.dataframe(pd.DataFrame(risks), use_container_width=True)
    else:
        st.info("No risk_mentions.json for this period yet.")

with tab3:
    if len(periods) < 2:
        st.info("Need at least 2 periods of data for this company to show trends/comparison.")
    else:
        json_paths = {p: company_dir / p / "extracted_metrics.json" for p in periods}
        json_paths = {p: path for p, path in json_paths.items() if path.exists()}
        if len(json_paths) >= 2:
            history_df = visualize.load_metrics_history(json_paths)
            metric_choice = st.selectbox("Metric", sorted(history_df["metric"].unique()))
            fig = visualize.trend_chart(history_df, metric_choice, selected_company)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough extracted metric files across periods yet.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠ This tool assists analysis; every figure should be spot-checked against "
    "the source filing before use in any real decision. See evaluation/ for "
    "accuracy scoring against manually-built ground truth."
)
