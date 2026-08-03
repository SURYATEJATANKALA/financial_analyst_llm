"""
visualize.py

Builds the three chart types the brief calls for:
  1. Metric trend lines over periods, per company (revenue, margin, EPS)
  2. Risk-mention frequency heatmap (rows=companies, columns=risk categories)
  3. Period-over-period deviation chart flagging the largest changes

Kept separate from the Streamlit app so these can also be generated
standalone into the outputs/ folder for the written report (brief wants
"visualizations" as a standalone deliverable, not just something living
inside the app).
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def load_metrics_history(json_paths: dict[str, Path]) -> pd.DataFrame:
    """json_paths: {period_label: path_to_extracted_metrics.json}"""
    rows = []
    for period, path in json_paths.items():
        with open(path) as f:
            metrics = json.load(f)
        for m in metrics:
            rows.append({"period": period, "metric": m["metric"], "value": m["value"], "unit": m["unit"]})
    return pd.DataFrame(rows)


def trend_chart(df: pd.DataFrame, metric_name: str, company: str) -> go.Figure:
    subset = df[df["metric"] == metric_name].sort_values("period")
    fig = px.line(
        subset, x="period", y="value", markers=True,
        title=f"{company} — {metric_name} over time",
    )
    fig.update_layout(yaxis_title=metric_name, xaxis_title="Reporting Period")
    return fig


def risk_heatmap(company_risk_counts: dict[str, dict[str, int]]) -> go.Figure:
    """company_risk_counts: {company: {category: count}}"""
    df = pd.DataFrame(company_risk_counts).T.fillna(0)
    fig = px.imshow(
        df, labels=dict(x="Risk Category", y="Company", color="Mentions"),
        text_auto=True, aspect="auto", color_continuous_scale="Reds",
        title="Risk Mention Frequency by Company and Category",
    )
    return fig


def deviation_chart(comparisons: list[dict], top_n: int = 10) -> go.Figure:
    """comparisons: list of dicts from compare_periods.MetricComparison (as dict)"""
    df = pd.DataFrame(comparisons)
    df = df[df["pct_change"].notna()].copy()
    df["abs_change"] = df["pct_change"].abs()
    df = df.sort_values("abs_change", ascending=False).head(top_n)

    fig = go.Figure(go.Bar(
        x=df["pct_change"], y=df["metric"], orientation="h",
        marker_color=["crimson" if v < 0 else "seagreen" for v in df["pct_change"]],
    ))
    fig.update_layout(
        title=f"Top {top_n} Period-over-Period Deviations",
        xaxis_title="% Change", yaxis_title="Metric",
    )
    return fig


if __name__ == "__main__":
    import argparse
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"

    parser = argparse.ArgumentParser(
        description="Generate standalone trend/risk-heatmap/deviation charts as HTML "
        "files from already-generated pipeline outputs."
    )
    parser.add_argument("--ticker", required=True, help="e.g. AAPL")
    parser.add_argument(
        "--periods", nargs="+", required=True,
        help="Period folder names under outputs/{ticker}/, in chronological order, e.g. 2023-09-30 2024-09-28",
    )
    parser.add_argument(
        "--comparison", required=True,
        help="Path to a compare_periods.py output JSON (list of MetricComparison dicts)",
    )
    parser.add_argument("--out-dir", default=None, help="Where to write chart HTML files (default: outputs/{ticker}/charts)")
    args = parser.parse_args()

    company_dir = OUTPUT_ROOT / args.ticker
    out_dir = Path(args.out_dir) if args.out_dir else company_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_paths = {p: company_dir / p / "extracted_metrics.json" for p in args.periods}
    missing = [p for p, path in json_paths.items() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing extracted_metrics.json for period(s): {missing}")

    history_df = load_metrics_history(json_paths)

    # 1. Trend charts — one per metric present in the data.
    trend_dir = out_dir / "trends"
    trend_dir.mkdir(exist_ok=True)
    for metric_name in sorted(history_df["metric"].unique()):
        fig = trend_chart(history_df, metric_name, args.ticker)
        safe_name = metric_name.lower().replace(" ", "_").replace("&", "and")
        fig.write_html(trend_dir / f"{safe_name}.html")
    print(f"Wrote {history_df['metric'].nunique()} trend chart(s) -> {trend_dir}")

    # 2. Risk-mention frequency heatmap — rows=periods (proxy for
    # company/period), columns=risk categories, since this run covers one
    # company across two periods rather than multiple companies.
    risk_counts = {}
    for period in args.periods:
        risk_path = company_dir / period / "risk_mentions.json"
        if risk_path.exists():
            with open(risk_path) as f:
                mentions_raw = json.load(f)
            counts = {}
            for m in mentions_raw:
                counts[m["category"]] = counts.get(m["category"], 0) + 1
            risk_counts[f"{args.ticker} {period}"] = counts
    if risk_counts:
        heatmap_fig = risk_heatmap(risk_counts)
        heatmap_path = out_dir / "risk_heatmap.html"
        heatmap_fig.write_html(heatmap_path)
        print(f"Wrote risk heatmap -> {heatmap_path}")
    else:
        print("No risk_mentions.json found for any period — skipping heatmap.")

    # 3. Period-over-period deviation chart.
    with open(args.comparison) as f:
        comparisons = json.load(f)
    dev_fig = deviation_chart(comparisons, top_n=len(comparisons))
    dev_path = out_dir / "deviation_chart.html"
    dev_fig.write_html(dev_path)
    print(f"Wrote deviation chart -> {dev_path}")
