"""Unified Streamlit dashboard for the Bosch manufacturing analytics project."""

from __future__ import annotations

import importlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.copilot import langchain_handbook  # noqa: E402
from src.copilot.query_engine import answer_question, read_sql  # noqa: E402
from src.dashboard.business_impact import calculate_business_impact  # noqa: E402


# Streamlit reruns the entrypoint in a long-lived Python process. Reload this
# project module so a dashboard update cannot call an older retriever signature.
langchain_handbook = importlib.reload(langchain_handbook)


DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "manufacturing_copilot.db"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs" / "final_deliverables"

st.set_page_config(
    page_title="Bosch Manufacturing Analytics Dashboard",
    page_icon=":material/manufacturing:",
    layout="wide",
)


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DATABASE_PATH, check_same_thread=False)


@st.cache_data
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    return read_sql(get_connection(), sql, params)


@st.cache_data
def load_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / relative_path)


@st.cache_data
def load_json(relative_path: str) -> dict:
    with (PROJECT_ROOT / relative_path).open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_optional_json(relative_path: str) -> dict:
    """Load an optional JSON artifact without crashing a deployed dashboard."""
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def money(value: float) -> str:
    return f"${value:,.0f}"


def get_gemini_configuration() -> tuple[str | None, str]:
    """Read deployment configuration without reading or displaying secrets."""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")
        secret_model = st.secrets.get("GEMINI_MODEL")
    except (FileNotFoundError, KeyError):
        secret_key = None
        secret_model = None
    api_key = str(secret_key or os.getenv("GEMINI_API_KEY") or "").strip() or None
    model = str(secret_model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
    return api_key, model


@st.cache_resource(show_spinner=False)
def get_handbook_retriever():
    """Build the LangChain handbook index once per application process."""
    return langchain_handbook.build_retriever()


def feature_display_name(feature: str) -> str:
    """Translate stored model keys into reader-facing labels for the dashboard."""
    labels = {
        "start_time": "Earliest Measurement Timestamp",
        "end_time": "Latest Measurement Timestamp",
        "cycle_time": "Observed Measurement Span",
        "processing_duration": "Observed Active Measurement Span",
        "waiting_time": "Observed Measurement Gap",
        "mean_waiting_time": "Mean Measurement Gap",
        "max_waiting_time": "Maximum Measurement Gap",
        "wait_event_count": "Measurement-Gap Event Count",
        "delay_ratio": "Relative Measurement-Gap Ratio",
        "observed_date_values": "Observed Measurement Timestamps",
        "station_count": "Stations with Recorded Measurements",
    }
    if feature in labels:
        return labels[feature]

    line_feature = re.fullmatch(r"line_(\d+)_(.+)", feature)
    if line_feature:
        line, suffix = line_feature.groups()
        return f"Line {line}: {labels.get(suffix, suffix.replace('_', ' ').title())}"

    station_feature = re.fullmatch(r"(L\d+_S\d+)_F(\d+)(?:__(is_missing))?", feature)
    if station_feature:
        station, feature_number, missing = station_feature.groups()
        suffix = " Missing-Measurement Indicator" if missing else " Measurement"
        return f"{station} Feature {feature_number}{suffix}"
    return feature


def line_filter() -> list[str]:
    lines = ["L0", "L1", "L2", "L3"]
    selected = st.sidebar.multiselect("Production lines", lines, default=lines)
    return selected or lines


def page_overview(lines: list[str]) -> None:
    st.header("Project KPI Overview")
    summary = query("SELECT * FROM production_summary ORDER BY display_order")
    baseline = query("SELECT * FROM executive_kpi_baseline").set_index("metric_key")
    source_catalog = query("SELECT * FROM source_catalog ORDER BY table_name")
    phase11 = load_json("reports/phase11_database_manifest.json")
    phase12 = load_json("reports/phase12_executive_dashboard_manifest.json")

    cols = st.columns(5)
    cols[0].metric("Historical failure rate", pct(baseline.loc["historical_failure_rate", "value"] * 100, 3))
    cols[1].metric("Validation MCC", f"{baseline.loc['model_mcc', 'value']:.3f}")
    cols[2].metric("Validation precision", pct(baseline.loc["model_precision", "value"] * 100, 1))
    cols[3].metric("Test products scored", f"{int(baseline.loc['test_products_scored', 'value']):,}")
    cols[4].metric("Predicted alerts", f"{int(baseline.loc['test_alerts', 'value']):,}")

    left, right = st.columns([1.15, 1])
    trend = query("SELECT * FROM failure_time_trends ORDER BY period_order")
    trend["period"] = trend["period_order"].map(lambda value: f"P{int(value):02d}")
    fig = px.line(
        trend,
        x="period",
        y="failure_rate_pct",
        markers=True,
        labels={"period": "Relative production period", "failure_rate_pct": "Failure rate (%)"},
        title="Failure trend across relative production time",
    )
    fig.add_hline(
        y=float(baseline.loc["historical_failure_rate", "value"]) * 100,
        line_dash="dot",
        annotation_text="Historical average",
    )
    fig.update_layout(height=390, showlegend=False)
    left.plotly_chart(fig, width="stretch")

    line_numbers = tuple(int(line[1:]) for line in lines)
    placeholders = ",".join("?" for _ in line_numbers)
    line_rates = query(
        f"""
        SELECT line, part_count, failure_count, failure_rate_pct, failure_rate_lift
        FROM line_failure_rates
        WHERE line IN ({placeholders})
        ORDER BY line
        """,
        line_numbers,
    )
    line_rates["line_label"] = "L" + line_rates["line"].astype(str)
    fig = px.bar(
        line_rates,
        x="line_label",
        y="failure_rate_pct",
        color="failure_rate_lift",
        color_continuous_scale="RdYlGn_r",
        labels={"line_label": "Line", "failure_rate_pct": "Failure rate (%)", "failure_rate_lift": "Lift"},
        title="Failure rate by production line",
    )
    fig.update_layout(height=390, coloraxis_colorbar_title="Lift")
    right.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Copilot prediction rows", f"{phase11['product_predictions']:,}")
    c2.metric("Failure trend periods", f"{phase12['failure_time_periods']}")
    c3.metric("SQLite tables", f"{len(source_catalog):,}")
    st.dataframe(
        summary[["metric", "value", "unit", "interpretation"]],
        width="stretch",
        hide_index=True,
    )


def page_prediction_model() -> None:
    st.header("Production Failure Prediction Model")
    model_metrics = query("SELECT * FROM model_metrics ORDER BY rank")
    improvement = load_csv("reports/phase6_model_improvement_summary.csv")
    advanced = query("SELECT * FROM advanced_ai_metrics ORDER BY mcc DESC")

    cols = st.columns(5)
    best = model_metrics.iloc[0]
    cols[0].metric("Selected model", best["model"])
    cols[1].metric("MCC", f"{best['mcc']:.3f}")
    cols[2].metric("Precision", f"{best['precision']:.3f}")
    cols[3].metric("Recall", f"{best['recall']:.3f}")
    cols[4].metric("PR-AUC", f"{best['pr_auc']:.3f}")

    left, right = st.columns(2)
    fig = px.bar(
        model_metrics,
        x="model",
        y="mcc",
        color="model",
        labels={"model": "Model", "mcc": "MCC"},
        title="Phase 6 model comparison",
    )
    fig.update_layout(height=420, showlegend=False)
    left.plotly_chart(fig, width="stretch")

    fig = px.scatter(
        model_metrics,
        x="recall",
        y="precision",
        size="mcc",
        color="model",
        hover_data=["f1", "pr_auc", "threshold"],
        title="Precision and recall trade-off",
    )
    fig.update_layout(height=420)
    right.plotly_chart(fig, width="stretch")

    st.subheader("Improvement Experiments")
    st.dataframe(
        improvement[["model", "production_safe", "mcc", "precision", "recall", "pr_auc", "notes"]],
        width="stretch",
        hide_index=True,
    )
    st.subheader("Advanced AI Diagnostics")
    st.dataframe(
        advanced[["model", "mcc", "precision", "recall", "pr_auc", "notes"]],
        width="stretch",
        hide_index=True,
    )
    st.warning(
        "The Phase 6 LightGBM remains the official production-safe model. "
        "Leaderboard leakage and validation-optimized blends are research-only."
    )


def page_product_families() -> None:
    st.header("Product Family Segmentation")
    families = load_csv("reports/phase5_product_family_failure_rates.csv")
    profiles = load_csv("reports/phase5_product_family_profiles.csv")

    cols = st.columns(4)
    top_family = families.sort_values("failure_rate_pct", ascending=False).iloc[0]
    cols[0].metric("Families", f"{families['final_product_family'].nunique()}")
    cols[1].metric("Highest-risk family", int(top_family["final_product_family"]))
    cols[2].metric("Family failure rate", pct(top_family["failure_rate_pct"], 3))
    cols[3].metric("Family size", f"{int(top_family['part_count']):,}")

    left, right = st.columns(2)
    fig = px.bar(
        families.sort_values("failure_rate_pct"),
        x="failure_rate_pct",
        y=families.sort_values("failure_rate_pct")["final_product_family"].astype(str),
        orientation="h",
        color="failure_lift_vs_overall",
        color_continuous_scale="RdYlGn_r",
        labels={"failure_rate_pct": "Failure rate (%)", "y": "Product family"},
        title="Failure rate by product family",
    )
    left.plotly_chart(fig, width="stretch")

    fig = px.scatter(
        families,
        x="avg_station_count",
        y="failure_rate_pct",
        size="part_count",
        color="avg_line_count",
        hover_name=families["final_product_family"].astype(str),
        labels={"avg_station_count": "Average station count", "failure_rate_pct": "Failure rate (%)"},
        title="Path complexity and family risk",
    )
    right.plotly_chart(fig, width="stretch")

    st.dataframe(families, width="stretch", hide_index=True)
    with st.expander("Family profile detail"):
        st.dataframe(profiles.head(100), width="stretch", hide_index=True)


def page_process_mining(lines: list[str]) -> None:
    st.header("Process Mining and Bottleneck Analysis")
    placeholders = ",".join("?" for _ in lines)
    bottlenecks = query(
        f"""
        SELECT *
        FROM bottlenecks
        WHERE substr(station,1,2) IN ({placeholders})
        ORDER BY bottleneck_rank
        """,
        tuple(lines),
    )
    throughput = query("SELECT * FROM throughput_efficiency ORDER BY split")
    routes = query("SELECT * FROM propagation_routes ORDER BY propagation_rank LIMIT 15")

    top = bottlenecks.iloc[0]
    cols = st.columns(5)
    cols[0].metric("Top bottleneck", top["station"])
    cols[1].metric("Bottleneck score", f"{top['bottleneck_score']:.1f}")
    cols[2].metric("Avg wait", f"{top['avg_waiting_time']:.2f}")
    cols[3].metric("P90 wait", f"{top['p90_waiting_time']:.2f}")
    cols[4].metric("Failure rate", pct(top["failure_rate_pct"], 3))

    fig = px.scatter(
        bottlenecks,
        x="avg_waiting_time",
        y="bottleneck_score",
        size="product_count",
        color="failure_rate_pct",
        hover_name="station",
        color_continuous_scale="YlOrRd",
        labels={"avg_waiting_time": "Average waiting time", "bottleneck_score": "Bottleneck score"},
        title="Volume, waiting time, and bottleneck score",
    )
    fig.update_layout(height=470)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    c1.dataframe(bottlenecks.head(15), width="stretch", hide_index=True)
    c2.dataframe(throughput, width="stretch", hide_index=True)
    st.subheader("Candidate Failure Propagation Routes")
    st.dataframe(routes, width="stretch", hide_index=True)
    st.caption("Routes are observational monitoring hypotheses, not causal proof.")


def page_root_cause(lines: list[str]) -> None:
    st.header("Model Explainability & Failure Drivers")
    placeholders = ",".join("?" for _ in lines)
    drivers = query("SELECT * FROM failure_drivers ORDER BY driver_rank LIMIT 25")
    root = query(
        f"""
        SELECT *
        FROM station_root_causes
        WHERE station LIKE 'L%_S%' AND substr(station,1,2) IN ({placeholders})
        ORDER BY total_mean_abs_shap DESC
        """,
        tuple(lines),
    )
    actions = query("SELECT * FROM engineer_actions")
    # Streamlit Cloud can retain the prior SQLite artifact briefly after a code
    # deployment. Support both the legacy and current action-plan schemas so the
    # page remains available while that artifact refreshes.
    action_priority = next(
        (
            column
            for column in ("predictive_signal_priority", "root_cause_priority")
            if column in actions.columns
        ),
        None,
    )
    if action_priority:
        actions = actions.sort_values(action_priority, ascending=False)

    if "feature_display_name" not in drivers.columns:
        drivers = drivers.assign(feature_display_name=drivers["feature"].map(feature_display_name))

    fig = px.bar(
        drivers.head(15).sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="feature_display_name",
        orientation="h",
        color="driver_type",
        labels={"mean_abs_shap": "Mean |SHAP|", "feature_display_name": "Predictive signal"},
        title="Top global SHAP predictive signals",
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    left.dataframe(root, width="stretch", hide_index=True)
    right.dataframe(actions, width="stretch", hide_index=True)
    st.warning(
        "SHAP explains model behavior. Engineering records are required before treating "
        "a predictive signal as a confirmed physical root cause. Timestamp-derived signals are relative, anonymized measurement indicators—not verified delays."
    )


def page_knowledge_graph(lines: list[str]) -> None:
    st.header("Knowledge Graph and Critical Nodes")
    placeholders = ",".join("?" for _ in lines)
    critical = query(
        f"""
        SELECT *
        FROM critical_nodes
        WHERE line IN ({placeholders})
        ORDER BY critical_rank
        """,
        tuple(lines),
    )
    routes = query("SELECT * FROM propagation_routes ORDER BY propagation_rank LIMIT 20")

    cols = st.columns(4)
    cols[0].metric("Critical stations shown", f"{len(critical):,}")
    cols[1].metric("Top critical node", critical.iloc[0]["station"])
    cols[2].metric("Top node score", f"{critical.iloc[0]['critical_node_score']:.1f}")
    cols[3].metric("Candidate routes", f"{len(routes):,}")

    left, right = st.columns(2)
    fig = px.bar(
        critical.head(15).sort_values("critical_node_score"),
        x="critical_node_score",
        y="station",
        orientation="h",
        color="line",
        labels={"critical_node_score": "Critical-node score"},
        title="Critical knowledge-graph nodes",
    )
    left.plotly_chart(fig, width="stretch")

    fig = px.scatter(
        critical,
        x="centrality_score",
        y="bottleneck_score",
        size="failure_rate_pct",
        color="line",
        hover_name="station",
        labels={"centrality_score": "Graph centrality score", "bottleneck_score": "Bottleneck score"},
        title="Graph centrality vs operational bottleneck",
    )
    right.plotly_chart(fig, width="stretch")
    st.dataframe(routes, width="stretch", hide_index=True)


def page_copilot() -> None:
    st.header("Gemini Handbook Copilot")
    st.caption(
        "Ask any project question in plain English. LangChain retrieves relevant Bosch handbook excerpts and Gemini "
        "writes the answer with numbered references."
    )
    examples = [
        "Explain the 8 product families in a non technical way.",
        "Why did we choose LightGBM as the production failure model?",
        "What are the main reasons products fail?",
        "What is the biggest bottleneck and why does it matter?",
        "Why are the high Kaggle leaderboard results not used in production?",
        "What did the advanced AI models add to the project?",
        "Explain this whole project to a plant manager.",
    ]
    selected = st.selectbox("Example questions", examples)
    question = st.text_area("Ask a question", value=selected, height=110)
    c1, c2 = st.columns([1, 4])
    ask = c1.button("Ask Agent", type="primary")
    st.caption(
        "[Bosch Handbook sources](https://github.com/kkr199/Bosch-Production-line-performance/tree/main/Bosch_Handbook_md)"
    )
    c2.caption(
        "Best for handbook guidance, non-technical explanations, model interpretation, "
        "process intelligence, and production-readiness questions."
    )

    if ask:
        api_key, model = get_gemini_configuration()
        if not api_key:
            st.error("Add GEMINI_API_KEY in Streamlit Secrets, then reboot the app.")
            return
        try:
            with st.spinner("Searching the handbook and drafting an answer..."):
                retriever = get_handbook_retriever()
                response = langchain_handbook.answer_handbook_question(
                    question,
                    api_key=api_key,
                    model=model,
                    retriever=retriever,
                )
        except langchain_handbook.HandbookCopilotError as error:
            st.error(str(error))
            return

        st.subheader("Answer")
        st.markdown(response.answer)
        st.caption(f"Model: {response.model} | LangChain in-memory handbook retrieval")
        if response.sources:
            st.subheader("References used")
            evidence_rows = [
                {
                    "reference": f"[{index}]",
                    "source": item.metadata.get("source", "Bosch handbook"),
                    "evidence": item.page_content[:900] + ("..." if len(item.page_content) > 900 else ""),
                }
                for index, item in enumerate(response.sources, start=1)
            ]
            st.dataframe(pd.DataFrame(evidence_rows), width="stretch", hide_index=True)

    with st.expander("Legacy reviewed template questions"):
        legacy_question = st.text_input("Template-backed question", value="Which stations have the highest failure rate?")
        if st.button("Run Template Answer"):
            answer = answer_question(get_connection(), legacy_question)
            st.markdown(f"### {answer.title}")
            st.write(answer.narrative)
            if not answer.data.empty:
                st.dataframe(answer.data, width="stretch", hide_index=True)
            st.caption(f"Reviewed intent: {answer.intent}. Uses deterministic SQL templates.")


def page_business_impact() -> None:
    st.header("Business Impact Scenario")
    baseline = query("SELECT * FROM executive_kpi_baseline").set_index("metric_key")
    c1, c2, c3, c4 = st.columns(4)
    volume = c1.number_input(
        "Production volume",
        min_value=10_000,
        max_value=20_000_000,
        value=int(baseline.loc["test_products_scored", "value"]),
        step=10_000,
    )
    failure_cost = c2.number_input("Cost per failure ($)", min_value=0.0, value=500.0, step=50.0)
    review_cost = c3.number_input("Cost per alert review ($)", min_value=0.0, value=20.0, step=5.0)
    effectiveness = c4.slider("Intervention effectiveness", 0, 100, 25, 5)

    impact = calculate_business_impact(
        production_volume=int(volume),
        failure_rate=float(baseline.loc["historical_failure_rate", "value"]),
        alert_rate=float(baseline.loc["test_alert_rate", "value"]),
        precision=float(baseline.loc["model_precision", "value"]),
        intervention_effectiveness=effectiveness / 100,
        cost_per_failure=float(failure_cost),
        cost_per_alert_review=float(review_cost),
    )

    cols = st.columns(5)
    cols[0].metric("Expected failures", f"{impact.expected_failures:,.0f}")
    cols[1].metric("Expected alerts", f"{impact.expected_alerts:,.0f}")
    cols[2].metric("Potentially prevented", f"{impact.potentially_prevented_failures:,.0f}")
    cols[3].metric("Gross avoided cost", money(impact.gross_avoided_failure_cost))
    cols[4].metric("Net estimated impact", money(impact.net_estimated_impact))

    fig = go.Figure(
        go.Waterfall(
            x=["Avoided failure cost", "Alert review cost", "Net impact"],
            y=[
                impact.gross_avoided_failure_cost,
                -impact.alert_review_cost,
                impact.net_estimated_impact,
            ],
            measure=["relative", "relative", "total"],
            increasing={"marker": {"color": "#2A7F62"}},
            decreasing={"marker": {"color": "#C74634"}},
            totals={"marker": {"color": "#355C7D"}},
        )
    )
    fig.update_layout(title="Illustrative impact bridge", yaxis_title="Estimated value ($)", height=430)
    st.plotly_chart(fig, width="stretch")
    st.info("This is an assumption-driven scenario. It is not realized savings from the Kaggle dataset.")


def page_deliverables() -> None:
    st.header("Final Deliverables and Project Documentation")
    manifest = load_optional_json("docs/final_deliverables/final_deliverables_manifest.json")
    deliverables = pd.DataFrame(manifest.get("deliverables", []))

    if "primary_artifacts" in deliverables:
        deliverables["primary_artifacts"] = deliverables["primary_artifacts"].apply(
            lambda values: "\n".join(values)
        )

    complete_count = (
        int((deliverables["status"] == "Complete").sum())
        if "status" in deliverables
        else 0
    )

    cols = st.columns(4)
    cols[0].metric("Deliverables", f"{len(deliverables)}")
    cols[1].metric("Complete", f"{complete_count}")
    cols[2].metric("Apps", "3")
    cols[3].metric("Reports package", "Ready" if not deliverables.empty else "Not included")

    if deliverables.empty:
        st.info("The optional final-deliverables package is not included in this deployment.")
    else:
        st.dataframe(deliverables, width="stretch", hide_index=True)

    files = [
        DOCS_DIR / "final_deliverables_index.md",
        DOCS_DIR / "Bosch_Production_Line_Performance_Research_Report.docx",
        DOCS_DIR / "Bosch_Production_Line_Performance_Presentation.pptx",
        DOCS_DIR / "Bosch_Production_Line_Performance_Final_Deliverables.zip",
    ]
    artifact_rows = [
        {"file": path.name, "path": str(path), "size_mb": path.stat().st_size / 1_000_000}
        for path in files
        if path.exists()
    ]
    if artifact_rows:
        st.dataframe(pd.DataFrame(artifact_rows), width="stretch", hide_index=True)


def main() -> None:
    st.title("Bosch Manufacturing Analytics Dashboard")

    if not DATABASE_PATH.exists():
        st.error("Missing SQLite database. Run `python src/data/phase11_manufacturing_copilot.py` first.")
        st.stop()

    lines = line_filter()
    st.sidebar.caption("Source: reviewed phase outputs and SQLite copilot database.")
    page = st.sidebar.radio(
        "Dashboard page",
        [
            "KPI Overview",
            "Prediction Model",
            "Product Families",
            "Process Mining",
            "Explainability",
            "Knowledge Graph",
            "Copilot",
            "Business Impact",
            "Deliverables",
        ],
    )

    if page == "KPI Overview":
        page_overview(lines)
    elif page == "Prediction Model":
        page_prediction_model()
    elif page == "Product Families":
        page_product_families()
    elif page == "Process Mining":
        page_process_mining(lines)
    elif page == "Explainability":
        page_root_cause(lines)
    elif page == "Knowledge Graph":
        page_knowledge_graph(lines)
    elif page == "Copilot":
        page_copilot()
    elif page == "Business Impact":
        page_business_impact()
    elif page == "Deliverables":
        page_deliverables()


if __name__ == "__main__":
    main()
