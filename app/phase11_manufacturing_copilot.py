"""Streamlit Manufacturing Copilot for Bosch production-line analytics."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.copilot.query_engine import answer_question, read_sql  # noqa: E402


DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "manufacturing_copilot.db"

st.set_page_config(
    page_title="Manufacturing Copilot",
    page_icon=":material/factory:",
    layout="wide",
)


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DATABASE_PATH, check_same_thread=False)


@st.cache_data
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    return read_sql(get_connection(), sql, params)


def percent(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def overview() -> None:
    summary = query("SELECT * FROM production_summary ORDER BY display_order")
    values = summary.set_index("metric")["value"]
    labels = summary.set_index("metric")["interpretation"]

    cols = st.columns(5)
    cols[0].metric("Historical failure rate", percent(values["Historical train failure rate"], 3))
    cols[1].metric("Validation MCC", f"{values['Validation MCC']:.3f}")
    cols[2].metric("Validation precision", percent(values["Validation precision"] * 100, 1))
    cols[3].metric("Validation recall", percent(values["Validation recall"] * 100, 1))
    cols[4].metric("Predicted test alerts", f"{int(values['Test predicted alerts']):,}")

    left, right = st.columns(2)
    station_rates = query(
        """
        SELECT station_key AS station, failure_rate_pct, failure_rate_lift, part_count
        FROM station_failure_rates ORDER BY failure_rate_pct DESC LIMIT 12
        """
    )
    fig = px.bar(
        station_rates.sort_values("failure_rate_pct"),
        x="failure_rate_pct",
        y="station",
        orientation="h",
        color="failure_rate_lift",
        color_continuous_scale="RdYlGn_r",
        labels={"failure_rate_pct": "Failure rate (%)", "station": "Station"},
        title="Highest observed station failure rates",
    )
    fig.update_layout(height=430, coloraxis_colorbar_title="Lift")
    left.plotly_chart(fig, width="stretch")

    bottlenecks = query(
        """
        SELECT station, bottleneck_score, avg_waiting_time, failure_rate_pct
        FROM bottlenecks ORDER BY bottleneck_rank LIMIT 12
        """
    )
    fig = px.scatter(
        bottlenecks,
        x="avg_waiting_time",
        y="bottleneck_score",
        size="bottleneck_score",
        color="failure_rate_pct",
        hover_name="station",
        color_continuous_scale="OrRd",
        labels={
            "avg_waiting_time": "Average waiting time",
            "bottleneck_score": "Bottleneck score",
            "failure_rate_pct": "Failure rate (%)",
        },
        title="Waiting-time and bottleneck priorities",
    )
    fig.update_layout(height=430)
    right.plotly_chart(fig, width="stretch")

    st.caption(
        f"Official risk model: {labels['Validation MCC']}. "
        "Failure and throughput metrics are historical or validation-derived, not live factory telemetry."
    )


def risk_monitor() -> None:
    st.subheader("Product Risk Monitor")
    split = st.segmented_control(
        "Population", options=["test", "validation"], default="test"
    )
    minimum_risk = st.slider("Minimum official failure probability", 0.0, 1.0, 0.70, 0.01)
    limit = st.select_slider("Rows", options=[25, 50, 100, 250], value=50)
    data = query(
        """
        SELECT Id, split, actual_response, failure_probability, predicted_failure,
               isolation_forest_anomaly_score, mlp_reconstruction_anomaly_score,
               graph_message_passing_failure_risk, trajectory_failure_risk
        FROM product_predictions
        WHERE split = ? AND failure_probability >= ?
        ORDER BY failure_probability DESC LIMIT ?
        """,
        (split, minimum_risk, limit),
    )
    if data.empty:
        st.info("No products meet the selected risk threshold.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Products shown", f"{len(data):,}")
    c2.metric("Mean official risk", percent(data["failure_probability"].mean() * 100, 1))
    c3.metric("Model alerts", f"{int(data['predicted_failure'].sum()):,}")
    st.dataframe(
        data,
        width="stretch",
        hide_index=True,
        column_config={
            "failure_probability": st.column_config.ProgressColumn(
                "Official risk", min_value=0.0, max_value=1.0, format="%.3f"
            )
        },
    )
    st.caption(
        "Advanced-AI columns are supporting diagnostics. Blank values mean Phase 10 scores "
        "were produced only for the bounded test preview."
    )


def root_causes() -> None:
    st.subheader("Model Explainability and Engineering Actions")
    stations = query(
        """
        SELECT DISTINCT station FROM station_root_causes
        WHERE station LIKE 'L%_S%' ORDER BY station
        """
    )["station"].tolist()
    station = st.selectbox("Station", stations, index=stations.index("L3_S32") if "L3_S32" in stations else 0)

    root = query(
        """
        SELECT station, feature_family, total_mean_abs_shap, top_driver,
               top_driver_type, recommended_action
        FROM station_root_causes WHERE station = ?
        ORDER BY total_mean_abs_shap DESC
        """,
        (station,),
    )
    operating = query(
        """
        SELECT failure_rate_pct, failure_lift, avg_waiting_time, p90_waiting_time,
               bottleneck_score, bottleneck_rank
        FROM bottlenecks WHERE station = ?
        """,
        (station,),
    )
    if not operating.empty:
        row = operating.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Failure rate", percent(row["failure_rate_pct"], 3))
        cols[1].metric("Failure lift", f"{row['failure_lift']:.2f}x")
        cols[2].metric("Average wait", f"{row['avg_waiting_time']:.2f}")
        cols[3].metric("Bottleneck rank", f"#{int(row['bottleneck_rank'])}")

    if root.empty:
        st.info("No station-specific SHAP root-cause row is available.")
    else:
        st.dataframe(root, width="stretch", hide_index=True)
        st.warning(
            "Model explanations identify associations worth investigating. Confirm sensor, "
            "maintenance, quality, and process records before declaring a physical root cause. Timestamp-derived features are temporal indicators, not verified delays."
        )

    drivers = query(
        """
        SELECT driver_rank, feature, driver_type, mean_abs_shap
        FROM failure_drivers WHERE station = ? ORDER BY driver_rank LIMIT 15
        """,
        (station,),
    )
    if not drivers.empty:
        fig = px.bar(
            drivers.sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            labels={"mean_abs_shap": "Mean |SHAP|", "feature": "Driver"},
            title=f"Model drivers for {station}",
        )
        st.plotly_chart(fig, width="stretch")


def process_intelligence() -> None:
    st.subheader("Process and Knowledge-Graph Intelligence")
    left, right = st.columns(2)
    critical = query(
        """
        SELECT critical_rank, station, line, critical_node_score, centrality_score,
               bottleneck_score, failure_rate_pct
        FROM critical_nodes ORDER BY critical_rank LIMIT 15
        """
    )
    fig = px.bar(
        critical.sort_values("critical_node_score"),
        x="critical_node_score",
        y="station",
        orientation="h",
        color="line",
        labels={"critical_node_score": "Critical-node score", "station": "Station"},
        title="Critical process nodes",
    )
    left.plotly_chart(fig, width="stretch")

    throughput = query("SELECT * FROM throughput_efficiency ORDER BY split")
    right.dataframe(throughput, width="stretch", hide_index=True)
    right.caption(
        "Throughput efficiency is derived from available timestamps. It is a comparative "
        "process proxy, not a formal OEE measure."
    )

    routes = query(
        """
        SELECT propagation_rank, candidate_route, route_score,
               minimum_transition_count, mean_transition_failure_rate_pct,
               mean_station_critical_score
        FROM propagation_routes ORDER BY propagation_rank LIMIT 20
        """
    )
    st.dataframe(routes, width="stretch", hide_index=True)
    st.caption("Routes are monitoring hypotheses from observed transitions, not causal proof.")


def copilot() -> None:
    st.subheader("Ask the Manufacturing Copilot")
    examples = [
        "Summarize production performance",
        "Which stations have the highest failure rate?",
        "What are the top bottlenecks?",
        "Why is L3_S32 risky?",
        "Show the most critical process nodes",
        "What are the likely failure propagation routes?",
        "How good is the production-safe model?",
    ]
    selected = st.selectbox("Example question", examples)
    question = st.text_input("Question", value=selected)
    if st.button("Ask", type="primary"):
        answer = answer_question(get_connection(), question)
        st.markdown(f"### {answer.title}")
        st.write(answer.narrative)
        if not answer.data.empty:
            st.dataframe(answer.data, width="stretch", hide_index=True)
        st.caption(f"Reviewed intent: {answer.intent}. Answers are generated from local SQLite data.")


def main() -> None:
    st.title("Manufacturing Copilot")
    st.caption("Failure risk, root-cause evidence, bottlenecks, and process intelligence")
    if not DATABASE_PATH.exists():
        st.error(
            "The Phase 11 database is missing. Run "
            "`python src/data/phase11_manufacturing_copilot.py` first."
        )
        st.stop()

    tabs = st.tabs(
        ["Overview", "Risk Monitor", "Explainability", "Process Intelligence", "Copilot"]
    )
    with tabs[0]:
        overview()
    with tabs[1]:
        risk_monitor()
    with tabs[2]:
        root_causes()
    with tabs[3]:
        process_intelligence()
    with tabs[4]:
        copilot()


if __name__ == "__main__":
    main()
