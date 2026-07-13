"""Executive manufacturing dashboard for Bosch production-line analytics."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.business_impact import calculate_business_impact  # noqa: E402


DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "manufacturing_copilot.db"

st.set_page_config(
    page_title="Manufacturing Executive Dashboard",
    page_icon=":material/analytics:",
    layout="wide",
)


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DATABASE_PATH, check_same_thread=False)


@st.cache_data
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, get_connection(), params=params)


def money(value: float) -> str:
    return f"${value:,.0f}"


def percent(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def selected_lines() -> list[str]:
    lines = ["L0", "L1", "L2", "L3"]
    chosen = st.sidebar.multiselect("Production lines", lines, default=lines)
    return chosen or lines


def executive_overview(lines: list[str]) -> None:
    placeholders = ",".join("?" for _ in lines)
    line_numbers = tuple(int(line[1:]) for line in lines)
    line_data = query(
        f"""
        SELECT line, part_count, failure_count, failure_rate_pct, failure_rate_lift
        FROM line_failure_rates WHERE line IN ({placeholders})
        ORDER BY line
        """,
        line_numbers,
    )
    station_data = query(
        f"""
        SELECT station_key, line, part_count, failure_count, failure_rate_pct,
               failure_rate_lift
        FROM station_failure_rates WHERE line IN ({placeholders})
        ORDER BY failure_rate_pct DESC
        """,
        line_numbers,
    )
    metrics = query("SELECT * FROM executive_kpi_baseline").set_index("metric_key")
    best_model = query("SELECT * FROM model_metrics ORDER BY rank LIMIT 1").iloc[0]
    throughput = query(
        "SELECT * FROM throughput_efficiency WHERE split='train'"
    ).iloc[0]

    weighted_failure_rate = (
        line_data["failure_count"].sum() / line_data["part_count"].sum() * 100
    )
    top_station = station_data.iloc[0]
    top_bottleneck = query(
        f"""
        SELECT * FROM bottlenecks
        WHERE substr(station,1,2) IN ({placeholders})
        ORDER BY bottleneck_rank LIMIT 1
        """,
        tuple(lines),
    ).iloc[0]

    cols = st.columns(6)
    cols[0].metric("Failure rate", percent(weighted_failure_rate, 3))
    cols[1].metric("Test products scored", f"{int(metrics.loc['test_products_scored','value']):,}")
    cols[2].metric("Predicted alerts", f"{int(metrics.loc['test_alerts','value']):,}")
    cols[3].metric("Validation MCC", f"{best_model['mcc']:.3f}")
    cols[4].metric("Top-risk station", top_station["station_key"], percent(top_station["failure_rate_pct"], 2))
    cols[5].metric("Top bottleneck", top_bottleneck["station"], f"Score {top_bottleneck['bottleneck_score']:.1f}")

    left, right = st.columns([1.15, 1])
    trend = query("SELECT * FROM failure_time_trends ORDER BY period_order")
    trend["period"] = trend["period_order"].map(lambda value: f"P{int(value):02d}")
    fig = px.line(
        trend,
        x="period",
        y="failure_rate_pct",
        markers=True,
        labels={"period": "Relative production period", "failure_rate_pct": "Failure rate (%)"},
        title="Failure rate across relative production time",
    )
    fig.add_hline(
        y=float(metrics.loc["historical_failure_rate", "value"]) * 100,
        line_dash="dot",
        annotation_text="Historical average",
    )
    fig.update_traces(line_color="#C74634", marker_color="#C74634")
    fig.update_layout(height=390, showlegend=False)
    left.plotly_chart(fig, width="stretch")

    line_chart = px.bar(
        line_data.assign(line_label="L" + line_data["line"].astype(str)),
        x="line_label",
        y="failure_rate_pct",
        color="failure_rate_lift",
        color_continuous_scale="RdYlGn_r",
        labels={"line_label": "Line", "failure_rate_pct": "Failure rate (%)", "failure_rate_lift": "Lift"},
        title="Failure rate by production line",
    )
    line_chart.update_layout(height=390, coloraxis_colorbar_title="Lift")
    right.plotly_chart(line_chart, width="stretch")

    st.caption(
        f"Average waiting time is {throughput['avg_waiting_time']:.2f} relative time units. "
        "Failure trends use ordered Bosch production timestamps rather than calendar dates."
    )


def station_heatmap(lines: list[str]) -> None:
    st.subheader("Station Risk Heatmap")
    placeholders = ",".join("?" for _ in lines)
    line_numbers = tuple(int(line[1:]) for line in lines)
    stations = query(
        f"""
        SELECT line, station, station_key, failure_rate_pct, failure_rate_lift,
               part_count
        FROM station_failure_rates WHERE line IN ({placeholders})
        """,
        line_numbers,
    )
    metric = st.segmented_control(
        "Heatmap metric",
        options=["Failure rate (%)", "Failure lift"],
        default="Failure rate (%)",
        key="heatmap_metric",
    )
    value_column = "failure_rate_pct" if metric == "Failure rate (%)" else "failure_rate_lift"
    matrix = stations.pivot(index="line", columns="station", values=value_column)
    matrix.index = [f"L{value}" for value in matrix.index]
    matrix.columns = [f"S{int(value)}" for value in matrix.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale="YlOrRd",
            colorbar={"title": metric},
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=f"{metric} by line and station",
        xaxis_title="Station",
        yaxis_title="Production line",
        height=420,
    )
    st.plotly_chart(fig, width="stretch")

    top = stations.sort_values(value_column, ascending=False).head(12)
    st.dataframe(
        top[["station_key", "part_count", "failure_rate_pct", "failure_rate_lift"]],
        width="stretch",
        hide_index=True,
    )


def bottleneck_analytics(lines: list[str]) -> None:
    st.subheader("Bottleneck Analytics")
    placeholders = ",".join("?" for _ in lines)
    bottlenecks = query(
        f"""
        SELECT station, product_count, avg_waiting_time, p90_waiting_time,
               failure_rate_pct, failure_lift, bottleneck_score, bottleneck_rank
        FROM bottlenecks
        WHERE substr(station,1,2) IN ({placeholders})
        ORDER BY bottleneck_rank
        """,
        tuple(lines),
    )
    fig = px.scatter(
        bottlenecks,
        x="avg_waiting_time",
        y="bottleneck_score",
        size="product_count",
        color="failure_rate_pct",
        hover_name="station",
        color_continuous_scale="YlOrRd",
        labels={
            "avg_waiting_time": "Average waiting time",
            "bottleneck_score": "Bottleneck score",
            "product_count": "Products",
            "failure_rate_pct": "Failure rate (%)",
        },
        title="Volume, waiting time, and bottleneck priority",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(bottlenecks.head(15), width="stretch", hide_index=True)


def shap_explanations() -> None:
    st.subheader("Model Failure Drivers")
    driver_count = st.slider("Drivers to display", 5, 20, 12)
    drivers = query(
        "SELECT * FROM failure_drivers ORDER BY driver_rank LIMIT ?",
        (driver_count,),
    )
    fig = px.bar(
        drivers.sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color="driver_type",
        labels={"mean_abs_shap": "Mean absolute SHAP", "feature": "Feature"},
        title="Global drivers of production-safe model predictions",
    )
    fig.update_layout(height=max(400, driver_count * 31), legend_title_text="Driver type")
    st.plotly_chart(fig, width="stretch")

    root_causes = query(
        """
        SELECT station, total_mean_abs_shap, top_driver, top_driver_type,
               recommended_action
        FROM station_root_causes
        WHERE station LIKE 'L%_S%'
        ORDER BY total_mean_abs_shap DESC
        """
    )
    st.dataframe(root_causes, width="stretch", hide_index=True)
    st.caption(
        "SHAP explains the model's prediction behavior. Engineering evidence is required "
        "before treating a driver as a confirmed physical root cause."
    )


def business_impact() -> None:
    st.subheader("Business Impact Scenario")
    baseline = query("SELECT * FROM executive_kpi_baseline").set_index("metric_key")
    st.caption("Adjust assumptions to estimate a scenario. These values are not realized savings.")

    c1, c2, c3, c4 = st.columns(4)
    volume = c1.number_input(
        "Production volume",
        min_value=10_000,
        max_value=20_000_000,
        value=int(baseline.loc["test_products_scored", "value"]),
        step=10_000,
    )
    failure_cost = c2.number_input(
        "Cost per failure ($)", min_value=0.0, value=500.0, step=50.0
    )
    review_cost = c3.number_input(
        "Cost per alert review ($)", min_value=0.0, value=20.0, step=5.0
    )
    effectiveness_pct = c4.slider(
        "Intervention effectiveness", 0, 100, 25, 5
    )

    impact = calculate_business_impact(
        production_volume=int(volume),
        failure_rate=float(baseline.loc["historical_failure_rate", "value"]),
        alert_rate=float(baseline.loc["test_alert_rate", "value"]),
        precision=float(baseline.loc["model_precision", "value"]),
        intervention_effectiveness=effectiveness_pct / 100,
        cost_per_failure=float(failure_cost),
        cost_per_alert_review=float(review_cost),
    )

    cols = st.columns(5)
    cols[0].metric("Expected failures", f"{impact.expected_failures:,.0f}")
    cols[1].metric("Expected alerts", f"{impact.expected_alerts:,.0f}")
    cols[2].metric("Potentially prevented", f"{impact.potentially_prevented_failures:,.0f}")
    cols[3].metric("Gross avoided cost", money(impact.gross_avoided_failure_cost))
    cols[4].metric("Net estimated impact", money(impact.net_estimated_impact))

    waterfall = go.Figure(
        go.Waterfall(
            x=["Avoided failure cost", "Alert review cost", "Net impact"],
            y=[
                impact.gross_avoided_failure_cost,
                -impact.alert_review_cost,
                impact.net_estimated_impact,
            ],
            measure=["relative", "relative", "total"],
            connector={"line": {"color": "#777"}},
            increasing={"marker": {"color": "#2A7F62"}},
            decreasing={"marker": {"color": "#C74634"}},
            totals={"marker": {"color": "#355C7D"}},
        )
    )
    waterfall.update_layout(
        title="Illustrative annualized impact bridge",
        yaxis_title="Estimated value ($)",
        height=430,
    )
    st.plotly_chart(waterfall, width="stretch")

    roi_text = (
        f"{impact.estimated_roi:.2f}x"
        if impact.estimated_roi is not None
        else "Not defined"
    )
    st.info(
        f"Scenario ROI: {roi_text}. Calculation uses the historical failure rate, "
        "test alert rate, validation precision, and the assumptions selected above."
    )


def main() -> None:
    st.title("Manufacturing Executive Dashboard")
    st.caption("Quality risk, process constraints, model drivers, and business impact")
    if not DATABASE_PATH.exists():
        st.error("Phase 11 database not found. Build the Manufacturing Copilot database first.")
        st.stop()

    lines = selected_lines()
    st.sidebar.caption(
        "Historical Bosch data. Test predictions are unlabeled; financial impact is scenario-based."
    )

    tabs = st.tabs(
        [
            "Executive Overview",
            "Station Heatmap",
            "Bottlenecks",
            "SHAP Drivers",
            "Business Impact",
        ]
    )
    with tabs[0]:
        executive_overview(lines)
    with tabs[1]:
        station_heatmap(lines)
    with tabs[2]:
        bottleneck_analytics(lines)
    with tabs[3]:
        shap_explanations()
    with tabs[4]:
        business_impact()


if __name__ == "__main__":
    main()
