"""Auditable natural-language query helpers for the Manufacturing Copilot."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import pandas as pd


STATION_PATTERN = re.compile(r"\bL\d+_S\d+\b", re.IGNORECASE)


@dataclass
class CopilotAnswer:
    title: str
    narrative: str
    data: pd.DataFrame
    intent: str


def read_sql(connection: sqlite3.Connection, query: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(query, connection, params=params)


def _station_from_question(question: str) -> str | None:
    match = STATION_PATTERN.search(question)
    return match.group(0).upper() if match else None


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def answer_question(connection: sqlite3.Connection, question: str) -> CopilotAnswer:
    """Map common manufacturing questions to reviewed, parameterized SQL."""
    text = question.strip().lower()
    station = _station_from_question(question)

    if any(phrase in text for phrase in ("summarize production", "production summary", "overall summary")):
        data = read_sql(
            connection,
            """
            SELECT metric, value, unit, interpretation
            FROM production_summary ORDER BY display_order
            """,
        )
        narrative = (
            "The historical train failure rate is shown with official validation quality, "
            "current test scoring volume, predicted alerts, bottleneck priority, and the "
            "timestamp-derived throughput proxy."
        )
        return CopilotAnswer("Production performance summary", narrative, data, "summary")

    if station and any(word in text for word in ("why", "root", "cause", "driver", "risk")):
        root = read_sql(
            connection,
            """
            SELECT station, top_driver, top_driver_type, total_mean_abs_shap,
                   recommended_action
            FROM station_root_causes
            WHERE station = ?
            ORDER BY total_mean_abs_shap DESC
            """,
            (station,),
        )
        operating = read_sql(
            connection,
            """
            SELECT station, failure_rate_pct, failure_lift, avg_waiting_time,
                   p90_waiting_time, bottleneck_score, bottleneck_rank
            FROM bottlenecks WHERE station = ?
            """,
            (station,),
        )
        data = root.merge(operating, on="station", how="outer")
        if data.empty:
            return CopilotAnswer(
                f"No reviewed evidence for {station}",
                "The database does not contain station-level evidence for this identifier.",
                data,
                "station_root_cause",
            )
        row = data.iloc[0]
        driver = row.get("top_driver")
        action = row.get("recommended_action")
        narrative = (
            f"{station} has a historical failure rate of "
            f"{_fmt(row.get('failure_rate_pct', float('nan')), 3)}% and a bottleneck score of "
            f"{_fmt(row.get('bottleneck_score', float('nan')))}. "
            f"The strongest reviewed model driver is {driver if pd.notna(driver) else 'not station-specific'}. "
            f"Recommended check: {action if pd.notna(action) else 'review queue, sensor, and process records.'} "
            "These are diagnostic associations and require engineering confirmation before corrective action."
        )
        return CopilotAnswer(f"Root-cause view: {station}", narrative, data, "station_root_cause")

    if any(word in text for word in ("bottleneck", "waiting", "queue", "slow")):
        data = read_sql(
            connection,
            """
            SELECT bottleneck_rank, station, bottleneck_score, avg_waiting_time,
                   p90_waiting_time, failure_rate_pct, failure_lift
            FROM bottlenecks ORDER BY bottleneck_rank LIMIT 10
            """,
        )
        top = data.iloc[0]
        narrative = (
            f"{top['station']} is the highest-ranked bottleneck with score "
            f"{_fmt(top['bottleneck_score'])}, average waiting time "
            f"{_fmt(top['avg_waiting_time'])}, and p90 waiting time "
            f"{_fmt(top['p90_waiting_time'])}."
        )
        return CopilotAnswer("Top production bottlenecks", narrative, data, "bottlenecks")

    if any(word in text for word in ("critical", "central", "node", "graph")):
        data = read_sql(
            connection,
            """
            SELECT critical_rank, station, line, critical_node_score, centrality_score,
                   bottleneck_score, failure_rate_pct
            FROM critical_nodes ORDER BY critical_rank LIMIT 10
            """,
        )
        top = data.iloc[0]
        narrative = (
            f"{top['station']} is the highest-priority knowledge-graph node with critical score "
            f"{_fmt(top['critical_node_score'])}. This combines network position, bottleneck "
            "evidence, failure rate, and root-cause importance."
        )
        return CopilotAnswer("Critical process nodes", narrative, data, "critical_nodes")

    if any(word in text for word in ("route", "path", "propagation", "flow")):
        data = read_sql(
            connection,
            """
            SELECT propagation_rank, candidate_route, route_score,
                   mean_transition_failure_rate_pct, mean_station_critical_score
            FROM propagation_routes ORDER BY propagation_rank LIMIT 10
            """,
        )
        top = data.iloc[0]
        narrative = (
            f"The leading candidate route is {top['candidate_route']} with route score "
            f"{_fmt(top['route_score'])}. It is a monitoring priority, not proof that failures "
            "causally propagate along the route."
        )
        return CopilotAnswer("Candidate failure-propagation routes", narrative, data, "routes")

    if any(word in text for word in ("model", "mcc", "precision", "recall", "performance")):
        data = read_sql(
            connection,
            """
            SELECT model, threshold, mcc, precision, recall, f1, pr_auc, rank
            FROM model_metrics ORDER BY rank
            """,
        )
        best = data.iloc[0]
        narrative = (
            f"The official production-safe model is {best['model']} with validation MCC "
            f"{best['mcc']:.3f}, precision {best['precision']:.3f}, recall "
            f"{best['recall']:.3f}, and PR-AUC {best['pr_auc']:.3f}."
        )
        return CopilotAnswer("Model validation summary", narrative, data, "model_performance")

    if any(word in text for word in ("driver", "shap", "feature", "root cause")):
        data = read_sql(
            connection,
            """
            SELECT driver_rank, feature, station, driver_type, mean_abs_shap,
                   mean_signed_shap
            FROM failure_drivers ORDER BY driver_rank LIMIT 15
            """,
        )
        top = data.iloc[0]
        narrative = (
            f"The strongest global failure driver is {top['feature']} with mean absolute "
            f"SHAP {top['mean_abs_shap']:.4f}. SHAP explains model behavior; it does not by "
            "itself establish physical causation."
        )
        return CopilotAnswer("Top failure drivers", narrative, data, "failure_drivers")

    if any(word in text for word in ("failure rate", "high risk station", "risky station")):
        data = read_sql(
            connection,
            """
            SELECT station_key AS station, line, part_count, failure_count, failure_rate_pct,
                   failure_rate_lift
            FROM station_failure_rates ORDER BY failure_rate_pct DESC LIMIT 10
            """,
        )
        top = data.iloc[0]
        narrative = (
            f"{top['station']} has the highest observed station failure rate at "
            f"{top['failure_rate_pct']:.3f}% ({top['failure_rate_lift']:.2f}x the overall rate)."
        )
        return CopilotAnswer("Highest-risk stations", narrative, data, "station_failure_rate")

    data = read_sql(
        connection,
        """
        SELECT metric, value, unit, interpretation
        FROM production_summary ORDER BY display_order
        """,
    )
    narrative = (
        "Here is the reviewed production summary. Ask about failure rates, bottlenecks, "
        "predictive signals, critical stations, process routes, or model validation for a focused answer."
    )
    return CopilotAnswer("Production performance summary", narrative, data, "summary")
