"""Prepare Phase 12 executive dashboard tables and documentation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "manufacturing_copilot.db"


def require_sources() -> None:
    required = [
        DATABASE_PATH,
        REPORTS_DIR / "phase3_first_time_failure_patterns.csv",
        REPORTS_DIR / "phase3_duration_failure_patterns.csv",
        REPORTS_DIR / "phase3_station_failure_rates.csv",
        REPORTS_DIR / "phase7_top_failure_drivers.csv",
        REPORTS_DIR / "phase8_bottleneck_scores.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase 12 sources:\n" + "\n".join(missing))


def add_order_columns(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "period_order", range(1, len(output) + 1))
    output["period_label"] = output[label_column].astype(str)
    return output


def prepare_dashboard_tables() -> dict[str, int | float | str]:
    require_sources()
    time_trend = add_order_columns(
        pd.read_csv(REPORTS_DIR / "phase3_first_time_failure_patterns.csv"),
        "first_time_bin",
    )
    duration_trend = add_order_columns(
        pd.read_csv(REPORTS_DIR / "phase3_duration_failure_patterns.csv"),
        "duration_bin",
    )

    with sqlite3.connect(DATABASE_PATH) as connection:
        time_trend.to_sql(
            "failure_time_trends", connection, if_exists="replace", index=False
        )
        duration_trend.to_sql(
            "failure_duration_trends", connection, if_exists="replace", index=False
        )

        model = pd.read_sql_query(
            "SELECT * FROM model_metrics ORDER BY rank LIMIT 1", connection
        ).iloc[0]
        throughput = pd.read_sql_query(
            "SELECT * FROM throughput_efficiency WHERE split='train'", connection
        ).iloc[0]
        top_station = pd.read_sql_query(
            "SELECT * FROM station_failure_rates ORDER BY failure_rate_pct DESC LIMIT 1",
            connection,
        ).iloc[0]
        top_bottleneck = pd.read_sql_query(
            "SELECT * FROM bottlenecks ORDER BY bottleneck_rank LIMIT 1", connection
        ).iloc[0]
        test_scoring = pd.read_sql_query(
            """
            SELECT COUNT(*) AS products, SUM(predicted_failure) AS alerts,
                   AVG(failure_probability) AS average_risk
            FROM product_predictions WHERE split='test'
            """,
            connection,
        ).iloc[0]

        baseline = pd.DataFrame(
            [
                ("historical_failure_rate", throughput["failure_rate_pct"] / 100, "decimal", "Train target rate"),
                ("model_precision", model["precision"], "decimal", "Official validation precision"),
                ("model_recall", model["recall"], "decimal", "Official validation recall"),
                ("model_mcc", model["mcc"], "score", "Official validation MCC"),
                ("test_alert_rate", test_scoring["alerts"] / test_scoring["products"], "decimal", "Selected threshold on unlabeled test data"),
                ("test_products_scored", test_scoring["products"], "products", "Full test population"),
                ("test_alerts", test_scoring["alerts"], "products", "Predicted failures"),
                ("top_station_failure_rate", top_station["failure_rate_pct"] / 100, "decimal", top_station["station_key"]),
                ("top_bottleneck_score", top_bottleneck["bottleneck_score"], "score", top_bottleneck["station"]),
            ],
            columns=["metric_key", "value", "unit", "context"],
        )
        baseline.to_sql(
            "executive_kpi_baseline", connection, if_exists="replace", index=False
        )

        connection.execute(
            "DELETE FROM source_catalog WHERE table_name IN "
            "('failure_time_trends','failure_duration_trends','executive_kpi_baseline')"
        )
        catalog = pd.DataFrame(
            [
                {
                    "table_name": "failure_time_trends",
                    "source_path": "reports/phase3_first_time_failure_patterns.csv",
                    "row_count": len(time_trend),
                    "loaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                },
                {
                    "table_name": "failure_duration_trends",
                    "source_path": "reports/phase3_duration_failure_patterns.csv",
                    "row_count": len(duration_trend),
                    "loaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                },
                {
                    "table_name": "executive_kpi_baseline",
                    "source_path": "Derived from Phase 3/6/8/11 reviewed outputs",
                    "row_count": len(baseline),
                    "loaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                },
            ]
        )
        catalog.to_sql(
            "source_catalog", connection, if_exists="append", index=False
        )
        connection.execute("PRAGMA optimize")

    summary = {
        "failure_time_periods": len(time_trend),
        "duration_periods": len(duration_trend),
        "executive_kpis": len(baseline),
        "historical_failure_rate_pct": float(throughput["failure_rate_pct"]),
        "top_risk_station": str(top_station["station_key"]),
        "top_bottleneck_station": str(top_bottleneck["station"]),
        "test_products_scored": int(test_scoring["products"]),
        "test_alerts": int(test_scoring["alerts"]),
    }
    return summary


def write_report(summary: dict[str, int | float | str]) -> None:
    report = f"""# Phase 12: Executive Dashboard

## Executive Purpose

The dashboard provides a summary-first view of manufacturing quality, failure
movement, station risk, bottlenecks, model explanations, and scenario-based
business impact.

## Headline Baseline

- Historical failure rate: {summary['historical_failure_rate_pct']:.3f}%
- Highest-risk station: `{summary['top_risk_station']}`
- Highest bottleneck station: `{summary['top_bottleneck_station']}`
- Test products scored: {summary['test_products_scored']:,}
- Model alerts on test population: {summary['test_alerts']:,}

## Dashboard Views

1. Executive overview and KPI scorecard.
2. Failure trend across ordered relative production-time periods.
3. Station heatmap by line and station.
4. Bottleneck analytics using timestamp-gap proxies, failure rate, and bottleneck score.
5. SHAP driver explanations from the production-safe Phase 6 model.
6. Business-impact scenario using user-controlled costs and intervention effectiveness.

## Interpretation Boundaries

- The failure trend uses relative Bosch production timestamps, not calendar dates.
- Test predictions do not have known outcomes.
- Business impact is a scenario, not realized savings.
- SHAP values explain model behavior and do not prove physical causation.
- Timestamp-derived features are relative measurement indicators, not verified
  production delays, queue times, or official start/end times.
- Throughput efficiency is a timestamp-derived proxy and is not OEE.
"""
    (REPORTS_DIR / "phase12_executive_dashboard_report.md").write_text(
        report, encoding="utf-8"
    )
    (REPORTS_DIR / "phase12_executive_dashboard_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    summary = prepare_dashboard_tables()
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
