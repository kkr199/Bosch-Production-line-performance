"""Build the Phase 11 SQLite database and supporting documentation."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATABASE_DIR = PROJECT_ROOT / "data" / "database"
DATABASE_PATH = DATABASE_DIR / "manufacturing_copilot.db"


TABLE_SOURCES = {
    "model_metrics": REPORTS_DIR / "phase6_model_comparison_metrics.csv",
    "failure_drivers": REPORTS_DIR / "phase7_top_failure_drivers.csv",
    "station_root_causes": REPORTS_DIR / "phase7_station_root_cause_report.csv",
    "engineer_actions": REPORTS_DIR / "phase7_engineer_action_plan.csv",
    "station_failure_rates": REPORTS_DIR / "phase3_station_failure_rates.csv",
    "line_failure_rates": REPORTS_DIR / "phase3_line_failure_rates.csv",
    "bottlenecks": REPORTS_DIR / "phase8_bottleneck_scores.csv",
    "throughput_efficiency": REPORTS_DIR / "phase8_throughput_efficiency.csv",
    "critical_nodes": REPORTS_DIR / "phase9_critical_nodes.csv",
    "propagation_routes": REPORTS_DIR / "phase9_failure_propagation_routes.csv",
    "advanced_ai_metrics": REPORTS_DIR / "phase10_advanced_ai_model_comparison.csv",
}


def require_sources() -> None:
    paths = list(TABLE_SOURCES.values()) + [
        PROCESSED_DIR / "phase6_test_predictions.csv",
        PROCESSED_DIR / "phase6_validation_dataset.csv",
        PROCESSED_DIR / "phase10_validation_advanced_ai_scores.csv",
        PROCESSED_DIR / "phase10_test_preview_advanced_ai_scores.csv",
        PROJECT_ROOT / "models" / "phase6_best_model.joblib",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase 11 source files:\n" + "\n".join(missing))


def write_frame(
    connection: sqlite3.Connection, table: str, frame: pd.DataFrame, chunksize: int = 25_000
) -> None:
    frame.to_sql(table, connection, if_exists="replace", index=False, chunksize=chunksize)


def build_prediction_table(connection: sqlite3.Connection) -> int:
    bundle = joblib.load(PROJECT_ROOT / "models" / "phase6_best_model.joblib")
    validation = pd.read_csv(PROCESSED_DIR / "phase6_validation_dataset.csv")
    feature_cols = bundle["feature_cols"]
    validation_probability = bundle["model"].predict_proba(validation[feature_cols])[:, 1]
    validation_predictions = pd.DataFrame(
        {
            "Id": validation["Id"].astype("int64"),
            "split": "validation",
            "actual_response": validation["Response"].astype("int8"),
            "failure_probability": validation_probability,
            "predicted_failure": (
                validation_probability >= float(bundle["best_threshold"])
            ).astype("int8"),
        }
    )
    validation_ai = pd.read_csv(
        PROCESSED_DIR / "phase10_validation_advanced_ai_scores.csv"
    ).drop(columns=["Response"], errors="ignore")
    validation_predictions = validation_predictions.merge(validation_ai, on="Id", how="left")

    test = pd.read_csv(PROCESSED_DIR / "phase6_test_predictions.csv")
    test_predictions = test.rename(
        columns={"predicted_failure_at_selected_threshold": "predicted_failure"}
    )
    test_predictions.insert(1, "split", "test")
    test_predictions.insert(2, "actual_response", np.nan)
    test_ai = pd.read_csv(PROCESSED_DIR / "phase10_test_preview_advanced_ai_scores.csv")
    test_predictions = test_predictions.merge(test_ai, on="Id", how="left")

    predictions = pd.concat([validation_predictions, test_predictions], ignore_index=True)
    write_frame(connection, "product_predictions", predictions)
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_product_predictions_id ON product_predictions(Id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_product_predictions_split_risk "
        "ON product_predictions(split, failure_probability DESC)"
    )
    return len(predictions)


def build_summary_table(connection: sqlite3.Connection) -> pd.DataFrame:
    model = pd.read_sql_query(
        "SELECT * FROM model_metrics ORDER BY rank LIMIT 1", connection
    ).iloc[0]
    train = pd.read_sql_query(
        "SELECT * FROM throughput_efficiency WHERE split='train'", connection
    ).iloc[0]
    bottleneck = pd.read_sql_query(
        "SELECT * FROM bottlenecks ORDER BY bottleneck_rank LIMIT 1", connection
    ).iloc[0]
    critical = pd.read_sql_query(
        "SELECT * FROM critical_nodes ORDER BY critical_rank LIMIT 1", connection
    ).iloc[0]
    predictions = pd.read_sql_query(
        """
        SELECT COUNT(*) AS products,
               SUM(predicted_failure) AS predicted_failures,
               AVG(failure_probability) AS avg_risk
        FROM product_predictions WHERE split='test'
        """,
        connection,
    ).iloc[0]

    summary = pd.DataFrame(
        [
            (1, "Historical train failure rate", train["failure_rate_pct"], "%", "Observed target rate"),
            (2, "Validation MCC", model["mcc"], "score", f"{model['model']} production-safe model"),
            (3, "Validation precision", model["precision"], "score", "Share of alerts that were failures"),
            (4, "Validation recall", model["recall"], "score", "Share of failures detected"),
            (5, "Test products scored", predictions["products"], "products", "Unlabeled production-like population"),
            (6, "Test predicted alerts", predictions["predicted_failures"], "products", "At selected validation threshold"),
            (7, "Average test risk", predictions["avg_risk"] * 100, "%", "Mean predicted failure probability"),
            (8, "Top bottleneck score", bottleneck["bottleneck_score"], "score", bottleneck["station"]),
            (9, "Top critical node score", critical["critical_node_score"], "score", critical["station"]),
            (10, "Average throughput efficiency", train["avg_throughput_efficiency"] * 100, "%", "Timestamp-derived proxy, not OEE"),
        ],
        columns=["display_order", "metric", "value", "unit", "interpretation"],
    )
    write_frame(connection, "production_summary", summary)
    return summary


def build_database() -> dict[str, int]:
    require_sources()
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    row_counts: dict[str, int] = {}
    with sqlite3.connect(DATABASE_PATH) as connection:
        for table, path in TABLE_SOURCES.items():
            frame = pd.read_csv(path)
            write_frame(connection, table, frame)
            row_counts[table] = len(frame)

        row_counts["product_predictions"] = build_prediction_table(connection)
        summary = build_summary_table(connection)
        row_counts["production_summary"] = len(summary)

        catalog = pd.DataFrame(
            [
                {
                    "table_name": table,
                    "source_path": str(path.relative_to(PROJECT_ROOT)),
                    "row_count": row_counts[table],
                    "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                for table, path in TABLE_SOURCES.items()
            ]
            + [
                {
                    "table_name": "product_predictions",
                    "source_path": "Phase 6 model + Phase 10 supporting scores",
                    "row_count": row_counts["product_predictions"],
                    "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "table_name": "production_summary",
                    "source_path": "Derived from reviewed Phase 3/6/8/9 outputs",
                    "row_count": row_counts["production_summary"],
                    "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            ]
        )
        write_frame(connection, "source_catalog", catalog)
        row_counts["source_catalog"] = len(catalog)
        connection.execute("PRAGMA optimize")

    return row_counts


def write_report(row_counts: dict[str, int]) -> None:
    report = f"""# Phase 11: Manufacturing Copilot

## Delivered Capabilities

1. Model and analytics outputs are consolidated in SQLite.
2. A Streamlit app provides production summaries, risk review, root-cause evidence,
   bottleneck analysis, process-graph insights, and natural-language questions.
3. Natural-language answers use reviewed intent templates and parameterized SQL.
4. Root-cause answers distinguish model associations from confirmed physical causes.
5. The production-safe Phase 6 LightGBM remains the official failure-risk model.

## Database

- Path: `data/database/manufacturing_copilot.db`
- Tables: {len(row_counts)}
- Product prediction rows: {row_counts['product_predictions']:,}
- Test population: all rows from `phase6_test_predictions.csv`
- Validation population: official Phase 6 validation set with known outcomes

## Natural-Language Examples

- Which stations have the highest failure rate?
- What are the top bottlenecks?
- Why is L3_S32 risky?
- Show the most critical process nodes.
- What are the likely failure propagation routes?
- How good is the production-safe model?
- Summarize production performance.

## Governance Notes

- The question engine is deterministic and offline; it does not send factory data to
  an external language model.
- SHAP, graph routes, and anomaly scores are diagnostic evidence, not causal proof.
- Phase 10 advanced-AI scores are supporting signals and do not replace the official
  Phase 6 probability.
- Throughput efficiency is a timestamp-derived proxy and must not be presented as OEE.
"""
    (REPORTS_DIR / "phase11_manufacturing_copilot_report.md").write_text(
        report, encoding="utf-8"
    )
    (REPORTS_DIR / "phase11_database_manifest.json").write_text(
        json.dumps(row_counts, indent=2), encoding="utf-8"
    )


def main() -> None:
    row_counts = build_database()
    write_report(row_counts)
    print(f"Created {DATABASE_PATH}")
    print(json.dumps(row_counts, indent=2))


if __name__ == "__main__":
    main()
