"""Synchronise selected LightGBM outcomes and importance data into the dashboard SQLite DB."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ADVANCED = ROOT / "adv ML Models"
REPORTS = ROOT / "reports"
DATABASE = ROOT / "data" / "database" / "manufacturing_copilot.db"
MODEL = ROOT / "models" / "advanced_ml_lightgbm_80_20_0_0025.joblib"
BENCHMARK = ADVANCED / "phase6_full_clean_split_rate_metrics_completed.csv"
OUTCOME = REPORTS / "advanced_ml_test_prediction_summary.csv"
PREDICTIONS = ROOT / "data" / "processed" / "advanced_ml_test_predictions.parquet"


def describe_feature(feature: str) -> tuple[str, str, str, str]:
    """Return display name, driver family, line and station for a model feature."""
    raw_station = re.fullmatch(r"(L\d+_S\d+)_F\d+(?:__(is_missing))?", feature)
    if raw_station:
        station, missing = raw_station.groups()
        suffix = " missing-measurement indicator" if missing else " measurement"
        return f"{station}{suffix}", "station measurement", station[:2], station
    line_feature = re.fullmatch(r"line_(\d+)_(.+)", feature)
    if line_feature:
        line, suffix = line_feature.groups()
        return f"Line {line} {suffix.replace('_', ' ')}", "line timing", f"L{line}", "line_level"
    if feature.startswith(("kmeans_", "hierarchical_", "path_", "product_")):
        return feature.replace("_", " ").title(), "path or family", "path", "path_level"
    if feature.endswith("__is_missing"):
        return feature.replace("__is_missing", " missing measurement").replace("_", " ").title(), "missingness", "other", "other"
    if any(token in feature for token in ("time", "duration", "waiting", "cycle", "delay", "station_count")):
        return feature.replace("_", " ").title(), "timing", "timing", "timing_level"
    return feature.replace("_", " ").title(), "other", "other", "other"


def recommended_action(driver_family: str) -> str:
    if driver_family in {"timing", "line timing"}:
        return "Treat as a temporal association; compare routes, families, and production windows before station-specific action."
    if driver_family == "path or family":
        return "Compare affected product families and process paths before changing an individual station."
    if driver_family == "missingness":
        return "Check skipped measurements, sensor availability, and alternate routing before drawing a process conclusion."
    return "Review measurement distributions, tooling, calibration records, and recent process changes before intervention."


def main() -> None:
    required = [MODEL, BENCHMARK, OUTCOME, PREDICTIONS, DATABASE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required dashboard artifact(s): " + ", ".join(missing))

    benchmark = pd.read_csv(BENCHMARK).query("status == 'completed'").sort_values(["mcc", "pr_auc"], ascending=False)
    best = benchmark.iloc[0]
    outcome = pd.read_csv(OUTCOME).iloc[0]
    average_risk_pct = float(pd.read_parquet(PREDICTIONS, columns=["failure_probability"])["failure_probability"].mean() * 100)

    artifact = joblib.load(MODEL)
    importance = pd.DataFrame({
        "feature": artifact["feature_cols"],
        "importance": artifact["model"].feature_importances_,
    }).query("importance > 0").sort_values("importance", ascending=False).reset_index(drop=True)
    importance["driver_rank"] = importance.index + 1
    importance[["feature_display_name", "driver_type", "line", "station"]] = importance["feature"].apply(
        lambda feature: pd.Series(describe_feature(feature))
    )
    importance["importance_pct"] = importance["importance"] / importance["importance"].sum() * 100
    importance["interpretation"] = importance["driver_type"].map(recommended_action)

    stations = (
        importance.groupby(["line", "station", "driver_type"], as_index=False)
        .agg(
            feature_count=("feature", "size"),
            total_importance=("importance", "sum"),
            total_importance_pct=("importance_pct", "sum"),
        )
        .sort_values("total_importance", ascending=False)
        .reset_index(drop=True)
    )
    top_feature = importance.sort_values("importance", ascending=False).drop_duplicates("station").set_index("station")
    stations["top_feature"] = stations["station"].map(top_feature["feature"])
    stations["top_feature_display_name"] = stations["station"].map(top_feature["feature_display_name"])
    stations["recommended_action"] = stations["driver_type"].map(recommended_action)
    stations["priority_rank"] = stations.index + 1

    summary_rows = [
        ("selected_model", str(best["model"]), "name", "Best completed validation configuration"),
        ("validation_mcc", str(float(best["mcc"])), "score", "80/20 holdout validation MCC"),
        ("validation_pr_auc", str(float(best["pr_auc"])), "score", "80/20 holdout validation PR-AUC"),
        ("validation_precision", str(float(best["precision"])), "score", "80/20 holdout validation precision"),
        ("validation_recall", str(float(best["recall"])), "score", "80/20 holdout validation recall"),
        ("decision_threshold", str(float(outcome["decision_threshold"])), "probability", "Training-derived operating threshold"),
        ("test_products_scored", str(int(outcome["test_products_scored"])), "products", "Full unlabeled test population"),
        ("test_alerts", str(int(outcome["test_alerts"])), "products", "Predicted alerts at operating threshold"),
        ("test_alert_rate", str(float(outcome["test_alert_rate_pct"]) / 100), "decimal", "Predicted alert rate"),
    ]

    with sqlite3.connect(DATABASE) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS advanced_model_summary (metric_key TEXT PRIMARY KEY, value TEXT NOT NULL, unit TEXT NOT NULL, context TEXT NOT NULL)")
        connection.execute("DELETE FROM advanced_model_summary")
        connection.executemany("INSERT INTO advanced_model_summary VALUES (?, ?, ?, ?)", summary_rows)
        connection.execute("UPDATE executive_kpi_baseline SET value = ?, context = ? WHERE metric_key = ?", (float(best["precision"]), "Selected LightGBM 80/20 holdout validation precision", "model_precision"))
        connection.execute("UPDATE executive_kpi_baseline SET value = ?, context = ? WHERE metric_key = ?", (float(best["recall"]), "Selected LightGBM 80/20 holdout validation recall", "model_recall"))
        connection.execute("UPDATE executive_kpi_baseline SET value = ?, context = ? WHERE metric_key = ?", (float(best["mcc"]), "Selected LightGBM 80/20 holdout validation MCC", "model_mcc"))
        connection.execute("UPDATE executive_kpi_baseline SET value = ?, context = ? WHERE metric_key = ?", (float(outcome["test_alert_rate_pct"]) / 100, "Selected LightGBM predicted alert rate", "test_alert_rate"))
        connection.execute("UPDATE executive_kpi_baseline SET value = ?, context = ? WHERE metric_key = ?", (int(outcome["test_alerts"]), "Selected LightGBM predicted alerts", "test_alerts"))
        updates = [
            (float(best["mcc"]), "Selected LightGBM 80/20 holdout validation", "Validation MCC"),
            (float(best["precision"]), "Selected LightGBM 80/20 holdout validation", "Validation precision"),
            (float(best["recall"]), "Selected LightGBM 80/20 holdout validation", "Validation recall"),
            (int(outcome["test_alerts"]), "Selected LightGBM at training-derived threshold", "Test predicted alerts"),
            (average_risk_pct, "Mean selected-LightGBM test probability", "Average test risk"),
        ]
        connection.executemany("UPDATE production_summary SET value = ?, interpretation = ? WHERE metric = ?", updates)
        importance.to_sql("advanced_model_feature_importance", connection, if_exists="replace", index=False)
        stations.to_sql("advanced_model_station_importance", connection, if_exists="replace", index=False)

    print(f"Synced {best['model']} MCC={best['mcc']:.6f}; {len(importance):,} non-zero feature importances; {len(stations):,} driver groups.")


if __name__ == "__main__":
    main()
