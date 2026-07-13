"""Phase 7 root cause analysis using the production-safe Phase 6 model."""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
SHAP_SAMPLE_ROWS = 8_000
LOCAL_EXPLANATION_ROWS = 1_000
FEATURE_RE = re.compile(r"^(?P<line>L\d+)_(?P<station>S\d+)_F(?P<feature>\d+)")


def load_phase6_assets() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    model_path = MODELS_DIR / "phase6_best_model.joblib"
    validation_path = PROCESSED_DIR / "phase6_validation_dataset.csv"
    train_path = PROCESSED_DIR / "phase6_train_dataset.csv"
    if not model_path.exists() or not validation_path.exists() or not train_path.exists():
        raise FileNotFoundError("Phase 7 requires Phase 6 production-safe model and train/validation datasets.")
    bundle = joblib.load(model_path)
    validation = pd.read_csv(validation_path)
    train = pd.read_csv(train_path)
    return bundle, train, validation


def unwrap_model(bundle: dict):
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"], model, feature_cols
    return model, model, feature_cols


def model_ready_matrix(pipeline, frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    if hasattr(pipeline, "named_steps") and "imputer" in pipeline.named_steps:
        imputed = pipeline.named_steps["imputer"].transform(x)
        return pd.DataFrame(imputed, columns=feature_cols, index=frame.index)
    return x


def parse_feature_origin(feature: str) -> dict[str, str]:
    base_feature = feature.replace("__is_missing", "")
    match = FEATURE_RE.match(base_feature)
    if match:
        line = match.group("line")
        station = match.group("station")
        return {
            "line": line,
            "station": f"{line}_{station}",
            "feature_family": "raw_measurement_missingness" if feature.endswith("__is_missing") else "raw_measurement",
        }
    if feature.startswith("line_"):
        line_token = feature.split("_")[1]
        line_label = f"L{line_token}" if line_token.isdigit() else "line_level"
        return {"line": line_label, "station": "line_level", "feature_family": "line_timing"}
    if "station" in feature or "path" in feature or "family" in feature:
        return {"line": "path", "station": "path_level", "feature_family": "path_or_family"}
    if "time" in feature or "duration" in feature or "delay" in feature or "cycle" in feature:
        return {"line": "timing", "station": "timing_level", "feature_family": "timing"}
    return {"line": "other", "station": "other", "feature_family": "categorical_or_other"}


def calculate_feature_importance(estimator, feature_cols: list[str]) -> pd.DataFrame:
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_).ravel()
    else:
        importance = np.zeros(len(feature_cols))
    rows = []
    for feature, value in zip(feature_cols, importance):
        rows.append({"feature": feature, "model_importance": float(value), **parse_feature_origin(feature)})
    report = pd.DataFrame(rows).sort_values("model_importance", ascending=False)
    report.to_csv(REPORTS_DIR / "phase7_feature_importance.csv", index=False)
    return report


def calculate_shap_explanations(estimator, x_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        shap_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_array = shap_values
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]

    mean_abs = np.abs(shap_array).mean(axis=0)
    mean_signed = shap_array.mean(axis=0)
    shap_report = pd.DataFrame(
        {
            "feature": x_sample.columns,
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
        }
    )
    origins = pd.DataFrame([parse_feature_origin(feature) for feature in shap_report["feature"]])
    shap_report = pd.concat([shap_report, origins], axis=1).sort_values("mean_abs_shap", ascending=False)
    shap_report.to_csv(REPORTS_DIR / "phase7_shap_global_importance.csv", index=False)

    local = pd.DataFrame(shap_array[:LOCAL_EXPLANATION_ROWS], columns=x_sample.columns, index=x_sample.index[:LOCAL_EXPLANATION_ROWS])
    local.insert(0, "sample_index", local.index)
    local.to_csv(REPORTS_DIR / "phase7_shap_local_values_sample.csv", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_array, x_sample, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase7_shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    top = shap_report.head(25).sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Top SHAP Failure Drivers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase7_top_shap_drivers.png", dpi=160, bbox_inches="tight")
    plt.close()

    return shap_report, local


def identify_top_failure_drivers(importance: pd.DataFrame, shap_report: pd.DataFrame) -> pd.DataFrame:
    merged = shap_report.merge(
        importance[["feature", "model_importance"]],
        on="feature",
        how="left",
    )
    merged["driver_rank"] = np.arange(1, len(merged) + 1)
    merged["driver_type"] = np.select(
        [
            merged["feature"].str.endswith("__is_missing"),
            merged["feature_family"].eq("line_timing"),
            merged["feature_family"].eq("timing"),
            merged["feature_family"].eq("path_or_family"),
            merged["feature_family"].eq("raw_measurement"),
        ],
        [
            "Missingness / skipped measurement signal",
            "Line-level timing signal",
            "Timing or delay signal",
            "Manufacturing path signal",
            "Numeric process measurement",
        ],
        default="Categorical or derived signal",
    )
    merged.head(100).to_csv(REPORTS_DIR / "phase7_top_failure_drivers.csv", index=False)
    return merged


def station_recommendation(station: str, rows: pd.DataFrame) -> str:
    features = " ".join(rows["feature"].head(8).tolist()).lower()
    if station in {"path_level", "timing_level", "line_level", "other"}:
        if "time" in features or "duration" in features or "waiting" in features:
            return "Review cycle-time, waiting-time, and queue behavior for products following this path."
        return "Use this as a cross-station signal; compare affected product families and paths before station-specific action."
    if "missing" in features:
        return "Check whether skipped measurements, sensor dropouts, or alternate routing through this station align with failures."
    if "time" in features or "duration" in features:
        return "Inspect station timing, queue buildup, operator handoff, and maintenance intervals for unusual delays."
    return "Review measurement distributions, tooling condition, calibration records, and recent process changes for this station."


def build_station_root_cause_reports(drivers: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    station_rows = (
        drivers.groupby(["line", "station", "feature_family"], dropna=False)
        .agg(
            feature_count=("feature", "count"),
            total_mean_abs_shap=("mean_abs_shap", "sum"),
            avg_mean_abs_shap=("mean_abs_shap", "mean"),
            top_driver=("feature", "first"),
            top_driver_type=("driver_type", "first"),
        )
        .reset_index()
        .sort_values("total_mean_abs_shap", ascending=False)
    )
    station_rows["recommended_action"] = [
        station_recommendation(station, drivers[drivers["station"] == station])
        for station in station_rows["station"]
    ]
    station_rows.to_csv(REPORTS_DIR / "phase7_station_root_cause_report.csv", index=False)

    failure_rate = validation["Response"].mean()
    summary_rows = []
    for station, group in station_rows.head(20).groupby("station"):
        summary_rows.append(
            {
                "station": station,
                "root_cause_priority": float(group["total_mean_abs_shap"].sum()),
                "primary_driver": group.iloc[0]["top_driver"],
                "recommended_action": group.iloc[0]["recommended_action"],
                "validation_failure_rate_pct": failure_rate * 100,
            }
        )
    pd.DataFrame(summary_rows).sort_values("root_cause_priority", ascending=False).to_csv(
        REPORTS_DIR / "phase7_engineer_action_plan.csv", index=False
    )
    return station_rows


def write_report(
    importance: pd.DataFrame,
    shap_report: pd.DataFrame,
    drivers: pd.DataFrame,
    station_report: pd.DataFrame,
    validation: pd.DataFrame,
) -> Path:
    report_path = REPORTS_DIR / "phase7_root_cause_analysis_report.md"
    top_driver = drivers.iloc[0]
    top_station = station_report.iloc[0]
    lines = [
        "# Phase 7: Root Cause Analysis",
        "",
        "## Model Used",
        "",
        "This phase uses the production-safe Phase 6 best model, not the Kaggle leaderboard/leak model. That makes the explanations more appropriate for future manufacturing projects and engineering discussions.",
        "",
        "## Validation Population",
        "",
        f"- Validation rows explained: {len(validation):,}",
        f"- Validation failure rate: {validation['Response'].mean() * 100:.3f}%",
        f"- SHAP sample size: {min(SHAP_SAMPLE_ROWS, len(validation)):,}",
        "",
        "## Top Failure Driver",
        "",
        f"The strongest global SHAP driver is `{top_driver['feature']}` with mean absolute SHAP {top_driver['mean_abs_shap']:.6f}.",
        "",
        "## Top Station / Root-Cause Area",
        "",
        f"The highest-priority root-cause area is `{top_station['station']}`, led by `{top_station['top_driver']}`.",
        "",
        "## Top 15 Failure Drivers",
        "",
        drivers.head(15)[
            ["driver_rank", "feature", "station", "driver_type", "mean_abs_shap", "mean_signed_shap"]
        ].to_markdown(index=False),
        "",
        "## Top 15 Station-Level Root Cause Priorities",
        "",
        station_report.head(15)[
            ["line", "station", "feature_family", "total_mean_abs_shap", "top_driver", "recommended_action"]
        ].to_markdown(index=False),
        "",
        "## Recommended Engineering Actions",
        "",
        "- Prioritize stations and path-level drivers with the largest total SHAP contribution.",
        "- For missingness drivers, check sensor availability, skipped operations, routing differences, and whether missing measurements represent a known process branch.",
        "- For timing drivers, inspect queue buildup, station waiting time, cycle-time drift, and recent maintenance windows.",
        "- For raw numeric drivers, review calibration, tooling condition, process limits, and distribution shifts for the named station measurements.",
        "- For path or product-family drivers, compare high-risk product routes against lower-risk routes and confirm whether routing policy or product mix explains the pattern.",
        "",
        "## Output Files",
        "",
        "- `reports/phase7_feature_importance.csv`",
        "- `reports/phase7_shap_global_importance.csv`",
        "- `reports/phase7_shap_local_values_sample.csv`",
        "- `reports/phase7_top_failure_drivers.csv`",
        "- `reports/phase7_station_root_cause_report.csv`",
        "- `reports/phase7_engineer_action_plan.csv`",
        "- `reports/figures/phase7_shap_summary.png`",
        "- `reports/figures/phase7_top_shap_drivers.png`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    bundle, train, validation = load_phase6_assets()
    estimator, pipeline, feature_cols = unwrap_model(bundle)
    importance = calculate_feature_importance(estimator, feature_cols)

    shap_sample = validation.sample(n=min(SHAP_SAMPLE_ROWS, len(validation)), random_state=RANDOM_STATE)
    x_sample = model_ready_matrix(pipeline, shap_sample, feature_cols)
    shap_report, _ = calculate_shap_explanations(estimator, x_sample)
    drivers = identify_top_failure_drivers(importance, shap_report)
    station_report = build_station_root_cause_reports(drivers, validation)
    report_path = write_report(importance, shap_report, drivers, station_report, validation)
    print(f"Phase 7 complete. Report written to {report_path}")


if __name__ == "__main__":
    main()
