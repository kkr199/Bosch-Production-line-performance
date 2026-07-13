"""Leaderboard-style Phase 6 model upgrade.

Public Bosch competition solutions relied heavily on production-order features:
parts close together in time or Id often share failure risk. These features are
useful for Kaggle scoring because train and test were sampled from the same
production period. Treat validation scores from this script as competition-style
estimates, not as a fully production-safe time-split estimate.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def best_threshold_metrics(y_true: pd.Series, scores: np.ndarray) -> dict[str, float]:
    thresholds = np.unique(np.quantile(scores, np.linspace(0.50, 0.999, 150)))
    best = None
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        row = {
            "threshold": float(threshold),
            "mcc": float(matthews_corrcoef(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        }
        if best is None or row["mcc"] > best["mcc"]:
            best = row
    assert best is not None
    best["pr_auc"] = float(average_precision_score(y_true, scores))
    return best


def load_base_order_frame(split: str) -> pd.DataFrame:
    phase4_cols = [
        "Id",
        "start_time",
        "end_time",
        "cycle_time",
        "station_count",
        "line_count",
        "path_complexity_score",
        "line_0_processing_duration",
        "line_1_processing_duration",
        "line_2_processing_duration",
        "line_3_processing_duration",
    ]
    phase4 = pd.read_csv(PROCESSED_DIR / f"phase4_{split}_engineered_features.csv", usecols=lambda c: c in phase4_cols + ["Response"])
    phase5 = pd.read_csv(
        PROCESSED_DIR / f"phase5_{split}_product_families.csv",
        usecols=["Id", "final_product_family", "kmeans_family", "dbscan_family", "hierarchical_family"],
    )
    frame = phase4.merge(phase5, on="Id", how="left")
    frame["split"] = split
    if "Response" not in frame.columns:
        frame["Response"] = np.nan
    return frame


def add_neighbor_delta_features(combined: pd.DataFrame, sort_cols: list[str], prefix: str, value_cols: list[str]) -> pd.DataFrame:
    ordered = combined.sort_values(sort_cols).copy()
    ordered[f"{prefix}_order_rank"] = np.arange(len(ordered), dtype=np.int32)
    known_response = ordered["Response"].fillna(0).astype(np.int8)
    ordered[f"{prefix}_known_fail_prev1"] = known_response.shift(1).fillna(0).astype(np.int8)
    ordered[f"{prefix}_known_fail_next1"] = known_response.shift(-1).fillna(0).astype(np.int8)
    ordered[f"{prefix}_known_fail_prev2"] = known_response.shift(2).fillna(0).astype(np.int8)
    ordered[f"{prefix}_known_fail_next2"] = known_response.shift(-2).fillna(0).astype(np.int8)
    ordered[f"{prefix}_known_fail_prev5_sum"] = known_response.shift(1).rolling(5, min_periods=1).sum().fillna(0)
    ordered[f"{prefix}_known_fail_next5_sum"] = (
        known_response.iloc[::-1].shift(1).rolling(5, min_periods=1).sum().iloc[::-1].fillna(0)
    )
    ordered[f"{prefix}_known_fail_prev20_sum"] = known_response.shift(1).rolling(20, min_periods=1).sum().fillna(0)
    ordered[f"{prefix}_known_fail_next20_sum"] = (
        known_response.iloc[::-1].shift(1).rolling(20, min_periods=1).sum().iloc[::-1].fillna(0)
    )

    for column in value_cols:
        values = pd.to_numeric(ordered[column], errors="coerce")
        ordered[f"{prefix}_{column}_diff_prev1"] = values - values.shift(1)
        ordered[f"{prefix}_{column}_diff_next1"] = values - values.shift(-1)
        ordered[f"{prefix}_{column}_diff_prev2"] = values - values.shift(2)
        ordered[f"{prefix}_{column}_diff_next2"] = values - values.shift(-2)

    keep_cols = ["Id"] + [column for column in ordered.columns if column.startswith(prefix)]
    return ordered[keep_cols]


def add_failure_distance_features(combined: pd.DataFrame) -> pd.DataFrame:
    ordered = combined.sort_values(["start_time", "Id"]).copy()
    start_values = ordered["start_time"].fillna(-1).to_numpy(dtype=np.float64)
    known_fail = ordered["Response"].fillna(0).to_numpy(dtype=np.int8)
    fail_positions = np.flatnonzero(known_fail == 1)
    fail_times = start_values[fail_positions]

    result = pd.DataFrame({"Id": ordered["Id"].to_numpy()})
    if len(fail_times) == 0:
        for column in [
            "time_to_prev_known_failure",
            "time_to_next_known_failure",
            "time_to_nearest_known_failure",
            "known_failures_window_0_02",
            "known_failures_window_0_05",
            "known_failures_window_0_10",
            "known_failures_window_0_50",
        ]:
            result[column] = 0
        return result

    insert_pos = np.searchsorted(fail_times, start_values)
    prev_idx = np.clip(insert_pos - 1, 0, len(fail_times) - 1)
    next_idx = np.clip(insert_pos, 0, len(fail_times) - 1)
    prev_dist = start_values - fail_times[prev_idx]
    next_dist = fail_times[next_idx] - start_values
    prev_dist[insert_pos == 0] = np.nan
    next_dist[insert_pos == len(fail_times)] = np.nan
    result["time_to_prev_known_failure"] = prev_dist
    result["time_to_next_known_failure"] = next_dist
    result["time_to_nearest_known_failure"] = np.nanmin(np.vstack([prev_dist, next_dist]), axis=0)

    for window in [0.02, 0.05, 0.10, 0.50]:
        left = np.searchsorted(fail_times, start_values - window, side="left")
        right = np.searchsorted(fail_times, start_values + window, side="right")
        counts = right - left
        counts = counts - known_fail
        safe_name = str(window).replace(".", "_")
        result[f"known_failures_window_{safe_name}"] = counts.astype(np.int16)

    return result


def build_leaderboard_features(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = PROCESSED_DIR / "phase6_leaderboard_boost_train_features.csv"
    test_path = PROCESSED_DIR / "phase6_leaderboard_boost_test_features.csv"
    if train_path.exists() and test_path.exists() and not force:
        return pd.read_csv(train_path), pd.read_csv(test_path)

    train = load_base_order_frame("train")
    test = load_base_order_frame("test")
    combined = pd.concat([train, test], ignore_index=True, sort=False)

    value_cols = [
        "Id",
        "start_time",
        "end_time",
        "cycle_time",
        "station_count",
        "line_count",
        "path_complexity_score",
        "line_0_processing_duration",
        "line_1_processing_duration",
        "line_2_processing_duration",
        "line_3_processing_duration",
        "final_product_family",
    ]
    feature_parts = [
        combined[["Id", "split"]],
        add_neighbor_delta_features(combined, ["Id"], "id_order", value_cols),
        add_neighbor_delta_features(combined, ["start_time", "Id"], "start_order", value_cols),
        add_neighbor_delta_features(combined, ["end_time", "Id"], "end_order", value_cols),
        add_failure_distance_features(combined),
    ]

    features = feature_parts[0]
    for part in feature_parts[1:]:
        features = features.merge(part, on="Id", how="left")

    numeric_cols = [column for column in features.columns if column not in {"Id", "split"}]
    for column in numeric_cols:
        values = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype=np.float32, copy=True)
        values[~np.isfinite(values)] = np.nan
        features[column] = values
    train_features = features[features["split"] == "train"].drop(columns=["split"])
    test_features = features[features["split"] == "test"].drop(columns=["split"])
    train_features.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)
    return train_features, test_features


def train_boost_models() -> tuple[pd.DataFrame, Path]:
    leaderboard_train, leaderboard_test = build_leaderboard_features()
    phase6_best = joblib.load(MODELS_DIR / "phase6_best_model.joblib")
    base_model = phase6_best["model"]
    base_feature_cols = phase6_best["feature_cols"]

    train_base = pd.read_csv(PROCESSED_DIR / "phase6_train_dataset.csv")
    valid_base = pd.read_csv(PROCESSED_DIR / "phase6_validation_dataset.csv")
    train = train_base[["Id", "Response"]].merge(leaderboard_train, on="Id", how="left")
    valid = valid_base[["Id", "Response"]].merge(leaderboard_train, on="Id", how="left")
    train["phase6_base_probability"] = base_model.predict_proba(
        train_base[base_feature_cols].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    valid["phase6_base_probability"] = base_model.predict_proba(
        valid_base[base_feature_cols].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    test_pred_base = pd.read_csv(PROCESSED_DIR / "phase6_test_predictions.csv", usecols=["Id", "failure_probability"])
    test_pred_base = test_pred_base.rename(columns={"failure_probability": "phase6_base_probability"})

    feature_cols = [column for column in train.columns if column not in {"Id", "Response"}]
    x_train = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_train = train["Response"].astype(np.int8)
    x_valid = valid[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_valid = valid["Response"].astype(np.int8)
    scale_pos_weight = (len(y_train) - int(y_train.sum())) / max(1, int(y_train.sum()))

    models = {
        "Leaderboard LightGBM": LGBMClassifier(
            n_estimators=650,
            learning_rate=0.025,
            num_leaves=64,
            min_child_samples=40,
            subsample=0.90,
            colsample_bytree=0.90,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "Leaderboard XGBoost": XGBClassifier(
            n_estimators=450,
            max_depth=7,
            learning_rate=0.035,
            subsample=0.90,
            colsample_bytree=0.90,
            min_child_weight=3,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_valid)[:, 1]
        metrics = best_threshold_metrics(y_valid, scores)
        rows.append({"model": name, **metrics})
        fitted[name] = model
        joblib.dump({"model": model, "feature_cols": feature_cols}, MODELS_DIR / f"phase6_{name.lower().replace(' ', '_')}.joblib")

    metrics_report = pd.DataFrame(rows).sort_values(["mcc", "pr_auc"], ascending=False)
    metrics_report["rank"] = range(1, len(metrics_report) + 1)
    metrics_path = REPORTS_DIR / "phase6_leaderboard_boost_metrics.csv"
    metrics_report.to_csv(metrics_path, index=False)

    best = metrics_report.iloc[0]
    best_name = str(best["model"])
    best_bundle = {
        "model": fitted[best_name],
        "feature_cols": feature_cols,
        "best_model_name": best_name,
        "best_threshold": float(best["threshold"]),
    }
    joblib.dump(best_bundle, MODELS_DIR / "phase6_leaderboard_boost_best_model.joblib")

    test_features = test_pred_base.merge(leaderboard_test, on="Id", how="left")
    test_scores = fitted[best_name].predict_proba(test_features[feature_cols].replace([np.inf, -np.inf], np.nan))[:, 1]
    pred = pd.DataFrame(
        {
            "Id": test_features["Id"].astype(np.int64),
            "failure_probability": test_scores,
            "predicted_failure_at_selected_threshold": (test_scores >= float(best["threshold"])).astype(np.uint8),
        }
    )
    pred_path = PROCESSED_DIR / "phase6_leaderboard_boost_test_predictions.csv"
    pred.to_csv(pred_path, index=False)
    return metrics_report, pred_path


def write_report(metrics_report: pd.DataFrame, pred_path: Path) -> Path:
    best = metrics_report.iloc[0]
    report_path = REPORTS_DIR / "phase6_leaderboard_boost_report.md"
    lines = [
        "# Phase 6 Leaderboard Boost",
        "",
        "## What Changed",
        "",
        "This extension adds public-solution-inspired order and leak-style features:",
        "",
        "- Previous/next known failure indicators after sorting by `Id`, `start_time`, and `end_time`.",
        "- Counts of known failures near each part's start time.",
        "- Distance to previous, next, and nearest known failure in production time.",
        "- Previous/next deltas for timing, path, line-duration, and product-family features.",
        "",
        "These features mirror the public Kaggle insight that failures cluster among nearby parts in production order. They are useful for competition scoring because train and test were randomly sampled from the same production period.",
        "",
        "## Validation Results",
        "",
        metrics_report.to_markdown(index=False),
        "",
        "## Selected Boost Model",
        "",
        (
            f"The selected boost model is **{best['model']}**, with MCC {best['mcc']:.4f}, "
            f"precision {best['precision']:.4f}, recall {best['recall']:.4f}, F1 {best['f1']:.4f}, "
            f"and PR-AUC {best['pr_auc']:.4f}."
        ),
        "",
        "## Important Caveat",
        "",
        "This is a competition-style validation result. Because order/leak features use nearby known training failures, the validation score is optimistic compared with a true future-time production deployment test.",
        "",
        "## Output Files",
        "",
        "- `data/processed/phase6_leaderboard_boost_train_features.csv`",
        "- `data/processed/phase6_leaderboard_boost_test_features.csv`",
        "- `data/processed/phase6_leaderboard_boost_test_predictions.csv`",
        "- `reports/phase6_leaderboard_boost_metrics.csv`",
        "- `models/phase6_leaderboard_boost_best_model.joblib`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_report, pred_path = train_boost_models()
    report_path = write_report(metrics_report, pred_path)
    print(f"Leaderboard boost complete. Report written to {report_path}")


if __name__ == "__main__":
    main()
