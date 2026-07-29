"""Benchmark all Phase 6 model families over numeric coverage thresholds.

Uses every labelled row in train_numeric.csv. Numeric screening happens only on
the 80% training partition; MCC and PR-AUC are measured on the untouched 20%
holdout. Outputs are written only inside this folder.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_DIR.parent
from selection_utils import (  # noqa: E402
    RAW_NUMERIC,
    load_fold_features,
    profile_training_fold,
    selected_features,
    validation_threshold,
)


THRESHOLDS = (0.001, 0.0025, 0.005, 0.01, 0.02)
RANDOM_STATE = 42
METRICS_PATH = OUTPUT_DIR / "full_training_model_threshold_metrics_live.csv"
LEGACY_METRICS_PATH = OUTPUT_DIR / "full_training_model_threshold_metrics.csv"


def model_factories(positive_weight: float) -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=250,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=60,
                        max_depth=14,
                        min_samples_leaf=25,
                        class_weight="balanced_subsample",
                        n_jobs=4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="aucpr",
            scale_pos_weight=positive_weight,
            tree_method="hist",
            n_jobs=4,
            random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=48,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=4,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            thread_count=4,
            verbose=False,
        ),
    }


def save_progress(rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(METRICS_PATH, index=False)


def load_completed_rows() -> list[dict[str, object]]:
    """Resume safely if an earlier live CSV was opened by another application."""
    for path in (METRICS_PATH, LEGACY_METRICS_PATH):
        if path.exists():
            return pd.read_csv(path).to_dict("records")
    return []


def run() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(exist_ok=True)
    target = pd.read_csv(RAW_NUMERIC, usecols=["Id", "Response"])
    target["Id"] = target["Id"].astype(np.int64)
    target["Response"] = target["Response"].astype(np.int8)
    train, valid = train_test_split(
        target, test_size=0.20, random_state=RANDOM_STATE, stratify=target["Response"]
    )
    print(f"Using all {len(target):,} labelled rows: {len(train):,} train / {len(valid):,} validation.", flush=True)
    print("Profiling numeric coverage and selection scores on training rows only...", flush=True)
    profile = profile_training_fold(set(train["Id"]), len(train))
    rows = load_completed_rows()
    completed = {
        (float(row["minimum_present_rate"]), str(row["model"]))
        for row in rows
        if str(row.get("status", "")) == "completed"
    }
    y_train = train["Response"].to_numpy()
    y_valid = valid["Response"].to_numpy()
    positive_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())

    for present_rate in THRESHOLDS:
        features, candidate_count = selected_features(profile, present_rate)
        print(f"\nThreshold {present_rate:g}: loading {len(features)} retained features...", flush=True)
        numeric = load_fold_features(set(train["Id"]) | set(valid["Id"]), features)
        x_train = numeric.reindex(train["Id"])[features]
        x_valid = numeric.reindex(valid["Id"])[features]

        for model_name, model in model_factories(positive_weight).items():
            if (present_rate, model_name) in completed:
                print(f"  Skipping {model_name}; completed result was recovered.", flush=True)
                continue
            print(f"  Fitting {model_name}...", flush=True)
            started = time.perf_counter()
            status = "completed"
            error = ""
            try:
                model.fit(x_train, y_train)
                train_scores = model.predict_proba(x_train)[:, 1]
                valid_scores = model.predict_proba(x_valid)[:, 1]
                cutoff = validation_threshold(train_scores, y_train)
                mcc = float(matthews_corrcoef(y_valid, (valid_scores >= cutoff).astype(np.uint8)))
                pr_auc = float(average_precision_score(y_valid, valid_scores))
                if model_name == "LightGBM":
                    suffix = str(present_rate).replace(".", "_")
                    joblib.dump(
                        {"model": model, "feature_cols": features, "minimum_present_rate": present_rate, "threshold": cutoff},
                        OUTPUT_DIR / f"lightgbm_present_rate_{suffix}.joblib",
                    )
            except Exception as exc:  # Preserve completed comparisons if one library cannot fit locally.
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                mcc = np.nan
                pr_auc = np.nan
                cutoff = np.nan
            rows.append(
                {
                    "minimum_present_rate": present_rate,
                    "minimum_observed_training_rows": int(len(train) * present_rate),
                    "candidate_feature_count": candidate_count,
                    "features_retained": len(features),
                    "model": model_name,
                    "validation_pr_auc": pr_auc,
                    "validation_mcc": mcc,
                    "training_prevalence_cutoff": cutoff,
                    "runtime_seconds": time.perf_counter() - started,
                    "status": status,
                    "error": error,
                }
            )
            save_progress(rows)
            print(f"    {status}", flush=True)
        del numeric, x_train, x_valid

    result = pd.DataFrame(rows).sort_values(["validation_mcc", "validation_pr_auc"], ascending=False)
    result.to_csv(METRICS_PATH, index=False)
    completed_result = result[result["status"] == "completed"]
    best = completed_result.iloc[0]
    lines = [
        "# Advanced ML Models: Full Training Dataset Benchmark",
        "",
        f"All {len(target):,} labelled numeric training rows were used in a fixed stratified 80/20 split. "
        "Coverage screening and target-informed ranking used the training partition only.",
        "",
        completed_result.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Best validation MCC",
        "",
        f"**{best['model']}** at a minimum present rate of **{best['minimum_present_rate']:.4g}** "
        f"achieved validation MCC **{best['validation_mcc']:.6f}** and PR-AUC **{best['validation_pr_auc']:.6f}**.",
    ]
    (OUTPUT_DIR / "full_training_model_threshold_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(run().to_string(index=False))
