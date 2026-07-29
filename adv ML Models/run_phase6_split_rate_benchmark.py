"""Full-data Phase 6 benchmark across split ratios and numeric coverage gates."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
from selection_utils import profile_training_fold, selected_features, validation_threshold, select_categories_on_train  # noqa: E402
from src.data.phase6_predictive_failure_modeling import build_model_dataset, find_dataset_path  # noqa: E402
from run_full_training_threshold_benchmark import model_factories  # noqa: E402

SPLITS = {"75/25": 0.25, "80/20": 0.20, "90/10": 0.10}
RATES = (0.001, 0.0025, 0.005, 0.01, 0.02)
RANDOM_STATE = 42
METRICS = OUT / "phase6_full_clean_split_rate_metrics.csv"


def save(rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(METRICS, index=False)


def run() -> pd.DataFrame:
    target = pd.read_csv(find_dataset_path("train_numeric.csv"), usecols=["Id", "Response"])
    target["Id"] = target["Id"].astype(np.int64); target["Response"] = target["Response"].astype(np.int8)
    rows: list[dict[str, object]] = []
    for split_label, valid_fraction in SPLITS.items():
        train, valid = train_test_split(target, test_size=valid_fraction, stratify=target["Response"], random_state=RANDOM_STATE)
        print(f"{split_label}: fitting numeric and categorical selection on {len(train):,} training rows...", flush=True)
        numeric_report = profile_training_fold(set(train["Id"]), len(train)); numeric_report["missing_rate"] = 1 - numeric_report["present_rate"]
        cats, levels = select_categories_on_train(train)
        y_train, y_valid = train["Response"].to_numpy(), valid["Response"].to_numpy()
        weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
        for rate in RATES:
            numeric, candidates = selected_features(numeric_report, rate)
            print(f"  {split_label}, present rate {rate:g}: building merged features...", flush=True)
            frame = build_model_dataset(target, numeric, cats, levels, numeric_report, "train").set_index("Id")
            features = [c for c in frame.columns if c != "Response"]
            x_train, x_valid = frame.reindex(train["Id"])[features], frame.reindex(valid["Id"])[features]
            for name, model in model_factories(weight).items():
                print(f"    {name}", flush=True); started = time.perf_counter()
                status, error = "completed", ""
                try:
                    model.fit(x_train, y_train); train_scores = model.predict_proba(x_train)[:, 1]; scores = model.predict_proba(x_valid)[:, 1]
                    cutoff = validation_threshold(train_scores, y_train); pred = (scores >= cutoff).astype(np.uint8)
                    mcc, pr_auc = float(matthews_corrcoef(y_valid, pred)), float(average_precision_score(y_valid, scores))
                    precision, recall = float(precision_score(y_valid, pred, zero_division=0)), float(recall_score(y_valid, pred, zero_division=0))
                except Exception as exc:
                    status, error, mcc, pr_auc, precision, recall, cutoff = "failed", f"{type(exc).__name__}: {exc}", np.nan, np.nan, np.nan, np.nan, np.nan
                rows.append({"split_ratio": split_label, "rows": len(target), "train_rows": len(train), "validation_rows": len(valid), "numeric_present_rate": rate, "numeric_candidates": candidates, "numeric_features": len(numeric), "categorical_columns": len(cats), "one_hot_levels": sum(map(len, levels.values())), "final_feature_count": len(features), "model": name, "mcc": mcc, "pr_auc": pr_auc, "precision": precision, "recall": recall, "runtime_seconds": time.perf_counter() - started, "status": status, "error": error})
                save(rows)
            del frame, x_train, x_valid
    result = pd.DataFrame(rows).sort_values(["mcc", "pr_auc"], ascending=False); save(result.to_dict("records")); return result


if __name__ == "__main__": print(run().to_string(index=False))
