"""Leakage-safe full-data replication of the earlier merged Phase 6 LightGBM pipeline."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
from selection_utils import profile_training_fold, selected_features, validation_threshold  # noqa: E402
from src.data.phase6_predictive_failure_modeling import build_model_dataset, find_dataset_path, read_header  # noqa: E402

RANDOM_STATE = 42
NUMERIC_PRESENT_RATE = 0.005  # Original Phase 6 gate being tested.
TOP_CATS = 20
TOP_LEVELS = 12


def select_categories_on_train(train: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """Select categorical columns and one-hot levels strictly from training IDs."""
    path = find_dataset_path("train_categorical.csv")
    columns = [c for c in read_header(path) if c != "Id"]
    lookup = train.set_index("Id")["Response"]
    stats = {c: [0, 0.0, 0, 0.0] for c in columns}  # present count/y, missing count/y
    for chunk in pd.read_csv(path, chunksize=20_000, dtype=str):
        chunk = chunk[chunk["Id"].astype(np.int64).isin(set(train["Id"]))]
        if chunk.empty:
            continue
        y = chunk["Id"].astype(np.int64).map(lookup).to_numpy(dtype=float)
        for c in columns:
            present = chunk[c].notna().to_numpy()
            stat = stats[c]
            stat[0] += int(present.sum()); stat[1] += float(y[present].sum())
            stat[2] += int((~present).sum()); stat[3] += float(y[~present].sum())
    ranked = []
    for c, (pc, py, mc, my) in stats.items():
        signal = abs(py / pc - my / mc) if pc and mc else 0.0
        if pc >= 200:
            ranked.append((signal, c))
    selected = [c for _, c in sorted(ranked, reverse=True)[:TOP_CATS]]
    level_counts = {c: {} for c in selected}
    train_ids = set(train["Id"])
    for chunk in pd.read_csv(path, usecols=["Id", *selected], chunksize=20_000, dtype=str):
        chunk = chunk[chunk["Id"].astype(np.int64).isin(train_ids)]
        if chunk.empty:
            continue
        y = chunk["Id"].astype(np.int64).map(lookup)
        for c in selected:
            grouped = pd.DataFrame({"v": chunk[c].fillna("__MISSING__"), "y": y}).groupby("v")["y"].agg(["size", "sum"])
            for value, row in grouped.iterrows():
                count, failures = level_counts[c].get(str(value), (0, 0.0))
                level_counts[c][str(value)] = (count + int(row["size"]), failures + float(row["sum"]))
    levels = {}
    for c, values in level_counts.items():
        ranked_levels = sorted(values.items(), key=lambda item: (item[1][1] / item[1][0], item[1][0]), reverse=True)
        levels[c] = [v for v, (count, _) in ranked_levels[:TOP_LEVELS] if count >= 50]
    return selected, levels


def precision_recall_at_k(y: np.ndarray, scores: np.ndarray) -> tuple[int, float, float]:
    k = max(1, int(np.ceil(len(y) * 0.01)))
    top = np.argsort(scores)[-k:]
    hits = int(y[top].sum())
    return k, hits / k, hits / max(1, int(y.sum()))


def run() -> dict[str, object]:
    target = pd.read_csv(find_dataset_path("train_numeric.csv"), usecols=["Id", "Response"])
    target["Id"] = target["Id"].astype(np.int64); target["Response"] = target["Response"].astype(np.int8)
    train, valid = train_test_split(target, test_size=0.20, stratify=target["Response"], random_state=RANDOM_STATE)
    print("Fitting numeric and categorical selection on training rows only...", flush=True)
    numeric_report = profile_training_fold(set(train["Id"]), len(train))
    numeric_report["missing_rate"] = 1 - numeric_report["present_rate"]
    numeric, numeric_candidates = selected_features(numeric_report, NUMERIC_PRESENT_RATE)
    cats, levels = select_categories_on_train(train)
    (OUT / "controlled_phase6_feature_selection.json").write_text(json.dumps({"numeric": numeric, "categorical": cats, "levels": levels}, indent=2))
    print("Building merged numeric, date/path, and one-hot categorical dataset...", flush=True)
    frame = build_model_dataset(target, numeric, cats, levels, numeric_report, "train").set_index("Id")
    features = [c for c in frame.columns if c != "Response"]
    x_train, x_valid = frame.reindex(train["Id"])[features], frame.reindex(valid["Id"])[features]
    y_train, y_valid = train["Response"].to_numpy(), valid["Response"].to_numpy()
    model = LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=48, subsample=0.85, colsample_bytree=0.85, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=4, verbose=-1)
    started = time.perf_counter(); model.fit(x_train, y_train)
    train_scores, valid_scores = model.predict_proba(x_train)[:, 1], model.predict_proba(x_valid)[:, 1]
    cutoff = validation_threshold(train_scores, y_train); k, precision_k, recall_k = precision_recall_at_k(y_valid, valid_scores)
    result = {"rows": len(target), "train_rows": len(train), "validation_rows": len(valid), "numeric_present_rate": NUMERIC_PRESENT_RATE, "numeric_candidates": numeric_candidates, "numeric_features": len(numeric), "categorical_columns": len(cats), "one_hot_levels": sum(map(len, levels.values())), "final_feature_count": len(features), "mcc": float(matthews_corrcoef(y_valid, valid_scores >= cutoff)), "pr_auc": float(average_precision_score(y_valid, valid_scores)), "precision_at_k": precision_k, "recall_at_k": recall_k, "k": k, "runtime_seconds": time.perf_counter() - started}
    pd.DataFrame([result]).to_csv(OUT / "controlled_phase6_replication_metrics.csv", index=False)
    joblib.dump({"model": model, "feature_cols": features, "threshold": cutoff, "metrics": result}, OUT / "controlled_phase6_lightgbm.joblib")
    return result


if __name__ == "__main__": print(pd.Series(run()).to_string())
