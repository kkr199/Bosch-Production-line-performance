"""Phase 6 predictive failure modeling for the Bosch dataset.

This phase uses raw numeric, categorical, and date-derived train/test inputs.
The full Bosch feature space is too large for a naive all-column one-hot model,
so this script first profiles correlations and missingness, then builds a
model-ready feature set from the strongest numeric, categorical, date, and path
features.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

CHUNKSIZE = 20_000
RANDOM_STATE = 42
NEGATIVE_SAMPLE_SIZE = 220_000
TOP_NUMERIC_FEATURES = 80
TOP_CATEGORICAL_COLUMNS = 20
TOP_CATEGORICAL_VALUES = 12
TOP_FINAL_CORRELATIONS = 150


def find_dataset_path(filename: str) -> Path:
    candidates = [
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "data" / "raw" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in project root or data/raw/")


def read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return next(csv.reader(file))


def ensure_prior_phase_outputs() -> None:
    required = [
        PROCESSED_DIR / "phase4_train_engineered_features.csv",
        PROCESSED_DIR / "phase4_test_engineered_features.csv",
        PROCESSED_DIR / "phase5_train_product_families.csv",
        PROCESSED_DIR / "phase5_test_product_families.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Phase 6 needs Phase 4 date features and Phase 5 product-family labels. Missing: "
            + ", ".join(str(path) for path in missing)
        )


def load_target() -> pd.DataFrame:
    return pd.read_csv(find_dataset_path("train_numeric.csv"), usecols=["Id", "Response"])


def profile_numeric_features(force: bool = False) -> pd.DataFrame:
    output_path = REPORTS_DIR / "phase6_numeric_correlation_report.csv"
    if output_path.exists() and not force:
        return pd.read_csv(output_path)

    numeric_path = find_dataset_path("train_numeric.csv")
    header = [column for column in read_header(numeric_path) if column not in {"Id", "Response"}]
    stats = {
        column: {
            "non_null_count": 0,
            "missing_count": 0,
            "sum_x": 0.0,
            "sum_x2": 0.0,
            "sum_y": 0.0,
            "sum_y2": 0.0,
            "sum_xy": 0.0,
            "missing_y_sum": 0.0,
        }
        for column in header
    }
    total_rows = 0

    reader = pd.read_csv(numeric_path, chunksize=CHUNKSIZE)
    for chunk in tqdm(reader, desc="Profiling numeric correlations", unit="chunk"):
        y = chunk["Response"].astype(np.float64)
        y_sum = float(y.sum())
        y2_sum = float((y * y).sum())
        total_rows += len(chunk)
        for column in header:
            values = chunk[column]
            mask = values.notna()
            non_null = int(mask.sum())
            missing = len(values) - non_null
            stat = stats[column]
            stat["missing_count"] += missing
            stat["missing_y_sum"] += float(y[~mask].sum()) if missing else 0.0
            if non_null == 0:
                continue
            x = values[mask].astype(np.float64)
            y_present = y[mask]
            stat["non_null_count"] += non_null
            stat["sum_x"] += float(x.sum())
            stat["sum_x2"] += float((x * x).sum())
            stat["sum_y"] += float(y_present.sum())
            stat["sum_y2"] += float((y_present * y_present).sum())
            stat["sum_xy"] += float((x * y_present).sum())

    rows = []
    global_response_rate = sum(stat["sum_y"] for stat in stats.values()) / max(
        1, sum(stat["non_null_count"] for stat in stats.values())
    )
    for column, stat in stats.items():
        n = stat["non_null_count"]
        numerator = n * stat["sum_xy"] - stat["sum_x"] * stat["sum_y"]
        denom_x = n * stat["sum_x2"] - stat["sum_x"] ** 2
        denom_y = n * stat["sum_y2"] - stat["sum_y"] ** 2
        value_corr = numerator / math.sqrt(denom_x * denom_y) if denom_x > 0 and denom_y > 0 else np.nan

        missing_rate = stat["missing_count"] / total_rows
        present_rate = 1 - missing_rate
        missing_failure_rate = (
            stat["missing_y_sum"] / stat["missing_count"] if stat["missing_count"] > 0 else np.nan
        )
        present_failure_rate = stat["sum_y"] / n if n > 0 else np.nan
        missing_signal = abs(missing_failure_rate - present_failure_rate) if stat["missing_count"] and n else 0.0

        rows.append(
            {
                "feature": column,
                "non_null_count": n,
                "missing_count": stat["missing_count"],
                "missing_rate": missing_rate,
                "present_rate": present_rate,
                "value_corr_with_response": value_corr,
                "abs_value_corr": abs(value_corr) if pd.notna(value_corr) else 0.0,
                "missing_failure_rate": missing_failure_rate,
                "present_failure_rate": present_failure_rate,
                "missing_signal_abs_diff": missing_signal,
                "global_response_rate_reference": global_response_rate,
            }
        )

    report = pd.DataFrame(rows).sort_values(
        ["abs_value_corr", "missing_signal_abs_diff"], ascending=False
    )
    report.to_csv(output_path, index=False)
    return report


def select_numeric_features(numeric_report: pd.DataFrame) -> list[str]:
    candidates = numeric_report[numeric_report["present_rate"] >= 0.005].copy()
    candidates["selection_score"] = candidates[["abs_value_corr", "missing_signal_abs_diff"]].max(axis=1)
    selected = candidates.sort_values("selection_score", ascending=False).head(TOP_NUMERIC_FEATURES)
    selected.to_csv(REPORTS_DIR / "phase6_selected_numeric_features.csv", index=False)
    return selected["feature"].tolist()


def profile_categorical_presence(target: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    output_path = REPORTS_DIR / "phase6_categorical_presence_report.csv"
    if output_path.exists() and not force:
        return pd.read_csv(output_path)

    categorical_path = find_dataset_path("train_categorical.csv")
    cat_columns = [column for column in read_header(categorical_path) if column != "Id"]
    target_lookup = target.set_index("Id")["Response"]
    stats = {column: {"present_count": 0, "present_y_sum": 0.0, "missing_count": 0, "missing_y_sum": 0.0} for column in cat_columns}

    reader = pd.read_csv(categorical_path, chunksize=8_000, dtype=str)
    for chunk in tqdm(reader, desc="Profiling categorical presence", unit="chunk"):
        y = chunk["Id"].astype(np.int64).map(target_lookup).astype(np.float64)
        for column in cat_columns:
            present = chunk[column].notna()
            present_count = int(present.sum())
            missing_count = len(chunk) - present_count
            stat = stats[column]
            stat["present_count"] += present_count
            stat["missing_count"] += missing_count
            stat["present_y_sum"] += float(y[present].sum()) if present_count else 0.0
            stat["missing_y_sum"] += float(y[~present].sum()) if missing_count else 0.0

    rows = []
    for column, stat in stats.items():
        present_rate = stat["present_count"] / max(1, stat["present_count"] + stat["missing_count"])
        present_failure_rate = stat["present_y_sum"] / stat["present_count"] if stat["present_count"] else np.nan
        missing_failure_rate = stat["missing_y_sum"] / stat["missing_count"] if stat["missing_count"] else np.nan
        signal = abs(present_failure_rate - missing_failure_rate) if pd.notna(missing_failure_rate) else 0.0
        rows.append(
            {
                "feature": column,
                "present_count": stat["present_count"],
                "missing_count": stat["missing_count"],
                "present_rate": present_rate,
                "present_failure_rate": present_failure_rate,
                "missing_failure_rate": missing_failure_rate,
                "presence_signal_abs_diff": signal,
            }
        )
    report = pd.DataFrame(rows).sort_values("presence_signal_abs_diff", ascending=False)
    report.to_csv(output_path, index=False)
    return report


def select_categorical_features(categorical_report: pd.DataFrame) -> list[str]:
    selected = categorical_report[
        (categorical_report["present_count"] >= 200)
        & (categorical_report["present_rate"].between(0.0001, 0.9999))
    ].head(TOP_CATEGORICAL_COLUMNS)
    if len(selected) < TOP_CATEGORICAL_COLUMNS:
        selected = categorical_report[categorical_report["present_count"] >= 200].head(TOP_CATEGORICAL_COLUMNS)
    selected.to_csv(REPORTS_DIR / "phase6_selected_categorical_features.csv", index=False)
    return selected["feature"].tolist()


def learn_categorical_levels(
    target: pd.DataFrame,
    selected_cat: list[str],
    force: bool = False,
) -> dict[str, list[str]]:
    output_path = REPORTS_DIR / "phase6_selected_categorical_levels.json"
    if output_path.exists() and not force:
        return json.loads(output_path.read_text(encoding="utf-8"))

    categorical_path = find_dataset_path("train_categorical.csv")
    target_lookup = target.set_index("Id")["Response"]
    counts: dict[str, dict[str, dict[str, float]]] = {column: {} for column in selected_cat}
    reader = pd.read_csv(categorical_path, usecols=["Id", *selected_cat], chunksize=20_000, dtype=str)
    for chunk in tqdm(reader, desc="Learning categorical levels", unit="chunk"):
        y = chunk["Id"].astype(np.int64).map(target_lookup).astype(np.float64)
        for column in selected_cat:
            values = chunk[column].fillna("__MISSING__")
            grouped = pd.DataFrame({"value": values, "Response": y}).groupby("value")["Response"].agg(["size", "sum"])
            for value, row in grouped.iterrows():
                bucket = counts[column].setdefault(str(value), {"count": 0.0, "failures": 0.0})
                bucket["count"] += float(row["size"])
                bucket["failures"] += float(row["sum"])

    level_map = {}
    for column, value_stats in counts.items():
        ranked = sorted(
            value_stats.items(),
            key=lambda item: (item[1]["failures"] / max(1.0, item[1]["count"]), item[1]["count"]),
            reverse=True,
        )
        level_map[column] = [value for value, stat in ranked[:TOP_CATEGORICAL_VALUES] if stat["count"] >= 50]

    output_path.write_text(json.dumps(level_map, indent=2), encoding="utf-8")
    return level_map


def sample_training_ids(target: pd.DataFrame) -> pd.DataFrame:
    positives = target[target["Response"] == 1]
    negatives = target[target["Response"] == 0].sample(
        n=min(NEGATIVE_SAMPLE_SIZE, int((target["Response"] == 0).sum())),
        random_state=RANDOM_STATE,
    )
    sampled = pd.concat([positives, negatives], axis=0).sample(frac=1, random_state=RANDOM_STATE)
    sampled.to_csv(PROCESSED_DIR / "phase6_sampled_training_ids.csv", index=False)
    return sampled


def load_numeric_features(ids: pd.DataFrame, selected_numeric: list[str], split: str) -> pd.DataFrame:
    path = find_dataset_path(f"{split}_numeric.csv")
    usecols = ["Id", *selected_numeric]
    if split == "train":
        usecols.append("Response")
    dtype_map = {column: "float32" for column in selected_numeric}
    frame = pd.read_csv(path, usecols=usecols, dtype=dtype_map)
    if split == "train":
        frame = ids[["Id", "Response"]].merge(frame.drop(columns=["Response"]), on="Id", how="left")
    return frame


def load_engineered_date_and_path_features(ids: pd.DataFrame, split: str) -> pd.DataFrame:
    phase4 = pd.read_csv(PROCESSED_DIR / f"phase4_{split}_engineered_features.csv")
    drop_cols = ["Response"] if "Response" in phase4.columns else []
    phase4 = phase4.drop(columns=drop_cols)
    phase5_cols = ["Id", "station_count", "line_count", "kmeans_family", "dbscan_family", "hierarchical_family", "final_product_family"]
    phase5 = pd.read_csv(PROCESSED_DIR / f"phase5_{split}_product_families.csv", usecols=phase5_cols)
    frame = phase4.merge(phase5, on="Id", how="left", suffixes=("", "_phase5"))
    if split == "train":
        frame = ids[["Id"]].merge(frame, on="Id", how="left")
    return frame


def load_categorical_ohe(
    ids: pd.DataFrame,
    selected_cat: list[str],
    level_map: dict[str, list[str]],
    split: str,
) -> pd.DataFrame:
    path = find_dataset_path(f"{split}_categorical.csv")
    chunks = []
    id_set = set(ids["Id"].tolist()) if split == "train" else None
    reader = pd.read_csv(path, usecols=["Id", *selected_cat], chunksize=30_000, dtype=str)
    for chunk in tqdm(reader, desc=f"One-hot encoding {split} categoricals", unit="chunk"):
        chunk["Id"] = chunk["Id"].astype(np.int64)
        if id_set is not None:
            chunk = chunk[chunk["Id"].isin(id_set)]
            if chunk.empty:
                continue
        encoded_parts = {"Id": chunk["Id"].to_numpy()}
        for column in selected_cat:
            values = chunk[column].fillna("__MISSING__")
            encoded_parts[f"{column}__is_missing"] = (values == "__MISSING__").astype(np.uint8).to_numpy()
            for level in level_map[column]:
                safe_level = str(level).replace(" ", "_").replace("/", "_")
                encoded_parts[f"{column}__eq_{safe_level}"] = (values == level).astype(np.uint8).to_numpy()
        encoded = pd.DataFrame(encoded_parts)
        chunks.append(encoded)
    if not chunks:
        return pd.DataFrame({"Id": ids["Id"]})
    frame = pd.concat(chunks, ignore_index=True)
    if split == "train":
        frame = ids[["Id"]].merge(frame, on="Id", how="left")
    return frame.fillna(0)


def add_missing_indicators(frame: pd.DataFrame, numeric_report: pd.DataFrame, selected_numeric: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    report = numeric_report.set_index("feature")
    for column in selected_numeric:
        if column in frame.columns and report.loc[column, "missing_rate"] > 0.05:
            frame[f"{column}__is_missing"] = frame[column].isna().astype(np.uint8)
    return frame


def build_model_dataset(
    ids: pd.DataFrame,
    selected_numeric: list[str],
    selected_cat: list[str],
    level_map: dict[str, list[str]],
    numeric_report: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    numeric = load_numeric_features(ids, selected_numeric, split)
    numeric = add_missing_indicators(numeric, numeric_report, selected_numeric)
    engineered = load_engineered_date_and_path_features(ids, split)
    categorical = load_categorical_ohe(ids, selected_cat, level_map, split)
    frame = numeric.merge(engineered, on="Id", how="left").merge(categorical, on="Id", how="left")
    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        frame = frame.loc[:, ~frame.columns.duplicated()]
    return frame


def split_train_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame, valid_frame = train_test_split(
        frame,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=frame["Response"],
    )
    train_frame.to_csv(PROCESSED_DIR / "phase6_train_dataset.csv", index=False)
    valid_frame.to_csv(PROCESSED_DIR / "phase6_validation_dataset.csv", index=False)
    return train_frame, valid_frame


def create_feature_correlation_report(train_frame: pd.DataFrame) -> pd.DataFrame:
    y = train_frame["Response"].astype(float)
    rows = []
    for column in train_frame.columns:
        if column in {"Id", "Response"}:
            continue
        values = pd.to_numeric(train_frame[column], errors="coerce")
        if values.notna().sum() < 50 or values.nunique(dropna=True) <= 1:
            continue
        corr = values.corr(y)
        rows.append(
            {
                "feature": column,
                "corr_with_response": corr,
                "abs_corr_with_response": abs(corr) if pd.notna(corr) else 0.0,
                "missing_rate": values.isna().mean(),
            }
        )
    report = pd.DataFrame(rows).sort_values("abs_corr_with_response", ascending=False)
    report.head(TOP_FINAL_CORRELATIONS).to_csv(REPORTS_DIR / "phase6_final_feature_correlation_report.csv", index=False)
    return report


def best_threshold_metrics(y_true: pd.Series, scores: np.ndarray) -> dict[str, float]:
    thresholds = np.unique(np.quantile(scores, np.linspace(0.50, 0.995, 80)))
    best = None
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        metrics = {
            "threshold": float(threshold),
            "mcc": float(matthews_corrcoef(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        }
        if best is None or metrics["mcc"] > best["mcc"]:
            best = metrics
    assert best is not None
    best["pr_auc"] = float(average_precision_score(y_true, scores))
    return best


def train_and_compare_models(train_frame: pd.DataFrame, valid_frame: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [column for column in train_frame.columns if column not in {"Id", "Response"}]
    x_train = train_frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_train = train_frame["Response"].astype(np.int8)
    x_valid = valid_frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_valid = valid_frame["Response"].astype(np.int8)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(1, pos)

    models = {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=700,
                        class_weight="balanced",
                        solver="saga",
                        n_jobs=-1,
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
                        n_estimators=80,
                        max_depth=14,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        eval_metric="aucpr",
                        scale_pos_weight=scale_pos_weight,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.04,
                        num_leaves=48,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        ),
        "CatBoost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostClassifier(
                        iterations=250,
                        depth=6,
                        learning_rate=0.05,
                        loss_function="Logloss",
                        eval_metric="PRAUC",
                        auto_class_weights="Balanced",
                        random_seed=RANDOM_STATE,
                        verbose=False,
                    ),
                ),
            ]
        ),
    }

    metrics_rows = []
    fitted_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_valid)[:, 1]
        metrics = best_threshold_metrics(y_valid, scores)
        metrics_rows.append({"model": name, **metrics})
        fitted_models[name] = model
        joblib.dump({"model": model, "feature_cols": feature_cols}, MODELS_DIR / f"phase6_{name.lower().replace(' ', '_')}.joblib")

    metrics_report = pd.DataFrame(metrics_rows).sort_values(["mcc", "pr_auc"], ascending=False)
    metrics_report["rank"] = range(1, len(metrics_report) + 1)
    metrics_report.to_csv(REPORTS_DIR / "phase6_model_comparison_metrics.csv", index=False)

    best_name = str(metrics_report.iloc[0]["model"])
    joblib.dump(
        {
            "model": fitted_models[best_name],
            "feature_cols": feature_cols,
            "best_model_name": best_name,
            "best_threshold": float(metrics_report.iloc[0]["threshold"]),
        },
        MODELS_DIR / "phase6_best_model.joblib",
    )
    return metrics_report


def score_kaggle_test_dataset(
    selected_numeric: list[str],
    selected_cat: list[str],
    level_map: dict[str, list[str]],
    numeric_report: pd.DataFrame,
) -> Path:
    test_ids = pd.read_csv(find_dataset_path("test_numeric.csv"), usecols=["Id"])
    test_frame = build_model_dataset(
        ids=test_ids,
        selected_numeric=selected_numeric,
        selected_cat=selected_cat,
        level_map=level_map,
        numeric_report=numeric_report,
        split="test",
    )
    test_frame.head(10_000).to_csv(PROCESSED_DIR / "phase6_test_dataset_preview.csv", index=False)

    best_bundle = joblib.load(MODELS_DIR / "phase6_best_model.joblib")
    feature_cols = best_bundle["feature_cols"]
    model = best_bundle["model"]
    threshold = float(best_bundle["best_threshold"])

    for column in feature_cols:
        if column not in test_frame.columns:
            test_frame[column] = 0
    scores = model.predict_proba(test_frame[feature_cols].replace([np.inf, -np.inf], np.nan))[:, 1]
    predictions = pd.DataFrame(
        {
            "Id": test_frame["Id"].astype(np.int64),
            "failure_probability": scores,
            "predicted_failure_at_selected_threshold": (scores >= threshold).astype(np.uint8),
        }
    )
    output_path = PROCESSED_DIR / "phase6_test_predictions.csv"
    predictions.to_csv(output_path, index=False)
    return output_path


def write_report(
    sampled_ids: pd.DataFrame,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    final_corr: pd.DataFrame,
    metrics_report: pd.DataFrame,
    test_predictions_path: Path,
) -> Path:
    report_path = REPORTS_DIR / "phase6_predictive_failure_modeling_report.md"
    best = metrics_report.iloc[0]
    lines = [
        "# Phase 6: Predictive Failure Modeling",
        "",
        "## Source Data Used",
        "",
        "- Raw `train_numeric.csv` and `test_numeric.csv` for numeric features and the train target.",
        "- Raw `train_categorical.csv` and `test_categorical.csv` for selected one-hot categorical features.",
        "- Raw-date-derived Phase 4 timing/path features from `train_date.csv` and `test_date.csv`.",
        "- Phase 5 product-family labels derived from raw station presence matrices.",
        "",
        "## Dataset Construction",
        "",
        f"- Sampled modeling rows: {len(sampled_ids):,}",
        f"- Train rows: {len(train_frame):,}",
        f"- Validation rows: {len(valid_frame):,}",
        f"- Kaggle test rows scored: {sum(1 for _ in test_predictions_path.open(encoding='utf-8')) - 1:,}",
        f"- Positive failures in sampled rows: {int(sampled_ids['Response'].sum()):,}",
        f"- Final model feature count: {len([c for c in train_frame.columns if c not in {'Id', 'Response'}]):,}",
        "",
        "Missing values were handled in two ways: important missingness was kept as explicit `__is_missing` features, while model input values were median-imputed inside each modeling pipeline.",
        "",
        "## Correlation Before Modeling",
        "",
        "The pipeline writes raw numeric, raw categorical-presence, and final model-feature correlation reports before training models.",
        "",
        final_corr.head(15).to_markdown(index=False),
        "",
        "## Model Comparison",
        "",
        metrics_report.to_markdown(index=False),
        "",
        "## Evaluation Status",
        "",
        "This run uses a validation split for model selection. The Kaggle `test_*.csv` files are unlabelled scoring inputs, not a held-out evaluation set. Per `docs/data_strategy_and_test_set_policy.md`, these metrics are experimental and must not be presented as a final production-performance estimate until a physically isolated labelled holdout is evaluated exactly once.",
        "",
        "## Selected Model",
        "",
        (
            f"The selected model is **{best['model']}**, ranked by MCC first and PR-AUC second. "
            f"It reached MCC {best['mcc']:.4f}, precision {best['precision']:.4f}, recall {best['recall']:.4f}, "
            f"F1 {best['f1']:.4f}, and PR-AUC {best['pr_auc']:.4f} on the validation split."
        ),
        "",
        "## Output Files",
        "",
        "- `data/processed/phase6_sampled_training_ids.csv`",
        "- `data/processed/phase6_train_dataset.csv`",
        "- `data/processed/phase6_validation_dataset.csv`",
        "- `data/processed/phase6_test_dataset_preview.csv`",
        "- `data/processed/phase6_test_predictions.csv`",
        "- `reports/phase6_numeric_correlation_report.csv`",
        "- `reports/phase6_categorical_presence_report.csv`",
        "- `reports/phase6_selected_numeric_features.csv`",
        "- `reports/phase6_selected_categorical_features.csv`",
        "- `reports/phase6_final_feature_correlation_report.csv`",
        "- `reports/phase6_model_comparison_metrics.csv`",
        "- `models/phase6_best_model.joblib`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_prior_phase_outputs()

    target = load_target()
    sampled_ids = sample_training_ids(target)

    numeric_report = profile_numeric_features()
    selected_numeric = select_numeric_features(numeric_report)

    categorical_report = profile_categorical_presence(target)
    selected_cat = select_categorical_features(categorical_report)
    level_map = learn_categorical_levels(target, selected_cat)

    train_path = PROCESSED_DIR / "phase6_train_dataset.csv"
    valid_path = PROCESSED_DIR / "phase6_validation_dataset.csv"
    metrics_path = REPORTS_DIR / "phase6_model_comparison_metrics.csv"
    best_model_path = MODELS_DIR / "phase6_best_model.joblib"

    if train_path.exists() and valid_path.exists():
        train_frame = pd.read_csv(train_path)
        valid_frame = pd.read_csv(valid_path)
    else:
        model_frame = build_model_dataset(
            ids=sampled_ids,
            selected_numeric=selected_numeric,
            selected_cat=selected_cat,
            level_map=level_map,
            numeric_report=numeric_report,
            split="train",
        )
        train_frame, valid_frame = split_train_validation(model_frame)

    final_corr_path = REPORTS_DIR / "phase6_final_feature_correlation_report.csv"
    if final_corr_path.exists():
        final_corr = pd.read_csv(final_corr_path)
    else:
        final_corr = create_feature_correlation_report(train_frame)

    if metrics_path.exists() and best_model_path.exists():
        metrics_report = pd.read_csv(metrics_path)
    else:
        metrics_report = train_and_compare_models(train_frame, valid_frame)
    test_predictions_path = score_kaggle_test_dataset(
        selected_numeric=selected_numeric,
        selected_cat=selected_cat,
        level_map=level_map,
        numeric_report=numeric_report,
    )
    report_path = write_report(sampled_ids, train_frame, valid_frame, final_corr, metrics_report, test_predictions_path)
    print(f"Phase 6 complete. Report written to {report_path}")


if __name__ == "__main__":
    main()
